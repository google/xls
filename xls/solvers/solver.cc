// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/solvers/solver.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/query_engine.h"

namespace xls::solvers {

/* static */ Predicate Predicate::IsEqualTo(Node* other) {
  return Predicate(PredicateKind::kEqualToNode, other);
}
/* static */ Predicate Predicate::IsExclusiveWith(Node* other) {
  return Predicate(PredicateKind::kExclusiveWithNode, other);
}
/* static */ Predicate Predicate::EqualToZero() {
  return Predicate(PredicateKind::kEqualToZero);
}
/* static */ Predicate Predicate::NotEqualToZero() {
  return Predicate(PredicateKind::kNotEqualToZero);
}
/* static */ Predicate Predicate::UnsignedGreaterOrEqual(Bits lower_bound) {
  return Predicate(PredicateKind::kUnsignedGreaterOrEqual, nullptr,
                   std::move(lower_bound));
}
/* static */ Predicate Predicate::UnsignedLessOrEqual(Bits upper_bound) {
  return Predicate(PredicateKind::kUnsignedLessOrEqual, nullptr,
                   std::move(upper_bound));
}
/* static */ Predicate Predicate::IsCompatibleWith(
    SharedTernaryTree ternaries) {
  return Predicate(PredicateKind::kCompatibleWithTernary, std::move(ternaries));
}
/* static */ Predicate Predicate::IsCompatibleWith(IntervalSetTree intervals) {
  return Predicate(PredicateKind::kCompatibleWithIntervalSet, intervals);
}

Predicate::Predicate(PredicateKind kind) : kind_(kind) {}

Predicate::Predicate(PredicateKind kind, Node* node)
    : kind_(kind), node_(node) {}

Predicate::Predicate(PredicateKind kind, Node* node, Bits value)
    : kind_(kind), node_(node), value_(std::move(value)) {}

Predicate::Predicate(PredicateKind kind, SharedTernaryTree ternaries)
    : kind_(kind), ternaries_(std::move(ternaries).ToOwned()) {}

Predicate::Predicate(PredicateKind kind, IntervalSetTree intervals)
    : kind_(kind), intervals_(std::move(intervals)) {}

Node* Predicate::node() const {
  CHECK(node_.has_value());
  return node_.value();
}

const Bits& Predicate::value() const {
  CHECK(value_.has_value());
  return value_.value();
}

TernaryTreeView Predicate::ternaries() const {
  CHECK(ternaries_.has_value());
  return ternaries_->AsView();
}

IntervalSetTreeView Predicate::intervals() const {
  CHECK(intervals_.has_value());
  return intervals_->AsView();
}

std::string Predicate::ToString() const {
  switch (kind_) {
    case PredicateKind::kEqualToZero:
      return "eq zero";
    case PredicateKind::kNotEqualToZero:
      return "ne zero";
    case PredicateKind::kEqualToNode:
      return absl::StrCat("eq ", node()->GetName());
    case PredicateKind::kExclusiveWithNode:
      return absl::StrCat("nand ", node()->GetName());
    case PredicateKind::kUnsignedGreaterOrEqual:
      return absl::StrCat("uge ", value_->ToDebugString());
    case PredicateKind::kUnsignedLessOrEqual:
      return absl::StrCat("ule ", value_->ToDebugString());
    case PredicateKind::kCompatibleWithTernary:
      return absl::StrCat(
          "compatible with ",
          ternaries_->ToString([](const TernaryVector& ternary) {
            return ::xls::ToString(ternary);
          }));
    case PredicateKind::kCompatibleWithIntervalSet:
      return absl::StrCat("compatible with ",
                          intervals_->ToString([](const IntervalSet& interval) {
                            return interval.ToString();
                          }));
  }
  return absl::StrFormat("<invalid predicate kind %d>",
                         static_cast<int>(kind_));
}

SolverFactoryRegistry& SolverFactoryRegistry::Get() {
  static auto* registry = new SolverFactoryRegistry();
  return *registry;
}

void SolverFactoryRegistry::Register(SolverKind kind, Factory factory) {
  factories_[kind] = std::move(factory);
}

absl::StatusOr<std::unique_ptr<Solver>> SolverFactoryRegistry::Create(
    SolverKind kind) const {
  auto it = factories_.find(kind);
  if (it == factories_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Solver implementation not registered for kind: %d",
                        static_cast<int>(kind)));
  }
  return it->second();
}

absl::StatusOr<std::unique_ptr<Solver>> CreateSolver(SolverKind kind) {
  return SolverFactoryRegistry::Get().Create(kind);
}

std::optional<std::vector<Node*>> CounterExampleAnnotator::NodeOrder(
    FunctionBase* fb) const {
  absl::StatusOr<std::vector<Node*>> topo_sort_nodes = TopoSort(fb);
  CHECK_OK(topo_sort_nodes);
  return *topo_sort_nodes;
}

std::optional<Value> CounterExampleAnnotator::CounterExampleValue(
    Node* node) const {
  auto it = absl::c_find_if(*fail_.counterexample, [&](const auto& pair) {
    const Node* counter_node = pair.first;
    return node->GetName() == counter_node->GetName();
  });
  if (it == fail_.counterexample->end()) {
    if (node->OpIn({Op::kParam, Op::kRegisterRead, Op::kStateRead, Op::kReceive,
                    Op::kInputPort, Op::kInstantiationInput})) {
      VLOG(1) << "No counterexample info found for input: " << node->GetName();
      return ZeroOfType(node->GetType());
    }
    return std::nullopt;
  }
  return it->second;
}

Annotation CounterExampleAnnotator::NodeAnnotation(Node* node) const {
  if (!fail_.counterexample.ok()) {
    return {};
  }
  auto paren = [](std::string_view sv) -> std::string {
    return absl::StrCat("(", sv, ")");
  };
  if (auto counter = CounterExampleValue(node); counter.has_value()) {
    counterexamples_[node] = *counter;
    return Annotation{.suffix = paren(counter->ToHumanString(format_))};
  }
  InterpreterEvents events;
  IrInterpreter interpreter(&counterexamples_, &events);
  if (!node->VisitSingleNode(&interpreter).ok()) {
    return {};
  }
  return Annotation{
      .suffix = paren(interpreter.ResolveAsValue(node).ToHumanString(format_))};
}

bool AbslParseFlag(std::string_view text, SolverKind* solver_kind,
                   std::string* error) {
  if (text == "unspecified" || text == "kUnspecified" ||
      text == "UNSPECIFIED") {
    *solver_kind = SolverKind::kUnspecified;
    return true;
  }
  if (text == "z3" || text == "kZ3" || text == "Z3") {
    *solver_kind = SolverKind::kZ3;
    return true;
  }
  if (text == "bitwuzla" || text == "kBitwuzla" || text == "BITWUZLA") {
    *solver_kind = SolverKind::kBitwuzla;
    return true;
  }
  *error = absl::StrCat("Unknown SolverKind: ", text);
  return false;
}

std::string AbslUnparseFlag(const SolverKind& solver_kind) {
  switch (solver_kind) {
    case SolverKind::kUnspecified:
      return "unspecified";
    case SolverKind::kZ3:
      return "z3";
    case SolverKind::kBitwuzla:
      return "bitwuzla";
  }
  return absl::StrCat(static_cast<int>(solver_kind));
}

absl::StatusOr<ProverResult> TryProve(
    FunctionBase* f, Node* subject, const Predicate& p,
    const SolverLimit& limit, bool allow_unsupported,
    absl::Span<const PredicateOfNode> assumptions, SolverKind kind) {
  XLS_ASSIGN_OR_RETURN(auto solver, CreateSolver(kind));
  return solver->TryProve(f, subject, p, limit, allow_unsupported, assumptions);
}

absl::StatusOr<ProverResult> TryProveCombination(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms,
    PredicateCombination combination, const SolverLimit& limit,
    bool allow_unsupported, absl::Span<const PredicateOfNode> assumptions,
    SolverKind kind) {
  XLS_ASSIGN_OR_RETURN(auto solver, CreateSolver(kind));
  return solver->TryProveCombination(f, terms, combination, limit,
                                     allow_unsupported, assumptions);
}

absl::StatusOr<ProverResult> TryProveDisjunction(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms,
    const SolverLimit& limit, bool allow_unsupported,
    absl::Span<const PredicateOfNode> assumptions, SolverKind kind) {
  return TryProveCombination(f, terms, PredicateCombination::kDisjunction,
                             limit, allow_unsupported, assumptions, kind);
}

absl::StatusOr<ProverResult> TryProveConjunction(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms,
    const SolverLimit& limit, bool allow_unsupported,
    absl::Span<const PredicateOfNode> assumptions, SolverKind kind) {
  return TryProveCombination(f, terms, PredicateCombination::kConjunction,
                             limit, allow_unsupported, assumptions, kind);
}

}  // namespace xls::solvers

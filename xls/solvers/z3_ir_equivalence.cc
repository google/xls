// Copyright 2023 The XLS Authors
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

#include "xls/solvers/z3_ir_equivalence.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {

namespace {

std::optional<std::string> GetAssertKey(const Assert* asrt) {
  if (asrt->label().has_value()) {
    return asrt->label();
  }
  if (asrt->original_label().has_value()) {
    return asrt->original_label();
  }
  return std::nullopt;
}

absl::StatusOr<std::vector<Node*>> BuildAssertChecks(
    Function* f, const absl::flat_hash_map<Node*, Node*>& cloned_from_b) {
  absl::flat_hash_map<std::string, Assert*> original_asserts;
  absl::flat_hash_map<std::string, Assert*> transformed_asserts;
  absl::flat_hash_set<Node*> b_nodes;
  b_nodes.reserve(cloned_from_b.size());
  for (const auto& [_, clone] : cloned_from_b) {
    b_nodes.insert(clone);
  }

  for (Node* node : f->nodes()) {
    if (!node->Is<Assert>()) {
      continue;
    }
    if (b_nodes.contains(node)) {
      continue;
    }
    std::optional<std::string> key = GetAssertKey(node->As<Assert>());
    if (!key.has_value()) {
      continue;
    }
    if (!original_asserts.emplace(*key, node->As<Assert>()).second) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Duplicate assert label `%s` in original function", *key));
    }
  }

  for (const auto& [original_node, clone_node] : cloned_from_b) {
    if (!original_node->Is<Assert>()) {
      continue;
    }
    std::optional<std::string> key = GetAssertKey(original_node->As<Assert>());
    if (!key.has_value()) {
      continue;
    }
    if (!transformed_asserts.emplace(*key, clone_node->As<Assert>()).second) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Duplicate assert label `%s` in transformed function", *key));
    }
  }

  std::vector<Node*> checks;
  checks.reserve(original_asserts.size() + transformed_asserts.size());
  for (const auto& [label, original_assert] : original_asserts) {
    auto transformed_it = transformed_asserts.find(label);
    if (transformed_it != transformed_asserts.end()) {
      XLS_ASSIGN_OR_RETURN(Node * equality,
                           f->MakeNodeWithName<CompareOp>(
                               SourceInfo(), original_assert->condition(),
                               transformed_it->second->condition(), Op::kEq,
                               absl::StrCat("assert_", label, "_match")));
      checks.push_back(equality);
      transformed_asserts.erase(transformed_it);
    } else {
      checks.push_back(original_assert->condition());
    }
  }
  for (const auto& [_, transformed_assert] : transformed_asserts) {
    checks.push_back(transformed_assert->condition());
  }
  return checks;
}

absl::StatusOr<Node*> CombineChecks(Function* f,
                                    absl::Span<Node* const> checks) {
  XLS_RET_CHECK(!checks.empty());
  if (checks.size() == 1) {
    return checks.front();
  }
  return f->MakeNodeWithName<NaryOp>(SourceInfo(), checks, Op::kAnd,
                                     "z3_equivalence_checks");
}

class RemoveAssertsPass : public OptimizationFunctionBasePass {
 public:
  RemoveAssertsPass()
      : OptimizationFunctionBasePass(
            "remove_asserts", "remove asserts for z3 equivalence checking") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& ctx) const override {
    bool changed = false;
    for (Node* n : ctx.TopoSort(f)) {
      if (n->Is<Assert>()) {
        XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
        XLS_RETURN_IF_ERROR(f->RemoveNode(n));
        changed = true;
      }
    }
    return changed;
  }
};
}  // namespace
absl::StatusOr<ProverResult> TryProveEquivalence(Function* a, Function* b,
                                                 bool ignore_asserts,
                                                 absl::Duration timeout) {
  std::unique_ptr<Package> to_test = std::make_unique<Package>(
      absl::StrFormat("%s_tester", a->package()->name()));
  XLS_ASSIGN_OR_RETURN(
      Function * to_test_func,
      a->Clone(absl::StrFormat("%s_test", a->name()), to_test.get()));

  if (!a->return_value()->GetType()->IsEqualTo(b->return_value()->GetType())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot prove equivalence of functions with differing "
                        "return types: %s vs %s",
                        a->return_value()->GetType()->ToString(),
                        b->return_value()->GetType()->ToString()));
  }

  if (a->params().size() != b->params().size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot prove equivalence of functions with differing "
                        "numbers of parameters: %d vs %d",
                        a->params().size(), b->params().size()));
  }

  for (int64_t i = 0; i < a->params().size(); ++i) {
    if (!a->params()[i]->GetType()->IsEqualTo(b->params()[i]->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot prove equivalence of functions with differing "
          "parameter %d types: %s vs %s",
          i, a->params()[i]->GetType()->ToString(),
          b->params()[i]->GetType()->ToString()));
    }
  }

  // Patch b into to_test. Wire up parameters to those at the same index in the
  // to_test_function.  We do this so we can test whether the two functions are
  // semantically equivalent by making a single Z3-AST function and checking a
  // single eq node's value.
  absl::flat_hash_map<Node*, Node*> node_map;
  for (Node* n : TopoSort(b)) {
    if (n->Is<Param>()) {
      XLS_ASSIGN_OR_RETURN(int64_t index, b->GetParamIndex(n->As<Param>()));
      node_map[n] = to_test_func->param(index);
      continue;
    }
    std::vector<Node*> new_ops;
    new_ops.reserve(n->operand_count());
    for (Node* op : n->operands()) {
      new_ops.push_back(node_map[op]);
    }
    XLS_ASSIGN_OR_RETURN(node_map[n],
                         n->CloneInNewFunction(new_ops, to_test_func));
  }

  // Add check

  Node* original_result = to_test_func->return_value();
  Node* transformed_result = node_map[b->return_value()];
  std::vector<Node*> checks;
  XLS_ASSIGN_OR_RETURN(Node * result_compare,
                       to_test_func->MakeNodeWithName<CompareOp>(
                           SourceInfo(), original_result, transformed_result,
                           Op::kEq, "TestCheck"));
  checks.push_back(result_compare);
  if (!ignore_asserts) {
    XLS_ASSIGN_OR_RETURN(auto assert_checks,
                         BuildAssertChecks(to_test_func, node_map));
    checks.insert(checks.end(), assert_checks.begin(), assert_checks.end());
  }
  XLS_ASSIGN_OR_RETURN(Node * new_ret, CombineChecks(to_test_func, checks));
  XLS_RETURN_IF_ERROR(to_test_func->set_return_value(new_ret));
  // Remove asserts prior to Z3 translation (the solver does not understand
  // them yet). If assert semantics are being checked, those checks have been
  // encoded into the return value already.
  OptimizationContext ctx;
  PassResults res;
  RemoveAssertsPass rap;
  XLS_RETURN_IF_ERROR(rap.Run(to_test.get(), {}, &res, ctx).status())
      << "Unable to remove asserts from function!";
  // Run prover
  XLS_ASSIGN_OR_RETURN(
      ProverResult base_result,
      TryProve(to_test_func, new_ret, Predicate::NotEqualToZero(), timeout));
  // remap parameters back tot he originals.
  return std::visit(
      Visitor{
          [](ProvenTrue t) -> absl::StatusOr<ProverResult> { return t; },
          [&](ProvenFalse f) -> absl::StatusOr<ProverResult> {
            if (f.counterexample.ok()) {
              absl::flat_hash_map<const Param*, Value> mapped_counterexample;
              for (const auto& [param, value] : *f.counterexample) {
                if (node_map.contains(param)) {
                  // from 'b'
                  mapped_counterexample[node_map[const_cast<Param*>(param)]
                                            ->As<Param>()] = value;
                } else {
                  // from 'a'
                  XLS_ASSIGN_OR_RETURN(
                      int64_t idx,
                      to_test_func->GetParamIndex(const_cast<Param*>(param)));
                  mapped_counterexample[a->param(idx)] = value;
                }
              }
              f.counterexample = mapped_counterexample;
            }
            return f;
          },
      },
      std::move(base_result));
}

absl::StatusOr<ProverResult> TryProveEquivalence(
    Function* original,
    const std::function<absl::Status(Package*, Function*)>& run_pass,
    absl::Duration timeout) {
  std::unique_ptr<Package> to_transform = std::make_unique<Package>(
      absl::StrFormat("%s_copy", original->package()->name()));
  XLS_ASSIGN_OR_RETURN(
      Function * to_transform_func,
      original->Clone(absl::StrFormat("%s_copy", original->name()),
                      to_transform.get()));
  XLS_RETURN_IF_ERROR(run_pass(to_transform.get(), to_transform_func));

  return TryProveEquivalence(original, to_transform_func, timeout);
}

}  // namespace xls::solvers::z3

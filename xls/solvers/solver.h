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

#ifndef XLS_SOLVERS_SOLVER_H_
#define XLS_SOLVERS_SOLVER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"
#include "xls/passes/query_engine.h"

namespace xls::solvers {

// Unified configuration for solver timeout and deterministic step limits.
struct SolverLimit {
  std::optional<absl::Duration> timeout;
  std::optional<int64_t>
      deterministic_limit;  // Maps to Z3 rlimit, CP-SAT conflict limits
};

// Supported Solver Backends
enum class SolverKind : uint8_t {
  kUnspecified,
  kZ3,
};

bool AbslParseFlag(std::string_view text, SolverKind* solver_kind,
                   std::string* error);

std::string AbslUnparseFlag(const SolverKind& solver_kind);

// Kinds of predicates we can compute about a subject node.
enum class PredicateKind : uint8_t {
  kEqualToZero,                // subject is zero
  kNotEqualToZero,             // subject is not zero
  kEqualToNode,                // subject and node are equal
  kExclusiveWithNode,          // at least one of subject and node is zero
  kUnsignedGreaterOrEqual,     // subject >= some given (constant) value
  kUnsignedLessOrEqual,        // subject <= some given (constant) value
  kCompatibleWithTernary,      // subject is compatible with given ternaries
  kCompatibleWithIntervalSet,  // subject is compatible with given interval sets
};

// Describes a predicate to compute about a subject node in an XLS IR function.
//
// Note: predicates currently implicitly refer to an (unreferenced) subject,
// like a return value, so you can make fairly context-free constructs like
// `Predicate::EqualToZero()`. (See `PredicateOfNode` for ways to explicitly
// provide a subject node for the predicate to act upon.)
class Predicate {
 public:
  // Returns a predicate that is true iff the subject is equal to `other`.
  static Predicate IsEqualTo(Node* other);

  // Returns a predicate that is true iff the subject is never true at the same
  // time as `other`.
  static Predicate IsExclusiveWith(Node* other);

  // Returns a predicate that is true iff the subject is compatible with the
  // given ternary; i.e., where the leaf ternaries have known bits, the subject
  // matches those bits.
  static Predicate IsCompatibleWith(SharedTernaryTree ternaries);

  // Returns a predicate that is true iff the subject is compatible with the
  // given interval sets; i.e., the value of each leaf is contained in the
  // corresponding interval set.
  static Predicate IsCompatibleWith(IntervalSetTree intervals);

  static Predicate EqualToZero();
  static Predicate NotEqualToZero();
  static Predicate UnsignedGreaterOrEqual(Bits lower_bound);
  static Predicate UnsignedLessOrEqual(Bits upper_bound);

  PredicateKind kind() const { return kind_; }

  // For predicates that refer to another node; e.g.
  // `Predicate::IsEqualTo(other)`, returns the node the predicate is comparing
  // to (`other` in this example).
  Node* node() const;

  // For predicates that have a bits value as part of the predicate payload,
  // returns the bits value; e.g. for
  // `Predicate::UnsignedGreaterOrEqual(my_bits)` returns the value of
  // `my_bits`.
  const Bits& value() const;

  // For predicates that have a ternary tree as part of the predicate payload
  // (e.g., `Predicate::IsCompatibleWith(ternary)`), returns the ternary tree
  // (`ternary` in this example).
  TernaryTreeView ternaries() const;

  // For predicates that have an interval set tree as part of the
  // predicate payload (e.g., `Predicate::IsCompatibleWith(intervals)`), returns
  // the interval set tree view (`intervals` in this example).
  IntervalSetTreeView intervals() const;

  std::string ToString() const;

 private:
  explicit Predicate(PredicateKind kind);
  Predicate(PredicateKind kind, Node* node);
  Predicate(PredicateKind kind, Node* node, Bits value);
  Predicate(PredicateKind kind, SharedTernaryTree ternaries);
  Predicate(PredicateKind kind, IntervalSetTree intervals);

  PredicateKind kind_;
  std::optional<Node*> node_;
  std::optional<Bits> value_;
  std::optional<TernaryTree> ternaries_;
  std::optional<IntervalSetTree> intervals_;
};

// Predicates generally don't encode a subject, they say things like "should be
// greater than zero" but the subject is implicit, e.g. a return value.
//
// This struct wraps a predicate with an explicit subject (that must be present
// inside of the associated function).
struct PredicateOfNode {
  Node* subject;
  Predicate p;
};

enum class PredicateCombination : uint8_t { kDisjunction, kConjunction };

using ProvenTrue = std::true_type;
struct ProvenFalse {
  // If available, a set of Values for the function's Params that implement the
  // counterexample; otherwise, an absl::Status documenting the failure to
  // translate the counterexample.
  absl::StatusOr<absl::flat_hash_map<Node*, Value>> counterexample =
      absl::UnimplementedError("no counterexample analysis attempted");

  // A message from the solver; usually a direct rendering of the solver
  // response.
  std::string message;
};
using ProverResult = std::variant<ProvenTrue, ProvenFalse>;

template <typename Sink>
void AbslStringify(Sink& sink, const ProverResult& p) {
  if (std::holds_alternative<ProvenTrue>(p)) {
    absl::Format(&sink, "[ProvenTrue]");
    return;
  }
  absl::Format(&sink, "[ProvenFalse: %s]", std::get<ProvenFalse>(p).message);
}

// Stateful, pre-translated solver instance bound to an XLS FunctionBase.
class SolverInstance {
 public:
  virtual ~SolverInstance() = default;

  virtual void SetLimit(const SolverLimit& limit) = 0;

  // Proves a node property on the pre-translated function.
  virtual absl::StatusOr<ProverResult> TryProve(
      Node* subject, const Predicate& p,
      absl::Span<const PredicateOfNode> assumptions = {}) = 0;

  // Proves a combination of properties on the pre-translated function.
  virtual absl::StatusOr<ProverResult> TryProveCombination(
      absl::Span<const PredicateOfNode> terms, PredicateCombination combination,
      absl::Span<const PredicateOfNode> assumptions = {}) = 0;

  // Returns solver-specific statistics if available.
  virtual absl::flat_hash_map<std::string, int64_t> GetStats() const {
    return {};
  }
};

// Generic solver runner representing a concrete backend (Z3, CP-SAT).
class Solver {
 public:
  virtual ~Solver() = default;

  virtual SolverKind kind() const = 0;

  // Creates a stateful instance that caches the translated function.
  virtual absl::StatusOr<std::unique_ptr<SolverInstance>> CreateSolverInstance(
      FunctionBase* f, bool allow_unsupported = false) = 0;

  // High-level, one-off helper functions.
  virtual absl::StatusOr<ProverResult> TryProve(
      FunctionBase* f, Node* subject, const Predicate& p,
      const SolverLimit& limit, bool allow_unsupported = false,
      absl::Span<const PredicateOfNode> assumptions = {}) = 0;

  virtual absl::StatusOr<ProverResult> TryProveCombination(
      FunctionBase* f, absl::Span<const PredicateOfNode> terms,
      PredicateCombination combination, const SolverLimit& limit,
      bool allow_unsupported = false,
      absl::Span<const PredicateOfNode> assumptions = {}) = 0;
};

// Decoupled Registry keeping Solver free of concrete linker references.
class SolverFactoryRegistry {
 public:
  using Factory = std::function<absl::StatusOr<std::unique_ptr<Solver>>()>;

  static SolverFactoryRegistry& Get();

  // Register is not thread-safe and is designed exclusively for
  // single-threaded static initialization (e.g., via static global variables).
  void Register(SolverKind kind, Factory factory);
  absl::StatusOr<std::unique_ptr<Solver>> Create(SolverKind kind) const;

 private:
  SolverFactoryRegistry() = default;
  absl::flat_hash_map<SolverKind, Factory> factories_;
};

// Public Creator Entry Point
absl::StatusOr<std::unique_ptr<Solver>> CreateSolver(SolverKind kind);

// Annotator that prints counterexample values for nodes in an IR entity;
// supports functions, procs, and blocks.
class CounterExampleAnnotator : public IrAnnotator {
 public:
  explicit CounterExampleAnnotator(
      const ProvenFalse& fail,
      FormatPreference format = FormatPreference::kDefault)
      : fail_(fail), format_(format) {}

  std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const override;
  Annotation NodeAnnotation(Node* node) const override;

  // Takes a node and if there is an explicit counter example value for it,
  // returns that value. By default returns nullopt for everything except Param
  // nodes where it returns the counterexample value of any param with the same
  // name or ZeroOfType if not available.
  virtual std::optional<Value> CounterExampleValue(Node* node) const;

 private:
  const ProvenFalse& fail_;
  FormatPreference format_;
  mutable absl::flat_hash_map<Node*, Value> counterexamples_;
};

}  // namespace xls::solvers

#endif  // XLS_SOLVERS_SOLVER_H_

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

#ifndef XLS_PASSES_CONTEXT_SENSITIVE_RANGE_QUERY_ENGINE_H_
#define XLS_PASSES_CONTEXT_SENSITIVE_RANGE_QUERY_ENGINE_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"

namespace xls {

// Generate range information which includes reasoning about the values given
// particular select branches have been chosen.
//
// By default this acts just like a normal RangeQueryEngine. The specialized
// information must be directly asked for.
//
// This works by enumerating all select cases, extracting range information
// given their selector is at the appropriate value and propagating that down
// for each single case. This means the engine is only able to provide
// information for a single case at a time.
class ContextSensitiveRangeQueryEngine final : public QueryEngine {
 public:
  ContextSensitiveRangeQueryEngine() = default;

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return base_case_ranges_.GetIntervals(node);
  }

  LeafTypeTree<IntervalSet> GetIntervalsGivenPredicates(
      Node* node, const absl::flat_hash_set<PredicateState>& state) const {
    return SpecializeGivenPredicate(state)->GetIntervals(node);
  }

  bool IsTracked(Node* node) const override {
    return base_case_ranges_.IsTracked(node);
  }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    return base_case_ranges_.GetTernary(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return base_case_ranges_.AtMostOneTrue(bits);
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return base_case_ranges_.AtLeastOneTrue(bits);
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return base_case_ranges_.KnownEquals(a, b);
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return base_case_ranges_.KnownNotEquals(a, b);
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return base_case_ranges_.Implies(a, b);
  }

  // NB This is implemented by BDD engine. The meaning of predicate_bit_values
  // here is literally just specific bits in various nodes each having a
  // particular value. The naming collision with PredicateState (ie the select
  // cases that are active) is unfortunate.
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return base_case_ranges_.ImpliedNodeValue(predicate_bit_values, node);
  }

  // NB This is implemented by BDD engine. The meaning of predicate_bit_values
  // here is literally just specific bits in various nodes each having a
  // particular value. The naming collision with PredicateState (ie the select
  // cases that are active) is unfortunate.
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return base_case_ranges_.ImpliedNodeTernary(predicate_bit_values, node);
  }

  // Specialize the query engine for the given predicate. For now only a state
  // set with a single element is supported. This is CHECK'd internally to avoid
  // surprising non-deterministic behavior. In the future we might relax this
  // restriction.
  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override;

 private:
  RangeQueryEngine base_case_ranges_;
  std::vector<std::unique_ptr<const RangeQueryEngine>> arena_;
  absl::flat_hash_map<PredicateState, const RangeQueryEngine*>
      one_hot_ranges_;
};

}  // namespace xls

#endif  // XLS_PASSES_CONTEXT_SENSITIVE_RANGE_QUERY_ENGINE_H_

// Copyright 2021 The XLS Authors
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

#ifndef XLS_PASSES_RANGE_QUERY_ENGINE_H_
#define XLS_PASSES_RANGE_QUERY_ENGINE_H_

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/passes/query_engine.h"

namespace xls {

using IntervalSetTree = LeafTypeTree<IntervalSet>;
using IntervalSetTreeView = LeafTypeTreeView<IntervalSet>;
using MutableIntervalSetTreeView = MutableLeafTypeTreeView<IntervalSet>;

class RangeQueryVisitor;

// Bundle of data that can be memoized.
struct RangeData {
  // TODO(google/xls#1090): TernaryVector is a std::vector<u8> basically and is
  // not very efficient. We should change this.
  // TODO(allight): We should maybe remove this or make it also a LTT
  std::optional<TernaryVector> ternary;
  IntervalSetTree interval_set;

  bool operator==(const RangeData& o) const {
    return ternary == o.ternary && interval_set == o.interval_set;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RangeData& g) {
    absl::Format(&sink, "[tern: %s, ist: %s]",
                 g.ternary ? ToString(*g.ternary) : "<nullopt>",
                 g.interval_set.ToString());
  }
};

// A helper for memoizing/restricting a range query run.
class RangeDataProvider {
 public:
  RangeDataProvider() = default;
  virtual ~RangeDataProvider() = default;

  // Get the a-priori known interval set tree for the node. Return nullopt if it
  // needs to be recalculated.
  virtual std::optional<RangeData> GetKnownIntervals(Node* node) = 0;

  // Iterate over a pre-computed function topological ordering.
  // It is required that any node which needs to have its intervals initialized
  // (either by the range-query engine itself or by calling 'GetKnownIntervals')
  // is included in this iteration.
  virtual absl::Status IterateFunction(DfsVisitor* visitor) = 0;
};

// Default empty-set givens. Nothing about the function is known.
class NoGivensProvider final : public RangeDataProvider {
 public:
  explicit NoGivensProvider(FunctionBase* function) : function_(function) {}

  std::optional<RangeData> GetKnownIntervals(Node* node) override {
    return std::nullopt;
  }

  absl::Status IterateFunction(DfsVisitor* visitor) override {
    return function_->Accept(visitor);
  }

 private:
  FunctionBase* function_;
};

// A query engine which tracks sets of intervals that a value can be in.
class RangeQueryEngine : public QueryEngine {
 public:
  // Create a `RangeQueryEngine` that contains no data.
  RangeQueryEngine() = default;
  RangeQueryEngine(RangeQueryEngine&&) = default;
  RangeQueryEngine(const RangeQueryEngine&) = default;
  RangeQueryEngine& operator=(const RangeQueryEngine&) = default;
  RangeQueryEngine& operator=(RangeQueryEngine&&) = default;

  // Populate the data in this `RangeQueryEngine` using the
  // given `FunctionBase*`;
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    NoGivensProvider givens(f);
    return PopulateWithGivens(givens);
  }

  // Populate the data in this `RangeQueryEngine givens` with the data returned
  // by `RangeGivensHelper` taken as a given. If `givens` is null proceed as
  // though there are no givens (ie `GetKnownIntervals` always returns
  // std::nullopt and `ShouldContinue` always returns true)
  absl::StatusOr<ReachedFixpoint> PopulateWithGivens(RangeDataProvider& givens);

  bool IsTracked(Node* node) const override {
    return known_bits_.contains(node);
  }

  // Check if there are explicit intervals associated with the node.
  bool HasExplicitIntervals(Node* node) const {
    return interval_sets_.contains(node);
  }

  // Check if the node has known intervals associated with it (either directly
  // or implicitly through known ternary bits).
  //
  // This is not the same as IsTracked since that (for complicated reasons which
  // relate to compatibility with the BDD query engine) actually checks whether
  // we have known ternary bits associated with the node. In the case of 'bits'
  // type values this is the same but in cases where the value is a tuple or an
  // array the value might have intervals associated with it but no actual
  // ternary bits (and thus not be 'tracked' as far as the query engine API is
  // concerned).
  //
  // TODO(allight): 2023-09-05, We should possibly rewrite these or at least
  // change the names.
  bool HasKnownIntervals(Node* node) const {
    return IsTracked(node) || HasExplicitIntervals(node);
  }

  std::optional<LeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    if (!node->GetType()->IsBits()) {
      return std::nullopt;
    }
    TernaryVector tvec = ternary_ops::FromKnownBits(known_bits_.at(node),
                                                    known_bit_values_.at(node));
    LeafTypeTree<TernaryVector> tree(node->GetType());
    tree.Set({}, tvec);
    return tree;
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return GetIntervalSetTree(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    int64_t maybe_one_count = 0;
    for (const TreeBitLocation& location : bits) {
      if (!IsKnown(location) || IsOne(location)) {
        maybe_one_count++;
      }
    }
    return maybe_one_count <= 1;
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    for (const TreeBitLocation& location : bits) {
      if (IsOne(location)) {
        return true;
      }
    }
    return false;
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return IsKnown(a) && IsKnown(b) && IsOne(a) == IsOne(b);
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return IsKnown(a) && IsKnown(b) && IsOne(a) != IsOne(b);
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return false;
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

  // Get the intervals associated with each leaf node in the type tree
  // associated with this node.
  IntervalSetTree GetIntervalSetTree(Node* node) const;

  // Get the intervals associated with each leaf node in the type tree
  // associated with this node as a view. Returns an error if
  // HasExplicitIntervals(node) is false.
  absl::StatusOr<IntervalSetTreeView> GetIntervalSetTreeView(Node* node) const {
    XLS_RET_CHECK(HasExplicitIntervals(node))
        << node << " does not have an existing interval-set tree";
    return interval_sets_.at(node).AsView();
  }

  // Set the intervals associated with the given node.
  //
  // This is primarily intended to be used in tests. Executing `Populate` may
  // overwrite the interval sets you define using this method, depending on what
  // node you defined.
  void SetIntervalSetTree(Node* node, const IntervalSetTree& interval_sets);
  void SetIntervalSetTree(Node* node, IntervalSetTree&& interval_sets);

  // Initialize a node's known bits.
  // This must be called before `SetIntervalSetTree`.
  void InitializeNode(Node* node);

 private:
  friend class RangeQueryVisitor;

  absl::flat_hash_map<Node*, Bits> known_bits_;
  absl::flat_hash_map<Node*, Bits> known_bit_values_;
  absl::flat_hash_map<Node*, IntervalSetTree> interval_sets_;
};

std::string IntervalSetTreeToString(const IntervalSetTree& tree);
std::ostream& operator<<(std::ostream& os, const IntervalSetTree& tree);

}  // namespace xls

#endif  // XLS_PASSES_RANGE_QUERY_ENGINE_H_

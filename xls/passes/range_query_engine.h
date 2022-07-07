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

#include <iosfwd>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/nodes.h"
#include "xls/passes/query_engine.h"

namespace xls {

using IntervalSetTree = LeafTypeTree<IntervalSet>;

class RangeQueryVisitor;

// A query engine which tracks sets of intervals that a value can be in.
class RangeQueryEngine : public QueryEngine {
 public:
  // Create a `RangeQueryEngine` that contains no data.
  RangeQueryEngine() {}

  // Populate the data in this `RangeQueryEngine` using the
  // given `FunctionBase*`;
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override {
    return known_bits_.contains(node);
  }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    XLS_CHECK(node->GetType()->IsBits());
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
    return absl::nullopt;
  }

  // Get the intervals associated with each leaf node in the type tree
  // associated with this node.
  IntervalSetTree GetIntervalSetTree(Node* node) const;

  // Set the intervals associated with the given node.
  //
  // This is primarily intended to be used in tests. Executing `Populate` may
  // overwrite the interval sets you define using this method, depending on what
  // node you defined.
  void SetIntervalSetTree(Node* node, const IntervalSetTree& interval_sets);

  // Initialize a node's known bits.
  // This must be called before `SetIntervalSetTree`.
  void InitializeNode(Node* node);

 private:
  friend class RangeQueryVisitor;

  absl::flat_hash_map<Node*, Bits> known_bits_;
  absl::flat_hash_map<Node*, Bits> known_bit_values_;
  absl::flat_hash_map<Node*, IntervalSetTree> interval_sets_;
};

// Reduce the size of the given `IntervalSet` to the given size.
// This is used to prevent the analysis from using too much memory and CPU.
//
// This works by choosing pairs of neighboring intervals that have small gaps
// and merging them by taking their convex hull, until only `size` intervals
// remain.
IntervalSet MinimizeIntervals(IntervalSet intervals, int64_t size = 16);

std::string IntervalSetTreeToString(const IntervalSetTree& tree);
std::ostream& operator<<(std::ostream& os, const IntervalSetTree& tree);

}  // namespace xls

#endif  // XLS_PASSES_RANGE_QUERY_ENGINE_H_

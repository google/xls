// Copyright 2020 The XLS Authors
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

#ifndef XLS_PASSES_BDD_QUERY_ENGINE_H_
#define XLS_PASSES_BDD_QUERY_ENGINE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A query engine which uses binary decision diagrams (BDDs) to analyze an XLS
// function. BDDs provide sharp analysis of bits values and relationships
// between bit values in the function (relative to ternary abstract evaluation).
// The downside is that BDDs can be slow in general and exponentially slow in
// particular for some operations such as arithmetic and comparison
// operations. For this reason, these operations are generally excluded from the
// analysis.
class BddQueryEngine : public QueryEngine {
 public:
  // `path_limit` is the maximum number of paths from the BDD node to the
  // terminals 0 and 1 to allow for a BDD expression before truncating it.
  // `node_filter` is an optional function which can be used to limit the nodes
  // which the BDD evaluates (returning false means the node will node be
  // evaluated). See BddFunction for details.
  explicit BddQueryEngine(int64_t path_limit = 0,
                          std::optional<std::function<bool(const Node*)>>
                              node_filter = std::nullopt)
      : path_limit_(path_limit), node_filter_(node_filter) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override {
    return known_bits_.contains(node);
  }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    CHECK(node->GetType()->IsBits());
    TernaryVector ternary =
        ternary_ops::FromKnownBits(known_bits_.at(node), bits_values_.at(node));
    LeafTypeTree<TernaryVector> result(node->GetType());
    result.Set({}, ternary);
    return result;
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override;
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;

  // Returns the underlying BddFunction representing the XLS function.
  const BddFunction& bdd_function() const { return *bdd_function_; }

 private:
  // Returns the underlying BDD. This method is const, but queries on a BDD
  // generally mutate the object. We sneakily avoid conflicts with C++ const
  // because the BDD is only held indirectly via pointers.
  // TODO(meheff): Enable queries on a BDD with out mutating the BDD itself.
  BinaryDecisionDiagram& bdd() const { return bdd_function_->bdd(); }

  // Returns the BDD node associated with the given bit.
  BddNodeIndex GetBddNode(const TreeBitLocation& location) const {
    CHECK(location.tree_index().empty());
    CHECK(location.node()->GetType()->IsBits());
    return bdd_function_->GetBddNode(location.node(), location.bit_index());
  }

  // A implies B  <=>  !(A && !B)
  bool Implies(const BddNodeIndex& a, const BddNodeIndex& b) const;

  // Returns true if the expression of the given BDD node exceeds the path
  // limit.
  // TODO(meheff): This should be part of the BDD itself where a query can be
  // performed and the BDD method returns a union of path limit exceeded or
  // the result of the query.
  bool ExceedsPathLimit(BddNodeIndex node) const {
    return path_limit_ > 0 && bdd().GetNode(node).path_count > path_limit_;
  }

  // The maximum number of paths in expression in the BDD before truncating.
  int64_t path_limit_;

  std::optional<std::function<bool(const Node*)>> node_filter_;

  // Indicates the bits at the output of each node which have known values.
  absl::flat_hash_map<Node*, Bits> known_bits_;

  // Indicates the values of bits at the output of each node (if known)
  absl::flat_hash_map<Node*, Bits> bits_values_;

  std::unique_ptr<BddFunction> bdd_function_;
};

}  // namespace xls

#endif  // XLS_PASSES_BDD_QUERY_ENGINE_H_

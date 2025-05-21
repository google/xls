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
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/lazy_query_engine.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

// Returns true if the given node is very cheap to evaluate using a
// BDD. Typically single-bit and logical operations are considered cheap as well
// as "free" operations like bitslice and concat.
bool IsCheapForBdds(const Node* node);

// A query engine which uses binary decision diagrams (BDDs) to analyze an XLS
// function. BDDs provide sharp analysis of bits values and relationships
// between bit values in the function (relative to ternary abstract evaluation).
// The downside is that BDDs can be slow in general and exponentially slow in
// particular for some operations such as arithmetic and comparison
// operations. For this reason, these operations are generally excluded from the
// analysis.
class BddQueryEngine
    : public LazyQueryEngine<std::vector<SaturatingBddNodeIndex>> {
 private:
  using Base = LazyQueryEngine<std::vector<SaturatingBddNodeIndex>>;
  using BddVector = std::vector<SaturatingBddNodeIndex>;
  using BddTree = LeafTypeTree<BddVector>;
  using BddTreeView = LeafTypeTreeView<BddVector>;
  using SharedBddTree = SharedLeafTypeTree<BddVector>;

 public:
  // The suggested default limit on the number of paths from a BDD node to the
  // BDD terminal nodes 0 and 1. If a BDD node associated with a particular bit
  // in the function ({Node*, bit index} pair) exceeds this value the bit's
  // representation in the BDD is replaced with a new BDD variable. This
  // provides a mechanism for limiting the growth of the BDD.
  static constexpr int64_t kDefaultPathLimit = 1024;

  // Returns an instance of the recommended default BddQueryEngine, using
  // kDefaultPathLimit and filtering to operate only on nodes that satisfy
  // IsCheapForBdds.
  static std::unique_ptr<BddQueryEngine> MakeDefault() {
    return std::make_unique<BddQueryEngine>(kDefaultPathLimit, IsCheapForBdds);
  }

  // `path_limit` is the maximum number of paths from the BDD node to the
  // terminals 0 and 1 to allow for a BDD expression before truncating it.
  // `node_filter` is an optional function which filters the nodes to be
  // evaluated. If this function returns false for a node then the node will not
  // be evaluated using BDDs. The node's bits will be new variables in the BDD
  // for which no information is known. If `node_filter` returns true, the node
  // still might *not* be evaluated because some kinds of nodes are never
  // evaluated for various reasons including computation expense.
  explicit BddQueryEngine(int64_t path_limit = 0,
                          std::optional<std::function<bool(const Node*)>>
                              node_filter = std::nullopt)
      : path_limit_(path_limit),
        node_filter_(node_filter),
        bdd_(std::make_unique<BinaryDecisionDiagram>()),
        evaluator_(
            std::make_unique<SaturatingBddEvaluator>(path_limit, bdd_.get())) {}

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override;

  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::btree_set<PredicateState>& state) const override;
  std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>&
          givens) const override;

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override;
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;

  bool IsAllZeros(Node* n) const override { return QueryEngine::IsAllZeros(n); }
  bool IsAllOnes(Node* n) const override { return QueryEngine::IsAllOnes(n); }
  bool IsFullyKnown(Node* n) const override {
    return QueryEngine::IsFullyKnown(n);
  }
  bool IsKnown(const TreeBitLocation& bit) const override {
    return KnownValue(bit).has_value();
  }
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override;
  std::optional<Value> KnownValue(Node* node) const override {
    return QueryEngine::KnownValue(node);
  }

  // Returns the underlying BDD. This method is const, but queries on a BDD
  // generally mutate the object. We sneakily avoid conflicts with C++ const
  // because the BDD is only held indirectly via pointers.
  // TODO(meheff): Enable queries on a BDD without mutating the BDD itself.
  BinaryDecisionDiagram& bdd() const { return *bdd_; }

  // Returns the BDD node associated with the given bit, if there is one;
  // otherwise returns std::nullopt.
  std::optional<BddNodeIndex> GetBddNode(
      const TreeBitLocation& location) const {
    std::optional<SharedBddTree> info = GetInfo(location.node());
    if (!info.has_value()) {
      return std::nullopt;
    }
    SaturatingBddNodeIndex node =
        info->Get(location.tree_index()).at(location.bit_index());
    if (std::holds_alternative<TooManyPaths>(node)) {
      return std::nullopt;
    }
    return std::get<BddNodeIndex>(node);
  }

 protected:
  BddTree ComputeInfo(
      Node* node,
      absl::Span<const BddTree* const> operand_infos) const override;

  absl::Status MergeWithGiven(BddVector& info,
                              const BddVector& given) const override;

 private:
  class AssumingQueryEngine;

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node, std::optional<BddNodeIndex> assumption) const;

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits,
                     std::optional<BddNodeIndex> assumption) const;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits,
                      std::optional<BddNodeIndex> assumption) const;
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node, std::optional<BddNodeIndex> assumption) const;
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node, std::optional<BddNodeIndex> assumption) const;
  bool KnownEquals(const TreeBitLocation& a, const TreeBitLocation& b,
                   std::optional<BddNodeIndex> assumption) const;
  bool KnownNotEquals(const TreeBitLocation& a, const TreeBitLocation& b,
                      std::optional<BddNodeIndex> assumption) const;

  bool IsKnown(const TreeBitLocation& bit,
               std::optional<BddNodeIndex> assumption) const;
  std::optional<bool> KnownValue(const TreeBitLocation& bit,
                                 std::optional<BddNodeIndex> assumption) const;
  std::optional<Value> KnownValue(Node* node,
                                  std::optional<BddNodeIndex> assumption) const;

  bool IsAllZeros(Node* n, std::optional<BddNodeIndex> assumption) const;
  bool IsAllOnes(Node* n, std::optional<BddNodeIndex> assumption) const;
  bool IsFullyKnown(Node* n, std::optional<BddNodeIndex> assumption) const;

  // A implies B  <=>  !(A && !B)
  bool Implies(const BddNodeIndex& a, const BddNodeIndex& b) const;

  // Returns true if the expression of the given BDD node exceeds the path
  // limit.
  // TODO(meheff): This should be part of the BDD itself where a query can be
  // performed and the BDD method returns a union of path limit exceeded or
  // the result of the query.
  bool ExceedsPathLimit(SaturatingBddNodeIndex node) const {
    if (path_limit_ <= 0) {
      return false;
    }
    if (std::holds_alternative<TooManyPaths>(node)) {
      return true;
    }
    return bdd().path_count(std::get<BddNodeIndex>(node)) > path_limit_;
  }

  // The maximum number of paths in expression in the BDD before truncating.
  int64_t path_limit_;

  std::optional<std::function<bool(const Node*)>> node_filter_;

  std::unique_ptr<BinaryDecisionDiagram> bdd_;
  std::unique_ptr<SaturatingBddEvaluator> evaluator_;

  // A map from bit locations to BDD variables; used when the BDD is saturated
  // to avoid creating new variables for the same bit.
  mutable absl::flat_hash_map<TreeBitLocation, BddNodeIndex> bit_variables_;
  BddNodeIndex GetVariableFor(TreeBitLocation location) const;

  // A map from nodes to BDD variables used to represent fully-unknown values;
  // used to avoid creating new variables for the same node.
  mutable absl::flat_hash_map<Node*, std::unique_ptr<BddTree>> node_variables_;
  BddTreeView GetVariablesFor(Node* node) const;
};

}  // namespace xls

#endif  // XLS_PASSES_BDD_QUERY_ENGINE_H_

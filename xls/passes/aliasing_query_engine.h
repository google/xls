// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_ALIASING_QUERY_ENGINE_H_
#define XLS_PASSES_ALIASING_QUERY_ENGINE_H_

#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/data_structures/union_find_map.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A wrapper that lets one add aliases for any node to another node to a query
// engine. This is useful if the underlying graph is being modified to add new
// nodes over time. Aliases can chain. Each node may only be added to alias set
// once.
class AliasingQueryEngine final : public QueryEngine {
 public:
  explicit AliasingQueryEngine(std::unique_ptr<QueryEngine> base)
      : base_(std::move(base)),
        alias_map_(std::make_shared<UnionFindMap<Node*, Node*>>()) {}

  QueryEngine& engine() { return *base_; }
  const QueryEngine& engine() const { return *base_; }

  // Mark new_node as an alias of alias_target and give it the same calculated
  // information.
  absl::Status AddAlias(Node* new_node, Node* alias_target) {
    XLS_RET_CHECK_EQ(new_node->GetType(), alias_target->GetType())
        << new_node << " aliasing to " << alias_target;
    XLS_RET_CHECK(!IsAliased(new_node)) << new_node << " Already aliased.";
    // Set old-node to map to itself (if not already present) in a group by
    // itself.
    alias_map_->Insert(alias_target, alias_target,
                       [](Node* ov, Node* nv) { return ov; });
    // Set new-node to map to itself in a group by itself.
    alias_map_->Insert(new_node, alias_target,
                       [](Node* ov, Node* nv) { return nv; });
    // group old and new node together both mapping to old-nodes result.
    XLS_RET_CHECK(alias_map_->Union(new_node, alias_target,
                                    [](Node* ov, Node* nv) { return nv; }))
        << "Unable to map alias of " << new_node;
    return absl::OkStatus();
  }
  bool IsAliased(Node* n) const {
    return alias_map_->Contains(n) && alias_map_->Find(n)->second != n;
  }
  // Get the node which will be used to look up data for 'orig'
  Node* UnaliasNode(Node* orig) const {
    return alias_map_->Contains(orig) ? alias_map_->Find(orig)->second : orig;
  }
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return base_->Populate(f);
  }
  bool IsTracked(Node* node) const override {
    return base_->IsTracked(UnaliasNode(node));
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    return base_->GetTernary(UnaliasNode(node));
  }
  AliasingQueryEngine UpdatableSpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const {
    return AliasingQueryEngine(base_->SpecializeGivenPredicate(state),
                               alias_map_);
  }
  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override {
    return std::unique_ptr<QueryEngine>(new AliasingQueryEngine(
        base_->SpecializeGivenPredicate(state), alias_map_));
  }
  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return base_->GetIntervals(UnaliasNode(node));
  }
  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return base_->AtMostOneTrue(UnaliasLocationList(bits));
  }
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return base_->AtLeastOneTrue(UnaliasLocationList(bits));
  }
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return base_->Implies(UnaliasLocation(a), UnaliasLocation(b));
  }
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return base_->ImpliedNodeValue(UnaliasPredicates(predicate_bit_values),
                                   UnaliasNode(node));
  }
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return base_->ImpliedNodeTernary(UnaliasPredicates(predicate_bit_values),
                                     UnaliasNode(node));
  }
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return base_->KnownEquals(UnaliasLocation(a), UnaliasLocation(b));
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return base_->KnownNotEquals(UnaliasLocation(a), UnaliasLocation(b));
  }
  bool AtMostOneBitTrue(Node* node) const override {
    return base_->AtMostOneBitTrue(UnaliasNode(node));
  }
  bool AtLeastOneBitTrue(Node* node) const override {
    return base_->AtLeastOneBitTrue(UnaliasNode(node));
  }
  bool ExactlyOneBitTrue(Node* node) const override {
    return base_->ExactlyOneBitTrue(UnaliasNode(node));
  }
  bool IsKnown(const TreeBitLocation& bit) const override {
    return base_->IsKnown(UnaliasLocation(bit));
  }
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override {
    return base_->KnownValue(UnaliasLocation(bit));
  }
  std::optional<Value> KnownValue(Node* node) const override {
    return base_->KnownValue(UnaliasNode(node));
  }
  bool IsAllZeros(Node* node) const override {
    return base_->IsAllZeros(UnaliasNode(node));
  }
  bool IsAllOnes(Node* node) const override {
    return base_->IsAllOnes(UnaliasNode(node));
  }
  bool IsFullyKnown(Node* node) const override {
    return base_->IsFullyKnown(UnaliasNode(node));
  }
  Bits MaxUnsignedValue(Node* node) const override {
    return base_->MaxUnsignedValue(UnaliasNode(node));
  }
  Bits MinUnsignedValue(Node* node) const override {
    return base_->MinUnsignedValue(UnaliasNode(node));
  }

 private:
  TreeBitLocation UnaliasLocation(const TreeBitLocation& loc) const {
    return TreeBitLocation(UnaliasNode(loc.node()), loc.bit_index(),
                           loc.tree_index());
  }
  std::vector<std::pair<TreeBitLocation, bool>> UnaliasPredicates(
      absl::Span<std::pair<TreeBitLocation, bool> const> predicate_bit_values)
      const {
    std::vector<std::pair<TreeBitLocation, bool>> preds;
    preds.reserve(predicate_bit_values.size());
    absl::c_transform(predicate_bit_values, std::back_inserter(preds),
                      [&](const std::pair<TreeBitLocation, bool>& pred)
                          -> std::pair<TreeBitLocation, bool> {
                        return {UnaliasLocation(pred.first), pred.second};
                      });
    return preds;
  }
  std::vector<TreeBitLocation> UnaliasLocationList(
      absl::Span<TreeBitLocation const> orig) const {
    std::vector<TreeBitLocation> res;
    res.reserve(orig.size());
    absl::c_transform(
        orig, std::back_inserter(res),
        [&](const TreeBitLocation& tbl) { return UnaliasLocation(tbl); });
    return res;
  }

  AliasingQueryEngine(std::unique_ptr<QueryEngine> base,
                      std::shared_ptr<UnionFindMap<Node*, Node*>> alias_map)
      : base_(std::move(base)), alias_map_(alias_map) {}
  std::unique_ptr<QueryEngine> base_;
  // Map from the 'new' node the old node which it shares its information with.
  // Shared so that specialized nodes qes can share the map.
  std::shared_ptr<UnionFindMap<Node*, Node*>> alias_map_;
};

}  // namespace xls

#endif  // XLS_PASSES_ALIASING_QUERY_ENGINE_H_

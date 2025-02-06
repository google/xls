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

#include "xls/passes/conditional_specialization_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Encapsulates a condition (a node holding a particular value) which can be
// assumed to be true at particular points in the graph.
struct Condition {
  Node* node;
  TernaryVector value;
  std::optional<IntervalSet> range;

  std::string ToString() const {
    const bool has_range = range.has_value() && !range->IsMaximal();
    const bool has_value = !ternary_ops::AllUnknown(value);
    if (!has_range && !has_value) {
      return absl::StrFormat("%s (unrestricted)", node->GetName());
    }
    if (!has_range) {
      return absl::StrFormat("%s==%s", node->GetName(), xls::ToString(value));
    }
    if (!has_value) {
      return absl::StrFormat("%s in %s", node->GetName(), range->ToString());
    }
    return absl::StrFormat("%s==%s & %s in %s", node->GetName(),
                           xls::ToString(value), node->GetName(),
                           range->ToString());
  }

  bool operator==(const Condition& other) const {
    if (node != other.node || value != other.value) {
      return false;
    }

    const bool has_range = range.has_value() && !range->IsMaximal();
    const bool other_has_range =
        other.range.has_value() && !other.range->IsMaximal();
    if (has_range != other_has_range) {
      return false;
    }
    if (!has_range) {
      return true;
    }
    return *range == *other.range;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Condition& c) {
    return H::combine(std::move(h), c.node, c.value, c.range);
  }
};

// A comparison functor for ordering Conditions. The functor orders conditions
// based on the topological order of the respective node, and then based on the
// condition's value. The motivation for ordering based on topological order is
// that the size of the set of conditions considered at any one time is bounded
// which means some conditions must be discarded.  We preferentially discard
// conditions of nodes which may be far away from the node being considered for
// transformation. Topological sort order gives a measure of distance for this
// purpose.
class ConditionCmp {
 public:
  explicit ConditionCmp(const absl::flat_hash_map<Node*, int64_t>& topo_index)
      : topo_index_(&topo_index) {}

  ConditionCmp(const ConditionCmp&) = default;
  ConditionCmp& operator=(const ConditionCmp&) = default;

  bool operator()(const Condition& a, const Condition& b) const {
    return std::pair(topo_index_->at(a.node), a.value) <
           std::pair(topo_index_->at(b.node), b.value);
  }

 private:
  const absl::flat_hash_map<Node*, int64_t>* topo_index_;
};

// A set of Conditions which can be assumed to hold at a particular point in the
// graph.
class ConditionSet {
 private:
  // The maximum size limit on the number of conditions to avoid superlinear
  // behavior.
  static constexpr int64_t kMaxConditions = 64;
  using ConditionVector = absl::InlinedVector<Condition, kMaxConditions + 1>;

 public:
  explicit ConditionSet(const absl::flat_hash_map<Node*, int64_t>& topo_index)
      : condition_cmp_(topo_index) {}

  ConditionSet(const ConditionSet&) = default;
  ConditionSet(ConditionSet&&) = default;
  ConditionSet& operator=(const ConditionSet&) = default;

  // Perform a set intersection with this set and `other` and assign the result
  // to this set.
  void Intersect(const ConditionSet& other) {
    ConditionVector original = std::move(conditions_);
    conditions_.clear();
    absl::c_set_intersection(other.conditions_, original,
                             std::inserter(conditions_, conditions_.begin()),
                             condition_cmp_);
    // Intersection should not increase set size.
    CHECK_LE(conditions_.size(), kMaxConditions);
  }

  // Perform a set union with this set and `other` and assign the result to this
  // set.
  void Union(const ConditionSet& other) {
    ConditionVector original = std::move(conditions_);
    conditions_.clear();
    absl::c_set_union(other.conditions_, original,
                      std::inserter(conditions_, conditions_.begin()),
                      condition_cmp_);
    while (conditions_.size() > kMaxConditions) {
      conditions_.pop_back();
    }
  }

  // Adds a condition to the set.  Note: it is possible to add conflicting
  // conditions to the set (pred==0 and pred==1). This is an indication that the
  // node is necessarily dead. Arbitrary transformation to dead code is legal so
  // arbitrarily picking one of the conflicting conditions and transforming
  // based on it is fine.
  void AddCondition(const Condition& condition) {
    VLOG(4) << absl::StreamFormat(
        "ConditionSet for (%s, %s) : %s", condition.node->GetName(),
        xls::ToString(condition.value), this->ToString());
    CHECK(!condition.node->Is<Literal>());
    if (auto it = absl::c_lower_bound(conditions_, condition, condition_cmp_);
        it == conditions_.end() || *it != condition) {
      conditions_.insert(it, condition);
    }
    // The conditions are ordering in topological sort order (based on
    // Condition.node) and transformation occurs in reverse topological sort
    // order so the most distant conditions should be at the end of the
    // condition set.  Just pop the last condition off the end if it exceeds the
    // limit.
    if (conditions_.size() > kMaxConditions) {
      conditions_.pop_back();
    }
    CHECK_LE(conditions_.size(), kMaxConditions);
  }

  absl::Span<const Condition> conditions() const { return conditions_; }

  bool empty() const { return conditions_.empty(); }

  std::string ToString() const {
    std::vector<std::string> pieces;
    pieces.reserve(conditions_.size());
    for (const Condition& condition : conditions_) {
      pieces.push_back(condition.ToString());
    }
    return absl::StrCat("(", absl::StrJoin(pieces, " & "), ")");
  }

  // Returns the conditions as predicates
  std::vector<std::pair<TreeBitLocation, bool>> GetPredicates() const {
    std::vector<std::pair<TreeBitLocation, bool>> predicates;
    for (const Condition& condition : conditions()) {
      for (int64_t i = 0; i < condition.node->BitCountOrDie(); ++i) {
        if (condition.value[i] == TernaryValue::kUnknown) {
          continue;
        }
        bool bit_value = (condition.value[i] == TernaryValue::kKnownOne);
        predicates.push_back({TreeBitLocation{condition.node, i}, bit_value});
      }
    }
    return predicates;
  }

  // Returns the conditions as givens
  absl::flat_hash_map<Node*, ValueKnowledge> GetAsGivens() const {
    absl::flat_hash_map<Node*, ValueKnowledge> givens;
    for (const Condition& condition : conditions()) {
      const bool condition_has_value =
          !ternary_ops::AllUnknown(condition.value);
      const bool condition_has_range =
          condition.range.has_value() && !condition.range->IsMaximal();
      if (!condition_has_value && !condition_has_range) {
        continue;
      }

      ValueKnowledge& given = givens[condition.node];
      if (condition_has_value) {
        if (given.ternary.has_value()) {
          if (absl::Status merged = ternary_ops::UpdateWithUnion(
                  given.ternary->Get({}), condition.value);
              !merged.ok()) {
            // This is impossible, as the conditions contradict each other. For
            // now, we can't do anything about this; it might be worth finding a
            // way to propagate this information.
            VLOG(1) << "Proved this condition set is impossible: "
                    << ToString();
            return {};
          }
        } else {
          given.ternary = TernaryTree::CreateSingleElementTree(
              condition.node->GetType(), std::move(condition.value));
        }
      }
      if (condition_has_range) {
        IntervalSet range = *condition.range;
        if (given.intervals.has_value()) {
          range = IntervalSet::Intersect(range, given.intervals->Get({}));
        }
        given.intervals = IntervalSetTree::CreateSingleElementTree(
            condition.node->GetType(), std::move(range));
      }
    }
    return givens;
  }

 private:
  ConditionCmp condition_cmp_;

  // Kept sorted at all times (according to `condition_cmp_`), retaining
  // only unique elements.
  ConditionVector conditions_;
};

// A map containing the set of conditions which can be assumed at each node
// (and at some edges). An example where a condition would be assigned to an
// edge rather than a node is a case arm of a select. In this case, a
// selector value can be assumed on the edge from the case operand to the
// select operation. In general, the condition cannot be assumed on the
// source node of the edge (case operand) because the source node may be
// used outside the case arm expression.
class ConditionMap {
 public:
  explicit ConditionMap(FunctionBase* f) {
    std::vector<Node*> topo_sort = TopoSort(f);
    for (int64_t i = 0; i < topo_sort.size(); ++i) {
      topo_index_[topo_sort[i]] = i;
      // Initially all node conditions are empty.
      node_conditions_.emplace(topo_sort[i],
                               std::make_unique<ConditionSet>(topo_index_));
    }
  }

  // Returns the condition set for the given node. Returns a mutable
  // reference as this is the mechanism for setting condition sets of nodes.
  ConditionSet& GetNodeConditionSet(Node* node) {
    auto it = node_conditions_.find(node);
    if (it == node_conditions_.end()) {
      it = node_conditions_.insert(
          it, {node, std::make_unique<ConditionSet>(topo_index_)});
    }
    return *it->second;
  }

  // Sets the condition set for the given edge where the edge extends to
  // `node` and operand index `operand_no`.
  void SetEdgeConditionSet(Node* node, int64_t operand_no,
                           ConditionSet condition_set) {
    VLOG(4) << absl::StrFormat("Setting conditions on %s->%s (operand %d): %s",
                               node->operand(operand_no)->GetName(),
                               node->GetName(), operand_no,
                               condition_set.ToString());
    std::pair<Node*, int64_t> key = {node, operand_no};
    CHECK(!edge_conditions_.contains(key));
    edge_conditions_.insert(
        {key, std::make_unique<ConditionSet>(std::move(condition_set))});
  }

  // Returns the conditions which can be assumed along the edge to `node`
  // from its operand index `operand_no`.
  const ConditionSet& GetEdgeConditionSet(Node* node, int64_t operand_no) {
    std::pair<Node*, int64_t> key = {node, operand_no};
    if (!edge_conditions_.contains(key)) {
      // There are no special conditions for this edge. Return the
      // conditions on the target of the edge which necessarily hold on the
      // edge as well.
      return GetNodeConditionSet(node);
    }
    return *edge_conditions_.at(key);
  }

  // Returns the conditions which can be assumed along the edge(s) from node
  // to user. This interface is asymmetric to SetEdgeCondition (which takes
  // a node and operand number) to make it easier to use because at a
  // particular node you have easy access to the user list but not the
  // operand number(s) associated with each user.
  const ConditionSet& GetEdgeConditionSet(Node* node, Node* user) {
    // Find the unique (if there is one) operand number of user which
    // corresponds to node.
    std::optional<int64_t> operand_index;
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == node) {
        if (operand_index.has_value()) {
          // `node` appears in multiple operands of `user`. Return the
          // assumptions that can be made at the node `user` itself. This is
          // typically not a strong conditions as might be assuming along
          // the edges.
          return GetNodeConditionSet(user);
        }
        operand_index = i;
      }
    }
    CHECK(operand_index.has_value()) << absl::StreamFormat(
        "%s is not a user of %s", user->GetName(), node->GetName());
    return GetEdgeConditionSet(user, *operand_index);
  }

  std::string ToString() const {
    std::stringstream os;
    os << "Node conditions:\n";
    for (const auto& [node, cond_set] : node_conditions_) {
      if (!cond_set->conditions().empty()) {
        os << absl::StrFormat("[%s]: %s", node->ToString(),
                              cond_set->ToString())
           << "\n";
      }
    }
    os << "Edge conditions:\n";
    for (const auto& [key, cond_set] : edge_conditions_) {
      if (!cond_set->conditions().empty()) {
        os << absl::StrFormat("[%s, %i]: %s", std::get<0>(key)->ToString(),
                              std::get<1>(key), cond_set->ToString())
           << "\n";
      }
    }
    return os.str();
  }

 private:
  // Index of each node in the function base in a topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index_;

  // Set of conditions which might be assumed at each node.
  absl::flat_hash_map<Node*, std::unique_ptr<ConditionSet>> node_conditions_;

  // Set of conditions which might be assumed at some edges. The key defines
  // an edge as (node, operand_no). If no key exists for an edge, then there
  // are no special conditions for the edge, and the conditions for the edge
  // are the same as the node.
  absl::flat_hash_map<std::pair<Node*, int64_t>, std::unique_ptr<ConditionSet>>
      edge_conditions_;

  // A cache of the set of conditions implied by a given condition. Avoids the
  // need to recompute this each time we rederive the same condition.
  absl::flat_hash_map<Condition, std::unique_ptr<ConditionSet>>
      implied_conditions_;
};

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns std::nullopt otherwise.
std::optional<Bits> ImpliedNodeValue(const ConditionSet& condition_set,
                                     Node* node,
                                     const QueryEngine& query_engine) {
  for (const Condition& condition : condition_set.conditions()) {
    if (condition.node == node && ternary_ops::IsFullyKnown(condition.value)) {
      VLOG(4) << absl::StreamFormat("%s trivially implies %s==%s",
                                    condition_set.ToString(), node->GetName(),
                                    xls::ToString(condition.value));
      return ternary_ops::ToKnownBitsValues(condition.value);
    }
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates =
      condition_set.GetPredicates();
  std::optional<Bits> implied_value =
      query_engine.ImpliedNodeValue(predicates, node);

  if (implied_value.has_value()) {
    VLOG(4) << absl::StreamFormat("%s implies %s==%v", condition_set.ToString(),
                                  node->GetName(), implied_value.value());
  }
  return implied_value;
}

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns std::nullopt otherwise.
std::optional<TernaryVector> ImpliedNodeTernary(
    const ConditionSet& condition_set, Node* node,
    const QueryEngine& query_engine) {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  TernaryVector result(node->BitCountOrDie(), TernaryValue::kUnknown);
  for (const Condition& condition : condition_set.conditions()) {
    if (condition.node == node) {
      VLOG(4) << absl::StreamFormat("%s trivially implies %s==%s",
                                    condition_set.ToString(), node->GetName(),
                                    xls::ToString(condition.value));
      if (absl::Status update_status =
              ternary_ops::UpdateWithUnion(result, condition.value);
          !update_status.ok()) {
        CHECK(absl::IsInvalidArgument(update_status));
        // This is impossible, as the conditions contradict each other. For
        // now, we can't do anything about this; it might be worth finding a
        // way to propagate this information.
        VLOG(1) << "Proved this condition is impossible: "
                << condition_set.ToString();
        return std::nullopt;
      }
    }
  }
  if (ternary_ops::IsFullyKnown(result)) {
    return result;
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates =
      condition_set.GetPredicates();
  std::optional<TernaryVector> implied_ternary =
      query_engine.ImpliedNodeTernary(predicates, node);
  if (implied_ternary.has_value()) {
    VLOG(4) << absl::StreamFormat("%s implies %s==%s", condition_set.ToString(),
                                  node->GetName(),
                                  xls::ToString(*implied_ternary));
    if (absl::Status update_status =
            ternary_ops::UpdateWithUnion(result, *implied_ternary);
        !update_status.ok()) {
      CHECK(absl::IsInvalidArgument(update_status));
      // This is impossible, as the conditions contradict each other. For
      // now, we can't do anything about this; it might be worth finding a
      // way to propagate this information.
      VLOG(1) << "Proved this condition is impossible: "
              << condition_set.ToString();
      return std::nullopt;
    }
  }

  return result;
}

// Returns the case arm node of the given select which is selected when the
// selector has the given value.
Node* GetSelectedCase(Select* select, const Bits& selector_value) {
  if (bits_ops::UGreaterThanOrEqual(selector_value, select->cases().size())) {
    return select->default_value().value();
  }
  // It is safe to convert to uint64_t because of the above check against
  // cases size.
  return select->get_case(selector_value.ToUint64().value());
}

std::optional<Node*> GetSelectedCase(PrioritySelect* select,
                                     const TernaryVector& selector_value) {
  for (int64_t i = 0; i < select->cases().size(); ++i) {
    if (selector_value[i] == TernaryValue::kUnknown) {
      // We can't be sure which case is selected.
      return std::nullopt;
    }
    if (selector_value[i] == TernaryValue::kKnownOne) {
      return select->get_case(i);
    }
  }
  // All bits of the selector are zero.
  return select->default_value();
}

struct ZeroValue : std::monostate {};
std::optional<std::variant<Node*, ZeroValue>> GetSelectedCase(
    OneHotSelect* ohs, const TernaryVector& selector_value) {
  if (!ternary_ops::IsFullyKnown(selector_value)) {
    // We can't be sure which case is selected.
    return std::nullopt;
  }
  Bits selector_bits = ternary_ops::ToKnownBitsValues(selector_value);
  if (selector_bits.PopCount() > 1) {
    // We aren't selecting just one state.
    return std::nullopt;
  }
  for (int64_t i = 0; i < selector_value.size(); ++i) {
    if (selector_bits.Get(i)) {
      return ohs->get_case(i);
    }
  }
  // All bits of the selector are zero.
  return ZeroValue{};
}

absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> AffectedBy(
    FunctionBase* f) {
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> affected_by;
  for (Node* node : TopoSort(f)) {
    for (Node* operand : node->operands()) {
      affected_by[operand].insert(node);
    }
  }
  return TransitiveClosure(affected_by);
}

absl::StatusOr<std::optional<Node*>> CheckMatch(Node* node,
                                                TernaryTreeView ternary,
                                                Node* user) {
  if (absl::c_all_of(ternary.elements(), [](TernarySpan entry) {
        return ternary_ops::AllUnknown(entry);
      })) {
    return std::nullopt;
  }
  LeafTypeTree<Bits> known_bits = leaf_type_tree::Map<Bits, TernaryVector>(
      ternary,
      [](TernarySpan entry) { return ternary_ops::ToKnownBits(entry); });
  XLS_ASSIGN_OR_RETURN(Node * bits_to_check,
                       GatherBits(node, known_bits.AsView()));
  InlineBitmap target_bitmap(bits_to_check->BitCountOrDie());
  int64_t target_index = 0;
  leaf_type_tree::ForEach(ternary, [&](TernarySpan entry) {
    for (TernaryValue value : entry) {
      if (value == TernaryValue::kUnknown) {
        continue;
      }
      target_bitmap.Set(target_index++, value == TernaryValue::kKnownOne);
    }
  });
  XLS_ASSIGN_OR_RETURN(
      Node * target,
      node->function_base()->MakeNode<Literal>(
          user->loc(), Value(Bits::FromBitmap(std::move(target_bitmap)))));
  return node->function_base()->MakeNode<CompareOp>(user->loc(), bits_to_check,
                                                    target, Op::kEq);
}

absl::StatusOr<std::optional<Node*>> CheckMatch(Node* node,
                                                IntervalSetTreeView intervals,
                                                Node* user) {
  if (absl::c_any_of(intervals.elements(), [](const IntervalSet& interval_set) {
        return interval_set.BitCount() > 0 && interval_set.IsEmpty();
      })) {
    // Matching is impossible. Return a literal 0.
    return node->function_base()->MakeNode<Literal>(user->loc(),
                                                    Value(UBits(0, 1)));
  }

  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> node_tree, ToTreeOfNodes(node));
  XLS_ASSIGN_OR_RETURN(
      LeafTypeTree<std::optional<Node*>> match_tree,
      (leaf_type_tree::ZipStatus<std::optional<Node*>, Node*, IntervalSet>(
          node_tree.AsView(), intervals.AsView(),
          [&](Node* leaf_node, IntervalSet interval_set)
              -> absl::StatusOr<std::optional<Node*>> {
            if (interval_set.BitCount() == 0 || interval_set.IsMaximal()) {
              return std::nullopt;
            }
            std::vector<Node*> interval_checks;
            interval_checks.reserve(interval_set.Intervals().size());
            for (const Interval& interval : interval_set.Intervals()) {
              std::optional<Node*> interval_check;

              if (!interval.LowerBound().IsZero()) {
                XLS_ASSIGN_OR_RETURN(
                    Node * lb, node->function_base()->MakeNode<Literal>(
                                   user->loc(), Value(interval.LowerBound())));
                XLS_ASSIGN_OR_RETURN(interval_check,
                                     node->function_base()->MakeNode<CompareOp>(
                                         user->loc(), leaf_node, lb, Op::kUGe));
              }

              if (!interval.UpperBound().IsAllOnes()) {
                XLS_ASSIGN_OR_RETURN(
                    Node * ub, node->function_base()->MakeNode<Literal>(
                                   user->loc(), Value(interval.UpperBound())));
                XLS_ASSIGN_OR_RETURN(Node * ub_check,
                                     node->function_base()->MakeNode<CompareOp>(
                                         user->loc(), leaf_node, ub, Op::kULe));
                if (interval_check.has_value()) {
                  XLS_ASSIGN_OR_RETURN(
                      interval_check,
                      node->function_base()->MakeNode<NaryOp>(
                          user->loc(),
                          absl::MakeConstSpan({*interval_check, ub_check}),
                          Op::kAnd));
                } else {
                  interval_check = ub_check;
                }
              }

              if (interval_check.has_value()) {
                interval_checks.push_back(*interval_check);
              }
            }
            return NaryAndIfNeeded(node->function_base(), interval_checks);
          })));

  if (absl::c_all_of(match_tree.elements(), [](std::optional<Node*> entry) {
        return !entry.has_value();
      })) {
    return std::nullopt;
  }

  std::vector<Node*> match_checks;
  match_checks.reserve(match_tree.elements().size());
  for (std::optional<Node*> entry : match_tree.elements()) {
    if (entry.has_value()) {
      match_checks.push_back(*entry);
    }
  }
  return NaryAndIfNeeded(node->function_base(), match_checks);
}

class ImpliedConditionCache {
 public:
  ImpliedConditionCache(FunctionBase* f, QueryEngine* query_engine)
      : query_engine_(query_engine) {
    std::vector<Node*> topo_sort = TopoSort(f);
    for (int64_t i = 0; i < topo_sort.size(); ++i) {
      topo_index_[topo_sort[i]] = i;
    }
  }

  const ConditionSet& GetImplied(const Condition& condition) {
    if (auto it = cache_.find(condition); it != cache_.end()) {
      return *it->second;
    }
    auto [it, _] =
        cache_.emplace(condition, std::make_unique<ConditionSet>(topo_index_));
    ConditionSet& implied_conditions = *it->second;

    implied_conditions.AddCondition(condition);

    if (condition.node->op() == Op::kNot &&
        !ternary_ops::AllUnknown(condition.value) &&
        !condition.node->operand(0)->Is<Literal>()) {
      Node* operand = condition.node->operand(0);

      VLOG(4) << "Lifting a known negated value: not(" << operand->GetName()
              << ") == " << xls::ToString(condition.value);

      TernaryVector negated = condition.value;
      for (int64_t i = 0; i < negated.size(); ++i) {
        if (negated[i] == TernaryValue::kKnownOne) {
          negated[i] = TernaryValue::kKnownZero;
        } else if (negated[i] == TernaryValue::kKnownZero) {
          negated[i] = TernaryValue::kKnownOne;
        }
      }
      implied_conditions.Union(
          GetImplied(Condition{.node = operand, .value = negated}));
    }

    if (condition.node->OpIn({Op::kAnd, Op::kOr, Op::kNand, Op::kNor})) {
      TernaryVector lifted_value = condition.node->OpIn({Op::kNand, Op::kNor})
                                       ? ternary_ops::Not(condition.value)
                                       : condition.value;
      TernaryValue lifted_bit = condition.node->OpIn({Op::kAnd, Op::kNand})
                                    ? TernaryValue::kKnownOne
                                    : TernaryValue::kKnownZero;
      TernaryValue non_lifted_bit = ternary_ops::Not(lifted_bit);

      if (absl::c_contains(lifted_value, lifted_bit)) {
        for (int64_t i = 0; i < lifted_value.size(); ++i) {
          if (lifted_value[i] == non_lifted_bit) {
            lifted_value[i] = TernaryValue::kUnknown;
          }
        }
        VLOG(4) << "Lifting known bits; " << OpToString(condition.node->op())
                << "("
                << absl::StrJoin(condition.node->operands(), ", ",
                                 [](std::string* out, Node* node) {
                                   absl::StrAppend(out, node->GetName());
                                 })
                << ") == " << xls::ToString(condition.value)
                << ", so all operands must match: "
                << xls::ToString(lifted_value);
        for (Node* operand : condition.node->operands()) {
          if (operand->Is<Literal>()) {
            continue;
          }
          implied_conditions.Union(
              GetImplied(Condition{.node = operand, .value = lifted_value}));
        }
      }
    }

    if ((condition.node->op() == Op::kEq &&
         ternary_ops::IsKnownOne(condition.value)) ||
        (condition.node->op() == Op::kNe &&
         ternary_ops::IsKnownZero(condition.value))) {
      Node* lhs = condition.node->operand(0);
      Node* rhs = condition.node->operand(1);

      VLOG(4) << "Converting a known equality to direct conditions: "
              << lhs->GetName() << " == " << rhs->GetName();

      if (std::optional<SharedLeafTypeTree<TernaryVector>> lhs_ternary =
              query_engine_->GetTernary(lhs);
          !rhs->Is<Literal>() && rhs->GetType()->IsBits() &&
          lhs_ternary.has_value() &&
          !ternary_ops::AllUnknown(lhs_ternary->Get({}))) {
        implied_conditions.Union(
            GetImplied(Condition{.node = rhs, .value = lhs_ternary->Get({})}));
      }
      if (std::optional<SharedLeafTypeTree<TernaryVector>> rhs_ternary =
              query_engine_->GetTernary(rhs);
          !lhs->Is<Literal>() && lhs->GetType()->IsBits() &&
          rhs_ternary.has_value() &&
          !ternary_ops::AllUnknown(rhs_ternary->Get({}))) {
        implied_conditions.Union(
            GetImplied(Condition{.node = lhs, .value = rhs_ternary->Get({})}));
      }
    }

    return implied_conditions;
  }

 private:
  QueryEngine* query_engine_;

  // Index of each node in the function base in a topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index_;

  absl::flat_hash_map<Condition, std::unique_ptr<ConditionSet>> cache_;
};

absl::StatusOr<bool> EliminateNoopNext(FunctionBase* f) {
  std::vector<Next*> to_remove;
  for (Node* n : f->nodes()) {
    if (!n->Is<Next>()) {
      continue;
    }
    Next* next = n->As<Next>();
    if (next->state_read() == next->value()) {
      to_remove.push_back(next);
    }
  }
  for (Next* next : to_remove) {
    XLS_RETURN_IF_ERROR(
        next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
    XLS_RETURN_IF_ERROR(f->RemoveNode(next));
  }
  return !to_remove.empty();
}

}  // namespace

absl::StatusOr<bool> ConditionalSpecializationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  if (options.eliminate_noop_next) {
    XLS_ASSIGN_OR_RETURN(changed, EliminateNoopNext(f));
  }

  std::vector<std::unique_ptr<QueryEngine>> query_engines;
  query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  if (use_bdd_) {
    query_engines.push_back(std::make_unique<BddQueryEngine>(
        BddFunction::kDefaultPathLimit, IsCheapForBdds));
  }

  UnionQueryEngine query_engine(std::move(query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  ConditionMap condition_map(f);
  ImpliedConditionCache condition_cache(f, &query_engine);

  std::optional<absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>>
      affected_by;

  // Iterate backwards through the graph because we add conditions at the case
  // arm operands of selects and propagate them upwards through the expressions
  // which compute the case arm.
  for (Node* node : ReverseTopoSort(f)) {
    ConditionSet& set = condition_map.GetNodeConditionSet(node);
    VLOG(4) << absl::StreamFormat("Considering node %s: %s", node->GetName(),
                                  set.ToString());

    if (node->Is<Invoke>()) {
      // The contents of an invoke may be side-effecting (e.g., the invoked
      // function might contain an assert), so don't assume any conditions for
      // this node or its predecessors.
      VLOG(4) << absl::StreamFormat(
          "Node %s is an invoke and could be side-effecting", node->GetName());
      continue;
    }
    if (OpIsSideEffecting(node->op()) &&
        !node->OpIn({Op::kSend, Op::kStateRead, Op::kNext})) {
      // Inputs to side-effecting operations should not change so don't assume
      // any conditions for this node or it's predecessors.
      VLOG(4) << absl::StreamFormat("Node %s is side-effecting",
                                    node->GetName());
      continue;
    }

    // Compute the intersection of the condition sets of the users of this node.
    //
    // If this node has an implicit use then we can't propagate any conditions
    // from the users because this value is unconditionally live and therefore
    // its computed value should not be changed. (However, we ignore the
    // implicit use of a StateRead that's only used to declare that the
    // corresponding state element is unchanged, since that's safe in this
    // context.)
    //
    // Similarly, if this node is a StateRead's predicate, then its value can
    // affect throughput and so shouldn't be changed.
    XLS_ASSIGN_OR_RETURN(
        bool has_real_implicit_use, [&]() -> absl::StatusOr<bool> {
          if (!f->HasImplicitUse(node)) {
            return false;
          }
          if (!node->Is<StateRead>()) {
            return true;
          }
          Proc* proc = f->AsProcOrDie();
          absl::btree_set<int64_t> next_state_indices =
              proc->GetNextStateIndices(node);
          if (next_state_indices.size() != 1) {
            return true;
          }
          int64_t next_state_index = *next_state_indices.begin();
          XLS_ASSIGN_OR_RETURN(int64_t index,
                               proc->GetStateElementIndex(
                                   node->As<StateRead>()->state_element()));
          return index != next_state_index;
        }());
    if (!has_real_implicit_use &&
        absl::c_none_of(node->users(),
                        [](Node* user) { return user->Is<StateRead>(); })) {
      VLOG(4) << absl::StreamFormat(
          "%s has no implicit use, computing intersection of conditions of "
          "users",
          node->GetName());
      bool first_user = true;
      for (Node* user : node->users()) {
        const ConditionSet& user_set =
            condition_map.GetEdgeConditionSet(node, user);
        VLOG(4) << "Conditions for edge " << node->GetName() << " -> "
                << user->GetName() << ": " << user_set.ToString();
        if (first_user) {
          set = user_set;
        } else {
          set.Intersect(user_set);
        }
        first_user = false;
      }
    }

    // If the only user of this node is a single select arm then add a condition
    // based on the selector value for this arm.
    if (node->Is<Select>()) {
      Select* select = node->As<Select>();
      // Don't bother specializing if the selector is a literal as this
      // results in a useless condition where we assume a literal has a
      // literal value.
      if (!select->selector()->Is<Literal>()) {
        for (int64_t case_no = 0; case_no < select->cases().size(); ++case_no) {
          ConditionSet edge_set = set;
          // If this case is selected, we know the selector is exactly
          // `case_no`.
          edge_set.Union(condition_cache.GetImplied(Condition{
              .node = select->selector(),
              .value = ternary_ops::BitsToTernary(
                  UBits(case_no, select->selector()->BitCountOrDie())),
          }));
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
      }
    }
    if (node->Is<OneHotSelect>()) {
      OneHotSelect* select = node->As<OneHotSelect>();
      if (!select->selector()->Is<Literal>()) {
        for (int64_t case_no = 0; case_no < select->cases().size(); ++case_no) {
          ConditionSet edge_set = set;
          // If this case is selected, we know the corresponding bit of the
          // selector is one.
          TernaryVector selector_value(select->selector()->BitCountOrDie(),
                                       TernaryValue::kUnknown);
          selector_value[case_no] = TernaryValue::kKnownOne;
          edge_set.Union(condition_cache.GetImplied(Condition{
              .node = select->selector(),
              .value = selector_value,
          }));
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
      }
    }
    if (node->Is<PrioritySelect>()) {
      PrioritySelect* select = node->As<PrioritySelect>();
      if (!select->selector()->Is<Literal>()) {
        for (int64_t case_no = 0; case_no < select->cases().size(); ++case_no) {
          ConditionSet edge_set = set;
          // If this case is selected, we know all the bits of the selector up
          // to and including `case_no`; all higher-priority bits are zero, and
          // this case's bit is one.
          Bits known_bits = Bits(select->selector()->BitCountOrDie());
          known_bits.SetRange(0, case_no + 1);
          Bits known_bits_values =
              Bits::PowerOfTwo(case_no, select->selector()->BitCountOrDie());
          edge_set.Union(condition_cache.GetImplied(Condition{
              .node = select->selector(),
              .value =
                  ternary_ops::FromKnownBits(known_bits, known_bits_values),
          }));
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
        ConditionSet edge_set = set;
        // If the default value is selected, we know all the bits of the
        // selector are zero.
        edge_set.Union(condition_cache.GetImplied(Condition{
            .node = select->selector(),
            .value = TernaryVector(select->selector()->BitCountOrDie(),
                                   TernaryValue::kKnownZero),
        }));
        condition_map.SetEdgeConditionSet(node, select->cases().size() + 1,
                                          std::move(edge_set));
      }
    }

    if (node->Is<ArrayUpdate>()) {
      ArrayUpdate* update = node->As<ArrayUpdate>();
      ConditionSet edge_set = set;
      Type* array_type = update->array_to_update()->GetType();
      for (Node* index : update->indices()) {
        if (index->Is<Literal>()) {
          continue;
        }

        const int64_t array_size = array_type->AsArrayOrDie()->size();
        if (Bits::MinBitCountUnsigned(array_size) > index->BitCountOrDie()) {
          continue;
        }

        // ArrayUpdate is a no-op if any index is out of range; as such, it only
        // cares about the update value if all indices are in range.
        edge_set.Union(condition_cache.GetImplied(Condition{
            .node = index,
            .value =
                TernaryVector(index->BitCountOrDie(), TernaryValue::kUnknown),
            .range = IntervalSet::Of({Interval::RightOpen(
                UBits(0, index->BitCountOrDie()),
                UBits(array_size, index->BitCountOrDie()))}),
        }));

        array_type = array_type->AsArrayOrDie()->element_type();
      }
      condition_map.SetEdgeConditionSet(node, ArrayUpdate::kUpdateValueOperand,
                                        std::move(edge_set));
    }

    if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      if (!send->predicate().has_value()) {
        continue;
      }

      Node* predicate = *send->predicate();
      if (predicate->Is<Literal>()) {
        continue;
      }

      VLOG(4) << absl::StreamFormat(
          "%s is a conditional send, assuming predicate %s is true for data %s",
          node->GetName(), predicate->GetName(), send->data()->GetName());

      ConditionSet edge_set = set;
      edge_set.Union(condition_cache.GetImplied(
          Condition{.node = predicate, .value = {TernaryValue::kKnownOne}}));
      condition_map.SetEdgeConditionSet(node, Send::kDataOperand,
                                        std::move(edge_set));
    }

    if (node->Is<Next>()) {
      Next* next = node->As<Next>();
      if (!next->predicate().has_value()) {
        continue;
      }

      Node* predicate = *next->predicate();
      if (predicate->Is<Literal>()) {
        continue;
      }

      VLOG(4) << absl::StreamFormat(
          "%s is a conditional next_value, assuming predicate %s is true for "
          "data %s",
          node->GetName(), predicate->GetName(), next->value()->GetName());

      ConditionSet edge_set = set;
      edge_set.Union(condition_cache.GetImplied(
          Condition{.node = predicate, .value = {TernaryValue::kKnownOne}}));
      condition_map.SetEdgeConditionSet(node, Next::kStateReadOperand,
                                        edge_set);
      condition_map.SetEdgeConditionSet(node, Next::kValueOperand,
                                        std::move(edge_set));
    }

    VLOG(4) << absl::StreamFormat("Conditions for %s : %s", node->GetName(),
                                  set.ToString());

    // Now specialize the node itself based on the conditions on its access.
    if (options.optimize_for_best_case_throughput && node->Is<StateRead>() &&
        !set.empty()) {
      StateRead* state_read = node->As<StateRead>();
      if (state_read->predicate().has_value()) {
        // For now, avoid specializing the predicate of an already-conditional
        // state read. This keeps us from getting into an infinite loop.
        continue;
      }

      // Record that this node is unused (including by next_value nodes) when
      // the condition set is not met.
      absl::flat_hash_map<Node*, ValueKnowledge> accessed_when =
          set.GetAsGivens();
      if (accessed_when.empty()) {
        continue;
      }

      std::vector<Node*> access_conditions;
      if (!affected_by.has_value()) {
        affected_by = AffectedBy(f);
      }
      for (auto& [src, given] : accessed_when) {
        if ((*affected_by)[node].contains(src)) {
          // The value of `src` depends on the value of `node`, so it's not
          // possible to specialize on `src` without creating a cycle.
          continue;
        }
        if (given.ternary.has_value()) {
          XLS_ASSIGN_OR_RETURN(std::optional<Node*> access_condition,
                               CheckMatch(src, given.ternary->AsView(), node));
          if (access_condition.has_value()) {
            access_conditions.push_back(*access_condition);
          }
        }
        if (given.intervals.has_value()) {
          XLS_ASSIGN_OR_RETURN(
              std::optional<Node*> access_condition,
              CheckMatch(src, given.intervals->AsView(), node));
          if (access_condition.has_value()) {
            access_conditions.push_back(*access_condition);
          }
        }
      }
      if (access_conditions.empty()) {
        continue;
      }

      VLOG(2) << absl::StreamFormat(
          "Specializing previously-unconditional state read %s; only accessed "
          "when: %s",
          node->GetName(), set.ToString());
      XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                           NaryAndIfNeeded(f, access_conditions));
      XLS_RETURN_IF_ERROR(state_read->SetPredicate(new_predicate));
      changed = true;
    }

    // Now specialize any operands (if possible) based on the conditions.
    if (node->Is<StateRead>()) {
      // We don't want to specialize the predicate of a state read; this can
      // reduce throughput.
      continue;
    }
    for (int64_t operand_no = 0; operand_no < node->operand_count();
         ++operand_no) {
      Node* operand = node->operand(operand_no);

      if (operand->Is<Literal>()) {
        // Operand is already a literal. Nothing to do.
        continue;
      }

      const ConditionSet& edge_set =
          condition_map.GetEdgeConditionSet(operand, node);
      VLOG(4) << absl::StrFormat("Conditions on edge %s -> %s: %s",
                                 operand->GetName(), node->GetName(),
                                 edge_set.ToString());
      if (edge_set.empty()) {
        continue;
      }

      if (node->Is<Next>() && operand_no == Next::kStateReadOperand) {
        // No point in specializing the state read, and it would make the node
        // invalid anyway; this is just a pointer to the state element.
        continue;
      }

      std::unique_ptr<QueryEngine> specialized_query_engine =
          query_engine.SpecializeGiven(edge_set.GetAsGivens());

      // First check to see if the condition set directly implies a value for
      // the operand. If so replace with the implied value.
      if (std::optional<Bits> implied_value =
              ImpliedNodeValue(edge_set, operand, *specialized_query_engine);
          implied_value.has_value()) {
        VLOG(3) << absl::StreamFormat("Replacing operand %d of %s with %v",
                                      operand_no, node->GetName(),
                                      implied_value.value());
        XLS_ASSIGN_OR_RETURN(
            Literal * literal,
            f->MakeNode<Literal>(operand->loc(), Value(implied_value.value())));
        XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(operand_no, literal));
        changed = true;
        continue;
      }

      // If `operand` is a select and any condition set of `node` implies the
      // selector value then we can wire the respective implied case directly to
      // that user. For example:
      //
      //         a   b                 a     b
      //          \ /                   \   / \
      //  s|t ->  sel0          s|t ->  sel0    \
      //           | \     =>            |      |
      //        c  |  \                  d   c  |
      //         \ |   d                      \ |
      //  s   ->  sel1                   s -> sel1
      //           |                            |
      //
      // This pattern is not handled elsewhere because `sel0` has other uses
      // than `sel1` so `sel0` does not inherit the selector condition `s==1`.
      //
      // It may be possible to bypass multiple selects so walk the edge up the
      // graph as far as possible. For example, in the diagram above `b` may
      // also be a select with a selector whose value is implied by `s`.
      //
      // This also applies to ANDs, ORs, and XORs, if the condition set implies
      // that all but one operand is the identity for the operation.
      if (operand->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel, Op::kAnd,
                         Op::kOr, Op::kXor})) {
        std::optional<Node*> replacement;
        Node* src = operand;
        while (src->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel, Op::kAnd,
                          Op::kOr, Op::kXor})) {
          if (src->Is<Select>()) {
            Select* select = src->As<Select>();
            if (select->selector()->Is<Literal>()) {
              break;
            }
            std::optional<Bits> implied_selector = ImpliedNodeValue(
                edge_set, select->selector(), *specialized_query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            Node* implied_case =
                GetSelectedCase(select, implied_selector.value());
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %v",
                operand->GetName(), node->GetName(),
                select->selector()->GetName(), select->GetName(),
                implied_selector.value());
            replacement = implied_case;
            src = implied_case;
          } else if (src->Is<PrioritySelect>()) {
            PrioritySelect* select = src->As<PrioritySelect>();
            if (select->selector()->Is<Literal>()) {
              break;
            }
            std::optional<TernaryVector> implied_selector = ImpliedNodeTernary(
                edge_set, select->selector(), *specialized_query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            std::optional<Node*> implied_case =
                GetSelectedCase(select, *implied_selector);
            if (!implied_case.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %s",
                operand->GetName(), node->GetName(),
                select->selector()->GetName(), select->GetName(),
                xls::ToString(*implied_selector));
            src = *implied_case;
            replacement = src;
          } else if (src->Is<OneHotSelect>()) {
            XLS_RET_CHECK(src->Is<OneHotSelect>());
            OneHotSelect* ohs = src->As<OneHotSelect>();
            if (ohs->selector()->Is<Literal>()) {
              break;
            }
            std::optional<TernaryVector> implied_selector = ImpliedNodeTernary(
                edge_set, ohs->selector(), *specialized_query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            for (int64_t case_no = 0; case_no < ohs->cases().size();
                 ++case_no) {
              if (implied_selector.value()[case_no] ==
                  TernaryValue::kKnownZero) {
                continue;
              }

              // This case could be selected - but if it's definitely zero when
              // selected, then we can ignore it.
              std::optional<Bits> implied_case = ImpliedNodeValue(
                  condition_map.GetEdgeConditionSet(ohs,
                                                    /*operand_no=*/case_no + 1),
                  ohs->cases()[case_no], *specialized_query_engine);
              if (implied_case.has_value() && implied_case->IsZero()) {
                implied_selector.value()[case_no] = TernaryValue::kKnownZero;
              }
            }
            std::optional<std::variant<Node*, ZeroValue>> implied_case =
                GetSelectedCase(ohs, *implied_selector);
            if (!implied_case.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %s",
                operand->GetName(), node->GetName(), ohs->selector()->GetName(),
                ohs->GetName(), xls::ToString(*implied_selector));
            if (std::holds_alternative<Node*>(*implied_case)) {
              src = std::get<Node*>(*implied_case);
            } else {
              XLS_RET_CHECK(std::holds_alternative<ZeroValue>(*implied_case));
              XLS_ASSIGN_OR_RETURN(
                  src,
                  f->MakeNode<Literal>(src->loc(), ZeroOfType(src->GetType())));
            }
            replacement = src;
          } else {
            XLS_RET_CHECK(src->OpIn({Op::kAnd, Op::kOr, Op::kXor}));
            auto is_identity = [&](const Bits& b) {
              if (src->op() == Op::kAnd) {
                return b.IsAllOnes();
              }
              return b.IsZero();
            };
            NaryOp* bitwise_op = src->As<NaryOp>();
            std::optional<Node*> nonidentity_operand = std::nullopt;
            for (Node* potential_src : bitwise_op->operands()) {
              XLS_RET_CHECK(potential_src->GetType()->IsBits());
              std::optional<Bits> implied_src = ImpliedNodeValue(
                  edge_set, potential_src, *specialized_query_engine);
              if (implied_src.has_value() && is_identity(*implied_src)) {
                continue;
              }
              if (nonidentity_operand.has_value()) {
                // There's more than one potentially-non-zero operand; we're
                // done, there's nothing to do.
                nonidentity_operand = std::nullopt;
                break;
              }
              nonidentity_operand = potential_src;
            }
            if (!nonidentity_operand.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply that bitwise operation "
                "%s has only one non-identity operand: %s",
                operand->GetName(), node->GetName(), bitwise_op->GetName(),
                nonidentity_operand.value()->GetName());
            src = *nonidentity_operand;
            replacement = src;
          }
        }
        if (replacement.has_value()) {
          VLOG(3) << absl::StreamFormat(
              "Replacing operand %d of %s with %s due to implied selector "
              "value(s)",
              operand_no, node->GetName(), replacement.value()->GetName());
          XLS_RETURN_IF_ERROR(
              node->ReplaceOperandNumber(operand_no, replacement.value()));
          changed = true;
        }
      }
    }
  }

  return changed;
}
absl::StatusOr<PassPipelineProto::Element>
ConditionalSpecializationPass::ToProto() const {
  // TODO(allight): This is not very elegant. Ideally the registry could handle
  // this? Doing it there would probably be even more weird though.
  PassPipelineProto::Element e;
  *e.mutable_pass_name() = use_bdd_ ? "cond_spec(Bdd)" : "cond_spec(noBdd)";
  return e;
}

XLS_REGISTER_MODULE_INITIALIZER(cond_spec, {
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(true)", true));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(Bdd)", true));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(false)", false));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(noBdd)", false));
});

}  // namespace xls

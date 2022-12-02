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

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_iterator.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {
namespace {

// Encapsulates a condition (a node holding a particular value) which can be
// assumed to be true at particular points in the graph.
struct Condition {
  Node* node;
  int64_t value;

  std::string ToString() const {
    return absl::StrFormat("%s==%d", node->GetName(), value);
  }

  bool operator==(const Condition& other) const {
    return node == other.node && value == other.value;
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
 public:
  // Use a btree to hold the conditions for a well defined order (determined by
  // ConditionCmp).
  using ConditionBTree = absl::btree_set<Condition, ConditionCmp>;

  explicit ConditionSet(const absl::flat_hash_map<Node*, int64_t>& topo_index)
      : condition_cmp_(topo_index), conditions_(condition_cmp_) {}

  ConditionSet(const ConditionSet&) = default;
  ConditionSet(ConditionSet&&) = default;
  ConditionSet& operator=(const ConditionSet&) = default;

  // Perform a set intersection with this set and `other` and assign the result
  // to this set.
  void Intersect(const ConditionSet& other) {
    ConditionBTree original = std::move(conditions_);
    conditions_.clear();
    std::set_intersection(other.conditions_.begin(), other.conditions_.end(),
                          original.begin(), original.end(),
                          std::inserter(conditions_, conditions_.begin()),
                          condition_cmp_);
    // Intersection should not increase set size.
    XLS_CHECK_LE(conditions_.size(), kMaxConditions);
  }

  // Adds a condition to the set.  Note: it is possible to add conflicting
  // conditions to the set (pred==0 and pred==1). This is an indication that the
  // node is necessarily dead. Arbitrary transformation to dead code is legal so
  // arbitrarily picking one of the conflicting conditions and transforming
  // based on it is fine.
  void AddCondition(const Condition& condition) {
    XLS_CHECK(!condition.node->Is<Literal>());
    conditions_.insert(condition);
    // The conditions are ordering in topological sort order (based on
    // Condition.node) and transformation occurs in reverse topological sort
    // order so the most distant conditions should be at the end of the
    // condition set.  Just pop the last condition off the end if it exceeds the
    // limit.
    if (conditions_.size() > kMaxConditions) {
      conditions_.erase(std::next(conditions_.end(), -1));
    }
    XLS_CHECK_LE(conditions_.size(), kMaxConditions);
  }

  const ConditionBTree& conditions() const { return conditions_; }

  std::string ToString() const {
    std::vector<std::string> pieces;
    for (const Condition& condition : conditions_) {
      pieces.push_back(condition.ToString());
    }
    return absl::StrCat("(", absl::StrJoin(pieces, " & "), ")");
  }

 private:
  // The maximum size limit on the number of conditions to avoid superlinear
  // behavior.
  static constexpr int64_t kMaxConditions = 64;
  ConditionCmp condition_cmp_;
  ConditionBTree conditions_;
};

// A map containing the set of conditions which can be assumed at each node (and
// at some edges). An example where a condition would be assigned to an edge
// rather than a node is a case arm of a select. In this case, a selector value
// can be assumed on the edge from the case operand to the select operation. In
// general, the condition cannot be assumed on the source node of the edge (case
// operand) because the source node may be used outside the case arm expression.
class ConditionMap {
 public:
  explicit ConditionMap(FunctionBase* f) {
    std::vector<Node*> topo_sort = TopoSort(f).AsVector();
    for (int64_t i = 0; i < topo_sort.size(); ++i) {
      topo_index_[topo_sort[i]] = i;
      // Initially all node conditions are empty.
      node_conditions_.emplace(topo_sort[i], topo_index_);
    }
  }

  // Returns the condition set for the given node. Returns a mutable reference
  // as this is the mechanism for setting condition sets of nodes.
  ConditionSet& GetNodeConditionSet(Node* node) {
    return node_conditions_.at(node);
  }

  // Sets the condition set for the given edge where the edge extends to `node`
  // and operand index `operand_no`.
  void SetEdgeConditionSet(Node* node, int64_t operand_no,
                           ConditionSet condition_set) {
    std::pair<Node*, int64_t> key = {node, operand_no};
    XLS_CHECK(!edge_conditions_.contains(key));
    edge_conditions_.insert({key, std::move(condition_set)});
  }

  // Returns the conditions which can be assumed along the edge(s) from node to
  // user. This interface is asymmetric to SetEdgeCondition (which takes a node
  // and operand number) to make it easier to use because at a particular node
  // you have easy access to the user list but not the operand number(s)
  // associated with each user.
  const ConditionSet& GetEdgeConditionSet(Node* node, Node* user) {
    // Find the unique (if there is one) operand number of user which
    // corresponds to node.
    std::optional<int64_t> operand_index;
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == node) {
        if (operand_index.has_value()) {
          // `node` appears in multiple operands of `user`. Return the
          // assumptions that can be made at the node `user` itself. This is
          // typically not a strong conditions as might be assuming along the
          // edges.
          return GetNodeConditionSet(user);
        }
        operand_index = i;
      }
    }
    XLS_CHECK(operand_index.has_value()) << absl::StreamFormat(
        "%s is not a user of %s", user->GetName(), node->GetName());
    std::pair<Node*, int64_t> key = {user, operand_index.value()};
    if (!edge_conditions_.contains(key)) {
      // There are no special conditions for this edge. Return the conditions on
      // the target of the edge which necessarily hold on the edge as well.
      return node_conditions_.at(user);
    }
    return edge_conditions_.at(key);
  }

 private:
  // Index of each node in the function base in a topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index_;

  // Set of conditions which might be assumed at each node.
  absl::flat_hash_map<Node*, ConditionSet> node_conditions_;

  // Set of conditions which might be assumed at some edges. The key defines an
  // edge as (node, operand_no). If no key exists for an edge, then there are no
  // special conditions for the edge, and the conditions for the edge are the
  // same as the node.
  absl::flat_hash_map<std::pair<Node*, int64_t>, ConditionSet> edge_conditions_;
};

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns abls::nullopt otherwise. `query_engine` can be
// null in which case BDD's are not used in the implication analysis.
std::optional<Bits> ImpliedNodeValue(const ConditionSet& condition_set,
                                      Node* node,
                                      const QueryEngine* query_engine) {
  for (const Condition& condition : condition_set.conditions()) {
    if (condition.node == node) {
      XLS_VLOG(3) << absl::StreamFormat("%s trivially implies %s==%d",
                                        condition_set.ToString(),
                                        node->GetName(), condition.value);
      return UBits(condition.value, node->BitCountOrDie());
    }
  }
  if (query_engine == nullptr) {
    return absl::nullopt;
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates;
  for (const Condition& condition : condition_set.conditions()) {
    for (int64_t i = 0; i < condition.node->BitCountOrDie(); ++i) {
      bool bit_value = i >= 64 ? false : ((condition.value >> i) & 1);
      predicates.push_back({TreeBitLocation{condition.node, i}, bit_value});
    }
  }
  std::optional<Bits> implied_value =
      query_engine->ImpliedNodeValue(predicates, node);

  if (implied_value.has_value()) {
    XLS_VLOG(3) << absl::StreamFormat("%s implies %s==%s",
                                      condition_set.ToString(), node->GetName(),
                                      implied_value->ToString());
  }
  return implied_value;
}

// Returns the case arm node of the given select which is selected when the
// selctor has the given value.
Node* GetSelectedCase(Select* select, const Bits& selector_value) {
  if (bits_ops::UGreaterThanOrEqual(selector_value, select->cases().size())) {
    return select->default_value().value();
  }
  // It is safe to convert to uint64_t because of the above check against cases
  // size.
  return select->get_case(selector_value.ToUint64().value());
}

}  // namespace

absl::StatusOr<bool> ConditionalSpecializationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  std::unique_ptr<BddQueryEngine> query_engine;
  if (use_bdd_) {
    query_engine = std::make_unique<BddQueryEngine>(
        BddFunction::kDefaultPathLimit, IsCheapForBdds);
    XLS_RETURN_IF_ERROR(query_engine->Populate(f).status());
  }

  ConditionMap condition_map(f);

  // Iterate backwards through the graph because we add conditions at the case
  // arm operands of selects and propagate them upwards through the expressions
  // which compute the case arm.
  bool changed = false;
  for (Node* node : ReverseTopoSort(f)) {
    ConditionSet& set = condition_map.GetNodeConditionSet(node);

    if (OpIsSideEffecting(node->op())) {
      // Inputs to side-effecting operations should not change so don't assume
      // any conditions for this node or it's predecessors.
      continue;
    }

    // Compute the intersection of the condition sets of the users of this node.
    //
    // If this node has an implicit use then we can't propagate any conditions
    // from the users because this value is unconditionally live and therefore
    // its computed value should not be changed.
    if (!f->HasImplicitUse(node)) {
      bool first_user = true;
      for (Node* user : node->users()) {
        if (first_user) {
          set = condition_map.GetEdgeConditionSet(node, user);
        } else {
          set.Intersect(condition_map.GetEdgeConditionSet(node, user));
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
          edge_set.AddCondition(Condition{select->selector(), case_no});
          XLS_VLOG(3) << absl::StreamFormat("ConditionSet for (%s, %d) : %s",
                                            node->GetName(), case_no + 1,
                                            edge_set.ToString());
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
      }
    }
    XLS_VLOG(3) << absl::StreamFormat("Conditions for %s : %s", node->GetName(),
                                      set.ToString());

    // Now specialize any operands (if possible) based on the conditions.
    for (int64_t operand_no = 0; operand_no < node->operand_count();
         ++operand_no) {
      Node* operand = node->operand(operand_no);

      if (operand->Is<Literal>()) {
        // Operand is already a literal. Nothing to do.
        continue;
      }

      const ConditionSet& edge_set =
          condition_map.GetEdgeConditionSet(operand, node);

      // First check to see if the condition set directly implies a value for
      // the operand. If so replace with the implied value.
      if (std::optional<Bits> implied_value =
              ImpliedNodeValue(edge_set, operand, query_engine.get());
          implied_value.has_value()) {
        XLS_VLOG(3) << absl::StreamFormat("Replacing operand %d of %s with %s",
                                          operand_no, node->GetName(),
                                          implied_value->ToString());
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
      if (operand->Is<Select>()) {
        std::optional<Node*> replacement;
        Node* src = operand;
        while (src->Is<Select>()) {
          Select* select = src->As<Select>();
          if (select->selector()->Is<Literal>()) {
            break;
          }
          std::optional<Bits> implied_selector = ImpliedNodeValue(
              edge_set, select->selector(), query_engine.get());
          if (!implied_selector.has_value()) {
            break;
          }
          Node* implied_case =
              GetSelectedCase(select, implied_selector.value());
          XLS_VLOG(4) << absl::StreamFormat(
              "Conditions for edge (%s, %s) imply selector %s of select %s has "
              "value %s",
              operand->GetName(), node->GetName(),
              select->selector()->GetName(), select->GetName(),
              implied_selector->ToString());
          replacement = implied_case;
          src = implied_case;
        }
        if (replacement.has_value()) {
          XLS_VLOG(3) << absl::StreamFormat(
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

}  // namespace xls

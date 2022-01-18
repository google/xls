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

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace {

// Encapsulates a condition (a node holding a particular value) which can be
// assumed to be true at particular points in the graph.
struct Condition {
  Node* node;
  int64_t value;

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
      pieces.push_back(absl::StrFormat("%s==%d", condition.node->GetName(),
                                       condition.value));
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

}  // namespace

absl::StatusOr<bool> ConditionalSpecializationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  bool changed = false;

  std::vector<Node*> topo_sort = TopoSort(f).AsVector();
  absl::flat_hash_map<Node*, int64_t> topo_index;
  for (int64_t i = 0; i < topo_sort.size(); ++i) {
    topo_index[topo_sort[i]] = i;
  }

  absl::flat_hash_map<Node*, ConditionSet> condition_sets;

  // Iterate backwards through the graph because we add conditions at the case
  // arm operands of selects and propagate them upwards through the expressions
  // which compute the case arm.
  for (auto it = topo_sort.rbegin(); it != topo_sort.rend(); ++it) {
    Node* node = *it;
    ConditionSet& set = condition_sets.emplace(node, topo_index).first->second;

    if (OpIsSideEffecting(node->op())) {
      // Inputs to side-effecting operations should not change so don't assume
      // any conditions for this node or it's predecessors.
      continue;
    }

    // Compute the intersection of the condition sets of the users of this node.
    bool first_user = true;
    for (Node* user : node->users()) {
      if (first_user) {
        set = condition_sets.at(user);
      } else {
        set.Intersect(condition_sets.at(user));
      }
      first_user = false;
    }

    // If the only user of this node is a single select arm then add a condition
    // based on the selector value for this arm.
    // TODO(meheff): 2021/12/21 Handle one-hot-selects.
    if (node->users().size() == 1) {
      Node* user = *(node->users().begin());
      if (user->Is<Select>() && user->OperandInstanceCount(node) == 1) {
        Select* select = user->As<Select>();
        // Don't bother specializing if the selector is a literal as this
        // results in a useless condition where we assume a literal has a
        // literal value.
        if (!select->selector()->Is<Literal>()) {
          for (int64_t case_no = 0; case_no < select->cases().size();
               ++case_no) {
            if (node == select->get_case(case_no)) {
              set.AddCondition(Condition{select->selector(), case_no});
            }
          }
        }
      }
    }

    XLS_VLOG(3) << absl::StreamFormat("Conditions for %s : %s", node->GetName(),
                                      set.ToString());

    // Now specialize any operands (if possible) based on the conditions. If any
    // of the operands of this node match any assumed conditions (node is the
    // same), then replace the operand with the literal value.
    for (int64_t operand_no = 0; operand_no < node->operand_count();
         ++operand_no) {
      Node* operand = node->operand(operand_no);
      for (const Condition& condition : set.conditions()) {
        if (operand == condition.node) {
          // Replace operand with literal of value condition.value.
          XLS_VLOG(3) << absl::StreamFormat(
              "Replacing operand %d of %s with %d", operand_no, node->GetName(),
              condition.value);
          XLS_ASSIGN_OR_RETURN(
              Literal * literal,
              f->MakeNode<Literal>(
                  operand->loc(),
                  Value(UBits(condition.value,
                              condition.node->BitCountOrDie()))));
          XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(operand_no, literal));
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace xls

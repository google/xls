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

#include "xls/passes/predicate_dominator_analysis.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xls/common/logging/logging.h"
#include "xls/common/strong_int.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/passes/predicate_state.h"

namespace xls {

namespace {
// head pointer type
XLS_DEFINE_STRONG_INT_TYPE(PredicateStackId, int64_t);

static std::ostream& operator<<(std::ostream& os,
                                const std::optional<PredicateStackId>& p) {
  if (p.has_value()) {
    return os << *p;
  }
  return os << "[Unassigned]";
}

static constexpr PredicateStackId kRootPredicateId(0);

// Node in the linked-list of all predicates we've visited.
//
// We assign each node a list of all the select predicates that guard the demand
// for that value, ordered from most-to-least specific. This struct is a node in
// that list and needs to be interpreted with the predicate_stacks_ list to
// provide the arena the 'previous' field (which is the linked-list successor
// pointer) is interpreted in. To avoid huge memory use all IR nodes share the
// list elements, with each just pointing to the list node which is their lists
// head. One way to think of this list is like a LISP list where the crd's are
// shared among many cars. This makes the structure somewhat resemble a reversed
// tree where there are no child-links. Note that as far as each node is
// concerned its just a linked list. The fact that some are pointed to by many
// other nodes is not visible to a list node.
//
// ## Algorithm
//
// Traverse the node graph from return values up towards parameters (in reverse
// topo-sort order). For each node merge the predicate stack of its users by
// finding their common predication path in the graph (ie suffix string).
// This is O(N^3) in the worst case (since its for each node (N) check each user
// (N) for the whole predicate stack size (N)) in a pathological case where
// stacked selects are collapsed. This would be a highly degenerate graph
// however. Normal graphs are generally not very top-heavy with most nodes only
// having a few (or zero) predicates guarding their use.
struct PredicateStackNode {
  // Select with the arm we are choosing.
  Node* const selector;
  // Which arm was chosen
  const PredicateState::ArmT arm;
  // Dominating selector
  const PredicateStackId previous;
  // How long the linked-list is.
  const size_t distance_to_root;

  bool IsRootPredicate() const {
    bool is_root = distance_to_root == 0;
    if (is_root) {
      CHECK_EQ(previous, kRootPredicateId);
      CHECK_EQ(selector, nullptr);
    }
    return is_root;
  }
};

class AnalysisHelper {
 public:
  static constexpr PredicateStackNode kRootPredicateStackNode{
      .selector = nullptr,
      .arm = 0,
      .previous = kRootPredicateId,
      .distance_to_root = 0};

  explicit AnalysisHelper(FunctionBase* func) : function_(func) {}

  absl::flat_hash_map<Node*, PredicateState> Analyze() {
    CHECK(node_states_.empty());
    CHECK(predicate_stacks_.empty());
    node_states_.reserve(function_->node_count());
    predicate_stacks_.push_back(kRootPredicateStackNode);
    // Run in reverse topo sort order. Handle users before the values they use.
    for (Node* node : ReverseTopoSort(function_)) {
      HandleNode(node);
    }

    // Every node should have a node-state, covert these into the PredicateState
    absl::flat_hash_map<Node*, PredicateState> result;
    result.reserve(node_states_.size());
    for (const auto& [node, head] : node_states_) {
      result[node] =
          (head == kRootPredicateId)
              ? PredicateState()
              : PredicateState(
                    predicate_stacks_[head.value()].selector->As<Select>(),
                    predicate_stacks_[head.value()].arm);
    }
    return result;
  }

  void HandleNode(Node* node) {
    // Handle return nodes.
    if (node->users().empty()) {
      node_states_[node] = kRootPredicateId;
      return;
    }
    std::optional<PredicateStackId> head;
    for (Node* user : node->users()) {
      if (user->Is<Select>()) {
        head = HandlePossibleNewArm(head, node, user->As<Select>());
      } else {
        head = HandleStandardNodeUse(head, node, user);
      }
      CHECK(head.has_value());
      node_states_[node] = *head;
    }
  }

  PredicateStackId HandleStandardNodeUse(std::optional<PredicateStackId> head,
                                         Node* node, Node* user) {
    // Join with other state.
    return FindJoinPoint(head, node_states_[user]);
  }

  // Select users specifically need to be handled specially because they have
  // new branches. We might just be (or include) a selector value however.
  PredicateStackId HandlePossibleNewArm(std::optional<PredicateStackId> head,
                                        Node* node, Select* user) {
    if (user->selector() == node) {
      // The selector is always demanded.  We don't need to check cases since
      // any uses in there would unify down to the selector's result anyway.
      // TODO(allight): 2023-07-27 For some optimizations it might be useful to
      // support asking about the set of predicates which protect specific edges
      // (or at least select edges) in which case we'd want to continue down to
      // create those predicate-stacks. Currently that isn't needed however.
      return HandleStandardNodeUse(head, node, user);
    }
    // Update the state to take into account a new arm.
    auto handle_arm = [&](Node* selected_value, PredicateState::ArmT arm) {
      if (selected_value != node) {
        return;
      }
      // cons this arm onto the predicate list of the user.
      PredicateStackId prev = node_states_[user];
      PredicateStackNode cons_set{
          .selector = user,
          .arm = arm,
          .previous = prev,
          .distance_to_root =
              predicate_stacks_[prev.value()].distance_to_root + 1};
      PredicateStackId new_id = NextId();
      predicate_stacks_.push_back(cons_set);
      // Find the actual join-point.
      head = FindJoinPoint(new_id, head);
    };
    for (int64_t i = 0; i < user->cases().size(); ++i) {
      handle_arm(user->cases()[i], /*arm=*/i);
    }
    if (user->default_value().has_value()) {
      handle_arm(*user->default_value(), DefaultArm{});
    }
    CHECK(head.has_value()) << user << " is marked as user of " << node
                            << " but no arms or conditions appear to use it?";
    return *head;
  }

  PredicateStackId NextId() const {
    return PredicateStackId(predicate_stacks_.size());
  }

  // Since set-ids monotonically increase we could do this with each node
  // holding a bit-map and count-leading-zeros?  Normally these select-chains
  // aren't terribly deep compared to the number of selects (i.e. most selects
  // are independent of one another on a structural level) so doing the list
  // traversal is better thanks to saving memory allocation.
  //
  // The pathological case where this functions non-optimally is just a huge
  // number of nested ifs.
  PredicateStackId FindJoinPoint(std::optional<PredicateStackId> a,
                                 std::optional<PredicateStackId> b) {
    CHECK(a.has_value() || b.has_value()) << "Both predicates unassigned!";
    if (a.has_value()) {
      CHECK_LT(a, NextId()) << "Invalid list head!";
    }
    if (b.has_value()) {
      CHECK_LT(b, NextId()) << "Invalid list head!";
    }
    // a & b are each cons-lists
    if (a == kRootPredicateId || b == kRootPredicateId) {
      // Don't need to explore to know root is end of the line.
      return kRootPredicateId;
    }
    if (a == b) {
      // At least one has_value from check above and they are equal so just pick
      // one.
      return a.value();
    }
    if (!a.has_value()) {
      return b.value();
    }
    if (!b.has_value()) {
      return a.value();
    }
    const PredicateStackNode& a_set = predicate_stacks_[a->value()];
    const PredicateStackNode& b_set = predicate_stacks_[b->value()];
    // Simplify by ensuring that b is always longer.
    if (a_set.distance_to_root > b_set.distance_to_root) {
      return FindJoinPoint(b, a);
    }
    // a_set is smaller than b_set
    size_t len_diff = b_set.distance_to_root - a_set.distance_to_root;
    PredicateStackId b_candidate = b.value();
    PredicateStackId a_candidate = a.value();
    // Get sublist of b that is same length as a
    for (size_t i = 0; i < len_diff; ++i) {
      // (tail b)
      CHECK_LT(b_candidate, NextId()) << "Invalid list head!";
      b_candidate = predicate_stacks_[b_candidate.value()].previous;
    }
    // Walk down both lists until we find a join.
    while (a_candidate != b_candidate) {
      CHECK_LT(a_candidate, NextId()) << "Invalid list head!";
      CHECK_LT(b_candidate, NextId()) << "Invalid list head!";
      CHECK_NE(a_candidate, kRootPredicateId)
          << "list element had invalid length!";
      CHECK_NE(b_candidate, kRootPredicateId)
          << "list element had invalid length!";
      a_candidate = predicate_stacks_[a_candidate.value()].previous;
      b_candidate = predicate_stacks_[b_candidate.value()].previous;
    }
    CHECK_LT(a_candidate, NextId()) << "Invalid list head!";
    return a_candidate;
  }

 private:
  FunctionBase* function_;
  // Map of node to the predicate list head they are guarded by.
  absl::flat_hash_map<Node*, PredicateStackId> node_states_;
  // Map from 'PredicateStackId' to the predicate node.
  std::vector<PredicateStackNode> predicate_stacks_;
};
}  // namespace

PredicateDominatorAnalysis PredicateDominatorAnalysis::Run(FunctionBase* f) {
  AnalysisHelper helper(f);
  return PredicateDominatorAnalysis(helper.Analyze());
}

}  // namespace xls

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

#include "xls/ir/node_iterator.h"

#include <deque>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

void NodeIterator::Initialize() {
  // For topological traversal we only add nodes to the order when all of its
  // users have been scheduled.
  //
  //       o    node, now ready, can be added to order!
  //      /|\
  //     v v v
  //     o o o  (users, all present in order)
  //
  // When a node is placed into the ordering, we place all of its operands into
  // the "pending_to_remaining_users" mapping if it is not yet present -- this
  // keeps track of how many more users must be seen (before that node is ready
  // to place into the ordering).
  //
  // NOTE: Initialize sorts reverse-topologically.  To sort topologically,
  // reverse the this->ordered_
  absl::flat_hash_map<Node*, int64_t> pending_to_remaining_users;
  pending_to_remaining_users.reserve(f_->node_count());
  std::deque<Node*> ready;

  ordered_ = std::make_unique<std::vector<Node*>>();
  ordered_->reserve(f_->node_count());

  auto is_scheduled = [&](Node* n) {
    auto it = pending_to_remaining_users.find(n);
    if (it == pending_to_remaining_users.end()) {
      return false;
    }
    return it->second < 0;
  };
  auto all_users_scheduled = [&](Node* n) {
    return absl::c_all_of(n->users(), is_scheduled);
  };
  auto bump_down_remaining_users = [&](Node* n) {
    XLS_CHECK(!n->users().empty());
    auto result = pending_to_remaining_users.insert({n, n->users().size()});
    auto it = result.first;
    int64_t& remaining_users = it->second;
    XLS_CHECK_GT(remaining_users, 0);
    remaining_users -= 1;
    XLS_VLOG(5) << "Bumped down remaining users for: " << n
                << "; now: " << remaining_users;
    if (remaining_users == 0) {
      ready.push_back(result.first->first);
      remaining_users -= 1;
    }
  };
  auto add_to_order = [&](Node* r) {
    XLS_VLOG(5) << "Adding node to order: " << r;
    DCHECK(all_users_scheduled(r)) << r << " users size: " << r->users().size();
    ordered_->push_back(r);

    // We want to be careful to only bump down our operands once, since we're a
    // single user, even though we may refer to them multiple times in our
    // operands sequence.
    absl::flat_hash_set<Node*> seen_operands;
    seen_operands.reserve(r->operand_count());
    for (auto it = r->operands().rbegin(); it != r->operands().rend(); ++it) {
      Node* operand = *it;
      if (auto [_, inserted] = seen_operands.insert(operand); inserted) {
        bump_down_remaining_users(operand);
      }
    }
  };

  auto seed_ready = [&](Node* n) {
    ready.push_front(n);
    XLS_CHECK(pending_to_remaining_users.insert({n, -1}).second);
  };

  auto is_return_value = [&](Node* n) {
    return n->function_base()->IsFunction() &&
           (n == n->function_base()->AsFunctionOrDie()->return_value());
  };

  Node* return_value = nullptr;
  for (Node* node : f_->nodes()) {
    if (node->users().empty()) {
      if (is_return_value(node)) {
        // Note: we special case the return value so it always comes at the
        // front.
        return_value = node;
      } else {
        DCHECK(all_users_scheduled(node));
        XLS_VLOG(5) << "At start node was ready: " << node;
        seed_ready(node);
      }
    }
  }

  if (return_value != nullptr) {
    XLS_VLOG(5) << "Maybe marking return value as ready: " << return_value;
    seed_ready(return_value);
  }

  while (!ready.empty()) {
    Node* r = ready.front();
    ready.pop_front();
    add_to_order(r);
  }

  if (ordered_->size() < f_->node_count()) {
    // Not all nodes have been placed indicating a cycle in the graph. Run a
    // trivial DFS visitor which will emit an error message displaying the
    // cycle.
    class CycleChecker : public DfsVisitorWithDefault {
      absl::Status DefaultHandler(Node* node) override {
        return absl::OkStatus();
      }
    };
    CycleChecker cycle_checker;
    XLS_CHECK_OK(f_->Accept(&cycle_checker));
    XLS_LOG(FATAL) << "Expected to find cycle in function base.";
  }
}

}  // namespace xls

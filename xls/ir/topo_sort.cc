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

#include "xls/ir/topo_sort.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

std::vector<Node*> ReverseTopoSort(FunctionBase* f,
                                   std::optional<absl::BitGenRef> randomizer) {
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
  // NOTE: sorts reverse-topologically.  To sort topologically, reverse the
  // result.
  absl::flat_hash_map<Node*, int64_t> pending_to_remaining_users;
  pending_to_remaining_users.reserve(f->node_count());
  std::deque<Node*> ready;

  std::vector<Node*> ordered;
  ordered.reserve(f->node_count());

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

  auto seed_ready = [&](Node* n) {
    ready.push_front(n);
    CHECK(pending_to_remaining_users.insert({n, -1}).second);
  };

  auto is_return_value = [&](Node* n) {
    return n->function_base()->IsFunction() &&
           (n == n->function_base()->AsFunctionOrDie()->return_value());
  };

  Node* return_value = nullptr;
  for (Node* node : f->nodes()) {
    if (node->users().empty()) {
      if (is_return_value(node)) {
        // Note: we special case the return value so it always comes at the
        // front.
        return_value = node;
      } else {
        DCHECK(all_users_scheduled(node));
        VLOG(5) << "At start node was ready: " << node;
        seed_ready(node);
      }
    }
  }

  if (return_value != nullptr) {
    VLOG(5) << "Maybe marking return value as ready: " << return_value;
    seed_ready(return_value);
  }

  std::optional<absl::flat_hash_map<Node*, int64_t>> random_priority;
  auto random_comparator = [&](Node* a, Node* b) {
    CHECK(random_priority.has_value());
    return random_priority->at(a) < random_priority->at(b);
  };
  if (randomizer.has_value()) {
    std::vector<Node*> random_order(f->nodes().begin(), f->nodes().end());
    absl::c_shuffle(random_order, *randomizer);
    for (int64_t i = 0; i < random_order.size(); ++i) {
      if (random_order[i] == return_value) {
        // Make sure the return value is always highest-priority, so it comes
        // first in the output.
        std::swap(random_order[i], random_order[random_order.size() - 1]);
      }
    }

    random_priority.emplace();
    random_priority->reserve(random_order.size());
    for (int64_t i = 0; i < random_order.size(); ++i) {
      CHECK(random_priority->emplace(random_order[i], i).second);
    }

    absl::c_make_heap(ready, random_comparator);
  }

  auto bump_down_remaining_users = [&](Node* n) {
    CHECK(!n->users().empty());
    auto result = pending_to_remaining_users.insert({n, n->users().size()});
    auto it = result.first;
    int64_t& remaining_users = it->second;
    CHECK_GT(remaining_users, 0);
    remaining_users -= 1;
    VLOG(5) << "Bumped down remaining users for: " << n
            << "; now: " << remaining_users;
    if (remaining_users == 0) {
      ready.push_back(result.first->first);
      if (random_priority.has_value()) {
        absl::c_push_heap(ready, random_comparator);
      }
      remaining_users -= 1;
    }
  };

  absl::flat_hash_set<Node*> seen_operands;
  auto add_to_order = [&](Node* r) {
    VLOG(5) << "Adding node to order: " << r;
    DCHECK(all_users_scheduled(r)) << r << " users size: " << r->users().size();
    ordered.push_back(r);

    // We share seen_operands across invocations of add_to_order to reduce
    // overhead of constructing/allocating a set each time. Clear it before
    // using it.
    seen_operands.clear();

    // We want to be careful to only bump down our operands once, since we're a
    // single user, even though we may refer to them multiple times in our
    // operands sequence.
    for (auto it = r->operands().rbegin(); it != r->operands().rend(); ++it) {
      Node* operand = *it;
      if (auto [_, inserted] = seen_operands.insert(operand); inserted) {
        bump_down_remaining_users(operand);
      }
    }
  };

  while (!ready.empty()) {
    Node* r;
    if (random_priority.has_value()) {
      absl::c_pop_heap(ready, random_comparator);
      r = ready.back();
      ready.pop_back();
    } else {
      r = ready.front();
      ready.pop_front();
    }
    add_to_order(r);
  }

  if (ordered.size() < f->node_count()) {
    // Not all nodes have been placed indicating a cycle in the graph. Run a
    // trivial DFS visitor which will emit an error message displaying the
    // cycle.
    class CycleChecker : public DfsVisitorWithDefault {
      absl::Status DefaultHandler(Node* node) override {
        return absl::OkStatus();
      }
    };
    CycleChecker cycle_checker;
    CHECK_OK(f->Accept(&cycle_checker));
    LOG(FATAL) << "Expected to find cycle in function base.";
  }

  return ordered;
}

std::vector<Node*> TopoSort(FunctionBase* f,
                            std::optional<absl::BitGenRef> randomizer) {
  std::vector<Node*> ordered = ReverseTopoSort(f, randomizer);
  std::reverse(ordered.begin(), ordered.end());
  return ordered;
}

}  // namespace xls

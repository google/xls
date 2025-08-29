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
#include <limits>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

std::vector<Node*> ReverseTopoSort(FunctionBase* f,
                                   std::optional<absl::BitGenRef> randomizer) {
  // We store both the resulting order & the nodes that are ready to be added to
  // the order in the same vector; the initial segment is the
  // finished order (in reverse topo-sort order), and `[num_ordered, end)` is
  // the worklist of nodes that are ready to be added to the order. The ready
  // nodes are in the order in which we found they were ready; any of them can
  // be added to the order without violating the reverse topo-sort constraint.
  int64_t num_ordered = 0;
  std::vector<Node*> result;
  result.reserve(f->node_count());
  auto ready_view = [&]() {
    return absl::MakeSpan(result).subspan(num_ordered);
  };

  // For topological traversal we only add nodes to the order when all of its
  // users have been scheduled.
  //
  //       o    node, now ready, can be added to order!
  //      /|\
  //     v v v
  //     o o o  (users, all present in order)
  //
  // We start by adding all nodes to the `remaining_users` mapping, and those
  // with no users to the `ready` stack (always putting the return value at the
  // end of the stack). The `remaining_users` mapping is used to track how many
  // users must be seen before a node is ready to be added to the order.
  //
  // NOTE: sorts reverse-topologically.  To sort topologically, reverse the
  // result.
  absl::flat_hash_map<Node*, int64_t> remaining_users;
  remaining_users.reserve(f->node_count());

  // Note: we special case the return value if it has no users, always putting
  // it on the `ready` queue first so it comes at the start of the order.
  Node* return_value =
      f->IsFunction() ? f->AsFunctionOrDie()->return_value() : nullptr;
  if (return_value != nullptr && return_value->users().empty()) {
    VLOG(5) << "Marking return value as ready: " << return_value;
    CHECK(remaining_users.emplace(return_value, 0).second);
    result.push_back(return_value);
  }
  for (Node* node : f->nodes_reversed()) {
    auto [it, inserted] = remaining_users.emplace(node, node->users().size());
    CHECK(inserted || (node == return_value && return_value->users().empty()));
    if (node == return_value) {
      continue;
    }
    const int64_t& node_remaining_users = it->second;

    if (node_remaining_users == 0) {
      VLOG(5) << "At start node was ready: " << node;
      result.push_back(node);
    }
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

    absl::Span<Node*> ready = ready_view();
    absl::c_make_heap(ready, random_comparator);
  }

  // For nodes with many operands, we de-duplicate operands using a hash set;
  // otherwise, we use a linear search over the operands we've already
  // processed. We share a single hash set to avoid re-allocating it for each
  // node.
  constexpr int64_t kMinOperandsForHashSetDeduplication = 16;
  absl::flat_hash_set<Node*> seen_operands;

  while (!ready_view().empty()) {
    if (random_priority.has_value()) {
      absl::Span<Node*> ready = ready_view();
      absl::c_pop_heap(ready, random_comparator);

      Node* removed = result.back();
      result.pop_back();
      result.insert(result.begin() + num_ordered, removed);
    }
    Node* r = result[num_ordered];

    VLOG(5) << "Adding node to order: " << r;
    ++num_ordered;

    const bool use_hash_set_deduplication =
        r->operand_count() >= kMinOperandsForHashSetDeduplication;
    if (use_hash_set_deduplication) {
      // Reset the hash set's contents *without* reducing its capacity, so we
      // can avoid unnecessary re-allocation.
      seen_operands.erase(seen_operands.begin(), seen_operands.end());

      seen_operands.reserve(r->operand_count());
    }
    for (int64_t operand_no = r->operand_count() - 1; operand_no >= 0;
         --operand_no) {
      Node* operand = r->operand(operand_no);
      if (use_hash_set_deduplication) {
        if (auto [_, inserted] = seen_operands.insert(operand); !inserted) {
          // We've already seen this operand.
          continue;
        }

      } else if (absl::c_linear_search(r->operands().subspan(operand_no + 1),
                                       operand)) {
        // We've already seen this operand.
        continue;
      }
      int64_t& operand_remaining_users = remaining_users.at(operand);
      DCHECK_GT(operand_remaining_users, 0);
      if (--operand_remaining_users == 0) {
        result.push_back(operand);
        if (random_priority.has_value()) {
          absl::Span<Node*> ready = ready_view();
          absl::c_push_heap(ready, random_comparator);
        }
      }
    }
  }

  if (num_ordered < f->node_count()) {
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

  return result;
}

std::vector<Node*> TopoSort(FunctionBase* f,
                            std::optional<absl::BitGenRef> randomizer) {
  std::vector<Node*> ordered = ReverseTopoSort(f, randomizer);
  std::reverse(ordered.begin(), ordered.end());
  return ordered;
}

std::vector<Node*> StableTopoSort(FunctionBase* f,
                                  absl::Span<int64_t const> reference_order) {
  std::vector<Node*> result;
  result.reserve(f->node_count());

  absl::flat_hash_map<Node*, int64_t> remaining_users;
  remaining_users.reserve(f->node_count());

  // Assign unique priorities to every node:
  // - The priority of nodes in the reference order is their index in the order.
  // - Nodes not in the reference order have higher priority than nodes in the
  //   reference order. This ensures that nodes not in the reference order do
  //   not interfere with ordering decisions made between nodes in the reference
  //   order by blocking the addition of reference order nodes to the ready
  //   list.
  absl::flat_hash_map<Node*, int64_t> priority;
  priority.reserve(f->node_count());
  absl::flat_hash_map<int64_t, int64_t> id_to_index;
  id_to_index.reserve(reference_order.size());
  for (int64_t i = 0; i < reference_order.size(); ++i) {
    id_to_index[reference_order[i]] = i;
  }
  int64_t next_priority = reference_order.size();
  for (Node* node : f->nodes()) {
    auto it = id_to_index.find(node->id());
    if (it != id_to_index.end()) {
      priority[node] = it->second;
    } else if (f->IsFunction() &&
               node == f->AsFunctionOrDie()->return_value()) {
      // Prefer return value at the end all else being equal.
      priority[node] = std::numeric_limits<int64_t>::max();
    } else {
      priority[node] = next_priority++;
    }
  }

  auto comparator = [&](Node* a, Node* b) {
    // A less-than comparator will put the max elment at the end of the heap
    // after c_pop_heap is called meaning higher priority nodes generally end up
    // later in the returned sort (after the final reversal).
    return priority.at(a) < priority.at(b);
  };

  std::vector<Node*> ready;
  for (Node* node : f->nodes_reversed()) {
    auto [it, inserted] = remaining_users.emplace(node, node->users().size());
    CHECK(inserted);
    const int64_t& node_remaining_users = it->second;
    if (node_remaining_users == 0) {
      VLOG(5) << "At start node was ready: " << node;
      ready.push_back(node);
    }
  }

  absl::c_make_heap(ready, comparator);

  absl::flat_hash_set<Node*> seen_operands;
  while (!ready.empty()) {
    absl::c_pop_heap(ready, comparator);
    Node* r = ready.back();
    ready.pop_back();
    result.push_back(r);

    VLOG(5) << "Adding node to order: " << r;

    seen_operands.clear();
    seen_operands.reserve(r->operand_count());
    for (int64_t operand_no = r->operand_count() - 1; operand_no >= 0;
         --operand_no) {
      Node* operand = r->operand(operand_no);
      if (auto [_, inserted] = seen_operands.insert(operand); !inserted) {
        continue;
      }
      int64_t& operand_remaining_users = remaining_users.at(operand);
      DCHECK_GT(operand_remaining_users, 0);
      if (--operand_remaining_users == 0) {
        ready.push_back(operand);
        absl::c_push_heap(ready, comparator);
      }
    }
  }

  if (result.size() < f->node_count()) {
    class CycleChecker : public DfsVisitorWithDefault {
      absl::Status DefaultHandler(Node* node) override {
        return absl::OkStatus();
      }
    };
    CycleChecker cycle_checker;
    CHECK_OK(f->Accept(&cycle_checker));
    LOG(FATAL) << "Expected to find cycle in function base.";
  }

  std::reverse(result.begin(), result.end());
  return result;
}

}  // namespace xls

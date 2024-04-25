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

#include "xls/delay_model/delay_heap.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace sched {

DelayHeap::PathLength DelayHeap::MaxAmongPredecessors(Node* node) const {
  PathLength length{0, 0};
  for (Node* p : predecessors(node)) {
    if (contains(p)) {
      length.critical_path_delay =
          std::max(length.critical_path_delay, CriticalPathDelay(p));
      length.longest_path =
          std::max(length.longest_path, path_lengths_.at(p).longest_path);
    }
  }
  return length;
}

absl::Status DelayHeap::Add(Node* node) {
  CHECK(!contains(node));
  DCHECK(!std::any_of(successors(node).begin(), successors(node).end(),
                      [&](Node* n) { return contains(n); }));
  XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                       delay_estimator_.GetOperationDelayInPs(node));
  PathLength path = MaxAmongPredecessors(node);
  path_lengths_[node] = {path.critical_path_delay + node_delay,
                         path.longest_path + 1};
  frontier_.insert(node);

  // Adding this node may have covered up other nodes previously on the
  // frontier. Remove these nodes.
  for (Node* p : predecessors(node)) {
    if (contains(p)) {
      auto it = frontier_.find(p);
      if (it != frontier_.end()) {
        frontier_.erase(it);
      }
    }
  }
  return absl::OkStatus();
}

DelayHeap::FrontierSet::iterator DelayHeap::Remove(Node* node) {
  DCHECK(!std::any_of(successors(node).begin(), successors(node).end(),
                      [&](Node* n) { return contains(n); }));
  // Node must be on the frontier to be erased.
  auto it = frontier_.find(node);
  CHECK(it != frontier_.end())
      << "Node " << node->GetName() << " is not on the frontier of the heap";

  // Removing nodes may have uncovered which are now at the frontier.
  for (Node* p : predecessors(node)) {
    if (contains(p) &&
        std::all_of(successors(p).begin(), successors(p).end(),
                    [&](Node* n) { return n == node || !contains(n); })) {
      frontier_.insert(p);
    }
  }

  auto next_it = std::next(it);
  frontier_.erase(it);
  path_lengths_.erase(node);
  return next_it;
}

absl::StatusOr<int64_t> DelayHeap::CriticalPathDelayAfterAdding(
    Node* node) const {
  CHECK(!contains(node));
  XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                       delay_estimator_.GetOperationDelayInPs(node));
  return std::max(CriticalPathDelay(),
                  MaxAmongPredecessors(node).critical_path_delay + node_delay);
}

int64_t DelayHeap::CriticalPathDelayAfterRemoving(Node* node) const {
  CHECK(contains(node));
  DCHECK_EQ(frontier_.count(node), 1);
  if (node != top()) {
    return CriticalPathDelay();
  }
  if (size() == 1) {
    return 0;
  }
  // Node is the top. After removing the top, the node with the next longest
  // critical path is either the second node on the frontier (if it exists) or a
  // predecessor of top.
  int64_t predecessor_delay = MaxAmongPredecessors(node).critical_path_delay;
  if (frontier_.size() == 1) {
    return predecessor_delay;
  }
  return std::max(CriticalPathDelay(*std::next(frontier_.begin())),
                  predecessor_delay);
}

std::string DelayHeap::ToString() const {
  std::string out;
  absl::StrAppend(&out, "Nodes in DelayHeap:\n");
  std::vector<Node*> nodes;
  for (const auto& pair : path_lengths_) {
    nodes.push_back(pair.first);
  }
  std::sort(nodes.begin(), nodes.end(), [&](Node* a, Node* b) {
    return path_lengths_.at(a) < path_lengths_.at(b) ||
           (path_lengths_.at(a) == path_lengths_.at(b) && (a->id() < b->id()));
  });
  for (Node* node : nodes) {
    absl::StrAppend(&out, "  ", node->GetName(), "\n");
  }
  absl::StrAppend(&out, "Frontier:\n");
  for (Node* node : frontier_) {
    absl::StrAppend(&out, "  ", node->GetName(), "\n");
  }
  return out;
}

absl::Span<Node* const> DelayHeap::GetUsersSpan(Node* node) const {
  auto it = users_vectors_.find(node);
  if (it != users_vectors_.end()) {
    return it->second;
  }
  std::vector<Node*>& users_vector = users_vectors_[node];
  users_vector.insert(users_vector.begin(), node->users().begin(),
                      node->users().end());
  return users_vector;
}

}  // namespace sched
}  // namespace xls

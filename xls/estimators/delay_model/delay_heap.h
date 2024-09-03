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

#ifndef XLS_ESTIMATORS_DELAY_MODEL_DELAY_HEAP_H_
#define XLS_ESTIMATORS_DELAY_MODEL_DELAY_HEAP_H_

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/node.h"

namespace xls {
namespace sched {

enum class Direction { kGrowsTowardUsers, kGrowsTowardOperands };

// A data structure for tracking the critical-path delay through an arbitrary
// set of nodes contained in an XLS function while allowing fast addition and
// removal of nodes to/from the set. An important use-case for this data
// structure is tracking the delay of a stage of a pipeline schedule while
// adding and removing nodes from the stage.
//
// A DelayHeap grows in a single direction, either Direction::kGrowsTowardUsers
// or Direction::kGrowsTowardOperands. If kGrowsTowardUsers, then nodes must be
// added in topological order. If kGrowsTowardOperands then nodes must be added
// in reverse topological order.
//
// In addition to the topological ordering constraint, there cannot exist a node
// outside the heap which is a strict predecessor and a strict successors of any
// node in the heap. With this definition, the allowed set of nodes in a
// DelayHeap are those which might constitute a feed-forward pipeline
// stage. There cannot exist a node outside the pipeline stage which is both a
// predecessor and successor of the pipeline stage as this would induce a cycle.
//
// A heap defines a frontier set of nodes which informally is the set nodes "on
// top" of the heap. More specifically, a node is in the frontier set if the
// node has no successors which are in the heap.
//
// For example, consider constructing a heap using nodes from an XLS function
// represented by the dependency graph below. This heap has direction
// kGrowsTowardUsers.
//
//    a
//   / \
//  b   c
//  | / | \
//  d   e  g
//   \ /   |
//    f    h
//
// Initially, the heap is empty. Nodes are added as follows with the respective
// state of the heap after adding (critical-path assumes each node has a delay
// of one):
//
// Add node 'b'.
//   heap nodes = {b}, frontier = {b}, critical-path delay = 1
// Add node 'c'.
//   heap nodes = {b, c}, frontier = {b, c}, critical-path delay = 1
// Add node 'd'.
//   heap nodes = {b, c, d}, frontier = {d}, critical-path delay = 2
// Add node 'e'.
//   heap nodes = {b, c, d, e}, frontier = {d, e}, critical-path delay = 2
// Add node 'f'.
//   heap nodes = {b, c, d, e, f}, frontier = {f}, critical-path delay = 3
class DelayHeap {
 public:
  using FrontierSet = std::set<Node*, std::function<bool(Node*, Node*)>>;

  // Constructs a new delay heap with the given direction and delay esimator to
  // use for nodes.
  explicit DelayHeap(Direction direction, const DelayEstimator& delay_estimator)
      : direction_(direction),
        frontier_([&](Node* a, Node* b) {
          // The frontier is ordered by critical path delay. Tie breaker
          // is node id for determinism.
          return path_lengths_.at(b) < path_lengths_.at(a) ||
                 (path_lengths_.at(a) == path_lengths_.at(b) &&
                  (a->id() < b->id()));
        }),
        delay_estimator_(delay_estimator) {}

  // Adds the given node to the data structure. Updates the frontier set and
  // critical-path delay of the heap. Necessarily the newly added node will be a
  // member of the frontier set. The complexity is logarithmic in the size of
  // the heap.
  absl::Status Add(Node* node);

  // Removes the given node to the data structure. Necessarily this node must be
  // in the frontier set of the heap. Updates the frontier set and critical-path
  // delay of the heap. Returns the iterator to the next node on the frontier.
  // The complexity is logarithmic in the size of the heap.
  //
  // Because newly exposed frontier nodes are necessarily added to the frontier
  // set *after* the removed node, the following code will remove all nodes
  // from the heap:
  //
  //   auto it = delay_heap.frontier().begin();
  //   while (it != delay_heap.frontier().end()) {
  //     it = delay_heap(Remove(*it));
  //   }
  FrontierSet::iterator Remove(Node* node);

  // Returns the number of nodes in the heap.
  int64_t size() const { return path_lengths_.size(); }

  // Returns the delay of the critical-path through the heap.
  int64_t CriticalPathDelay() const {
    return size() == 0 ? 0 : CriticalPathDelay(top());
  }

  // Returns the delay of the critical-path up through the given node. This
  // includes the delay of the node itself.
  int64_t CriticalPathDelay(Node* node) const {
    return path_lengths_.at(node).critical_path_delay;
  }

  // Returns true if the given node is contained in the heap.
  bool contains(Node* node) const { return path_lengths_.contains(node); }

  // Returns the critical path delay through the heap after adding the given
  // node.
  absl::StatusOr<int64_t> CriticalPathDelayAfterAdding(Node* node) const;

  // Returns the critical path delay through the heap after removing the given
  // node. The node must be in the frontier set.
  int64_t CriticalPathDelayAfterRemoving(Node* node) const;

  // Returns the direction in which the heap grows.
  Direction direction() const { return direction_; }

  // Returns set of frontier nodes of the heap. The set is ordered from largest
  // critical-path delay to smallest.
  const FrontierSet& frontier() const { return frontier_; }

  // Returns a node in the frontier set which has the largest critical-path
  // delay.
  Node* top() const { return *frontier_.begin(); }

  std::string ToString() const;

 private:
  // Data structure containing the critical-path delay and number of nodes in
  // the longest path by node count.
  struct PathLength {
    // Less than operator is used to sort the frontier set. The longest path
    // component is used to ensure that the set is in a (reverse) topological
    // sort in the face of zero-delay nodes.
    bool operator<(const PathLength& other) const {
      return critical_path_delay < other.critical_path_delay ||
             (critical_path_delay == other.critical_path_delay &&
              longest_path < other.longest_path);
    }
    bool operator==(const PathLength& other) const {
      return critical_path_delay == other.critical_path_delay &&
             longest_path == other.longest_path;
    }

    int64_t critical_path_delay;
    int64_t longest_path;
  };

  // Returns a span containing the users of 'node'. Because Node::users() is not
  // a vector, this function may create vector containing the users and store
  // the value in a cache for later use.
  absl::Span<Node* const> GetUsersSpan(Node* node) const;

  // Returns the predecessors of the given node. The predecessors are the graph
  // neighbors of the given node in the opposite direction of the direction the
  // heap grows.
  absl::Span<Node* const> predecessors(Node* node) const {
    return direction_ == Direction::kGrowsTowardUsers ? node->operands()
                                                      : GetUsersSpan(node);
  }

  // Returns the successors of the given node. The successors are the graph
  // neighbors of the given node in the opposite direction of the direction the
  // heap grows.
  absl::Span<Node* const> successors(Node* node) const {
    return direction_ == Direction::kGrowsTowardUsers ? GetUsersSpan(node)
                                                      : node->operands();
  }

  // Returns the maximum critical-path and path length among the predecessors of
  // the given node.
  PathLength MaxAmongPredecessors(Node* node) const;

  // The direction the heap grows in.
  Direction direction_;

  // The set of nodes on the frontier of the heap.
  FrontierSet frontier_;

  const DelayEstimator& delay_estimator_;

  // A map from node in the heap to the longest path length value for the node.
  absl::flat_hash_map<Node*, PathLength> path_lengths_;

  // A cache containing vectors with the users of each node.
  mutable absl::flat_hash_map<Node*, std::vector<Node*>> users_vectors_;
};

}  // namespace sched
}  // namespace xls

#endif  // XLS_ESTIMATORS_DELAY_MODEL_DELAY_HEAP_H_

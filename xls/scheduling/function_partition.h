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

#ifndef XLS_SCHEDULING_FUNCTION_PARTITION_H_
#define XLS_SCHEDULING_FUNCTION_PARTITION_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xls/ir/node.h"

namespace xls {
namespace sched {

// Splits the given set of nodes 'partitionable_nodes' into two partitions such
// that the sum of the bit counts of nodes live across the partition boundary is
// minimized. For a partition of nodes into sets A and B, the nodes live across
// the partition boundary are those which satisfy either of the following
// conditions:
//
//  (1) is an operand of a node in B and is not in B.
//  (2) is a use of a node in A and is not in A
//
// Note that a node live across the partition boundary need not be in the
// set 'partitionable_nodes'.
//
// The partition is a dicut (directed cut) of 'partitionable_nodes'. For
// partitions A and B, edges can only extend from nodes in A to nodes in B. No
// edge may extend from a node in B to a node in A.
//
// The set of nodes 'partitionable_nodes' must satisfy the following constraint:
// any path between nodes in 'partitionable_nodes' in the XLS function can only
// include nodes in 'partitionable_nodes'. This same constraint is satisfied by
// any set of nodes which may form a stage in a feed-forward pipeline, or any
// set of nodes which are contiguous in a topological sort.
//
// The cost of this partition (sum of the bit count of nodes live across the
// partition boundary) is precisely the number of flops required if registers
// are inserted between the nodes of A and the nodes of B.
//
// Returns the two partitions as a std::pair. The first element is the
// predecessor partition of the dicut (partition A in the example above).
std::pair<std::vector<Node*>, std::vector<Node*>> MinCostFunctionPartition(
    FunctionBase* f, absl::Span<Node* const> partitionable_nodes);

}  // namespace sched
}  // namespace xls

#endif  // XLS_SCHEDULING_FUNCTION_PARTITION_H_

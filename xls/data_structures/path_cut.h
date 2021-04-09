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

#ifndef XLS_DATA_STRUCTURES_PATH_CUT_H_
#define XLS_DATA_STRUCTURES_PATH_CUT_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_message.h"
#include "xls/common/logging/logging.h"
#include "xls/common/strong_int.h"

namespace xls {

DEFINE_STRONG_INT_TYPE(PathNodeId, int32_t);
DEFINE_STRONG_INT_TYPE(PathEdgeId, int32_t);
DEFINE_STRONG_INT_TYPE(PathNodeWeight, int32_t);
DEFINE_STRONG_INT_TYPE(PathEdgeWeight, int32_t);

// A type of nonempty path/linear graphs, i.e.: graphs that look like:
//
//   0   1   2   3   4   5   6
//   o───o───o───o───o───o───o
//     0   1   2   3   4   5
//
// with weights on both the nodes and edges.
class PathGraph {
 public:
  // Create a new nonempty path graph. There must be exactly one more node
  // weight than edge weight in the two input vectors, since a path graph always
  // has one more node than edge.
  static absl::StatusOr<PathGraph> Create(
      absl::Span<const PathNodeWeight> node_weights,
      absl::Span<const PathEdgeWeight> edge_weights) {
    if (edge_weights.size() + 1 != node_weights.size()) {
      return absl::InternalError(
          absl::StrFormat("Size of node weights (%d) must be 1 greater than "
                          "size of edge weights (%d).",
                          node_weights.size(), edge_weights.size()));
    }
    PathGraph result;
    result.node_weights_ =
        std::vector<PathNodeWeight>(node_weights.begin(), node_weights.end());
    result.edge_weights_ =
        std::vector<PathEdgeWeight>(edge_weights.begin(), edge_weights.end());
    return result;
  }

  // Returns the number of nodes, which is always one greater than the number of
  // edges in the path graph.
  int32_t NumNodes() const { return node_weights_.size(); }

  // Returns the number of edges, which is always one fewer than the number of
  // nodes in the path graph.
  int32_t NumEdges() const { return edge_weights_.size(); }

  // Returns the node weight associated with the given node.
  PathNodeWeight WeightOfNode(PathNodeId node) const {
    return node_weights_.at(static_cast<int32_t>(node));
  }

  // Returns the edge weight associated with the given edge.
  PathEdgeWeight WeightOfEdge(PathEdgeId edge) const {
    return edge_weights_.at(static_cast<int32_t>(edge));
  }

  // Returns the node that comes before the given edge.
  //
  // Precondition: the given edge is a valid edge.
  PathNodeId EdgeSource(PathEdgeId edge) const {
    XLS_CHECK_GE(static_cast<int32_t>(edge), 0);
    XLS_CHECK_LT(static_cast<int32_t>(edge), edge_weights_.size());
    return PathNodeId(static_cast<int32_t>(edge));
  }

  // Returns the node that comes after the given edge.
  //
  // Precondition: the given edge is a valid edge.
  PathNodeId EdgeTarget(PathEdgeId edge) const {
    XLS_CHECK_GE(static_cast<int32_t>(edge), 0);
    XLS_CHECK_LT(static_cast<int32_t>(edge), edge_weights_.size());
    return PathNodeId(static_cast<int32_t>(edge) + 1);
  }

  // Returns the edge after the given node, if there is one.
  absl::optional<PathEdgeId> NodeSuccessorEdge(PathNodeId node) const {
    XLS_CHECK_GE(static_cast<int32_t>(node), 0);
    XLS_CHECK_LT(static_cast<int32_t>(node), node_weights_.size());
    if (static_cast<int32_t>(node) == node_weights_.size() - 1) {
      return absl::nullopt;
    }
    return PathEdgeId(static_cast<int32_t>(node));
  }

  // Returns the edge previous to the given node, if there is one.
  absl::optional<PathEdgeId> NodePredecessorEdge(PathNodeId node) const {
    XLS_CHECK_GE(static_cast<int32_t>(node), 0);
    XLS_CHECK_LT(static_cast<int32_t>(node), node_weights_.size());
    if (static_cast<int32_t>(node) <= 0) {
      return absl::nullopt;
    }
    return PathEdgeId(static_cast<int32_t>(node) - 1);
  }

 private:
  PathGraph() = default;
  std::vector<PathNodeWeight> node_weights_;
  std::vector<PathEdgeWeight> edge_weights_;
};

// A partition of the set of nodes into equivalence classes that are contiguous.
//
// The nodes within each equivalence class should be in increasing order, and
// the equivalence classes themselves should be sorted in lexicographic order.
using PathCut = std::vector<std::vector<PathNodeId>>;

// Display a `PathCut` as a string.
std::string PathCutToString(const PathCut& cut);

// Convert a list of "cut edges" into a `PathCut`, i.e.: if the given `Path` has
// six edges, and the given list of cut edges is `[2, 4]`, then the returned
// partition will be `[[0, 1, 2], [3, 4], [5]]`.
PathCut CutEdgesToPathCut(const PathGraph& path,
                          absl::Span<const PathEdgeId> cut_edges);

// Compute the optimal (in terms of minimizing the total weight of cut edges)
// `PathCut` for a given `Path`, assuming that the sum of the node weights in
// each piece of the `PathCut` must be less than or equal to the
// given `PathNodeWeight`.
//
// Returns `absl::nullopt` when there is no cut that satisfies the constraint
// given by `maximum_weight`.
//
// The algorithm is based on https://cs.stackexchange.com/a/138417
absl::optional<PathCut> ComputePathCut(const PathGraph& path,
                                       PathNodeWeight maximum_weight);

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_PATH_CUT_H_

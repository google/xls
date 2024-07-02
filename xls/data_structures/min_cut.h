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

#ifndef XLS_DATA_STRUCTURES_MIN_CUT_H_
#define XLS_DATA_STRUCTURES_MIN_CUT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/common/strong_int.h"

namespace xls {
namespace min_cut {

XLS_DEFINE_STRONG_INT_TYPE(NodeId, int32_t);
XLS_DEFINE_STRONG_INT_TYPE(EdgeId, int32_t);

// A directed edge in the graph used for computing mincuts.
struct Edge {
  NodeId from;
  NodeId to;
  int64_t weight;

  // The unique ID of the edge. IDs are numbered sequentially from zero.
  EdgeId id;
};

// A data structure representing a directed graph (not necessarily acyclic) for
// the purposes of computing a min-cut.
class Graph {
 public:
  // Adds a node to the graph and returns the node's unique id. Node unique IDs
  // are numbered sequentially from zero. The optional name is used only for
  // generating the ToString output.
  NodeId AddNode(std::string name = "");

  // Adds a edge extending from 'from' to 'to' of the given weight.
  EdgeId AddEdge(NodeId from, NodeId to, int64_t weight);

  // Returns the set of edges extending from the given node.
  absl::Span<const EdgeId> successors(NodeId node) const {
    return successors_[static_cast<int64_t>(node)];
  }

  // Returns the edge with the given unique ID.
  const Edge& edge(EdgeId id) const { return edges_[static_cast<int64_t>(id)]; }

  // Returns the number of edges/nodes in the graph.
  int64_t edge_count() const { return edges_.size(); }
  int64_t node_count() const { return successors_.size(); }

  // Returns the maximum value of any node/edge unique ID.
  EdgeId max_edge_id() const { return EdgeId(edges_.size() - 1); }
  NodeId max_node_id() const { return NodeId(successors_.size() - 1); }

  std::string ToString() const;
  std::string name(NodeId node) const;
  std::string name(EdgeId edge) const;

 private:
  // The set of all edges in the graph. The vector is indexed by EdgeId.
  std::vector<Edge> edges_;

  // The set of edges extending from each node. The outer vector is indexed by
  // NodeId.
  std::vector<std::vector<EdgeId>> successors_;

  absl::flat_hash_map<NodeId, std::string> node_names_;
};

struct GraphCut {
  // The total weight of the cut.
  int64_t weight;

  // The set of nodes in the partition containing the 'source' node of the cut.
  std::vector<NodeId> source_partition;

  // The set of nodes in the partition containing the 'sink' node of the cut.
  std::vector<NodeId> sink_partition;

  std::string ToString(const Graph& graph) const;
};

// Computes a minimum cut of the given graph where source and sink are in
// different partitions. The cut is returned as a partitioning of the nodes of
// the graph into two sets of nodes on either side of the cut. The min cut is
// found via the Ford-Fulkerson method using Dinic's algorithm. This results in
// a worst case run time of O(V^2 * E).
GraphCut MinCutBetweenNodes(const Graph& graph, NodeId source, NodeId sink);

}  // namespace min_cut
}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_MIN_CUT_H_

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

#include "xls/data_structures/min_cut.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"

namespace xls {
namespace min_cut {

std::string GraphCut::ToString(const Graph& graph) const {
  std::vector<std::string> lines;
  lines.push_back(absl::StrFormat("weight: %d", weight));
  lines.push_back(absl::StrFormat("source partition:"));
  for (NodeId node : source_partition) {
    lines.push_back(absl::StrFormat("  %s", graph.name(node)));
  }
  lines.push_back(absl::StrFormat("sink partition:"));
  for (NodeId node : sink_partition) {
    lines.push_back(absl::StrFormat("  %s", graph.name(node)));
  }
  return absl::StrJoin(lines, "\n");
}

NodeId Graph::AddNode(std::string name) {
  NodeId id(successors_.size());
  successors_.push_back({});
  if (!name.empty()) {
    node_names_[id] = name;
  }
  return id;
}

EdgeId Graph::AddEdge(NodeId from, NodeId to, int64_t weight) {
  EdgeId id(edges_.size());
  edges_.push_back({from, to, weight, id});
  successors_.at(static_cast<int64_t>(from)).push_back(id);
  return id;
}

std::string Graph::name(NodeId node) const {
  if (node_names_.contains(node)) {
    return node_names_.at(node);
  }
  return absl::StrCat(static_cast<int64_t>(node));
}

std::string Graph::name(EdgeId edge) const {
  return absl::StrFormat("%s->%s",
                         name(edges_[static_cast<int64_t>(edge)].from),
                         name(edges_[static_cast<int64_t>(edge)].to));
}

std::string Graph::ToString() const {
  std::string out = "Graph:\n";
  for (int64_t i = 0; i < node_count(); ++i) {
    NodeId node(i);
    absl::StrAppendFormat(
        &out, "  Node %s: %s\n", name(node),
        absl::StrJoin(successors_[i], ", ", [&](std::string* out, EdgeId e_id) {
          absl::StrAppendFormat(out, "%s[%d]", name(edge(e_id).to),
                                edge(e_id).weight);
        }));
  }
  return out;
}

namespace {

// Struct representing an edge in the residual graph.
struct ResidualEdge {
  NodeId from;
  NodeId to;
  int64_t capacity;

  // The id of the other edge of the forward/backward pair of edges created in
  // the residual graph for each edge in the original graph.
  EdgeId dual_edge;
};

// Data structure representing the residual graph. The residual graph is a data
// structure used in the min-cut algorithm which mirrors the input graph. The
// nodes in the two graphs are identical, but each edge in the input graph
// corresponds to two edges in the residual graph: one edge aligned with the
// original edge and one edge in the backwards direction. Each residual edge has
// a capacity which is a function of the original edge weight and the current
// flow along the edge.
class ResidualGraph {
 public:
  explicit ResidualGraph(const Graph& graph) {
    // There are exactly twice as many edges in the residual graph because each
    // edge in the original graph maps to a forward and backward edge in the
    // residual graph. The forward edge in the original graph has the same
    // EdgeId as its corresponding edge in the original graph.
    edges_.resize(graph.edge_count() * 2);
    successors_.resize(graph.node_count());
    for (EdgeId edge_id = EdgeId{0}; edge_id <= graph.max_edge_id();
         edge_id += EdgeId{1}) {
      const Edge& edge = graph.edge(edge_id);
      EdgeId backward_edge_id{int64_t{edge_id} + graph.edge_count()};

      // Add the forward edge to the residual graph. It has an inital capacity
      // equal to the weight of the edge in the original graph.
      edges_[int64_t{edge_id}] =
          ResidualEdge{edge.from, edge.to, edge.weight, backward_edge_id};
      successors_[int64_t{edge.from}].push_back(edge_id);

      // Add the backward edge to the residual graph. It has an inital capacity
      // of zero.
      edges_[int64_t{backward_edge_id}] =
          ResidualEdge{edge.to, edge.from, 0, edge_id};
      successors_[int64_t{edge.to}].push_back(backward_edge_id);
    }
  }

  // Returns the edges extended from the given node.
  absl::Span<const EdgeId> successors(NodeId node) const {
    return successors_[int64_t{node}];
  }

  // Push flow along the given edge. The capacity of this edge is reduced and
  // the capacity of the dual edge is increased.
  void PushFlow(int64_t amount, ResidualEdge* residual_edge) {
    CHECK_GE(residual_edge->capacity, amount);
    residual_edge->capacity -= amount;
    edge(residual_edge->dual_edge).capacity += amount;
  }

  const ResidualEdge& edge(EdgeId id) const { return edges_[int64_t{id}]; }
  ResidualEdge& edge(EdgeId id) { return edges_[int64_t{id}]; }

 private:
  // The set of all edges in the graph. The vector is indexed by EdgeId.
  std::vector<ResidualEdge> edges_;

  // Edges extending from each node. Outer vector is indexed by NodeId.
  std::vector<std::vector<EdgeId>> successors_;
};

// Returns a string representation of the graph which includes residual capacity
// of each edge.
std::string GraphWithFlowToString(const Graph& graph,
                                  const ResidualGraph& residual_graph) {
  std::string out = "Graph:\n";
  for (NodeId n = NodeId(0); n <= graph.max_node_id(); ++n) {
    absl::StrAppendFormat(
        &out, "  %s : %s\n", graph.name(n),
        absl::StrJoin(
            graph.successors(n), ", ", [&](std::string* out, EdgeId e_id) {
              const Edge& e = graph.edge(e_id);
              absl::StrAppendFormat(
                  out, "%s[%d/%d]", graph.name(e.to),
                  e.weight - residual_graph.edge(e_id).capacity, e.weight);
            }));
  }
  return out;
}

// Finds an augmenting path from source to sink in the graph (path of edges with
// non-zero residual capacity) and increases the flow along each edge in the
// path. The amount of flow increase is equal to the minimum residual capacity
// of any edge in the path. The augmenting path is found via Dinic's algorithm
// which finds a shortest augmenting path via BFS search. Returns the amount of
// augmented flow. If no augmenting path is found then zero is returned.
int64_t AugmentFlow(const Graph& graph, NodeId source, NodeId sink,
                    ResidualGraph* residual_graph) {
  XLS_VLOG_LINES(4, GraphWithFlowToString(graph, *residual_graph));

  // Perform a BFS search from flow to find the shortest augmenting path (by
  // edge count) from source to sink. For each edge traversed in the BFS search,
  // 'path_back' holds the previous edge in the path from the source along with
  // the residual capacity of that path..
  absl::flat_hash_map<NodeId, std::pair<ResidualEdge*, int64_t>> path_back;
  std::vector<NodeId> frontier = {source};
  while (!frontier.empty()) {
    std::vector<NodeId> next_frontier;
    // Try to add extend the BFS-constructed paths from the source with the
    // given edge.
    auto maybe_extend_frontier = [&](ResidualEdge* e) {
      if (path_back.contains(e->to)) {
        return;
      }
      int64_t path_capacity = path_back.contains(e->from)
                                  ? path_back.at(e->from).second
                                  : std::numeric_limits<int64_t>::max();
      // The residual capacity of a path is the minimum residual capacity of any
      // edge in the path.
      path_capacity = std::min(path_capacity, e->capacity);
      path_back[e->to] = {e, path_capacity};
      next_frontier.push_back(e->to);
      VLOG(5) << "  Added " << graph.name(e->to) << " to the frontier";
    };

    for (NodeId node : frontier) {
      VLOG(5) << "Visiting " << graph.name(node);
      for (EdgeId edge_id : residual_graph->successors(node)) {
        ResidualEdge& edge = residual_graph->edge(edge_id);
        VLOG(5) << absl::StreamFormat(
            "  Traversing %s->%s", graph.name(edge.from), graph.name(edge.to));
        CHECK_GE(edge.capacity, 0);
        if (edge.capacity > 0) {
          maybe_extend_frontier(&edge);
          if (edge.to == sink) {
            // Found augmenting path. As this is the first path we've found in a
            // BFS search it is necessarily a shortest path. Walk back along
            // path and augment the flow.
            int64_t augmented_flow_amount = path_back.at(sink).second;
            CHECK_GT(augmented_flow_amount, 0);
            NodeId n = sink;
            while (n != source) {
              ResidualEdge* e = path_back.at(n).first;
              residual_graph->PushFlow(augmented_flow_amount, e);
              CHECK_GE(e->capacity, 0);
              n = e->from;
            }
            if (VLOG_IS_ON(4)) {
              VLOG(4) << "Augmented flow: " << augmented_flow_amount;
              std::vector<const ResidualEdge*> augmented_path;
              NodeId n = sink;
              while (n != source) {
                const ResidualEdge* e = path_back.at(n).first;
                augmented_path.push_back(e);
                n = e->from;
              }
              std::reverse(augmented_path.begin(), augmented_path.end());
              VLOG(4) << "Augmented path: "
                      << absl::StrJoin(
                             augmented_path, ", ",
                             [&](std::string* out, const ResidualEdge* e) {
                               absl::StrAppendFormat(
                                   out, "%s->%s [new capacity %d]",
                                   graph.name(e->from), graph.name(e->to),
                                   e->capacity);
                             });
            }
            return augmented_flow_amount;
          }
        }
      }
    }

    frontier = std::move(next_frontier);
    next_frontier.clear();
  }

  VLOG(4) << "No augmenting path found";
  return 0;
}

}  // namespace

GraphCut MinCutBetweenNodes(const Graph& graph, NodeId source, NodeId sink) {
  // This loop is the core of the Ford-Fulkerson method. Starting with zero flow
  // on all edges, flow is increased along a path from source to sink with
  // residual capacity (called an augmenting path). When no further augmenting
  // paths exist, flow has been maximized. 'flow' is the flow of each edge in
  // the graph indexed by EdgeId.
  ResidualGraph residual_graph(graph);
  while (AugmentFlow(graph, source, sink, &residual_graph) > 0) {
  }

  // Once a maximum flow is found, walk the residual graph from the source. All
  // reachable nodes form one partition.
  absl::flat_hash_set<NodeId> reachable_from_source;
  std::deque<NodeId> frontier = {source};
  reachable_from_source.insert(source);
  while (!frontier.empty()) {
    NodeId node = frontier.front();
    frontier.pop_front();
    for (EdgeId successor_edge_id : residual_graph.successors(node)) {
      const ResidualEdge& edge = residual_graph.edge(successor_edge_id);
      if (edge.capacity > 0 && !reachable_from_source.contains(edge.to)) {
        reachable_from_source.insert(edge.to);
        frontier.push_back(edge.to);
      }
    }
  }
  CHECK(!reachable_from_source.contains(sink));

  GraphCut min_cut;
  min_cut.weight = 0;
  for (NodeId node_id = NodeId(0); node_id <= graph.max_node_id(); ++node_id) {
    if (reachable_from_source.contains(node_id)) {
      min_cut.source_partition.push_back(node_id);
    } else {
      min_cut.sink_partition.push_back(node_id);
    }
    for (EdgeId edge_id : graph.successors(node_id)) {
      const Edge& edge = graph.edge(edge_id);
      if (reachable_from_source.contains(edge.from) &&
          !reachable_from_source.contains(edge.to)) {
        min_cut.weight += edge.weight;
      }
    }
  }

  XLS_VLOG_LINES(4, min_cut.ToString(graph));

  return min_cut;
}

}  // namespace min_cut
}  // namespace xls

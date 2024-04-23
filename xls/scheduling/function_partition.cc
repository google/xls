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

#include "xls/scheduling/function_partition.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xls/data_structures/min_cut.h"
#include "xls/ir/function.h"

namespace xls {
namespace sched {

std::pair<std::vector<Node*>, std::vector<Node*>> MinCostFunctionPartition(
    FunctionBase* f, absl::Span<Node* const> partitionable_nodes) {
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Computing min-cut of function " << f->name()
            << ", partitionable nodes:";
    for (Node* node : partitionable_nodes) {
      VLOG(4) << "  " << node->GetName();
    }
  }

  absl::flat_hash_set<Node*> partitionable_nodes_set(
      partitionable_nodes.begin(), partitionable_nodes.end());
  CHECK_EQ(partitionable_nodes_set.size(), partitionable_nodes.size());

  min_cut::Graph graph;
  auto source = graph.AddNode("source");
  auto sink = graph.AddNode("sink");
  std::vector<Node*> xls_nodes_in_mincut_graph;

  // Maps to/from XLS Nodes to nodes in the mincut graph.
  absl::flat_hash_map<Node*, min_cut::NodeId> xls_to_mincut_node;
  absl::flat_hash_map<min_cut::NodeId, Node*> mincut_to_xls_node;

  const int64_t kMaxWeight = std::numeric_limits<int64_t>::max();

  // Adds an edge to the mincut graph. To enforce that the cut is a dicut (no
  // circular dependencies between the two partitions), add an opposing edge of
  // maximum weight.
  auto add_edge = [&](min_cut::NodeId src, min_cut::NodeId tgt,
                      int64_t weight) {
    graph.AddEdge(src, tgt, weight);
    graph.AddEdge(tgt, src, kMaxWeight);
  };

  auto add_node_to_mincut_graph = [&](Node* node) {
    CHECK(!xls_to_mincut_node.contains(node));
    min_cut::NodeId graph_node_id = graph.AddNode(node->GetName());
    xls_to_mincut_node[node] = graph_node_id;
    mincut_to_xls_node[graph_node_id] = node;
    xls_nodes_in_mincut_graph.push_back(node);
    return graph_node_id;
  };

  for (Node* node : partitionable_nodes) {
    min_cut::NodeId node_id = add_node_to_mincut_graph(node);
    if (node->Is<Param>()) {
      // Add a maximum weight edge from the artificial source node to each
      // parameter node. This forces the cut to be below the parameter nodes.
      add_edge(source, node_id, kMaxWeight);
    }

    // Add a node in the mincut graph for each operand of a node in the
    // partitionable set which is itself not in the partitionable set. These
    // correspond to values flowing into the partitionable set. Add an edge of
    // maximum weight from the source node to each of these operand nodes to
    // enforce that the cut goes below these nodes.
    for (Node* operand : node->operands()) {
      if (!partitionable_nodes_set.contains(operand) &&
          !xls_to_mincut_node.contains(operand)) {
        min_cut::NodeId operand_node = add_node_to_mincut_graph(operand);
        add_edge(source, operand_node, kMaxWeight);
      }
    }

    // Add a node in the mincut graph for each user of the partitionable set
    // which is itself not in the partitionable set. These correspond to values
    // flowing out of the partitionable set. Add an edge of maximum weight to
    // the sink node from each of these users to enforce that the cut goes above
    // these nodes.
    for (Node* user : node->users()) {
      if (!partitionable_nodes_set.contains(user) &&
          !xls_to_mincut_node.contains(user)) {
        min_cut::NodeId user_node = add_node_to_mincut_graph(user);
        add_edge(user_node, sink, kMaxWeight);
      }
    }
  }

  // Returns the edge weight of an edge extending from the given node. Fanout is
  // the number of successors of the corresponding mincut node.
  auto edge_weight = [&](Node* node, int64_t fan_out) {
    const int64_t kWeightFactor = 1024 * 1024;
    return (node->GetType()->GetFlatBitCount() * kWeightFactor + fan_out / 2) /
           fan_out;
  };

  // Add edges corresponding to edges in the XLS function. kWeightFactor is a
  // value which is multiplied times the bit-width of an XLS graph edge to
  // compute the weight of an mincut graph edge. This factor is necessary to
  // reduce rounding error because bit-widths are divided by fan-out to avoid
  // double-counting non-unit fanout nodes in the min-cut cost computation.
  for (Node* node : xls_nodes_in_mincut_graph) {
    std::vector<min_cut::NodeId> successors;
    for (Node* user : node->users()) {
      if (xls_to_mincut_node.contains(user)) {
        successors.push_back(xls_to_mincut_node.at(user));
      }
    }
    if (successors.empty()) {
      continue;
    }
    if (successors.size() == 1) {
      add_edge(xls_to_mincut_node.at(node), successors.front(),
               edge_weight(node, /*fan_out=*/1));
      continue;
    }
    // The node in the mincut graph has a fanout greater than one. To avoid
    // double counting of bit-widths in the cut, we need divide the edge weight
    // by the fan-out amount and add a new node with a corresponding fan-in. For
    // example, given a node 'x' in the XLS graph with a fan out of three and
    // bit-count of C:
    //
    //      x
    //    / | \
    //   a  b  c
    //
    // We create the following construct in the mincut graph where each edge has
    // a weight of C/3 (multiplied by kWeightFactor to minimize rounding error):
    //
    //        x
    //      / | \ kWeightFactor * C/3
    //     /  |  \
    //    a   b   c
    //     \  |  /
    //      \ | / kWeightFactor * C/3
    //     x_fanin
    //
    min_cut::NodeId node_sink = graph.AddNode(node->GetName() + "_fanin");
    int64_t weight = edge_weight(node, /*fan_out=*/successors.size());
    for (min_cut::NodeId successor : successors) {
      add_edge(xls_to_mincut_node.at(node), successor, weight);
      add_edge(successor, node_sink, weight);
    }
  }

  min_cut::GraphCut graph_cut =
      min_cut::MinCutBetweenNodes(graph, source, sink);

  // Map the mincut graph partition back to the XLS graph.
  std::pair<std::vector<Node*>, std::vector<Node*>> partitions;
  for (min_cut::NodeId node_id : graph_cut.source_partition) {
    if (mincut_to_xls_node.contains(node_id)) {
      Node* node = mincut_to_xls_node.at(node_id);
      if (partitionable_nodes_set.contains(node)) {
        partitions.first.push_back(node);
      }
    }
  }
  for (min_cut::NodeId node_id : graph_cut.sink_partition) {
    if (mincut_to_xls_node.contains(node_id)) {
      Node* node = mincut_to_xls_node.at(node_id);
      if (partitionable_nodes_set.contains(node)) {
        partitions.second.push_back(node);
      }
    }
  }
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Before cut";
    for (Node* node : partitions.first) {
      VLOG(4) << "  " << node->GetName();
    }
    VLOG(4) << "After cut";
    for (Node* node : partitions.second) {
      VLOG(4) << "  " << node->GetName();
    }
  }
  return partitions;
}

}  // namespace sched
}  // namespace xls

// Copyright 2022 The XLS Authors
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

#include "xls/passes/minimal_feedback_arcs.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/token_provenance_analysis.h"

namespace xls {
namespace {
// Get channel_id for send_receive channels. If node is not a send/receive, or
// if the channel used by node is not send_receive, returns nullopt.
std::optional<int64_t> GetInternalChannelId(Node* node) {
  if (!IsChannelNode(node)) {
    return std::nullopt;
  }
  absl::StatusOr<Channel*> ch = GetChannelUsedByNode(node);
  if (!ch.ok()) {
    return std::nullopt;
  }
  if (ch.value()->supported_ops() != ChannelOps::kSendReceive) {
    return std::nullopt;
  }
  return ch.value()->id();
}

// Map storing directed edges mapping a head/tail node to its set of
// tails/heads. Uses btree_map and btree_set sorted by Node::NodeIdLessThan to
// make iteration order stable.
using StableEdgeMap =
    absl::btree_map<Node*, absl::btree_set<Node*, Node::NodeIdLessThan>,
                    Node::NodeIdLessThan>;

// Directed graph representing connectivity of interal channel operations (i.e.
// channel send/receives on kSendReceive channels). An edge exists from a->b if
// there is a data (or token) dependency between channel operations or between a
// send and a receive on the same channel. For example, for internal channels w/
// id 0 and 1:
//   proc a(...) {
//     send_token0: token = send(..., channel_id=0)
//     after_all_sends: token = after_all(send_token, another_send_token)
//     send_token1: token = send(..., channel_id=1)
//     ...
//   }
//   proc b(...) {
//     receive_token: (token, bits[32]) = receive(..., channel_id=0)
//     ...
//   }
//
// The above example has an edge from send_token0 to send_token1 (b/c of the
// dependency through after_all_sends) and an edge from send_token0 to
// receive_token1 (b/c they are a send/receive pair on the same channel).
//
// The graph is represented with two edge maps: one with edge heads as keys
// and tails as values (predecessor_edges) and the other with edge tails as
// keys and heads as values (successor_edges). We keep both predecessor and
// successor maps because identifying sinks is trivial in the successor map but
// expensive in the predecessor map and identifying sources vice versa. Graph
// operations will need to take care to keep the two maps consistent.
struct InterProcConnectivityGraph {
  StableEdgeMap predecessor_edges;
  StableEdgeMap successor_edges;
};

absl::StatusOr<InterProcConnectivityGraph> MakeInterProcConnectivityGraph(
    const Package* p) {
  // We represent the connectivity graph with both a successor- and predecessor-
  // map- see comments for InterProcConnectivityGraph above for more info about
  // why.
  StableEdgeMap predecessor_edges;
  StableEdgeMap successor_edges;

  // Use these channel_id -> send/receive maps to eventually build up the
  // inter-proc connections. Once we've looped through every proc, these maps
  // will be joined by channel id and the (send, receive) pairs will be added to
  // the edge maps.
  absl::btree_map<int64_t, absl::btree_set<Receive*, Node::NodeIdLessThan>>
      channel_id_to_internal_receive;
  absl::btree_map<int64_t, absl::btree_set<Send*, Node::NodeIdLessThan>>
      channel_id_to_internal_send;

  std::vector<NodeAndPredecessors> token_dag;
  for (FunctionBase* fb : p->GetFunctionBases()) {
    // Insert predecessors for every node into predecessor_edges. It is only
    // supposed to contain internal channel operations, so we will eventually
    // remove all other nodes, but we initially keep around the other nodes
    // while building the graph so we can resolve dependencies through
    // intermediate nodes, e.g. after_alls and tuples.
    for (Node* node : TopoSort(fb)) {
      // This is a btree_set<Node*> w/ the comparison function used in
      // StableEdgeMap.
      StableEdgeMap::value_type::second_type node_predecessors;
      std::optional<int64_t> channel_id = GetInternalChannelId(node);
      if (channel_id.has_value()) {
        // node is an internal channel operation, so add it as a predecessor.
        node_predecessors.insert(node);
      } else {
        // Node is not an internal channel operation, so add its transitive
        // predecessors.
        // It's OK to insert if it isn't already there- the copy will be a no-op
        // and intermediate nodes will be removed later.
        StableEdgeMap::value_type::second_type& transitive_predecessors =
            predecessor_edges[node];
        std::copy(transitive_predecessors.begin(),
                  transitive_predecessors.end(),
                  std::inserter(node_predecessors, node_predecessors.end()));
      }
      for (Node* successor : node->users()) {
        predecessor_edges[successor].insert(node_predecessors.begin(),
                                            node_predecessors.end());
      }
      if (!channel_id.has_value()) {
        continue;
      }
      switch (node->op()) {
        case Op::kSend: {
          channel_id_to_internal_send[*channel_id].insert(node->As<Send>());
          break;
        }
        case Op::kReceive: {
          channel_id_to_internal_receive[*channel_id].insert(
              node->As<Receive>());
          break;
        }
        default:
          return absl::InternalError("Expected send or receive.");
      }
    }
  }
  if (channel_id_to_internal_send.size() !=
      channel_id_to_internal_receive.size()) {
    return absl::InternalError(
        absl::StrFormat("Number of internal sends (%d) should match number of "
                        "internal receives (%d).",
                        channel_id_to_internal_send.size(),
                        channel_id_to_internal_receive.size()));
  }

  // Previously, we added edges between channel operations within a
  // FunctionBase. Now, add edges between sends and receives on the same
  // channel. If there are multiple sends or receives on the same channel, add
  // an edge from every send to each receive.
  for (auto [channel_id, send_nodes] : channel_id_to_internal_send) {
    auto itr = channel_id_to_internal_receive.find(channel_id);
    if (itr == channel_id_to_internal_receive.end()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, p->GetChannel(channel_id));
      return absl::InternalError(absl::StrFormat(
          "Channel %s (channel_id=%d) had a send but no receive.",
          channel->name(), channel_id));
    }
    for (Send* send_node : send_nodes) {
      for (Receive* receive_node : itr->second) {
        predecessor_edges[receive_node].insert(send_node);
      }
    }
  }

  // Clear intermediate nodes.
  absl::erase_if(predecessor_edges, [](const StableEdgeMap::value_type& edges) {
    return !GetInternalChannelId(edges.first).has_value();
  });

  //  Mirror the predecessor map onto the successor map.
  for (const auto& [node, predecessors] : predecessor_edges) {
    successor_edges.insert({node, {}});
    for (Node* predecessor : predecessors) {
      successor_edges[predecessor].insert(node);
    }
  }

  return InterProcConnectivityGraph{
      .predecessor_edges = std::move(predecessor_edges),
      .successor_edges = std::move(successor_edges)};
}

// Returns a linear arrangement that approximately minimizes the feedback arc
// set. The algorithm is due to P. Eades, X. Lin, and W. F. Smyth, "A fast and
// effective heuristic for the feedback arc set problem."
std::vector<Node*> GreedyFAS(InterProcConnectivityGraph& graph) {
  StableEdgeMap& predecessor_edges = graph.predecessor_edges;
  StableEdgeMap& successor_edges = graph.successor_edges;

  // The return value is the concatenation of s1 and s2. s1 is built up by
  // appending newly removed nodes to the end, while s2 is built up by
  // prepending newly removed nodes to the beginning. We store s2 in reverse
  // order because appending is cheap for std::vector. We eventually
  // std::reverse_copy() s2_rev to the end of s1.
  std::vector<Node*> s1, s2_rev;
  // Keep a set of nodes that have been removed. When keys are removed from the
  // predecessor/successor map, you need to remove values from the
  // successor/predecessor map. To keep things simple, we keep a list of all
  // nodes that have been removed and absl::erase_if every element contained in
  // removed_nodes.
  absl::flat_hash_set<Node*> removed_nodes;
  auto clear_removed_values =
      [&removed_nodes](StableEdgeMap::value_type::second_type& values) {
        absl::erase_if(values, [&removed_nodes](Node* node) {
          return removed_nodes.contains(node);
        });
      };
  auto clear_removed_keys = [&removed_nodes](StableEdgeMap& map) {
    absl::erase_if(map, [&removed_nodes](StableEdgeMap::const_reference edge) {
      return removed_nodes.contains(edge.first);
    });
  };

  auto node_is_sink = [&successor_edges](Node* node) {
    auto itr = successor_edges.find(node);
    return itr == successor_edges.end() || itr->second.empty();
  };

  while (!(predecessor_edges.empty() && successor_edges.empty())) {
    // Start by removing sinks (nodes with no successors).
    bool saw_sink = true;
    while (saw_sink) {
      saw_sink = false;
      clear_removed_keys(successor_edges);
      for (auto& [node, successors] : successor_edges) {
        clear_removed_values(successors);
        if (successors.empty()) {  // has no successors, is a sink
          saw_sink = true;
          removed_nodes.insert(node);
          s2_rev.push_back(node);
        }
      }
    }
    // After removing sinks, remove sources (nodes with no predecessors).
    bool saw_source = true;
    while (saw_source) {
      saw_source = false;
      clear_removed_keys(predecessor_edges);
      for (auto& [node, predecessors] : predecessor_edges) {
        clear_removed_values(predecessors);
        if (!node_is_sink(node) &&
            predecessors.empty()) {  // has no predecessors, is a source
          saw_source = true;
          removed_nodes.insert(node);
          s1.push_back(node);
        }
      }
    }
    // Once sinks and sources are removed, we remove the node with the maximum
    // degree. We want don't consider proc-internal nodes and external channel
    // operations here- this is where the cycle breaking happens, and we want to
    // break cycles at internal channel boundaries. The other nodes will become
    // sinks/sources eventually after removing the internal channel operations.
    // sources after removing them)
    auto max_itr = std::max_element(
        predecessor_edges.begin(), predecessor_edges.end(),
        [&successor_edges](const auto& lhs, const auto& rhs) {
          // We should only see internal channel ops at this point.
          CHECK(GetInternalChannelId(lhs.first).has_value());
          CHECK(GetInternalChannelId(rhs.first).has_value());

          // outdegree is size of the successor set.
          int64_t lhs_outdegree = successor_edges.at(lhs.first).size();
          int64_t rhs_outdegree = successor_edges.at(rhs.first).size();
          // lhs and rhs are the predecessors, so indegree is the size of the
          // predecessor set.
          int64_t lhs_indegree = static_cast<int64_t>(lhs.second.size());
          int64_t rhs_indegree = static_cast<int64_t>(rhs.second.size());
          // degree = outdegree - indegree
          int64_t lhs_degree = lhs_outdegree - lhs_indegree;
          int64_t rhs_degree = rhs_outdegree - rhs_indegree;
          return lhs_degree < rhs_degree;
        });
    if (max_itr != predecessor_edges.end()) {
      CHECK(GetInternalChannelId(max_itr->first).has_value());
      removed_nodes.insert(max_itr->first);
      s1.push_back(max_itr->first);
      successor_edges.erase(max_itr->first);
      predecessor_edges.erase(max_itr);  // invalidates max_itr
    }
  }
  // Concatenate s1 and s2_rev (stored in reverse).
  std::reverse_copy(s2_rev.begin(), s2_rev.end(), std::back_inserter(s1));

  return s1;
}
}  // namespace

absl::StatusOr<absl::flat_hash_set<Channel*>> MinimalFeedbackArcs(
    const Package* p) {
  XLS_ASSIGN_OR_RETURN(InterProcConnectivityGraph graph,
                       MakeInterProcConnectivityGraph(p));

  // Make a copy of successors before calling GreedyFAS because it will mutate
  // the graph. We keep the successors to find back-edges in the linear
  // arrangement.
  StableEdgeMap successors_copy = graph.successor_edges;

  if (VLOG_IS_ON(3)) {
    XLS_VLOG(3) << "Predecessors:";
    for (const auto& [key, values] : graph.predecessor_edges) {
      XLS_VLOG(3) << absl::StreamFormat("\t%v: {%s}", *key,
                                        absl::StrJoin(values, ", "));
    }
    XLS_VLOG(3) << "Successors:";
    for (const auto& [key, values] : successors_copy) {
      XLS_VLOG(3) << absl::StreamFormat("\t%v: {%s}", *key,
                                        absl::StrJoin(values, ", "));
    }
  }

  std::vector<Node*> arrangement = GreedyFAS(graph);
  XLS_VLOG(3) << absl::StreamFormat("Arrangement s: [%s]\n",
                                    absl::StrJoin(arrangement, ", "));

  // The feedback nodes are those that have successors that occur earlier in the
  // arrangement. Add those nodes' associated channels to the result set.
  absl::flat_hash_set<Node*> seen;
  absl::flat_hash_set<Channel*> result;
  for (Node* node : arrangement) {
    std::optional<int64_t> node_channel_id = GetInternalChannelId(node);
    XLS_RET_CHECK(node_channel_id.has_value());
    seen.insert(node);

    auto itr = successors_copy.find(node);
    if (itr == successors_copy.end()) {
      continue;
    }
    XLS_VLOG(5) << absl::StreamFormat("successors to %v are {%s}\n", *node,
                                      absl::StrJoin(itr->second, ", "));
    // Find internal channel operations that have already been seen and add them
    // to the result set.
    for (Node* successor : itr->second) {
      XLS_RET_CHECK(IsChannelNode(successor));
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(successor));
      if (seen.contains(successor)) {
        result.insert(ch);
      }
    }
  }
  XLS_VLOG(3) << absl::StreamFormat("minimal feedback arc set: {%s}\n",
                                    absl::StrJoin(result, ", "));
  return result;
}
}  // namespace xls

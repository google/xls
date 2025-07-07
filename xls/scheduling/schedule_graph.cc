// Copyright 2025 The XLS Authors
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

#include "xls/scheduling/schedule_graph.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/topo_sort.h"

namespace xls {
namespace {

// Return a vector of the unique members of `nodes`. Order is preserved.
std::vector<Node*> UniquifyNodes(absl::Span<Node* const> nodes) {
  std::vector<Node*> uniqued_nodes;
  absl::flat_hash_set<Node*> set;
  for (Node* node : nodes) {
    auto [_, inserted] = set.insert(node);
    if (inserted) {
      uniqued_nodes.push_back(node);
    }
  }
  return uniqued_nodes;
};

struct SendReceivePair {
  std::optional<Send*> send;
  std::optional<Receive*> receive;
};

// Return the set of send/receive pairs for each ChannelInstance in the
// elaboration. If a channel instance is on the interface of the top proc, only
// one of the send/receive fields will be present.
absl::StatusOr<absl::flat_hash_map<ChannelInstance*, SendReceivePair>>
GetSendReceivePairs(const ProcElaboration& elab) {
  absl::flat_hash_map<ChannelInstance*, SendReceivePair> pairs;
  for (Proc* proc : elab.procs()) {
    for (Node* node : proc->nodes()) {
      if (node->Is<ChannelNode>()) {
        ChannelNode* channel_node = node->As<ChannelNode>();
        XLS_ASSIGN_OR_RETURN(
            ChannelInterface * interface,
            proc->GetChannelInterface(channel_node->channel_name(),
                                      channel_node->direction()));
        for (ChannelInstance* instance :
             elab.GetInstancesOfChannelInterface(interface)) {
          if (node->Is<Send>()) {
            pairs[instance].send = node->As<Send>();
          } else {
            pairs[instance].receive = node->As<Receive>();
          }
        }
      }
    }
  }
  return pairs;
}

}  // namespace

bool IsUntimed(Node* node) { return node->Is<Literal>(); }

std::string ScheduleGraph::ToString() const {
  std::vector<std::string> lines;
  lines.push_back(absl::StrCat(
      "ScheduleGraph(", IsFunctionBaseScoped() ? "FunctionBase " : "Package ",
      name(), "):"));
  auto node_name = [](Node* node) {
    return absl::StrCat(node->function_base()->name(), "::", node->GetName());
  };
  for (const ScheduleNode& node : nodes()) {
    auto attribute_string = [](bool value, std::string_view name) {
      return value ? absl::StrCat(" [", name, "]") : "";
    };
    lines.push_back(absl::StrCat(
        "  ", node_name(node.node),
        attribute_string(node.schedule_in_first_stage,
                         "schedule_in_first_stage"),
        attribute_string(node.schedule_in_last_stage, "schedule_in_last_stage"),
        attribute_string(node.is_live_in, "is_live_in"),
        attribute_string(node.is_live_out, "is_live_out"), ":"));
    for (Node* successor : node.successors) {
      lines.push_back(absl::StrCat("    ", node_name(successor)));
    }
    for (const ScheduleBackedge& backedge : backedges()) {
      if (backedge.source == node.node) {
        lines.push_back(absl::StrCat("    ", backedge.destination->GetName(),
                                     " [backedge]"));
      }
    }
  }
  return absl::StrJoin(lines, "\n");
}

ScheduleGraph ScheduleGraph::Create(
    FunctionBase* f, const absl::flat_hash_set<Node*>& dead_after_synthesis) {
  std::vector<ScheduleBackedge> backedges;
  std::vector<ScheduleNode> nodes;
  nodes.reserve(f->node_count());
  for (Node* node : TopoSort(f)) {
    nodes.push_back(ScheduleNode{
        .node = node,
        .predecessors = std::vector<Node*>(node->operands().begin(),
                                           node->operands().end()),
        .successors =
            std::vector<Node*>(node->users().begin(), node->users().end()),
        .is_dead_after_synthesis = dead_after_synthesis.contains(node),
        // For functions, all parameter nodes must be scheduled in the first
        // stage of the pipeline...
        .schedule_in_first_stage = f->IsFunction() && node->Is<Param>(),
        // ... and the return value must be scheduled in the final stage, unless
        // it's a parameter.
        .schedule_in_last_stage =
            f->IsFunction() && f->AsFunctionOrDie()->return_value() == node &&
            !node->Is<Param>(),
        .is_live_in = false,
        .is_live_out = f->IsFunction() && f->HasImplicitUse(node)});
  }

  // Proc state is represented as backedges in the graph.
  if (f->IsProc()) {
    for (Next* next : f->AsProcOrDie()->next_values()) {
      StateRead* state_read = next->state_read()->As<StateRead>();
      backedges.push_back(
          ScheduleBackedge{.source = next,
                           .destination = state_read,
                           .distance = LessThanInitiationInterval()});
    }
  }

  return ScheduleGraph(f->name(), f, std::move(nodes), std::move(backedges));
}

absl::StatusOr<ScheduleGraph> ScheduleGraph::CreateSynchronousGraph(
    Package* p, absl::Span<Channel* const> loopback_channels,
    const ProcElaboration& elab,
    const absl::flat_hash_set<Node*>& dead_after_synthesis) {
  // TODO(https://github.com/google/xls/issues/2175): Multiply-instantiated
  // procs are not supported.
  for (Proc* proc : elab.procs()) {
    if (elab.GetInstances(proc).size() != 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "Proc `%s` is instantiated multiple times", proc->name()));
    }
  }

  // TODO(https://github.com/google/xls/issues/2175): Loopback channels are not
  // yet supported.
  if (!loopback_channels.empty()) {
    return absl::UnimplementedError("Loopback channels are not yet supported.");
  }

  absl::flat_hash_map<ChannelInstance*, SendReceivePair> send_recv_pairs;
  XLS_ASSIGN_OR_RETURN(send_recv_pairs, GetSendReceivePairs(elab));

  // Nodes which are live in/out of the graph. These are the receive/send nodes
  // which are connected to the top-level interface.
  absl::flat_hash_set<Node*> live_in_nodes;
  absl::flat_hash_set<Node*> live_out_nodes;

  absl::flat_hash_map<Node*, int64_t> remaining_predecessors;
  std::deque<Node*> ready_nodes;

  // Compute the set of successors and predecessors in the dataflow graph. The
  // dataflow graph includes all nodes in all procs of the elaboration. The
  // edges are the set of normal dataflow edges and non-loopback channels (which
  // are edges from sends to receives).
  absl::flat_hash_map<Node*, std::vector<Node*>> predecessor_map;
  absl::flat_hash_map<Node*, std::vector<Node*>> successor_map;
  std::vector<ScheduleBackedge> backedges;
  for (Proc* proc : elab.procs()) {
    for (Node* node : proc->nodes()) {
      predecessor_map[node];
      successor_map[node];

      std::vector<Node*>& predecessors = predecessor_map.at(node);
      std::vector<Node*>& successors = successor_map.at(node);

      predecessors = UniquifyNodes(node->operands());
      successors =
          std::vector<Node*>(node->users().begin(), node->users().end());

      if (node->Is<ChannelNode>()) {
        ChannelNode* channel_node = node->As<ChannelNode>();
        XLS_ASSIGN_OR_RETURN(
            ChannelInterface * interface,
            proc->GetChannelInterface(channel_node->channel_name(),
                                      channel_node->direction()));
        absl::Span<ChannelInstance* const> channel_instances =
            elab.GetInstancesOfChannelInterface(interface);
        XLS_RET_CHECK_EQ(channel_instances.size(), 1);
        ChannelInstance* channel_instance = channel_instances.front();
        if (absl::c_find(loopback_channels, channel_instance->channel) ==
            loopback_channels.end()) {
          // Non-loopback channels are dataflow edges in the graph. Loopback
          // channels are handled elsewhere as explicit backedges in the graph.
          const SendReceivePair& pair = send_recv_pairs.at(channel_instance);
          if (channel_node->direction() == ChannelDirection::kSend) {
            if (pair.receive.has_value()) {
              successors.push_back(*pair.receive);
            } else {
              // `node` sends on a top-level interface channel.
              XLS_RET_CHECK(elab.IsTopInterfaceChannel(channel_instance));
              live_out_nodes.insert(node);
            }
          } else {
            if (pair.send.has_value()) {
              predecessors.push_back(*pair.send);
            } else {
              // `node` receives from a top-level interface channel.
              XLS_RET_CHECK(elab.IsTopInterfaceChannel(channel_instance));
              live_in_nodes.insert(node);
            }
          }
        }
      }

      remaining_predecessors[node] = predecessors.size();
      if (predecessors.empty()) {
        ready_nodes.push_back(node);
      }
    }

    for (Next* next : proc->next_values()) {
      backedges.push_back(
          ScheduleBackedge{.source = next,
                           .destination = next->state_read(),
                           .distance = LessThanInitiationInterval()});
    }
  }

  // Compute a toposort of the nodes in the dataflow graph. The ScheduleNodes
  // in ScheduleGraph.nodes_ must be in toposort order.
  std::vector<ScheduleNode> nodes;
  nodes.reserve(remaining_predecessors.size());
  while (!ready_nodes.empty()) {
    Node* node = ready_nodes.front();
    ready_nodes.pop_front();

    nodes.push_back(ScheduleNode{
        .node = node,
        .predecessors = predecessor_map.at(node),
        .successors = successor_map.at(node),
        .is_dead_after_synthesis = dead_after_synthesis.contains(node),
        .schedule_in_first_stage = false,
        .schedule_in_last_stage = false,
        .is_live_in = live_in_nodes.contains(node),
        .is_live_out = live_out_nodes.contains(node)});

    for (Node* successor : successor_map.at(node)) {
      if (--remaining_predecessors[successor] == 0) {
        ready_nodes.push_front(successor);
      }
    }
  }

  return ScheduleGraph(p->name(), p, std::move(nodes), std::move(backedges));
}

}  // namespace xls

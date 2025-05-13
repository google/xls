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
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/topo_sort.h"

namespace xls {

std::string ScheduleGraph::ToString() const {
  std::vector<std::string> lines;
  lines.push_back(absl::StrCat("ScheduleGraph(", name(), "):"));
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
  ScheduleGraph graph;
  graph.name_ = f->name();
  graph.ir_scope_ = f;
  graph.nodes_.reserve(f->node_count());
  for (Node* node : TopoSort(f)) {
    graph.nodes_.push_back(ScheduleNode{
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
      graph.backedges_.push_back(
          ScheduleBackedge{.source = next,
                           .destination = state_read,
                           .distance = LessThanInitiationInterval()});
    }
  }

  graph.node_map_.reserve(graph.nodes_.size());
  for (int64_t i = 0; i < graph.nodes_.size(); ++i) {
    graph.node_map_[graph.nodes_[i].node] = i;
  }

  return graph;
}

}  // namespace xls

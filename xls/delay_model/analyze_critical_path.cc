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

#include "xls/delay_model/analyze_critical_path.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_info.pb.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"

namespace xls {

absl::StatusOr<std::vector<CriticalPathEntry>> AnalyzeCriticalPath(
    FunctionBase* f, std::optional<int64_t> clock_period_ps,
    const DelayEstimator& delay_estimator) {
  struct NodeEntry {
    Node* node;

    // Delay of the node.
    int64_t node_delay;

    // The delay of the critical path in the graph up to and including this node
    // (includes this node's delay).
    int64_t critical_path_delay;

    // The predecessor on the critical path through this node.
    std::optional<Node*> critical_path_predecessor;

    // Whether this node was delayed by a cycle boundary.
    bool delayed_by_cycle_boundary;
  };

  // Map from each node to it's corresponding entry.
  absl::flat_hash_map<Node*, NodeEntry> node_entries;

  // The node with the greatest critical path delay.
  std::optional<NodeEntry> latest_entry;

  for (Node* node : TopoSort(f)) {
    NodeEntry& entry = node_entries[node];
    entry.node = node;

    // The maximum delay from any path up to but not including `node`.
    int64_t max_path_delay = 0;
    for (Node* operand : node->operands()) {
      int64_t operand_path_delay = node_entries.at(operand).critical_path_delay;
      if (operand_path_delay >= max_path_delay) {
        max_path_delay = node_entries.at(operand).critical_path_delay;
        entry.critical_path_predecessor = operand;
      }
    }
    XLS_ASSIGN_OR_RETURN(entry.node_delay,
                         delay_estimator.GetOperationDelayInPs(node));

    // If the dependency straddles a clock boundary we have to make our delay
    // start from the clock time.
    entry.delayed_by_cycle_boundary = false;
    if (clock_period_ps.has_value() &&
        (((max_path_delay + entry.node_delay) / clock_period_ps.value()) >
         (max_path_delay / clock_period_ps.value()))) {
      int64_t new_max_path_delay = RoundDownToNearest(
          max_path_delay + entry.node_delay, clock_period_ps.value());
      CHECK_GT(new_max_path_delay, max_path_delay);
      max_path_delay = new_max_path_delay;
      entry.delayed_by_cycle_boundary = true;
    }
    entry.critical_path_delay = max_path_delay + entry.node_delay;

    if (!latest_entry.has_value() ||
        latest_entry->critical_path_delay <= entry.critical_path_delay) {
      latest_entry = entry;
    }
  }

  // Starting with the operation with the longest pat hdelay, walk back up its
  // critical path constructing CriticalPathEntry's as we go.
  std::vector<CriticalPathEntry> critical_path;
  XLS_RET_CHECK(latest_entry.has_value());
  NodeEntry* entry = &(latest_entry.value());
  while (true) {
    critical_path.push_back(CriticalPathEntry{
        .node = entry->node,
        .node_delay_ps = entry->node_delay,
        .path_delay_ps = entry->critical_path_delay,
        .delayed_by_cycle_boundary = entry->delayed_by_cycle_boundary});
    if (!entry->critical_path_predecessor.has_value()) {
      break;
    }
    entry = &node_entries.at(entry->critical_path_predecessor.value());
  }

  return std::move(critical_path);
}

std::string CriticalPathToString(
    absl::Span<const CriticalPathEntry> critical_path,
    std::optional<std::function<std::string(Node*)>> extra_info) {
  std::string result;
  for (const CriticalPathEntry& entry : critical_path) {
    absl::StrAppendFormat(&result, " %6dps (+%3dps)%s: %s\n",
                          entry.path_delay_ps, entry.node_delay_ps,
                          entry.delayed_by_cycle_boundary ? "!" : "",
                          entry.node->ToStringWithOperandTypes());
    if (extra_info.has_value()) {
      absl::StrAppend(&result, extra_info.value()(entry.node));
    }
  }
  return result;
}

CriticalPathProto CriticalPathToProto(
    absl::Span<const CriticalPathEntry> critical_path) {
  CriticalPathProto proto;
  for (const CriticalPathEntry& entry : critical_path) {
    DelayInfoNodeProto* node = proto.add_nodes();
    node->set_op(ToOpProto(entry.node->op()));
    node->set_node_delay_ps(entry.node_delay_ps);
    node->set_total_delay_ps(entry.path_delay_ps);
    node->set_id(entry.node->id());
    node->set_ir(entry.node->ToStringWithOperandTypes());
  }
  proto.set_total_delay_ps(critical_path.front().path_delay_ps);
  return proto;
}

}  // namespace xls

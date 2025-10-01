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

#include "xls/estimators/delay_model/analyze_critical_path.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_info.pb.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"

namespace xls {

namespace {

absl::StatusOr<NodeDelayEntries> AccumulateNodeDelays(
    FunctionBase* f, std::optional<int64_t> clock_period_ps,
    const DelayEstimator& delay_estimator,
    absl::AnyInvocable<bool(Node*)>& source_filter,
    absl::AnyInvocable<bool(Node*)>& sink_filter) {
  NodeDelayEntries entries;
  entries.topo_sorted_nodes = TopoSort(f);

  for (Node* node : entries.topo_sorted_nodes) {
    if (!source_filter(node) &&
        !absl::c_any_of(node->operands(), [&](Node* operand) {
          return entries.node_entries.contains(operand);
        })) {
      // This node is neither a source nor on a path from a source.
      continue;
    }
    NodeDelayEntry& entry = entries.node_entries[node];
    entry.node = node;

    // The maximum delay from any path up to but not including `node`.
    int64_t max_path_delay = 0;
    for (Node* operand : node->operands()) {
      auto it = entries.node_entries.find(operand);
      if (it == entries.node_entries.end()) {
        // This operand is neither a source nor on a path from a source.
        continue;
      }
      int64_t operand_path_delay = it->second.critical_path_delay;
      if (operand_path_delay >= max_path_delay) {
        max_path_delay = operand_path_delay;
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

    if (!sink_filter(node)) {
      continue;
    }
    if (!entries.latest.has_value() ||
        entries.latest->critical_path_delay <= entry.critical_path_delay) {
      entries.latest = entry;
    }
  }
  return entries;
}

}  // anonymous namespace

absl::StatusOr<std::vector<CriticalPathEntry>> AnalyzeCriticalPath(
    FunctionBase* f, std::optional<int64_t> clock_period_ps,
    const DelayEstimator& delay_estimator,
    absl::AnyInvocable<bool(Node*)> source_filter,
    absl::AnyInvocable<bool(Node*)> sink_filter) {
  XLS_ASSIGN_OR_RETURN(NodeDelayEntries entries,
                       AccumulateNodeDelays(f, clock_period_ps, delay_estimator,
                                            source_filter, sink_filter));

  // `latest_entry` has no value for empty FunctionBases or if the source & sink
  // filters removed all nodes.
  if (!entries.latest.has_value()) {
    return std::vector<CriticalPathEntry>();
  }

  // Starting with the operation with the longest path delay, walk back up its
  // critical path constructing CriticalPathEntry's as we go.
  std::vector<CriticalPathEntry> critical_path;
  NodeDelayEntry* entry = &(entries.latest.value());
  while (true) {
    critical_path.push_back(CriticalPathEntry{
        .node = entry->node,
        .node_delay_ps = entry->node_delay,
        .path_delay_ps = entry->critical_path_delay,
        .delayed_by_cycle_boundary = entry->delayed_by_cycle_boundary});
    if (!entry->critical_path_predecessor.has_value()) {
      break;
    }
    entry = &entries.node_entries.at(entry->critical_path_predecessor.value());
  }

  return std::move(critical_path);
}

absl::StatusOr<absl::flat_hash_map<Node*, int64_t>> SlackFromCriticalPath(
    FunctionBase* f, std::optional<int64_t> clock_period_ps,
    const DelayEstimator& delay_estimator,
    absl::AnyInvocable<bool(Node*)> source_filter,
    absl::AnyInvocable<bool(Node*)> sink_filter) {
  XLS_ASSIGN_OR_RETURN(NodeDelayEntries entries,
                       AccumulateNodeDelays(f, clock_period_ps, delay_estimator,
                                            source_filter, sink_filter));

  absl::flat_hash_map<Node*, int64_t> node_slack;
  for (auto node_iter = entries.topo_sorted_nodes.rbegin();
       node_iter != entries.topo_sorted_nodes.rend(); ++node_iter) {
    Node* node = *node_iter;
    if (!entries.node_entries.contains(node)) {
      continue;
    }
    const NodeDelayEntry& node_entry = entries.node_entries.at(node);

    int64_t min_slack = std::numeric_limits<int64_t>::max();
    bool has_any_users = false;
    for (Node* user : node->users()) {
      if (!entries.node_entries.contains(user)) {
        continue;
      }
      has_any_users = true;

      int64_t max_other_operand_delay = node_entry.critical_path_delay;
      for (Node* operand : user->operands()) {
        if (!entries.node_entries.contains(operand)) {
          continue;
        }
        max_other_operand_delay =
            std::max(max_other_operand_delay,
                     entries.node_entries.at(operand).critical_path_delay);
      }

      // A node's slack w.r.t a user is the user's slack plus how much less this
      // node's delay is than the largest delay of the user's other operands.
      min_slack =
          std::min(min_slack, node_slack[user] + max_other_operand_delay -
                                  node_entry.critical_path_delay);
    }

    // If at the end of the def-use chain, the slack is how much less this
    // node's delay is than the critical path delay.
    node_slack[node] =
        has_any_users
            ? min_slack
            : std::max((int64_t)0, entries.latest->critical_path_delay -
                                       node_entry.critical_path_delay);
  }

  return node_slack;
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

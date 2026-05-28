// Copyright 2026 The XLS Authors
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

#include "xls/scheduling/random_scheduler.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

absl::StatusOr<std::vector<Node*>> RandomScheduler::ShuffleNodes() {
  std::vector<Node*> nodes;
  nodes.reserve(asap_.graph().nodes().size());
  absl::c_transform(
      asap_.graph().nodes(), std::back_inserter(nodes),
      [](const ScheduleNode& schedule_node) { return schedule_node.node; });
  absl::c_shuffle(nodes, bitgen_);
  return nodes;
}

absl::StatusOr<int64_t> RandomScheduler::GetRandomCycle(Node* node, int64_t low,
                                                        int64_t high) {
  return absl::Uniform<int64_t>(bitgen_, low, high + 1);
}

absl::StatusOr<ScheduleCycleMap> RandomScheduler::Schedule(
    std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
    SchedulingFailureBehavior failure_behavior,
    std::optional<int64_t> worst_case_throughput) {
  XLS_RET_CHECK(
      std::holds_alternative<FunctionBase*>(asap_.graph().ir_scope()));
  auto* f = std::get<FunctionBase*>(asap_.graph().ir_scope());
  // Use ASAP schedule to get max pipeline stages.
  if (!pipeline_stages.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        auto asap_schedule,
        asap_.Schedule(pipeline_stages, clock_period_ps, failure_behavior,
                       worst_case_throughput));
    pipeline_stages = absl::c_max_element(asap_schedule,
                                          [](auto item1, auto item2) {
                                            return item1.second < item2.second;
                                          })
                          ->second +
                      1;
    VLOG(5) << "ASAP Says the schedule is "
            << f->DumpIr(ScheduleCycleAnnotator(asap_schedule));
    VLOG(5) << "Max pipeline stages: " << *pipeline_stages;
  }
  XLS_ASSIGN_OR_RETURN(sched::ScheduleBounds bounds,
                       asap_.ComputeBounds(pipeline_stages, clock_period_ps,
                                           worst_case_throughput));
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> all_nodes, ShuffleNodes());
  for (auto* node : all_nodes) {
    if (bounds.lb(node) == bounds.ub(node)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        int64_t cycle, GetRandomCycle(node, bounds.lb(node), bounds.ub(node)));
    VLOG(5) << "Scheduling node " << node->ToString() << "[" << bounds.lb(node)
            << ", " << bounds.ub(node) << "] into " << cycle;
    bounds.AddConstraint(NodeInCycleConstraint{node, cycle});
    XLS_RETURN_IF_ERROR(
        ASAPSchedulerWrapper::TightenBounds(bounds, f, pipeline_stages));
  }
  // Every node is either fully constrained or explicitly put into a random
  // cycle.
  ScheduleCycleMap cycle_map;
  cycle_map.reserve(all_nodes.size());
  for (auto* node : all_nodes) {
    cycle_map[node] = bounds.lb(node);
  }
  return cycle_map;
}

}  // namespace xls

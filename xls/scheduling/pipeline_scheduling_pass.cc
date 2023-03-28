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

#include "xls/scheduling/pipeline_scheduling_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

absl::StatusOr<bool> PipelineSchedulingPass::RunInternal(
    SchedulingUnit<>* unit, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  XLS_RET_CHECK_NE(options.delay_estimator, nullptr);

  absl::flat_hash_map<Node*, int64_t> schedule_cycle_map_before;
  SchedulingOptions scheduling_options = options.scheduling_options;
  if (unit->schedule.has_value()) {
    for (int64_t c = 0; c < unit->schedule.value().length(); ++c) {
      for (Node* node : unit->schedule.value().nodes_in_cycle(c)) {
        schedule_cycle_map_before[node] = c;
        scheduling_options.add_constraint(NodeInCycleConstraint(node, c));
      }
    }
  }

  FunctionBase* f = nullptr;
  if (unit->schedule.has_value()) {
    f = unit->schedule.value().function_base();
  } else {
    std::optional<FunctionBase*> top = unit->ir->GetTop();
    XLS_RET_CHECK(top.has_value())
        << "Package " << unit->name() << " needs a top function/proc.";
    f = top.value();
  }

  XLS_ASSIGN_OR_RETURN(
      unit->schedule,
      PipelineSchedule::Run(f, *options.delay_estimator, scheduling_options));

  absl::flat_hash_map<Node*, int64_t> schedule_cycle_map_after;
  for (int64_t c = 0; c < unit->schedule.value().length(); ++c) {
    for (Node* node : unit->schedule.value().nodes_in_cycle(c)) {
      schedule_cycle_map_after[node] = c;
    }
  }

  return schedule_cycle_map_before != schedule_cycle_map_after;
}

}  // namespace xls

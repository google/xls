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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

namespace {

// Adds cycle constraints from a PipelineSchedule into SchedulingOptions.
void AddCycleConstraints(const PipelineSchedule& schedule,
                         SchedulingOptions& scheduling_options) {
  for (int64_t c = 0; c < schedule.length(); ++c) {
    for (Node* node : schedule.nodes_in_cycle(c)) {
      scheduling_options.add_constraint(NodeInCycleConstraint(node, c));
    }
  }
}
}  // namespace

absl::StatusOr<bool> PipelineSchedulingPass::RunInternal(
    Package* package, const SchedulingPassOptions& options,
    PassResults* results, SchedulingContext& context) const {
  XLS_RET_CHECK_NE(options.delay_estimator, nullptr);
  bool changed = false;

  XLS_ASSIGN_OR_RETURN(std::vector<FunctionBase*> schedulable_functions,
                       context.GetSchedulableFunctions());

  // Scheduling of procs with proc-scoped channels requires an elaboration.
  std::optional<ProcElaboration> elab;
  if (package->ChannelsAreProcScoped() && !schedulable_functions.empty() &&
      schedulable_functions.front()->IsProc()) {
    XLS_ASSIGN_OR_RETURN(Proc * top, package->GetTopAsProc());
    XLS_ASSIGN_OR_RETURN(elab, ProcElaboration::Elaborate(top));
  }
  std::optional<const ProcElaboration*> elab_opt =
      elab.has_value() ? std::optional<const ProcElaboration*>(&elab.value())
                       : std::nullopt;

  for (FunctionBase* f : schedulable_functions) {
    if (f->ForeignFunctionData().has_value()) {
      continue;
    }
    absl::flat_hash_map<Node*, int64_t> schedule_cycle_map_before;
    SchedulingOptions scheduling_options = options.scheduling_options;
    auto schedule_itr = context.schedules().find(f);
    if (schedule_itr != context.schedules().end()) {
      const PipelineSchedule& schedule = schedule_itr->second;
      schedule_cycle_map_before = schedule.GetCycleMap();
      if (!scheduling_options.use_fdo()) {
        AddCycleConstraints(schedule, scheduling_options);
      }
    }
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        options.synthesizer == nullptr
            ? RunPipelineSchedule(f, *options.delay_estimator,
                                  scheduling_options, elab_opt)
            : RunPipelineScheduleWithFdo(f, *options.delay_estimator,
                                         scheduling_options,
                                         *options.synthesizer, elab_opt));

    // Compute `changed` before moving schedule into context.schedules.
    changed = changed || (schedule_cycle_map_before != schedule.GetCycleMap());

    context.schedules().insert_or_assign(schedule_itr, f, std::move(schedule));
  }
  return changed;
}

}  // namespace xls

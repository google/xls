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
    SchedulingUnit* unit, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  XLS_RET_CHECK(!unit->schedule.has_value())
      << "Package " << unit->name() << " already has a schedule.";
  XLS_RET_CHECK_NE(options.delay_estimator, nullptr);
  Function* entry;
  if (options.scheduling_options.entry().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        entry,
        unit->package->GetFunction(options.scheduling_options.entry().value()));
  } else {
    XLS_ASSIGN_OR_RETURN(entry, unit->package->EntryFunction());
  }
  XLS_ASSIGN_OR_RETURN(unit->schedule,
                       PipelineSchedule::Run(entry, *options.delay_estimator,
                                             options.scheduling_options));
  return true;
}

}  // namespace xls

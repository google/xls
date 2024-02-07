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

#include "xls/scheduling/scheduling_checker.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

absl::Status SchedulingChecker::Run(SchedulingUnit* unit,
                                    const SchedulingPassOptions& options,
                                    SchedulingPassResults* results) const {
  XLS_RETURN_IF_ERROR(VerifyPackage(unit->GetPackage()));
  XLS_ASSIGN_OR_RETURN(std::vector<FunctionBase*> schedulable_functions,
                       unit->GetSchedulableFunctions());
  XLS_RET_CHECK_GT(schedulable_functions.size(), 0);
  for (FunctionBase* fb : schedulable_functions) {
    XLS_RET_CHECK_EQ(fb->package(), unit->GetPackage());
    auto itr = unit->schedules().find(fb);
    if (itr == unit->schedules().end()) {
      XLS_RET_CHECK(unit->schedules().empty()) << absl::StreamFormat(
          "Schedulable function %v not found in non-empty schedules map", *fb);
      continue;
    }
    const PipelineSchedule& schedule = itr->second;
    XLS_RET_CHECK_EQ(schedule.function_base(), fb);
    XLS_RETURN_IF_ERROR(schedule.Verify());
    // TODO(meheff): Add check to ensure schedule matches the specified
    // SchedulingOptions. For example, number pipeline_stages, clock_period,
    // etc.
  }
  return absl::OkStatus();
}

}  // namespace xls

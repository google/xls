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
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

absl::Status SchedulingChecker::Run(Package* package,
                                    const SchedulingPassOptions& options,
                                    PassResults* results,
                                    SchedulingContext& context) const {
  XLS_RETURN_IF_ERROR(VerifyPackage(package));
  XLS_ASSIGN_OR_RETURN(std::vector<FunctionBase*> schedulable_functions,
                       context.GetSchedulableFunctions());
  XLS_RET_CHECK_GT(schedulable_functions.size(), 0);
  for (FunctionBase* fb : schedulable_functions) {
    XLS_RET_CHECK_EQ(fb->package(), package);
    if (!context.package_schedule().HasSchedule(fb)) {
      XLS_RET_CHECK(context.package_schedule().GetSchedules().empty())
          << absl::StreamFormat(
                 "Schedulable function %v not found in non-empty schedules map",
                 *fb);
      continue;
    }
    const PipelineSchedule& schedule =
        context.package_schedule().GetSchedule(fb);
    XLS_RET_CHECK_EQ(schedule.function_base(), fb);
    XLS_RETURN_IF_ERROR(schedule.Verify());
    // TODO(meheff): Add check to ensure schedule matches the specified
    // SchedulingOptions. For example, number pipeline_stages, clock_period,
    // etc.
  }
  return absl::OkStatus();
}

}  // namespace xls

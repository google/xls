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

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/verifier.h"

namespace xls {

absl::Status SchedulingChecker::Run(SchedulingUnit* unit,
                                    const SchedulingPassOptions& options,
                                    SchedulingPassResults* results) const {
  XLS_RETURN_IF_ERROR(VerifyPackage(unit->package));
  if (unit->schedule.has_value()) {
    Function* entry;
    if (options.scheduling_options.entry().has_value()) {
      XLS_ASSIGN_OR_RETURN(entry,
                           unit->package->GetFunction(
                               options.scheduling_options.entry().value()));
    } else {
      XLS_ASSIGN_OR_RETURN(entry, unit->package->EntryFunction());
    }
    XLS_RET_CHECK_EQ(entry, unit->schedule->function());
    XLS_RETURN_IF_ERROR(unit->schedule->Verify());
    // TODO(meheff): Add check to ensure schedule matches the specified
    // SchedulingOptions. For example, number pipeline_stages, clock_period,
    // etc.
  }
  return absl::OkStatus();
}

}  // namespace xls

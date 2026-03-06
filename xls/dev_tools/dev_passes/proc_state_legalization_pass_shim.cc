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

#include "xls/dev_tools/dev_passes/proc_state_legalization_pass_shim.h"

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {
absl::StatusOr<bool> ProcStateLegalizationPassShim::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* pass_results, OptimizationContext& context) const {
  SchedulingContext sched_context = SchedulingContext::CreateForWholePackage(p);
  XLS_ASSIGN_OR_RETURN(bool res,
                       proc_state_sched_pass_.Run(p, SchedulingPassOptions(),
                                                  pass_results, sched_context));
  return res;
}

}  // namespace xls

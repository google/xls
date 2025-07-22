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

#include <utility>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {
absl::StatusOr<bool> ProcStateLegalizationPassShim::RunOnFunctionBaseInternal(
    FunctionBase* fb, const OptimizationPassOptions& options,
    PassResults* pass_results, OptimizationContext& context) const {
  SchedulingContext sched_context =
      SchedulingContext::CreateForSingleFunction(fb);
  PassResults results;
  if (pass_results) {
    results.invocation = std::move(pass_results->invocation);
  }
  XLS_ASSIGN_OR_RETURN(
      bool res, proc_state_sched_pass_.RunOnFunctionBase(
                    fb, SchedulingPassOptions(), &results, sched_context));
  if (pass_results) {
    pass_results->invocation = std::move(results.invocation);
  }
  return res;
}

}  // namespace xls

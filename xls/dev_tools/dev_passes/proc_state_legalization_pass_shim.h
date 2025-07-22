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

#ifndef XLS_DEV_TOOLS_DEV_PASSES_PROC_STATE_LEGALIZATION_PASS_SHIM_H_
#define XLS_DEV_TOOLS_DEV_PASSES_PROC_STATE_LEGALIZATION_PASS_SHIM_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/proc_state_legalization_pass.h"
namespace xls {

// Compatibility shim to use the scheduling pass 'ProcStateLegalizationPass' in
// the optimization pass pipeline for modernizing procs.
class ProcStateLegalizationPassShim : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "proc_state_legalization_shim";
  ProcStateLegalizationPassShim()
      : OptimizationFunctionBasePass(kName, "Proc State Legalization Pass") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* fb, const OptimizationPassOptions& options,
      PassResults* pass_results, OptimizationContext& context) const override;

 private:
  ProcStateLegalizationPass proc_state_sched_pass_;
};

}  // namespace xls

#endif  // XLS_DEV_TOOLS_DEV_PASSES_PROC_STATE_LEGALIZATION_PASS_SHIM_H_

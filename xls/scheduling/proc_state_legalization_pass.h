// Copyright 2024 The XLS Authors
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

#ifndef XLS_SCHEDULING_PROC_STATE_LEGALIZATION_PASS_H_
#define XLS_SCHEDULING_PROC_STATE_LEGALIZATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

// Pass which legalizes state parameters by introducing a default `next_value`
// node that fires if no other `next_value` node will fire.
//
// Attempts to recognize scenarios where no default is needed (even using Z3 to
// attempt to prove that one of the explicit `next_value` nodes is guaranteed to
// fire), but may produce dead `next_value` nodes in some scenarios. However,
// this should have a minimal impact on QoR.
class ProcStateLegalizationPass
    : public SchedulingOptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "proc_state_legal";

  ProcStateLegalizationPass()
      : SchedulingOptimizationFunctionBasePass(kName,
                                               "Proc State Legalization") {}
  ~ProcStateLegalizationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, SchedulingUnit* s, const SchedulingPassOptions& options,
      SchedulingPassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_PROC_STATE_LEGALIZATION_PASS_H_

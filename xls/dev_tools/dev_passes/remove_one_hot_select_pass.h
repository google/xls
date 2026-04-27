// Copyright 2026 The XLS Authors
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

#ifndef XLS_DEV_TOOLS_DEV_PASSES_REMOVE_ONE_HOT_SELECT_PASS_H_
#define XLS_DEV_TOOLS_DEV_PASSES_REMOVE_ONE_HOT_SELECT_PASS_H_

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which replaces OneHotSelects with equivalent Selects without regard to
// profitability concerns. This pass is meant for investigation and analysis use
// only and should not be used for normal optimization pipelines.
//
// To avoid extreme proliferation of nodes, we only translate OneHotSelects with
// a selector bitwidth <= 16.
class RemoveOneHotSelectPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "remove_one_hot_sel";
  static constexpr int64_t kMaxBits = 16;

  RemoveOneHotSelectPass()
      : OptimizationFunctionBasePass(kName, "One Hot Select Removal Pass") {}
  ~RemoveOneHotSelectPass() = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* pass_results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_DEV_TOOLS_DEV_PASSES_REMOVE_ONE_HOT_SELECT_PASS_H_

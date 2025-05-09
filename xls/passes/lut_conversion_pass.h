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

#ifndef XLS_PASSES_LUT_CONVERSION_PASS_H_
#define XLS_PASSES_LUT_CONVERSION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which opportunistically converts nodes to lookup tables (selects) where
// we can prove it's beneficial.
class LutConversionPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "lut_conversion";
  explicit LutConversionPass()
      : OptimizationFunctionBasePass(kName, "LUT Conversion") {}
  ~LutConversionPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_LUT_CONVERSION_PASS_H_

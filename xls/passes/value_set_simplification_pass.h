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

#ifndef XLS_PASSES_VALUE_SET_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_VALUE_SET_SIMPLIFICATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Simplifies operations where an operand can only take a small set of values
// (determined by range/interval analysis) into a select of cheaper operations.
class ValueSetSimplificationPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "value_set_simp";
  explicit ValueSetSimplificationPass()
      : OptimizationFunctionBasePass(kName, "Value Set Simplification") {}
  ~ValueSetSimplificationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_VALUE_SET_SIMPLIFICATION_PASS_H_

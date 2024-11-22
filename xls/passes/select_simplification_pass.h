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

#ifndef XLS_PASSES_SELECT_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_SELECT_SIMPLIFICATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Base class which simplifies selects and one-hot-selects. Example
// optimizations include removing dead arms and eliminating selects with
// constant selectors.
class SelectSimplificationPassBase : public OptimizationFunctionBasePass {
 public:
  ~SelectSimplificationPassBase() override = default;

 protected:
  explicit SelectSimplificationPassBase(std::string_view short_name,
                                        std::string_view name,
                                        bool with_range_analysis = false)
      : OptimizationFunctionBasePass(short_name, name),
        range_analysis_(with_range_analysis) {}

  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;

  bool range_analysis_;
};

// Pass which simplifies selects and one-hot-selects. Example optimizations
// include removing dead arms and eliminating selects with constant selectors.
// Uses ternary analysis to determine possible values.
class SelectSimplificationPass : public SelectSimplificationPassBase {
 public:
  static constexpr std::string_view kName = "select_simp";
  SelectSimplificationPass()
      : SelectSimplificationPassBase(kName, "Select Simplification",
                                     /*with_range_analysis=*/false) {}
  ~SelectSimplificationPass() override = default;
};

// Pass which simplifies selects and one-hot-selects. Example optimizations
// include removing dead arms and eliminating selects with constant selectors.
// Uses range analysis to determine possible values.
class SelectRangeSimplificationPass : public SelectSimplificationPassBase {
 public:
  static constexpr std::string_view kName = "select_range_simp";
  SelectRangeSimplificationPass()
      : SelectSimplificationPassBase(kName, "Select Range Simplification",
                                     /*with_range_analysis=*/true) {}
  ~SelectRangeSimplificationPass() override = default;
};

}  // namespace xls

#endif  // XLS_PASSES_SELECT_SIMPLIFICATION_PASS_H_

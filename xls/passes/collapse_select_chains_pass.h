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

#ifndef XLS_PASSES_COLLAPSE_SELECT_CHAINS_PASS_H_
#define XLS_PASSES_COLLAPSE_SELECT_CHAINS_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which collapses chains of selects with disjoint selectors into a single
// one-hot-select.
//
// Chains of binary `select` operations, particularly where one case of a
// `select` feeds into another `select` (e.g.,
// `sel(pred_a, {val_a, sel(pred_b, {val_b, default_val})})`), are transformed
// into a single `one_hot_select` if the selectors are provably disjoint (at
// most one can be true at any given time). This simplifies the IR structure and
// can lead to more efficient hardware implementations.
class CollapseSelectChainsPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "collapse_select_chains";
  explicit CollapseSelectChainsPass()
      : OptimizationFunctionBasePass(kName,
                                     "BDD-based Select Chain Collapsing") {}
  ~CollapseSelectChainsPass() override = default;

  RedundancyGuard GetRedundancyGuard(
      const OptimizationPassOptions& options,
      OptimizationContext& context) const override {
    return RedundancyGuard::CanSkip();
  }

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_COLLAPSE_SELECT_CHAINS_PASS_H_

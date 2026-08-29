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

#ifndef XLS_PASSES_CONCAT_SELECT_REMOVAL_PASS_H_
#define XLS_PASSES_CONCAT_SELECT_REMOVAL_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
namespace xls {

// Replaces a concat of selects with the same selectors with a single select of
// concats. This seems to be optimized better by synthesis tools.
//
// For example,
//   (concat (select a b c)  (select a d e))
// would be replaced with
//   (select a (concat b d) (concat c e))
//
// Or
//   (concat X (select a b c) (select a d e) Y)
// would be replaced with
//   (Concat X (select a (concat b d) (concat c e)) Y)
//
// We will not add additional bit-slices to support things like
//   (concat (select a b c) X (select a d e))
// TODO(allight): Should we do this?
//
// Nor will we combine selects with unlike selectors.
//   (concat (select a b c) (select d e f))
class ConcatSelectRemovalPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "concat_select_removal";
  ConcatSelectRemovalPass()
      : OptimizationFunctionBasePass(kName, "Concat Select Removal") {}
  ~ConcatSelectRemovalPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CONCAT_SELECT_REMOVAL_PASS_H_

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

#ifndef XLS_PASSES_LOOP_IDIOM_PASS_H_
#define XLS_PASSES_LOOP_IDIOM_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Recognizes common loop idioms in `counted_for` loops and rewrites them into
// more direct IR forms BEFORE the loop is unrolled.
//
// First target: a zero-filled array shift (DSLX `data << shift` expressed as a
// `for` loop with `result[i] = if i < shift { false } else { data[i-shift] }`).
// The counted_for body lowers to an `array_update` of
// `sel(ult(i, shift), [array_index(data, [i - shift]), 0])`; rewriting that to
// a packed `shll` + per-bit `bit_slice` shape lets codegen emit a native shift
// instead of per-element index muxes.
class LoopIdiomPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "loop_idiom";
  LoopIdiomPass()
      : OptimizationFunctionBasePass(kName, "Recognize loop idioms") {}

  bool IsIdempotent() const override { return true; }

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

#endif  // XLS_PASSES_LOOP_IDIOM_PASS_H_

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

#ifndef XLS_PASSES_BITWISE_RECOMBINATION_PASS_H_
#define XLS_PASSES_BITWISE_RECOMBINATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which recombines concats of contiguous bit slices & inverted bit slices
// from a single source node (and constants) into masked bitwise operations.
//
// The pass uses 1D dynamic programming optimal partitioning to balance
// readability:
//   - Identifies localized sub-runs that can be merged into clean 1-gate masks
//     (nibble-aligned or periodic, e.g. x ^ 0x5555).
//   - Naturally peels boundary constants (e.g. address alignment or padding)
//     to form clean hybrid concats like {x[31:8] ^ 24'h555555, 8'b0}.
//   - Preserves coarse struct/bus layouts where non-aligned hex masks would be
//     unreadable.
//   - Recombines dense bit-salad into 2-gate operations when the splinter count
//     is high.
//
// Recombined operations include:
//   - all inverted slices                 => not(x)
//   - raw slices + inverted slices        => (x ^ mask)
//   - raw slices + 0 literals             => (x & mask)
//   - raw slices + 1 literals             => (x | mask)
//   - raw slices + 0s + 1s (no inv)       => ((x & mask_and) | mask_or)
//   - inverted + 1s (no raw)              => (not(x) | mask_or)
//   - raw + inverted + 1s                 => ((x ^ mask_xor) | mask_or)
//   - inverted + 0s (no raw)              => (not(x) & mask_and)
//   - raw + inverted + 0s                 => ((x ^ mask_xor) & mask_and)
//   - inverted + 0s + 1s (no raw)         => ((not(x) & mask_and) | mask_or)
//   - all four (raw, inv, 0s, 1s)         => ((x & mask_and) ^ mask_xor)
//
// This is typically run late in the compiler pipeline (e.g. in codegen 1.5
// block cleanup and optimization) to fold un-cancelled bitwise splits back into
// clean, idiomatic hardware operations before Verilog emission.
class BitwiseRecombinationPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "bitwise_recombine";
  explicit BitwiseRecombinationPass()
      : OptimizationFunctionBasePass(kName, "bitwise recombination") {}
  ~BitwiseRecombinationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_BITWISE_RECOMBINATION_PASS_H_

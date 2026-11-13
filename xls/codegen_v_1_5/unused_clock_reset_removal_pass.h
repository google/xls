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

#ifndef XLS_CODEGEN_V_1_5_UNUSED_CLOCK_RESET_REMOVAL_PASS_H_
#define XLS_CODEGEN_V_1_5_UNUSED_CLOCK_RESET_REMOVAL_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

// Removes clock and reset ports from instantiated (non-top) blocks which do
// not use them (https://github.com/google/xls/issues/4635).
//
// Block conversion adds clock and reset ports to every block up front. A
// block whose logic ends up purely combinational (e.g. a stateless proc
// lowered into a single stage with no I/O flops) keeps those ports even
// though nothing in the block reads them. For nested blocks the parent then
// drives a clock/reset into a module which never uses it, which trips
// unused-signal warnings in downstream tools.
//
// The clock port is removed if the block has no registers, no clocked
// instantiations (FIFOs, delay lines, blocks with a clock port, extern
// instantiations) and no asserts/covers/traces (those are emitted differently
// depending on the presence of a clock). The reset port is removed if the
// port has no users. Blocks are processed children-first so removals
// propagate up the hierarchy. The top block is never changed so the
// user-facing module interface stays stable.
class UnusedClockResetRemovalPass : public BlockConversionPass {
 public:
  static constexpr std::string_view kName = "unused_clock_reset_removal";

  UnusedClockResetRemovalPass()
      : BlockConversionPass(
            kName, "Remove unused clock/reset ports from non-top blocks") {}

 protected:
  absl::StatusOr<bool> RunInternal(
      Package* package, const BlockConversionPassOptions& options,
      PassResults* results, BlockConversionContext& context) const override;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_UNUSED_CLOCK_RESET_REMOVAL_PASS_H_

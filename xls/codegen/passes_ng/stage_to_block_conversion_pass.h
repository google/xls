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

#ifndef XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_PASS_H_
#define XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_PASS_H_

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

// Converts the group of procs created by stage conversion to blocks.
class StageToBlockConversionPass : public CodegenPass {
 public:
  StageToBlockConversionPass()
      : CodegenPass("stage_to_block", "Stage to block conversion pass") {}
  ~StageToBlockConversionPass() override = default;

  absl::StatusOr<bool> RunInternal(Package* absl_nonnull package,
                                   const CodegenPassOptions& options,
                                   PassResults* absl_nonnull results,
                                   CodegenContext& context) const override;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_PASS_H_

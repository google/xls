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

#ifndef XLS_CODEGEN_V_1_5_GLOBAL_CHANNEL_BLOCK_STITCHING_PASS_H_
#define XLS_CODEGEN_V_1_5_GLOBAL_CHANNEL_BLOCK_STITCHING_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

// Stitches blocks based on procs with global channels under one top-level
// container block. This pass is a no-op for packages using proc-scoped
// channels; for those packages, `ProcInstantiationLoweringPass` is the rough
// counterpart.
class GlobalChannelBlockStitchingPass : public BlockConversionPass {
 public:
  GlobalChannelBlockStitchingPass()
      : BlockConversionPass(
            "global_channel_block_stitching",
            "Stitch blocks based on procs with global channels.") {}
  ~GlobalChannelBlockStitchingPass() override = default;

  absl::StatusOr<bool> RunInternal(Package* package,
                                   const BlockConversionPassOptions& options,
                                   PassResults* results) const final;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_GLOBAL_CHANNEL_BLOCK_STITCHING_PASS_H_

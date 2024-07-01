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

#include "xls/codegen/block_stitching_pass.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

absl::StatusOr<bool> BlockStitchingPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  // No need to stitch blocks when we don't have 2+ blocks.
  if (unit->package->blocks().size() < 2) {
    return false;
  }
  return absl::UnimplementedError("Block stitching not implemented yet.");
}

}  // namespace xls::verilog

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

#include "xls/codegen/update_channel_metadata_pass.h"

#include "absl/status/statusor.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/status_macros.h"

namespace xls::verilog {

absl::StatusOr<bool> UpdateChannelMetadataPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  bool changed = false;

  for (auto& [fb, block] : unit->function_base_to_block()) {
    if (fb->IsProc()) {
      XLS_RETURN_IF_ERROR(UpdateChannelMetadata(
          unit->GetMetadataForBlock(block).streaming_io_and_pipeline, block));
    }
  }

  return changed;
}

}  // namespace xls::verilog

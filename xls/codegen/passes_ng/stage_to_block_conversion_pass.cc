// Copyright 2025 The XLS Authors
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

#include "xls/codegen/passes_ng/stage_to_block_conversion_pass.h"

#include <optional>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

absl::StatusOr<bool> StageToBlockConversionPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  VLOG(3) << "Converting stage ir to block ir:";
  XLS_VLOG_LINES(3, package->DumpIr());

  const CodegenOptions& codegen_options = options.codegen_options;
  StageConversionMetadata& stage_conversion_metadata =
      context.stage_conversion_metadata();
  BlockConversionMetadata& block_conversion_metadata =
      context.block_conversion_metadata();

  std::optional<FunctionBase*> top = package->GetTop();
  XLS_RET_CHECK(top.has_value());

  // TODO(tedhong): 2025-06-03 - Enable for procs.
  if (top.value()->IsProc()) {
    return false;
  }

  XLS_ASSIGN_OR_RETURN(
      ProcMetadata * top_metadata,
      stage_conversion_metadata.GetTopProcMetadata(top.value()));

  // Create the skeleton block hierarchy.
  XLS_ASSIGN_OR_RETURN(Block * top_block, CreateBlocksForProcHierarchy(
                                              codegen_options, *top_metadata,
                                              stage_conversion_metadata,
                                              block_conversion_metadata));
  context.SetTopBlock(top_block);

  XLS_RETURN_IF_ERROR(AddResetAndClockPortsToBlockHierarchy(
                          codegen_options, *top_metadata,
                          stage_conversion_metadata, block_conversion_metadata)
                          .status());

  // Fill in the blocks.
  VLOG(3) << "Converting stage ir to block ir - after set up:";
  XLS_VLOG_LINES(3, package->DumpIr());

  XLS_RETURN_IF_ERROR(ConvertProcHierarchyToBlocks(
                          codegen_options, *top_metadata,
                          stage_conversion_metadata, block_conversion_metadata)
                          .status());

  return true;
}

}  // namespace xls::verilog

// Copyright 2021 The XLS Authors
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

#include "xls/codegen/passes_ng/stage_to_block_conversion.h"

#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::verilog {

absl::StatusOr<Block*> CreateBlocksForProcHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata) {
  if (!top_metadata.IsTop()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConvertProcHierarchyToBlocks proc %s must be a "
                        "top-level proc created by stage conversion.",
                        top_metadata.proc()->name()));
  }

  Package* package = top_metadata.proc()->package();

  // TODO(tedhong): 2024-12-20 - Add support for nested pipelines.
  //
  // For now, the only children of a top-level proc should be stages.
  XLS_ASSIGN_OR_RETURN(std::vector<ProcMetadata*> stages,
                       stage_conversion_metadata.GetChildrenOf(&top_metadata));

  for (ProcMetadata* proc_metadata : stages) {
    Proc* proc = proc_metadata->proc();
    Block* block = proc->package()->AddBlock(
        std::make_unique<Block>(proc->name(), package));

    block_conversion_metadata.AssociateWithNewBlock(proc_metadata, block);

    XLS_VLOG_LINES(2, block->DumpIr());
  }

  // Create and stitch together top level block.
  Proc* top_proc = top_metadata.proc();
  Block* top_block = top_proc->package()->AddBlock(
      std::make_unique<Block>(top_metadata.proc()->name(), package));

  block_conversion_metadata.AssociateWithNewBlock(&top_metadata, top_block);

  return top_block;
}

absl::StatusOr<Block*> AddResetAndClockPortsToBlockHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata) {
  XLS_ASSIGN_OR_RETURN(std::vector<ProcMetadata*> stages,
                       stage_conversion_metadata.GetChildrenOf(&top_metadata));

  for (ProcMetadata* proc_metadata : stages) {
    XLS_ASSIGN_OR_RETURN(
        BlockMetadata * block_metadata,
        block_conversion_metadata.GetBlockMetadata(proc_metadata));

    XLS_RET_CHECK_OK(MaybeAddResetPort(block_metadata->block(), options));
    XLS_RET_CHECK_OK(MaybeAddClockPort(block_metadata->block(), options));
  }

  XLS_ASSIGN_OR_RETURN(
      BlockMetadata * top_block_metadata,
      block_conversion_metadata.GetBlockMetadata(&top_metadata));

  XLS_RET_CHECK_OK(MaybeAddResetPort(top_block_metadata->block(), options));
  XLS_RET_CHECK_OK(MaybeAddClockPort(top_block_metadata->block(), options));

  return top_block_metadata->block();
}

}  // namespace xls::verilog

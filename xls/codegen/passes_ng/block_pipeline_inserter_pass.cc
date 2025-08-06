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

#include "xls/codegen/passes_ng/block_pipeline_inserter_pass.h"

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/block_pipeline_inserter.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

absl::StatusOr<bool> BlockPipelineInserterPass::RunInternal(
    Package* absl_nonnull package, const CodegenPassOptions& options,
    PassResults* absl_nonnull results, CodegenContext& context) const {
  // TODO(tedhong): 2025-06-03 - Remove once procs are supported.
  if (!context.HasTopBlock()) {
    return false;
  }

  VLOG(3) << "Inserting pipeline registers to block ir:";
  XLS_VLOG_LINES(3, package->DumpIr());

  const CodegenOptions& codegen_options = options.codegen_options;
  BlockConversionMetadata& block_conversion_metadata =
      context.block_conversion_metadata();

  XLS_RET_CHECK(context.HasTopBlock());
  XLS_ASSIGN_OR_RETURN(
      BlockMetadata * block_metadata,
      block_conversion_metadata.GetBlockMetadata(context.top_block()));

  XLS_RETURN_IF_ERROR(
      InsertPipelineIntoBlock(codegen_options, *block_metadata).status());

  return true;
}

}  // namespace xls::verilog

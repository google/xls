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

#ifndef XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_H_
#define XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"

namespace xls::verilog {
// Creates new blocks for each proc in the proc hierarchy under the given
// proc associated with top_metadata.  The proc hierarchy is assumed to be
// a proc hierarchy previously created with Stage Conversion.
absl::StatusOr<Block*> CreateBlocksForProcHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata);

// Depending on codegen options, add in reset and clock ports to the block
// hierarchy.
absl::StatusOr<Block*> AddResetAndClockPortsToBlockHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata);

// Converts the proc hierarchy beneath the given top-level proc to blocks owned
// by the same package as the proc.
//
// The top-level proc must be a proc created by stage conversion.
//
// Metadata associated with the procs should be present in top_metadata and
// stage_conversion_metadata, and metadata associated with the newly created
// blocks will be created in block_conversion_metadata.
absl::StatusOr<Block*> ConvertProcHierarchyToBlocks(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_H_

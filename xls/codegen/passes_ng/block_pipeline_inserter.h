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

#ifndef XLS_CODEGEN_PASSES_NG_BLOCK_PIPELINE_INSERTER_H_
#define XLS_CODEGEN_PASSES_NG_BLOCK_PIPELINE_INSERTER_H_

#include "absl/status/statusor.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/codegen/codegen_options.h"
#include "xls/ir/block.h"

namespace xls::verilog {

// Inserts pipeline flops/fifos in between stages of a block.
//
// The top-level proc must be a proc created by stage conversion.
// The top-level block must be a block created by stage to block conversion.
absl::StatusOr<Block*> InsertPipelineIntoBlock(
    const CodegenOptions& options, BlockMetadata& top_block_metadata);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_BLOCK_PIPELINE_INSERTER_H_

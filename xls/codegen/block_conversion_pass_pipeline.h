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

#ifndef XLS_CODEGEN_BLOCK_CONVERSION_PASS_PIPELINE_H_
#define XLS_CODEGEN_BLOCK_CONVERSION_PASS_PIPELINE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Creates a CodegenPassUnit for the block conversion pass pipeline.
//
// Also, initializes the package with blocks for eventual block conversion.
// TODO(tedhong): 2024-11-22 - Make this a separate pass.
//
absl::StatusOr<CodegenPassUnit> CreateBlocksFor(
    const PackagePipelineSchedules& schedules, const CodegenOptions& options,
    Package* package);

// Returns the block conversion pass pipeline which runs on a package and
// lowers the IR to Block IR in prepration for eventual lowering to Verilog.
//
// After BlockConversion further Block passes in Codegen Pass are needed
// before BlockGeneration can lower the Block IR to Verilog.
std::unique_ptr<CodegenCompoundPass> CreateBlockConversionPassPipeline(
    const CodegenOptions& options);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_BLOCK_CONVERSION_PASS_PIPELINE_H_

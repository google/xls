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

#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/passes/optimization_pass.h"

namespace xls::verilog {

// Returns the block conversion pass pipeline which runs on a package and
// lowers the IR to Block IR in preparation for eventual lowering to Verilog.
//
// After BlockConversion further Block passes in Codegen Pass are needed
// before BlockGeneration can lower the Block IR to Verilog.
std::unique_ptr<CodegenCompoundPass> CreateBlockConversionPassPipeline(
    const CodegenOptions& options, OptimizationContext& context);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_BLOCK_CONVERSION_PASS_PIPELINE_H_

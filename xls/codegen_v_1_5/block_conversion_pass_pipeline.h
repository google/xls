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

#ifndef XLS_CODEGEN_CODEGEN_V_1_5_BLOCK_CONVERSION_PASS_PIPELINE_H_
#define XLS_CODEGEN_CODEGEN_V_1_5_BLOCK_CONVERSION_PASS_PIPELINE_H_

#include <memory>

#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/passes/optimization_pass.h"

namespace xls::codegen {

// Returns a pipeline which converts an unscheduled IR package into a standard
// block.
std::unique_ptr<BlockConversionCompoundPass> CreateBlockConversionPassPipeline(
    OptimizationContext& opt_context);

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_CODEGEN_V_1_5_BLOCK_CONVERSION_PASS_PIPELINE_H_

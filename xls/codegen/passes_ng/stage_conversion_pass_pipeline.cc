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

// Create a standard pipeline of passes. This pipeline should
// be used in the main driver as well as in testing.

#include "xls/codegen/passes_ng/stage_conversion_pass_pipeline.h"

#include <memory>

#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/stage_conversion_pass.h"
#include "xls/passes/optimization_pass.h"

namespace xls::verilog {

std::unique_ptr<CodegenCompoundPass> CreateStageConversionPassPipeline(
    const CodegenOptions& options, OptimizationContext& context) {
  auto top = std::make_unique<CodegenCompoundPass>(
      "codegen_stage_conversion",
      "Top level codegen IR to Stage IR pass pipeline");

  top->Add<StageConversionPass>();

  return top;
}

}  // namespace xls::verilog

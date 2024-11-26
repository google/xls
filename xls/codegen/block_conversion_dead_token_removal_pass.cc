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

#include "xls/codegen/block_conversion_dead_token_removal_pass.h"

#include <memory>

#include "xls/codegen/codegen_checker.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_wrapper_pass.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"

namespace xls::verilog {

BlockConversionDeadTokenRemovalPass::BlockConversionDeadTokenRemovalPass()
    : CodegenCompoundPass(
          BlockConversionDeadTokenRemovalPass::kName,
          "Dead token removal during block-conversion process") {
  AddInvariantChecker<CodegenChecker>();
  Add<CodegenWrapperPass>(std::make_unique<DataflowSimplificationPass>());
  Add<CodegenWrapperPass>(std::make_unique<DeadCodeEliminationPass>());
  Add<RegisterLegalizationPass>();
  Add<CodegenWrapperPass>(std::make_unique<DeadCodeEliminationPass>());
}

}  // namespace xls::verilog

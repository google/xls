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

// Create a standard pipeline of passes. This pipeline should
// be used in the main driver as well as in testing.

#include "xls/codegen/block_conversion_pass_pipeline.h"

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xls/codegen/block_conversion_dead_token_removal_pass.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/convert_ir_to_blocks_passes.h"
#include "xls/codegen/mark_channel_fifos_pass.h"
#include "xls/codegen/passes_ng/block_pipeline_inserter_pass.h"
#include "xls/codegen/passes_ng/flow_control_simplification_pass.h"
#include "xls/codegen/passes_ng/metadata_conversion_pass.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_pass.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

// Creates blocks for funcs/procs that are not handled by the NG passes.
//
// Currently NG passes can only handle functions, so procs go through a
// a different set of passes.
//
// This pass creates the top block and the provenance to the source proc.
// `ConvertFuncsToPipelinedBlocksPass` and `ConvertProcsToPipelinedBlocksPass`
// will key off the proventance and fill in the rest of the block details.
class NonNGBlockCreationPass : public CodegenPass {
 public:
  NonNGBlockCreationPass()
      : CodegenPass("non_ng_block_creation", "Non-NG Block creation pass") {}
  ~NonNGBlockCreationPass() override = default;

  absl::StatusOr<bool> RunInternal(Package* package,
                                   const CodegenPassOptions& options,
                                   PassResults* results,
                                   CodegenContext& context) const override {
    // Skip if the NG passes were used (and hence the top block has been
    // created).
    if (context.HasTopBlock()) {
      return false;
    }

    // Create the top block first.
    XLS_RET_CHECK(package->GetTop().has_value());
    FunctionBase* top = *package->GetTop();

    const CodegenOptions& codegen_options = options.codegen_options;

    std::string module_name(SanitizeVerilogIdentifier(
        codegen_options.module_name().value_or(top->name())));

    Block* top_block =
        package->AddBlock(std::make_unique<Block>(module_name, package));
    context.SetTopBlock(top_block);

    // We use a uniquer here because the top block name comes from the codegen
    // option's `module_name` field (if set). A non-top proc could have the same
    // name, so the name uniquer will ensure that the sub-block gets a suffix if
    // needed. Note that the NameUniquer's sanitize performs a different
    // function from `SanitizeVerilogIdentifier()`, which is used to ensure that
    // identifiers are OK for RTL.
    NameUniquer block_name_uniquer("__");
    XLS_RET_CHECK_EQ(block_name_uniquer.GetSanitizedUniqueName(module_name),
                     module_name);

    // Create all sub-blocks and make associations between the function,
    // schedule and block.
    for (auto& [fb, schedule] : context.function_base_to_schedule()) {
      std::string sub_block_name =
          (fb == top) ? module_name
                      : block_name_uniquer.GetSanitizedUniqueName(
                            SanitizeVerilogIdentifier(fb->name()));
      Block* block = (fb == top) ? top_block
                                 : package->AddBlock(std::make_unique<Block>(
                                       sub_block_name, package));

      block->SetFunctionBaseProvenance(fb);
    }

    context.GcMetadata();

    return true;
  }
};

std::unique_ptr<CodegenCompoundPass> CreateBlockConversionPassPipeline(
    const CodegenOptions& options, OptimizationContext& context) {
  auto top = std::make_unique<CodegenCompoundPass>(
      "codegen", "Top level codegen IR to Block pass pipeline");

  if (options.generate_combinational()) {
    top->Add<NonNGBlockCreationPass>();
    top->Add<ConvertFuncsToCombinationalBlocksPass>();
    top->Add<ConvertProcsToCombinationalBlocksPass>();
  } else {
    // New NG Passes
    top->Add<StageToBlockConversionPass>();
    top->Add<BlockPipelineInserterPass>();
    top->Add<FlowControlSimplificationPass>();
    top->Add<MetadataConversionPass>();

    // Passes to handle funcs/procs not handles by the NG passes.
    //
    // These passes do not touch blocks created by the NG passes, because
    // the blocks created by the NG passes do not set the provenance.
    //
    // TODO(tedhong): 2025-06-03 - Remove these passes once the NG passes
    // can handle all cases.
    top->Add<NonNGBlockCreationPass>();
    top->Add<MarkChannelFifosPass>();
    top->Add<ConvertFuncsToPipelinedBlocksPass>();
    top->Add<ConvertProcsToPipelinedBlocksPass>();
  }

  top->Add<BlockConversionDeadTokenRemovalPass>(context);

  return top;
}

}  // namespace xls::verilog

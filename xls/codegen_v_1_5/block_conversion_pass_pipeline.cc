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

#include "xls/codegen_v_1_5/block_conversion_pass_pipeline.h"

#include <memory>

#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_wrapper_pass.h"
#include "xls/codegen_v_1_5/block_finalization_pass.h"
#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"
#include "xls/codegen_v_1_5/flow_control_insertion_pass.h"
#include "xls/codegen_v_1_5/function_io_lowering_pass.h"
#include "xls/codegen_v_1_5/global_channel_block_stitching_pass.h"
#include "xls/codegen_v_1_5/idle_insertion_pass.h"
#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"
#include "xls/codegen_v_1_5/proc_instantiation_lowering_pass.h"
#include "xls/codegen_v_1_5/register_cleanup_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/codegen_v_1_5/side_effect_condition_pass.h"
#include "xls/codegen_v_1_5/signature_generation_pass.h"
#include "xls/codegen_v_1_5/state_to_register_io_lowering_pass.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/optimization_pass.h"

namespace xls::codegen {

std::unique_ptr<BlockConversionCompoundPass> CreateBlockConversionPassPipeline(
    OptimizationContext& opt_context) {
  auto top = std::make_unique<BlockConversionCompoundPass>(
      "block_conversion", "Top level codegen v1.5 block conversion pipeline");

  // Convert Procs and Functions ScheduledProc/ScheduledFunction.
  top->Add<SchedulingPass>();

  // Convert ScheduledProc/ScheduledFunction to ScheduledBlock.
  top->Add<ScheduledBlockConversionPass>();

  // Update assert conditions to be guarded by stage_done signals.
  top->Add<SideEffectConditionPass>();

  // Add pipeline registers for data flowing between stages.
  top->Add<PipelineRegisterInsertionPass>();

  // Lower state reads/writes to register read/writes.
  top->Add<StateToRegisterIoLoweringPass>();

  // Lower channel I/O to ports.
  top->Add<ChannelToPortIoLoweringPass>();

  // Lower params and return values to ports.
  top->Add<FunctionIOLoweringPass>();

  // Insert flow control between stages.
  top->Add<FlowControlInsertionPass>();

  // Add idle signal output if requested.
  top->Add<IdleInsertionPass>();

  // Lower proc instantiations to block instantiations.
  top->Add<ProcInstantiationLoweringPass>();

  // Stitch blocks that were lowered from procs using global channels.
  top->Add<GlobalChannelBlockStitchingPass>();

  // Add module signatures.
  top->Add<SignatureGenerationPass>();

  // Lower scheduled block to standard block, inlining each stage.
  top->Add<BlockFinalizationPass>();

  // Clean up unused registers & load-enable bits (including flow-control
  // registers).
  top->Add<RegisterCleanupPass>();

  // Clean up identity placeholders and unnecessary array/tuple manipulation.
  top->Add<BlockConversionWrapperPass>(std::make_unique<IdentityRemovalPass>(),
                                       opt_context);
  top->Add<BlockConversionWrapperPass>(
      std::make_unique<DataflowSimplificationPass>(), opt_context);

  // Remove anything we created & then left dead.
  top->Add<BlockConversionWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>(), opt_context);

  return top;
}

}  // namespace xls::codegen

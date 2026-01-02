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
#include "xls/codegen_v_1_5/idle_insertion_pass.h"
#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"
#include "xls/codegen_v_1_5/register_cleanup_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/codegen_v_1_5/state_to_register_io_lowering_pass.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
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

  // Lower state reads/writes to register read/writes.
  top->Add<StateToRegisterIoLoweringPass>();

  // Lower channel I/O to ports.
  top->Add<ChannelToPortIoLoweringPass>();

  // Lower params and return values to ports.
  top->Add<FunctionIOLoweringPass>();

  // Insert flow control between stages.
  top->Add<FlowControlInsertionPass>();

  // Add pipeline registers for data flowing between stages.
  top->Add<PipelineRegisterInsertionPass>();

  // Add idle signal output if requested.
  top->Add<IdleInsertionPass>();

  // Lower scheduled block to standard block, inlining each stage.
  top->Add<BlockFinalizationPass>();

  // Clean up unused registers & load-enable bits (including flow-control
  // registers).
  top->Add<RegisterCleanupPass>();

  // Clean up unnecessary array/tuple manipulation.
  top->Add<BlockConversionWrapperPass>(
      std::make_unique<DataflowSimplificationPass>(), opt_context);

  // Remove anything we created & then left dead.
  top->Add<BlockConversionWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>(), opt_context);

  return top;
}

}  // namespace xls::codegen

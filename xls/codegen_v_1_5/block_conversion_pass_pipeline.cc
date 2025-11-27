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
#include "xls/codegen_v_1_5/block_finalization_pass.h"
#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"
#include "xls/codegen_v_1_5/flow_control_insertion_pass.h"
#include "xls/codegen_v_1_5/param_and_return_value_lowering_pass.h"
#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/codegen_v_1_5/state_to_register_io_lowering_pass.h"

namespace xls::codegen {

std::unique_ptr<BlockConversionCompoundPass>
CreateBlockConversionPassPipeline() {
  auto top = std::make_unique<BlockConversionCompoundPass>(
      "block_conversion", "Top level codegen v1.5 block conversion pipeline");

  // Convert IR to scheduled IR.
  top->Add<SchedulingPass>();

  // Lower state reads/writes to register read/writes.
  top->Add<StateToRegisterIoLoweringPass>();

  // Lower channel I/O to ports.
  top->Add<ChannelToPortIoLoweringPass>();

  // Lower params and return values to ports.
  top->Add<ParamAndReturnValueLoweringPass>();

  // Insert flow control between stages.
  top->Add<FlowControlInsertionPass>();

  // Add pipeline registers for data flowing between stages.
  top->Add<PipelineRegisterInsertionPass>();

  // Lower scheduled block to standard block, inlining each stage.
  top->Add<BlockFinalizationPass>();

  return top;
}

}  // namespace xls::codegen

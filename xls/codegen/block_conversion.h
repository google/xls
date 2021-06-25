// Copyright 2021 The XLS Authors
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

#ifndef XLS_CODEGEN_BLOCK_CONVERSION_H_
#define XLS_CODEGEN_BLOCK_CONVERSION_H_

#include "absl/status/statusor.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Converts a function into a block of the given name (the function name is
// ignored). Function arguments become input ports, function return value
// becomes an output port. Returns a pointer to the block.
absl::StatusOr<Block*> FunctionToBlock(Function* f,
                                       absl::string_view block_name);

// Converts a function in a pipelined (stateless) block. The pipeline is
// constructed using the given schedule. Registers are inserted between each
// stage. Inputs and outputs are not flopped.
absl::StatusOr<Block*> FunctionToPipelinedBlock(
    const PipelineSchedule& schedule, Function* f,
    absl::string_view block_name);

// Converts the given proc to a combinational block. Proc must be stateless
// (state type is an empty tuple). Streaming channels must have ready-valid flow
// control (FlowControl::kReadyValid). Receives/sends of these streaming
// channels become input/output ports with additional ready/valid ports for flow
// control. Receives/sends of single-value channels become input/output ports in
// the returned block.
absl::StatusOr<Block*> ProcToCombinationalBlock(Proc* proc,
                                                absl::string_view block_name);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_BLOCK_CONVERSION_H_

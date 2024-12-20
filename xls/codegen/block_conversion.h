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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// If specified by user, returns the PackageInterFaceProto::Function for the
// function with the given name.
std::optional<PackageInterfaceProto::Function> FindFunctionInterface(
    const std::optional<PackageInterfaceProto>& src,
    std::string_view func_name);

// Converts a function or proc to a pipelined block. The pipeline is constructed
// using the given schedule. Registers are inserted between each stage.
// If `f` is already a Block, an error is returned.
absl::StatusOr<CodegenPassUnit> FunctionBaseToPipelinedBlock(
    const PipelineSchedule& schedule, const CodegenOptions& options,
    FunctionBase* f);

// Converts every scheduled function/proc in `package` to pipelined blocks.
absl::StatusOr<CodegenPassUnit> PackageToPipelinedBlocks(
    const PackagePipelineSchedules& schedule, const CodegenOptions& options,
    Package* package);

// Converts a function into a pipelined block.
//
// Parameters:
//  input - schedule: Schedule for the to-be-created block.
//  input - options:  Codegen options.
//  inout - unit:     Metadata for codegen passes.
//  input - function: Function to convert to a pipelined block.
//  inout - block:    Destination block, should be empty.
absl::Status SingleFunctionToPipelinedBlock(const PipelineSchedule& schedule,
                                            const CodegenOptions& options,
                                            CodegenPassUnit& unit, Function* f,
                                            absl::Nonnull<Block*> block);

// Converts a function into a combinational block. Function arguments become
// input ports, function return value becomes an output port. Returns a pointer
// to the block.
absl::StatusOr<CodegenPassUnit> FunctionToCombinationalBlock(
    Function* f, const CodegenOptions& options);

// Converts the given proc to a combinational block. Proc must be stateless
// (state type is an empty tuple). Streaming channels must have ready-valid flow
// control (FlowControl::kReadyValid). Receives/sends of these streaming
// channels become input/output ports with additional ready/valid ports for flow
// control. Receives/sends of single-value channels become input/output ports in
// the returned block.
absl::StatusOr<CodegenPassUnit> ProcToCombinationalBlock(
    Proc* proc, const CodegenOptions& options);

// Converts the given function or proc to a combinational block. See
// FunctionToCombinationalBlock() or ProcToCombinationalBlock() for more info.
absl::StatusOr<CodegenPassUnit> FunctionBaseToCombinationalBlock(
    FunctionBase* f, const CodegenOptions& options);

// Adds a register between the node and all its downstream users.
// Returns the new register added.
absl::StatusOr<RegisterRead*> AddRegisterAfterNode(
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior,
    std::optional<Node*> load_enable, Node* node, Block* block);

// Add a zero-latency buffer after a set of data/valid/ready signal.
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
// Updates valid_nodes with the additional nodes associated with valid
// registers.
absl::StatusOr<Node*> AddZeroLatencyBufferToRDVNodes(
    Node* from_data, Node* from_valid, Node* from_rdy,
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes);

// Clones every node in the given proc into the given block. Some nodes are
// handled specially.  See CloneNodesIntoBlockHandler for details.
absl::StatusOr<StreamingIOPipeline> CloneProcNodesIntoBlock(
    Proc* proc, const CodegenOptions& options, Block* block);

absl::StatusOr<
    std::tuple<StreamingIOPipeline, std::optional<ConcurrentStageGroups>>>
CloneNodesIntoPipelinedBlock(const PipelineSchedule& schedule,
                             const CodegenOptions& options, Block* block);

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
absl::Status AddCombinationalFlowControl(
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    std::vector<std::optional<Node*>>& stage_valid,
    const CodegenOptions& options, Block* block);

// Update io channel metadata with latest information from block conversion.
absl::Status UpdateChannelMetadata(const StreamingIOPipeline& io, Block* block);

std::string PipelineSignalName(std::string_view root, int64_t stage);
absl::StatusOr<std::string> StreamingIOName(Node* node);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_BLOCK_CONVERSION_H_

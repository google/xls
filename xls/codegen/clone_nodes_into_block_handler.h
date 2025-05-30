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

#ifndef XLS_CODEGEN_CLONE_NODES_INTO_BLOCK_HANDLER_H_
#define XLS_CODEGEN_CLONE_NODES_INTO_BLOCK_HANDLER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {
// Clones every node in the given func/proc into the given block. Some nodes are
// handled specially:
//
// * Proc token parameter becomes an operandless AfterAll operation in the
//   block.
// * Proc state parameter (which must be an empty tuple) becomes a Literal
//   operation in the block.
// * Receive operations become InputPorts.
// * Send operations become OutputPorts.
// * Function parameters become InputPorts.
// * The Function return value becomes an OutputPort.
//
// GetResult() returns a StreamingIOPipeline which
//   1. Contains the InputPorts and OutputPorts created from
//      Send/Receive operations of streaming channels
//   2. Contains a list of PipelineRegisters per stage of the pipeline.
//
// Example input proc:
//
//   chan x_ch(bits[32], kind=streaming, flow_control=single_value, id=0, ...)
//   chan y_ch(bits[32], kind=streaming, flow_control=single_value, id=1, ...)
//
//   proc foo(tkn: token, st: (), init=42) {
//     rcv_x: (token, bits[32]) = receive(tkn, channel=x_ch)
//     rcv_x_token: token = tuple_index(rcv_x, index=0)
//     x: bits[32] = tuple_index(rcv_x, index=1)
//     not_x: bits[32] = not(x)
//     snd_y: token = send(rcv_x_token, not_x, channel=y_ch)
//     next (tkn, snd_y)
//   }
//
// Resulting block:
//
//  block (x: bits[32], y: bits[32]) {
//    x: bits[32] = input_port(name=x)
//    not_x: bits[32] = not(x)
//    y: bits[32] = output_port(not_x, name=x)
//  }
//
// Ready/valid flow control including inputs ports and output ports are added
// later.
class CloneNodesIntoBlockHandler {
 public:
  // Initialize this object with the proc/function, the block the
  // proc/function should be cloned into, and the stage_count.
  //
  // If the block is to be a combinational block, stage_count should be
  // set to 0;
  CloneNodesIntoBlockHandler(
      FunctionBase* proc_or_function, int64_t stage_count,
      const CodegenOptions& options, Block* block,
      std::optional<const PackageSchedule*> schedule = std::nullopt,
      std::optional<const ProcElaboration*> elab = std::nullopt);

  // Add ports to the block corresponding to the channels on the interface of
  // the proc. Should only be called for procs.
  absl::Status AddChannelPortsAndFifoInstantiations();

  // Literals are special because they exist outside of the pipeline/schedule.
  absl::Status AddLiterals();

  // Adds the block instantiations corresponding to the proc instantiations
  // within the proc being converted.  Should only be called for procs. This is
  // a noop if channels are not proc-scoped.
  absl::Status AddBlockInstantiations(
      const absl::flat_hash_map<FunctionBase*, Block*>& converted_blocks);

  // Add channel port metadata to the block.
  absl::Status AddBlockMetadata();

  // For a given set of sorted nodes, process and clone them into the
  // block.
  absl::Status CloneNodes(absl::Span<Node* const> sorted_nodes, int64_t stage);

  // Add pipeline registers. A register is needed for each node which is
  // scheduled at or before this cycle and has a use after this cycle.
  absl::Status AddNextPipelineStage(int64_t stage);

  // If a function, create an output port for the function's return.
  absl::Status AddOutputPortsIfFunction(std::string_view output_port_name);

  // Figure out based on the reads and writes to state variables what stages are
  // mutually exclusive with one another.
  absl::Status MarkMutualExclusiveStages(int64_t stage_count);

  // Return structure describing streaming io ports and pipeline registers.
  StreamingIOPipeline GetResult() { return result_; }

  std::optional<ConcurrentStageGroups> GetConcurrentStages() {
    return concurrent_stages_;
  }

  // Abstraction representing the signals of a channel interface (ports or fifo
  // instantiation input/outputs).
  struct ChannelConnection {
   public:
    ChannelRef channel;
    ChannelDirection direction;
    ConnectionKind kind;
    std::optional<ChannelNode*> channel_node;
    Node* data;
    std::optional<Node*> valid;
    std::optional<Node*> ready;

    std::optional<ChannelInterface*> instantiated_channel_interface;
    std::optional<Block*> instantiated_block;

    // Replaces the value driving the data/ready/valid port with the given
    // node.
    absl::Status ReplaceDataSignal(Node* value) const;
    absl::Status ReplaceValidSignal(Node* value) const;
    absl::Status ReplaceReadySignal(Node* value) const;
  };

 private:
  // Don't clone state read operations. Instead replace with a RegisterRead
  // operation.
  absl::StatusOr<Node*> HandleStateRead(Node* node, Stage stage);

  // Replace function parameters with input ports.
  absl::StatusOr<Node*> HandleFunctionParam(Node* node);

  // Replace next values with a RegisterWrite.
  absl::Status HandleNextValue(Node* node, Stage stage);

  // Don't clone Receive operations. Instead replace with a tuple
  // containing the Receive's token operand and an InputPort operation.
  //
  // Both data and valid ports are created in this function.  See
  // MakeInputValidPortsForInputChannels() for additional handling of
  // the valid signal.
  //
  // In the case of handling non-blocking receives, the logic to adapt
  // data to a tuple of (data, valid) is added here.
  absl::StatusOr<Node*> HandleReceiveNode(Receive* receive, int64_t stage,
                                          const ChannelConnection& connection);

  // Don't clone Send operations. Instead replace with an OutputPort
  // operation in the block.
  absl::StatusOr<Node*> HandleSendNode(Send* send, int64_t stage,
                                       const ChannelConnection& connection);

  // Clone the operation from the source to the block as is.
  absl::StatusOr<Node*> HandleGeneralNode(Node* node);

  // Create a pipeline register for the given node.
  //
  // Returns a PipelineRegister whose reg_read field can be used
  // to chain dependent ops to.
  absl::StatusOr<PipelineRegister> CreatePipelineRegister(std::string_view name,
                                                          Node* node,
                                                          Stage stage_write);

  // Creates pipeline registers for a given node.
  //
  // Depending on the type of node, multiple pipeline registers
  // may be created.
  //  1. Each pipeline register added will be added to pipeline_register_list
  //     which is passed in by reference.
  //  2. Logic may be inserted after said registers  so that a single node with
  //     the same type as the input node is returned.
  //
  absl::StatusOr<Node*> CreatePipelineRegistersForNode(
      std::string_view base_name, Node* node, Stage stage,
      std::vector<PipelineRegister>& pipeline_registers_list);

  Block* block() const { return block_; };

  // Returns the schedule for the FunctionBase being lowered.
  const PipelineSchedule& GetSchedule() const {
    return schedule_.value()->GetSchedule(function_base_);
  }

  bool is_proc_;
  FunctionBase* function_base_;

  const CodegenOptions& options_;

  Block* block_;
  std::optional<const PackageSchedule*> schedule_;
  std::optional<const ProcElaboration*> elab_;
  std::optional<ConcurrentStageGroups> concurrent_stages_;
  StreamingIOPipeline result_;
  absl::flat_hash_map<Node*, Node*> node_map_;
  absl::flat_hash_map<std::pair<std::string, ChannelDirection>,
                      ChannelConnection>
      channel_connections_;
};
}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CLONE_NODES_INTO_BLOCK_HANDLER_H_

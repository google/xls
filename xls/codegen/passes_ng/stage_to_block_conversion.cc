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

#include "xls/codegen/passes_ng/stage_to_block_conversion.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/passes_ng/block_channel_adapter.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls::verilog {
namespace {

// Structure to bundle up the names for a RDV Node
struct RDVNodeGroupName {
  std::string ready;
  std::string data;
  std::string valid;
};

RDVNodeGroupName CreatePortNamesForChannel(
    const CodegenOptions& options, const ChannelInterface* chan_interface) {
  // Retrieve the suffix for the ports associated with the channel.
  std::string_view ready_suffix = options.streaming_channel_ready_suffix();
  std::string_view data_suffix =
      (chan_interface->kind() == ChannelKind::kStreaming)
          ? options.streaming_channel_data_suffix()
          : "";
  std::string_view valid_suffix = options.streaming_channel_valid_suffix();

  // Construct names for the ports.
  std::string port_ready_name =
      absl::StrCat(chan_interface->name(), ready_suffix);
  std::string port_data_name =
      absl::StrCat(chan_interface->name(), data_suffix);
  std::string port_valid_name =
      absl::StrCat(chan_interface->name(), valid_suffix);

  return RDVNodeGroupName{.ready = port_ready_name,
                          .data = port_data_name,
                          .valid = port_valid_name};
}

class StageToBlockCloner {
 public:
  // Initializes this object with the metadata for the stage to be cloned,
  // and the metadata for the block to be populated.
  StageToBlockCloner(const CodegenOptions& options,
                     const ProcMetadata& proc_metadata,
                     BlockMetadata& block_metadata)
      : options_(options),
        proc_metadata_(proc_metadata),
        block_metadata_(block_metadata) {}

  // For a given set of sorted nodes, process and clone them into the
  // block.  Also perform basic hookup of valid and ready signals.
  //
  // Channel op nodes are delegated to HandlReceive() and HandleSend().
  absl::Status RunForStage() {
    Proc* proc = proc_metadata_.proc();

    XLS_RETURN_IF_ERROR(CreateInterfaceChannelRefsForBlock());

    for (const Node* node : TopoSort(proc)) {
      Node* block_node = nullptr;

      if (node->Is<ChannelNode>()) {
        if (node->Is<Receive>()) {
          XLS_ASSIGN_OR_RETURN(block_node, HandleReceiveNode(node));
        } else {
          XLS_ASSIGN_OR_RETURN(block_node, HandleSendNode(node));
        }
      } else {
        XLS_ASSIGN_OR_RETURN(block_node, HandleGeneralNode(node));
      }

      proc_ir_to_block_ir_node_map_[node] = block_node;
    }

    XLS_RETURN_IF_ERROR(WireValidSignals());
    XLS_RETURN_IF_ERROR(WireReadySignals());

    return absl::OkStatus();
  }

 private:
  absl::StatusOr<BlockRDVSlot> CreateSendPortsForChannelRef(
      const SendChannelInterface* chan_interface, Block* block) {
    RDVNodeGroupName port_names =
        CreatePortNamesForChannel(options_, chan_interface);

    // Create the ports.
    // For the data port, attach to a temporary literal 0.
    XLS_ASSIGN_OR_RETURN(Node * literal_data_0,
                         block->MakeNode<xls::Literal>(
                             SourceInfo(), ZeroOfType(chan_interface->type())));

    // For the ready port, attach to a temporary literal 1.
    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));

    XLS_ASSIGN_OR_RETURN(OutputPort * output_data_port,
                         block->AddOutputPort(port_names.data, literal_data_0));
    XLS_ASSIGN_OR_RETURN(OutputPort * output_valid_port,
                         block->AddOutputPort(port_names.valid, literal_1));

    XLS_ASSIGN_OR_RETURN(InputPort * input_ready_port,
                         block->AddInputPort(port_names.ready,
                                             block->package()->GetBitsType(1)));

    return BlockRDVSlot::CreateSendSlot(
        "send",
        RDVNodeGroup{input_ready_port, output_data_port, output_valid_port},
        block);
  }

  absl::StatusOr<BlockRDVSlot> CreateReceivePortsForChannelRef(
      const ReceiveChannelInterface* chan_interface, Block* block) {
    RDVNodeGroupName port_names =
        CreatePortNamesForChannel(options_, chan_interface);

    // Create the ports. For the ready port and attach to a temporary literal 1.
    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_ASSIGN_OR_RETURN(
        InputPort * input_data_port,
        block->AddInputPort(port_names.data, chan_interface->type()));
    XLS_ASSIGN_OR_RETURN(InputPort * input_valid_port,
                         block->AddInputPort(port_names.valid,
                                             block->package()->GetBitsType(1)));
    XLS_ASSIGN_OR_RETURN(OutputPort * output_ready_port,
                         block->AddOutputPort(port_names.ready, literal_1));

    // Add in a slot for the channel
    return BlockRDVSlot::CreateReceiveSlot(
        "receive",
        RDVNodeGroup{output_ready_port, input_data_port, input_valid_port},
        block);
  }

  absl::Status CreateInterfaceChannelRefsForBlock() {
    Block* block = block_metadata_.block();
    Proc* proc = proc_metadata_.proc();

    for (const ChannelInterface* chan_interface : proc->interface()) {
      if (chan_interface->direction() == ChannelDirection::kSend) {
        auto send_chan_interface =
            down_cast<const SendChannelInterface*>(chan_interface);

        XLS_ASSIGN_OR_RETURN(
            BlockRDVSlot slot,
            CreateSendPortsForChannelRef(send_chan_interface, block));

        block_metadata_.AddChannelMetadata(
            BlockChannelMetadata(send_chan_interface).AddSlot(std::move(slot)));

      } else {
        auto receive_chan_interface =
            down_cast<const ReceiveChannelInterface*>(chan_interface);

        XLS_ASSIGN_OR_RETURN(
            BlockRDVSlot slot,
            CreateReceivePortsForChannelRef(receive_chan_interface, block));

        block_metadata_.AddChannelMetadata(
            BlockChannelMetadata(receive_chan_interface)
                .AddSlot(std::move(slot)));
      }
    }

    return absl::OkStatus();
  }

  // Wires the valid signals for the block.
  //
  // And all input channel valids and attach to all output channel valids.
  absl::Status WireValidSignals() {
    Block* block = block_metadata_.block();

    std::vector<Node*> active_valids;

    // Assign all_active_inputs_valid to 1 in case there are no inputs.
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_inputs_valid,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));

    for (std::unique_ptr<BlockChannelMetadata>& input :
         block_metadata_.inputs()) {
      XLS_RET_CHECK(input->adapter().has_value());
      RDVAdapter& adapter = input->adapter().value();

      if (adapter.channel_predicate() != nullptr) {
        // Logic for the active valid signal for a Receive operation with a
        // predicate `pred`.
        //
        //   active = !pred | pred && valid
        //          = !pred | valid
        XLS_ASSIGN_OR_RETURN(
            Node * not_pred,
            block->MakeNode<UnOp>(SourceInfo(), adapter.channel_predicate(),
                                  Op::kNot));

        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        if (!adapter.channel_predicate()->HasAssignedName()) {
          not_pred->SetName(absl::StrFormat(
              "%s_not_pred", input->channel_interface()->name()));
        }

        std::vector<Node*> operands = {not_pred, adapter.valid()};
        XLS_ASSIGN_OR_RETURN(
            Node * active_valid,
            block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));

        // not_pred will have an assigned name or be inlined, so only check
        // the ready port. If it has an assigned name, just let everything
        // inline. Otherwise, give a descriptive name.
        if (!adapter.valid()->HasAssignedName()) {
          active_valid->SetName(absl::StrFormat(
              "%s_active_valid", input->channel_interface()->name()));
        }

        active_valids.push_back(active_valid);
      } else {
        // No predicate is the same as pred = true, so
        // active = !pred | valid = !true | valid = false | valid = valid
        active_valids.push_back(adapter.valid());
      }
    }

    XLS_ASSIGN_OR_RETURN(
        all_active_inputs_valid,
        block->MakeNode<NaryOp>(SourceInfo(), active_valids, Op::kAnd));

    for (std::unique_ptr<BlockChannelMetadata>& output :
         block_metadata_.outputs()) {
      std::vector<Node*> output_valid_operands = {all_active_inputs_valid};

      XLS_RET_CHECK(output->adapter().has_value());
      RDVAdapter& adapter = output->adapter().value();

      if (adapter.channel_predicate() != nullptr) {
        output_valid_operands.push_back(adapter.channel_predicate());
      }

      XLS_ASSIGN_OR_RETURN(Node * output_valid,
                           block->MakeNode<NaryOp>(
                               SourceInfo(), output_valid_operands, Op::kAnd));

      // Replace existing output with the new valid signal.
      XLS_RETURN_IF_ERROR(adapter.SetValid(output_valid));
    }

    return absl::OkStatus();
  }

  // Wires the ready signals for the block.
  //
  // And all output channel ready signals and attach to all input channel
  // readys.
  absl::Status WireReadySignals() {
    Block* block = block_metadata_.block();

    std::vector<Node*> active_readys;

    // Assign all_active_readys_valid to 1 in case there are no outputs
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_outputs_ready,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));

    for (std::unique_ptr<BlockChannelMetadata>& output :
         block_metadata_.outputs()) {
      XLS_RET_CHECK(output->adapter().has_value());
      RDVAdapter& adapter = output->adapter().value();

      if (adapter.channel_predicate() != nullptr) {
        // Logic for the active ready signal for a send operation with a
        // predicate `pred`.
        //
        //   active = !pred | pred && valid
        //          = !pred | valid
        XLS_ASSIGN_OR_RETURN(
            Node * not_pred,
            block->MakeNode<UnOp>(SourceInfo(), adapter.channel_predicate(),
                                  Op::kNot));

        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        if (!adapter.channel_predicate()->HasAssignedName()) {
          not_pred->SetName(absl::StrFormat(
              "%s_not_pred", output->channel_interface()->name()));
        }

        std::vector<Node*> operands = {not_pred, adapter.ready()};
        XLS_ASSIGN_OR_RETURN(
            Node * active_ready,
            block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));

        // not_pred will have an assigned name or be inlined, so only check
        // the ready port. If it has an assigned name, just let everything
        // inline. Otherwise, give a descriptive name.
        if (!adapter.ready()->HasAssignedName()) {
          active_ready->SetName(absl::StrFormat(
              "%s_active_ready", output->channel_interface()->name()));
        }

        active_readys.push_back(active_ready);
      } else {
        // No predicate is the same as pred = true, so
        // active = !pred | valid = !true | valid = false | valid = valid
        active_readys.push_back(adapter.ready());
      }
    }
    XLS_ASSIGN_OR_RETURN(
        all_active_outputs_ready,
        block->MakeNode<NaryOp>(SourceInfo(), active_readys, Op::kAnd));

    for (std::unique_ptr<BlockChannelMetadata>& input :
         block_metadata_.inputs()) {
      std::vector<Node*> input_ready_operands = {all_active_outputs_ready};

      XLS_RET_CHECK(input->adapter().has_value());
      RDVAdapter& adapter = input->adapter().value();

      if (adapter.channel_predicate() != nullptr) {
        input_ready_operands.push_back(adapter.channel_predicate());
      }

      XLS_ASSIGN_OR_RETURN(Node * input_ready,
                           block->MakeNode<NaryOp>(
                               SourceInfo(), input_ready_operands, Op::kAnd));

      // Replace existing output with the new valid signal.
      XLS_RETURN_IF_ERROR(adapter.SetReady(input_ready));
    }
    return absl::OkStatus();
  }

  // Don't clone Receive operations. Instead replace with a tuple
  // containing the Receive's token operand and an InputPort operation.
  //
  // In the case of handling non-blocking receives, the logic to adapt
  // data to a tuple of (data, valid) is added here.
  //
  // Streaming and non-streaming channels are handled identically here
  // (except for naming), and both create data, valid, and ready ports.
  //
  // Future passes will remove any unneeded ports for single-value channels.
  absl::StatusOr<Node*> HandleReceiveNode(const Node* node) {
    Block* block = block_metadata_.block();
    const Receive* receive = node->As<Receive>();

    XLS_ASSIGN_OR_RETURN(ChannelInterface * chan_interface,
                         receive->GetChannelInterface());

    // Retrieve the slot associated with the channel.
    XLS_ASSIGN_OR_RETURN(BlockChannelMetadata * block_metadata,
                         block_metadata_.GetChannelMetadata(chan_interface));

    // Add in the receive adapter after the slot.  There should be one at
    // this point.
    XLS_RET_CHECK(block_metadata->slots().size() == 1);

    BlockRDVSlot& slot = block_metadata->slots()[0];

    XLS_ASSIGN_OR_RETURN(
        RDVAdapter adapter,
        RDVAdapter::CreateReceiveAdapter(slot, receive,
                                         proc_ir_to_block_ir_node_map_, block));

    Node* channel_op_value = adapter.channel_op_value();
    XLS_RETURN_IF_ERROR(block_metadata->AddAdapter(std::move(adapter)));

    return channel_op_value;
  }

  // Don't clone Send operations. Instead replace with an OutputPort
  // operation in the block.
  //
  // Streaming and non-streaming channels are handled identically here
  // (except for naming), and both create data, valid, and ready ports.
  //
  // Future passes will remove any unneeded ports for single-value channels.
  absl::StatusOr<Node*> HandleSendNode(const Node* node) {
    Block* block = block_metadata_.block();
    const Send* send = node->As<Send>();

    XLS_ASSIGN_OR_RETURN(ChannelInterface * chan_interface,
                         send->GetChannelInterface());

    // Retrieve the slot associated with the channel.
    XLS_ASSIGN_OR_RETURN(BlockChannelMetadata * block_metadata,
                         block_metadata_.GetChannelMetadata(chan_interface));

    // Add in the send adapter after the slot.  There should be one at
    // this point.
    XLS_RET_CHECK(block_metadata->slots().size() == 1);

    BlockRDVSlot& slot = block_metadata->slots()[0];

    XLS_ASSIGN_OR_RETURN(RDVAdapter adapter,
                         RDVAdapter::CreateSendAdapter(
                             slot, send, proc_ir_to_block_ir_node_map_, block));

    // Update the adapter's data node.
    auto send_data_node = proc_ir_to_block_ir_node_map_.find(send->data());
    XLS_RET_CHECK(send_data_node != proc_ir_to_block_ir_node_map_.end());
    XLS_RET_CHECK_OK(adapter.SetData(send_data_node->second));

    Node* channel_op_value = adapter.channel_op_value();
    XLS_RETURN_IF_ERROR(block_metadata->AddAdapter(std::move(adapter)));

    return channel_op_value;
  }

  // Clone the operation from the source to the block as is.
  absl::StatusOr<Node*> HandleGeneralNode(const Node* node) {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(proc_ir_to_block_ir_node_map_.at(operand));
    }
    return node->CloneInNewFunction(new_operands, block_metadata_.block());
  }

  const CodegenOptions& options_;
  const ProcMetadata& proc_metadata_;
  BlockMetadata& block_metadata_;

  // Map from node in the stage proc to node in the block.
  absl::flat_hash_map<const Node*, Node*> proc_ir_to_block_ir_node_map_;
};

}  // namespace

absl::StatusOr<Block*> CreateBlocksForProcHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata) {
  if (!top_metadata.IsTop()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConvertProcHierarchyToBlocks proc %s must be a "
                        "top-level proc created by stage conversion.",
                        top_metadata.proc()->name()));
  }

  Package* package = top_metadata.proc()->package();

  // TODO(tedhong): 2024-12-20 - Add support for nested pipelines.
  //
  // For now, the only children of a top-level proc should be stages.
  XLS_ASSIGN_OR_RETURN(std::vector<ProcMetadata*> stages,
                       stage_conversion_metadata.GetChildrenOf(&top_metadata));

  for (ProcMetadata* proc_metadata : stages) {
    Proc* proc = proc_metadata->proc();
    Block* block = proc->package()->AddBlock(
        std::make_unique<Block>(proc->name(), package));

    block_conversion_metadata.AssociateWithNewBlock(proc_metadata, block);
  }

  // Create and stitch together top level block.
  Proc* top_proc = top_metadata.proc();
  Block* top_block = top_proc->package()->AddBlock(
      std::make_unique<Block>(top_metadata.proc()->name(), package));

  block_conversion_metadata.AssociateWithNewBlock(&top_metadata, top_block);

  return top_block;
}

absl::StatusOr<Block*> ConvertProcHierarchyToBlocks(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata) {
  XLS_ASSIGN_OR_RETURN(std::vector<ProcMetadata*> stages,
                       stage_conversion_metadata.GetChildrenOf(&top_metadata));

  for (ProcMetadata* proc_metadata : stages) {
    XLS_ASSIGN_OR_RETURN(
        BlockMetadata * block_metadata,
        block_conversion_metadata.GetBlockMetadata(proc_metadata));

    XLS_RETURN_IF_ERROR(
        StageToBlockCloner(options, *proc_metadata, *block_metadata)
            .RunForStage());
  }

  // TODO(tedhong): 2025-04-08 - Stitch together top level block.
  XLS_ASSIGN_OR_RETURN(
      BlockMetadata * top_block_metadata,
      block_conversion_metadata.GetBlockMetadata(&top_metadata));

  return top_block_metadata->block();
}

absl::StatusOr<Block*> AddResetAndClockPortsToBlockHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata) {
  XLS_ASSIGN_OR_RETURN(std::vector<ProcMetadata*> stages,
                       stage_conversion_metadata.GetChildrenOf(&top_metadata));

  for (ProcMetadata* proc_metadata : stages) {
    XLS_ASSIGN_OR_RETURN(
        BlockMetadata * block_metadata,
        block_conversion_metadata.GetBlockMetadata(proc_metadata));

    XLS_RET_CHECK_OK(MaybeAddResetPort(block_metadata->block(), options));
    XLS_RET_CHECK_OK(MaybeAddClockPort(block_metadata->block(), options));
  }

  XLS_ASSIGN_OR_RETURN(
      BlockMetadata * top_block_metadata,
      block_conversion_metadata.GetBlockMetadata(&top_metadata));

  XLS_RET_CHECK_OK(MaybeAddResetPort(top_block_metadata->block(), options));
  XLS_RET_CHECK_OK(MaybeAddClockPort(top_block_metadata->block(), options));

  return top_block_metadata->block();
}

}  // namespace xls::verilog

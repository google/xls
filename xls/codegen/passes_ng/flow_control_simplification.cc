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

#include "xls/codegen/passes_ng/flow_control_simplification.h"

#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls::verilog {

absl::Status RemoveReadyBackpressure(BlockMetadata& top_block_metadata) {
  Block* top_block = top_block_metadata.block();

  for (std::unique_ptr<BlockChannelMetadata>& send_channel :
       top_block_metadata.outputs()) {
    Node* ready_port = send_channel->slots().back().GetPorts().ready;

    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        top_block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));

    XLS_RETURN_IF_ERROR(ready_port->ReplaceUsesWith(literal_1));
  }

  for (std::unique_ptr<BlockChannelMetadata>& receive_channel :
       top_block_metadata.inputs()) {
    Node* ready_port = receive_channel->slots().front().GetPorts().ready;

    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        top_block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_RETURN_IF_ERROR(ready_port->ReplaceOperandNumber(0, literal_1));
  }

  return absl::OkStatus();
}

absl::Status RemoveValidSignals(BlockMetadata& top_block_metadata) {
  Block* top_block = top_block_metadata.block();

  for (std::unique_ptr<BlockChannelMetadata>& send_channel :
       top_block_metadata.outputs()) {
    Node* valid_port = send_channel->slots().back().GetPorts().valid;

    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        top_block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_RETURN_IF_ERROR(valid_port->ReplaceOperandNumber(0, literal_1));
  }

  for (std::unique_ptr<BlockChannelMetadata>& receive_channel :
       top_block_metadata.inputs()) {
    Node* valid_port = receive_channel->slots().front().GetPorts().valid;

    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        top_block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_RETURN_IF_ERROR(valid_port->ReplaceUsesWith(literal_1));
  }

  return absl::OkStatus();
}

absl::Status RemoveUnusedInputPorts(Block* absl_nonnull top_block) {
  std::vector<InputPort*> ports_to_remove;

  for (InputPort* port : top_block->GetInputPorts()) {
    if (port->IsDead()) {
      ports_to_remove.push_back(port);
    }
  }

  for (InputPort* port : ports_to_remove) {
    XLS_RETURN_IF_ERROR(top_block->RemoveNode(port));
  }

  return absl::OkStatus();
}

absl::Status RemoveConstantOutputPorts(Block* absl_nonnull top_block) {
  std::vector<OutputPort*> ports_to_remove;

  for (OutputPort* port : top_block->GetOutputPorts()) {
    if (port->operand(0)->Is<Literal>()) {
      ports_to_remove.push_back(port);
    }
  }

  for (OutputPort* port : ports_to_remove) {
    XLS_RETURN_IF_ERROR(top_block->RemoveNode(port));
  }

  return absl::OkStatus();
}

}  // namespace xls::verilog

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

#include "xls/codegen/passes_ng/block_channel_adapter.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/block_utils.h"
#include "xls/common/status/ret_check.h"
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
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::verilog {
namespace {

std::string CreateAdapterBufferName(std::string_view port_name,
                                    RDVAdapter::AdapterType adapter_type) {
  return absl::StrCat(
      "__", port_name, "_",
      adapter_type == RDVAdapter::AdapterType::kSend ? "send" : "receive",
      "_buf");
}

absl::StatusOr<RDVNodeGroup> CreateAdapterBufferBank(
    BlockRDVSlot& slot, RDVAdapter::AdapterType adapter_type,
    const SourceInfo& loc, Block* block) {
  RDVNodeGroup ports = slot.GetPorts();
  RDVNodeGroup bank = slot.GetUpstreamBufferBank();

  // Buffer the channel ready/data/and valid signals.
  XLS_ASSIGN_OR_RETURN(
      Node * ready_buf,
      CreateBufferAfter(
          CreateAdapterBufferName(ports.ready->GetNameView(), adapter_type),
          bank.ready, loc, block));
  XLS_ASSIGN_OR_RETURN(
      Node * data_buf,
      CreateBufferBefore(
          CreateAdapterBufferName(ports.data->GetNameView(), adapter_type),
          bank.data, loc, block));
  XLS_ASSIGN_OR_RETURN(
      Node * valid_buf,
      CreateBufferBefore(
          CreateAdapterBufferName(ports.valid->GetNameView(), adapter_type),
          bank.valid, loc, block));

  return RDVNodeGroup{ready_buf, data_buf, valid_buf};
}

}  // namespace

absl::StatusOr<RDVAdapter> RDVAdapter::CreateSendAdapter(
    BlockRDVSlot& slot, const Send* ABSL_NONNULL send,
    const IrToBlockIrMap& node_map, Block* ABSL_NONNULL block) {
  const SourceInfo& loc = send->loc();

  // Buffer the channel ready/data/and valid signals.
  XLS_ASSIGN_OR_RETURN(
      RDVNodeGroup buf_bank,
      CreateAdapterBufferBank(slot, AdapterType::kSend, loc, block));

  // Map the Send node to the token operand of the Send in the
  // block.
  auto send_tok_iter = node_map.find(send->token());
  XLS_RET_CHECK(send_tok_iter != node_map.end());
  Node* send_tok = send_tok_iter->second;

  XLS_ASSIGN_OR_RETURN(Node * channel_op_value,
                       block->MakeNode<UnOp>(
                           /*loc=*/loc, send_tok, Op::kIdentity));

  // Create a the predicate signals.
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  Node* channel_predicate = literal_1;
  if (send->predicate().has_value()) {
    auto predicate_iter = node_map.find(send->predicate().value());
    XLS_RET_CHECK(predicate_iter != node_map.end());
    channel_predicate = predicate_iter->second;
  }

  return RDVAdapter(AdapterType::kSend, buf_bank, channel_op_value,
                    channel_predicate);
}

absl::StatusOr<RDVAdapter> RDVAdapter::CreateInterfaceSendAdapter(
    BlockRDVSlot& slot, Block* ABSL_NONNULL block) {
  const SourceInfo loc;

  // Buffer the channel ready/data/and valid signals.
  XLS_ASSIGN_OR_RETURN(
      RDVNodeGroup buf_bank,
      CreateAdapterBufferBank(slot, AdapterType::kSend, loc, block));

  // Create a the predicate and channel op value signals.
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));
  Node* channel_predicate = literal_1;
  XLS_ASSIGN_OR_RETURN(
      Node * channel_op_value,
      block->MakeNode<xls::Literal>(SourceInfo(), Value::Tuple({})));

  return RDVAdapter(AdapterType::kSend, buf_bank, channel_op_value,
                    channel_predicate);
}

absl::StatusOr<RDVAdapter> RDVAdapter::CreateReceiveAdapter(
    BlockRDVSlot& slot, const Receive* ABSL_NONNULL receive,
    const IrToBlockIrMap& node_map, Block* ABSL_NONNULL block) {
  const SourceInfo& loc = receive->loc();

  // Buffer the channel ready/data/and valid signals.
  XLS_ASSIGN_OR_RETURN(
      RDVNodeGroup buf_bank,
      CreateAdapterBufferBank(slot, AdapterType::kReceive, loc, block));

  // If blocking, return a tuple of (token, data), and if non-blocking
  // return a tuple of (token, data, valid).
  Node* channel_op_value;

  auto recv_tok_iter = node_map.find(receive->operand(0));
  XLS_RET_CHECK(recv_tok_iter != node_map.end());
  Node* recv_tok = recv_tok_iter->second;

  if (receive->is_blocking()) {
    XLS_ASSIGN_OR_RETURN(
        channel_op_value,
        block->MakeNode<Tuple>(loc,
                               std::vector<Node*>({recv_tok, buf_bank.data})));
  } else {
    // Ensure that the output of the receive is zero when the data is not
    // valid or the predicate is false.
    XLS_ASSIGN_OR_RETURN(
        channel_op_value,
        block->MakeNode<Tuple>(loc, std::vector<Node*>({recv_tok, buf_bank.data,
                                                        buf_bank.valid})));
  }

  // Create a valid and predicate signals.
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  Node* channel_valid = receive->is_blocking() ? buf_bank.valid : literal_1;

  Node* channel_predicate = literal_1;
  if (receive->predicate().has_value()) {
    auto predicate_iter = node_map.find(receive->predicate().value());
    XLS_RET_CHECK(predicate_iter != node_map.end());
    channel_predicate = predicate_iter->second;
  }

  return RDVAdapter(AdapterType::kReceive,
                    RDVNodeGroup{buf_bank.ready, buf_bank.data, channel_valid},
                    channel_op_value, channel_predicate);
}

absl::StatusOr<RDVAdapter> RDVAdapter::CreateInterfaceReceiveAdapter(
    BlockRDVSlot& slot, Block* ABSL_NONNULL block) {
  SourceInfo loc;

  // Buffer the channel ready/data/and valid signals.
  XLS_ASSIGN_OR_RETURN(
      RDVNodeGroup buf_bank,
      CreateAdapterBufferBank(slot, AdapterType::kReceive, loc, block));

  // Create a valid and predicate signals.
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));
  Node* channel_predicate = literal_1;

  return RDVAdapter(AdapterType::kReceive, buf_bank,
                    /*channel_op_value=*/buf_bank.data, channel_predicate);
}

}  // namespace xls::verilog

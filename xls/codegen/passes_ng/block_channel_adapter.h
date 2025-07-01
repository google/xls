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

#ifndef XLS_CODEGEN_PASSES_NG_BLOCK_CHANNEL_ADAPTER_H_
#define XLS_CODEGEN_PASSES_NG_BLOCK_CHANNEL_ADAPTER_H_

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls::verilog {

// Adapts the RDV signals to the signals needed for a send or receive operation.
//
// From a send slot, inserts the following logic to support returning a token
// value and operations to rewire data and valid.
//               [Adapter]  [     SLOT     ] r]
//                           left     right
// Port Ready ----<|----------<|-------<|---
// Port Data  ----|>----------|>-------|>---
// Port Valid ----|>----------|>-------|>---
//      token <----
//
// From a receive slot, inserts the following logic to support returning the
// receive's value and to support blocking/non-blocking receives.
//
// After the adapter, non-blocking receives have a valid signal of 1.
//
//              [     SLOT     ]  [Adapter]
//                left     right
// Port Ready -----<|-------<|-----<|--- Available to re-assign with SetReady()
// Port Data  -----|>-------|>------+--- channel_op_value (token, data, (opt)
// valid). Port Valid -----|>-------|>------|
//                                 token
class RDVAdapter {
 public:
  // Used to map IR nodes to Block IR nodes.
  using IrToBlockIrMap = absl::flat_hash_map<const Node*, Node*>;

  enum class AdapterType {
    kSend,
    kReceive,
  };

  // Creates a send adapter for a given slot and operation.
  static absl::StatusOr<RDVAdapter> CreateSendAdapter(
      BlockRDVSlot& slot, const Send* ABSL_NONNULL send,
      const IrToBlockIrMap& node_map, Block* ABSL_NONNULL block);

  // Creates a receive adapter for a given slot and operation.
  static absl::StatusOr<RDVAdapter> CreateReceiveAdapter(
      BlockRDVSlot& slot, const Receive* ABSL_NONNULL receive,
      const IrToBlockIrMap& node_map, Block* ABSL_NONNULL block);

  // Creates a send adapter for a given slot that is not associated with a
  // send operation.
  static absl::StatusOr<RDVAdapter> CreateInterfaceSendAdapter(
      BlockRDVSlot& slot, Block* ABSL_NONNULL block);

  // Creates a receive adapter for a given slot that is not associated with
  // a receive operation.
  static absl::StatusOr<RDVAdapter> CreateInterfaceReceiveAdapter(
      BlockRDVSlot& slot, Block* ABSL_NONNULL block);

  // Rewire the ready signal to a new source.
  absl::Status SetReady(Node* new_ready_src) {
    if (adapter_type_ == AdapterType::kSend) {
      return absl::InvalidArgumentError("Cannot set ready on a send adapter.");
    }

    return rdv_bufs_.ready->ReplaceOperandNumber(0, new_ready_src);
  }

  // Rewire the data signal to a new source.
  absl::Status SetData(Node* ABSL_NONNULL new_data_src) {
    if (adapter_type_ == AdapterType::kReceive) {
      return absl::InvalidArgumentError(
          "Cannot set data on a receive adapter.");
    }

    return rdv_bufs_.data->ReplaceOperandNumber(0, new_data_src);
  }

  // Rewire the valid signal to a new source.
  absl::Status SetValid(Node* ABSL_NONNULL new_valid_src) {
    if (adapter_type_ == AdapterType::kReceive) {
      return absl::InvalidArgumentError(
          "Cannot set valid on a receive adapter.");
    }

    return rdv_bufs_.valid->ReplaceOperandNumber(0, new_valid_src);
  }

  // Returns the data node for the adapter.
  Node* data() { return rdv_bufs_.data; }

  // Returns the ready node for the adapter.
  Node* ready() { return rdv_bufs_.ready; }

  // Returns the valid node for the adapter.
  Node* valid() { return rdv_bufs_.valid; }

  // Getter for the channel op value node.
  Node* channel_op_value() { return channel_op_value_; }

  // Getter for the predicate node of the channel.
  Node* channel_predicate() { return channel_predicate_; }

  // Returns the type of adapter this is.
  AdapterType adapter_type() const { return adapter_type_; }

 private:
  RDVAdapter(AdapterType adapter_type, RDVNodeGroup rdv_bufs,
             Node* channel_op_value, Node* channel_predicate)
      : adapter_type_(adapter_type),
        rdv_bufs_(rdv_bufs),
        channel_op_value_(channel_op_value),
        channel_predicate_(channel_predicate) {}

  // The type of adapter this is.
  AdapterType adapter_type_;

  // The ready-data-valid block buffers that are the adapter's internal
  // view of the channel.
  RDVNodeGroup rdv_bufs_;

  // Block node representing the return value of the send/receive op.
  Node* channel_op_value_ = nullptr;

  // Block node representing the predicate passed to the send/receive op.
  Node* channel_predicate_ = nullptr;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_BLOCK_CHANNEL_ADAPTER_H_

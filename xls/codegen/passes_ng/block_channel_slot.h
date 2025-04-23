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

#ifndef XLS_CODEGEN_PASSES_NG_BLOCK_CHANNEL_SLOT_H_
#define XLS_CODEGEN_PASSES_NG_BLOCK_CHANNEL_SLOT_H_

#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"

namespace xls::verilog {

// A group of associated ready-data-valid nodes.
struct RDVNodeGroup {
  Node* ready;
  Node* data;
  Node* valid;
};

// Creates and groups together a set of nodes of a RDV channel at the XLS block
// level to support the insertion of different RDV adapters/logic/fifos on the
// particular channel.
//
// The following are the different views supported by this class:
//   1. Ready-Valid-Data ports at the interface of the block.
//   2. Ready-Valid-Data nodes where additional slots or adapters can be
//      inserted.
//
// Pictorially slots can be created at the interface of a block
// or between blocks:
//
//      ------------             ----------
//                 |            |
//          ---    |     ---    |    ---
//         | S |->-R----| S |---R->-| S |
//         | l |   |    | l |   |   | l |
// Send--->| o |->-D----| o |---D->-| o |--->Receive
//         | t |   |    | t |   |   | t |
//         |(1)|->-V----|(2)|---V->-|(3)|
//          ---    |     ---    |    ---
//                 |            |
//      -----------              ----------
//
// The forward direction is from the direction of the send to receive.
//
// For slots between blocks (Slot-2 above), they are not associated with
// ports, but with a set of nodes between the two ports.
class BlockRDVSlot {
 public:
  // Create a slot for a send channel.
  //
  // This will also insert a pair of buffers to serve as attachment points for
  // any additional slot logic.
  //
  // -----<|-------<|----Node Ready
  // -----|>-------|>----Node Data
  // -----|>-------|>----Node Valid
  //
  static absl::StatusOr<BlockRDVSlot> CreateSendSlot(std::string_view name,
                                                     RDVNodeGroup rdv_group,
                                                     Block* ABSL_NONNULL block);

  // Create a slot for a receive channel.
  static absl::StatusOr<BlockRDVSlot> CreateReceiveSlot(
      std::string_view name, RDVNodeGroup rdv_group, Block* ABSL_NONNULL block);

  // Returns the name of this slot.
  std::string_view name() const { return name_; }

  // Returns the upstream bank (towards the send op) side of buffers.
  RDVNodeGroup GetUpstreamBufferBank() { return upstream_buffer_bank_; }

  // Returns the downstream bank (towards the receive op) side of buffers.
  RDVNodeGroup GetDownstreamBufferBank() { return downstream_buffer_bank_; }

  // Returns the set of ports this slot is associated with.
  RDVNodeGroup GetPorts() { return port_nodes_; }

 private:
  BlockRDVSlot(std::string_view name, RDVNodeGroup upstream_buffer_bank,
               RDVNodeGroup downstream_buffer_bank, RDVNodeGroup port_nodes)
      : name_(name),
        upstream_buffer_bank_(upstream_buffer_bank),
        downstream_buffer_bank_(downstream_buffer_bank),
        port_nodes_(port_nodes) {}

  // Name of the slot, used to prefix ops/nodes associated with this slot.
  std::string name_;

  // These are the nodes associated with each of the banks of buffers.
  RDVNodeGroup upstream_buffer_bank_;
  RDVNodeGroup downstream_buffer_bank_;

  // These are the ports/nodes associated with the channel.
  //
  // These will be ports for slots at the interface of a block and nodes for
  // slots between blocks.
  RDVNodeGroup port_nodes_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_BLOCK_CHANNEL_SLOT_H_

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

#include "xls/codegen/passes_ng/block_channel_slot.h"

#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/passes_ng/block_utils.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::verilog {
namespace {

enum class SlotSide { kUpstream = 0, kDownstream };

std::string CreateSlotBufferName(std::string_view port_name,
                                 std::string_view slot_name,
                                 SlotSide slot_side) {
  return absl::StrCat(
      "__", port_name, "_slot_", slot_name, "_buf_",
      slot_side == SlotSide::kUpstream ? "upstream" : "downstream");
}

}  // namespace

absl::StatusOr<BlockRDVSlot> BlockRDVSlot::CreateSendSlot(
    std::string_view name, RDVNodeGroup rdv_group, Block* ABSL_NONNULL block) {
  // Create buffers
  //    upstream     downstream
  // -----<|-------<|----Port Ready
  // -----|>-------|>----Port Data
  // -----|>-------|>----Port Valid

  // Construct names for the ports.
  std::string_view ready_port_name = rdv_group.ready->GetNameView();
  std::string_view data_port_name = rdv_group.data->GetNameView();
  std::string_view valid_port_name = rdv_group.valid->GetNameView();

  // Construct names for buffers placed before/after the ports.
  std::string ready_buf_upstream_name =
      CreateSlotBufferName(ready_port_name, name, SlotSide::kUpstream);
  std::string data_buf_upstream_name =
      CreateSlotBufferName(data_port_name, name, SlotSide::kUpstream);
  std::string valid_buf_upstream_name =
      CreateSlotBufferName(valid_port_name, name, SlotSide::kUpstream);
  std::string ready_buf_downstream_name =
      CreateSlotBufferName(ready_port_name, name, SlotSide::kDownstream);
  std::string data_buf_downstream_name =
      CreateSlotBufferName(data_port_name, name, SlotSide::kDownstream);
  std::string valid_buf_downstream_name =
      CreateSlotBufferName(valid_port_name, name, SlotSide::kDownstream);

  // Create the buffer pairs.
  XLS_ASSIGN_OR_RETURN(
      Node * ready_buf_downstream,
      CreateBufferAfter(ready_buf_downstream_name, rdv_group.ready,
                        rdv_group.ready->loc(), block));
  XLS_ASSIGN_OR_RETURN(
      Node * ready_buf_upstream,
      CreateBufferAfter(ready_buf_upstream_name, ready_buf_downstream,
                        rdv_group.ready->loc(), block));

  XLS_ASSIGN_OR_RETURN(
      Node * data_buf_downstream,
      CreateBufferBefore(data_buf_downstream_name, rdv_group.data,
                         rdv_group.data->loc(), block));
  XLS_ASSIGN_OR_RETURN(
      Node * data_buf_upstream,
      CreateBufferBefore(data_buf_upstream_name, data_buf_downstream,
                         rdv_group.data->loc(), block));

  XLS_ASSIGN_OR_RETURN(
      Node * valid_buf_downstream,
      CreateBufferBefore(valid_buf_downstream_name, rdv_group.valid,
                         rdv_group.valid->loc(), block));
  XLS_ASSIGN_OR_RETURN(
      Node * valid_buf_upstream,
      CreateBufferBefore(valid_buf_upstream_name, valid_buf_downstream,
                         rdv_group.valid->loc(), block));

  return BlockRDVSlot(
      name,
      RDVNodeGroup{ready_buf_upstream, data_buf_upstream, valid_buf_upstream},
      RDVNodeGroup{ready_buf_downstream, data_buf_downstream,
                   valid_buf_downstream},
      rdv_group);
}

absl::StatusOr<BlockRDVSlot> BlockRDVSlot::CreateReceiveSlot(
    std::string_view name, RDVNodeGroup rdv_group, Block* ABSL_NONNULL block) {
  // Create buffers
  //                upstream     downstream
  // Port Ready -----<|-------<|----
  // Port Data  -----|>-------|>----
  // Port Valid -----|>-------|>----

  // Construct names for the ports.
  std::string_view ready_port_name = rdv_group.ready->GetNameView();
  std::string_view data_port_name = rdv_group.data->GetNameView();
  std::string_view valid_port_name = rdv_group.valid->GetNameView();

  // Construct names for buffers placed before/after the ports.
  std::string ready_buf_upstream_name =
      CreateSlotBufferName(ready_port_name, name, SlotSide::kUpstream);
  std::string data_buf_upstream_name =
      CreateSlotBufferName(data_port_name, name, SlotSide::kUpstream);
  std::string valid_buf_upstream_name =
      CreateSlotBufferName(valid_port_name, name, SlotSide::kUpstream);
  std::string ready_buf_downstream_name =
      CreateSlotBufferName(ready_port_name, name, SlotSide::kDownstream);
  std::string data_buf_downstream_name =
      CreateSlotBufferName(data_port_name, name, SlotSide::kDownstream);
  std::string valid_buf_downstream_name =
      CreateSlotBufferName(valid_port_name, name, SlotSide::kDownstream);

  // Create the buffer pairs.
  XLS_ASSIGN_OR_RETURN(
      Node * ready_buf_upstream,
      CreateBufferBefore(ready_buf_upstream_name, rdv_group.ready,
                         rdv_group.ready->loc(), block));
  XLS_ASSIGN_OR_RETURN(
      Node * ready_buf_downstream,
      CreateBufferBefore(ready_buf_downstream_name, ready_buf_upstream,
                         rdv_group.ready->loc(), block));

  XLS_ASSIGN_OR_RETURN(Node * data_buf_upstream,
                       CreateBufferAfter(data_buf_upstream_name, rdv_group.data,
                                         rdv_group.data->loc(), block));
  XLS_ASSIGN_OR_RETURN(
      Node * data_buf_downstream,
      CreateBufferAfter(data_buf_downstream_name, data_buf_upstream,
                        rdv_group.data->loc(), block));

  XLS_ASSIGN_OR_RETURN(
      Node * valid_buf_upstream,
      CreateBufferAfter(valid_buf_upstream_name, rdv_group.valid,
                        rdv_group.valid->loc(), block));
  XLS_ASSIGN_OR_RETURN(
      Node * valid_buf_downstream,
      CreateBufferAfter(valid_buf_downstream_name, valid_buf_upstream,
                        rdv_group.valid->loc(), block));

  return BlockRDVSlot(
      name,
      RDVNodeGroup{ready_buf_upstream, data_buf_upstream, valid_buf_upstream},
      RDVNodeGroup{ready_buf_downstream, data_buf_downstream,
                   valid_buf_downstream},
      rdv_group);
}

}  // namespace xls::verilog

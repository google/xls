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

#include "xls/codegen/passes_ng/block_pipeline_inserter.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
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
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls::verilog {
namespace {

// Adds a register between the node and all its downstream users.
// Returns the new register added.
absl::StatusOr<RegisterRead*> AddRegisterAfterNode(
    std::string_view name_prefix, std::optional<Node*> load_enable,
    const std::optional<Value>& reset_value, Node* node) {
  Block* block = node->function_base()->AsBlockOrDie();

  // If there is a reset value, there must be a reset port.
  XLS_RET_CHECK(!reset_value.has_value() || block->GetResetPort().has_value());

  std::string name = absl::StrFormat("__%s_reg", name_prefix);

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block->AddRegister(name, node->GetType(), reset_value));

  XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                       block->MakeNodeWithName<RegisterRead>(
                           /*loc=*/node->loc(),
                           /*reg=*/reg,
                           /*name=*/name));

  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(reg_read));

  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<RegisterWrite>(
                              /*loc=*/node->loc(),
                              /*data=*/node,
                              /*load_enable=*/load_enable,
                              /*reset=*/reset_value.has_value()
                                  ? block->GetResetPort()
                                  : std::nullopt,
                              /*reg=*/reg)
                          .status());

  return reg_read;
}

// Given a slot, insert a pipeline register with basic flow control.
//
// Before:
// -----<|-------<|----Node Ready
// -----|>-------|>----Node Data
// -----|>-------|>----Node Valid
//
// After:
//                          | data |
// -----|>------------------| flop |-------|>---- Node Data
//
// Node Ready (Upstream/Left   ^ data_load_en
// -----<|-----------------    |
//                        |    |
//            ------|-)   |    |
//            |     |& )--+-----
//            |  ---|-)
//            |  |
//            |  |
//            |  |
//            |  |
//            |  |  /--|-+-----------------<|---- Node Ready (Downstream/Right)
//       -----|--+-< or|
//       |    |     \--|--
//       v    |          |   not_valid
//  vld_ld_en |          o
//            |          ^
//            | | vld  | |
// -----|>----+-| flop |-------------------|>---- Node Valid
//
//
// Flops are inserted between the data and valid paths while flow control is
// inserted into the ready path as well as driving the flop load enables:
//   valid_load_enable = valid && ready || ! valid = ready || ! valid.
//   data_load_enable = valid_load_enable && valid.
absl::Status InsertPipelineFlop(const CodegenOptions& options,
                                BlockRDVSlot& slot,
                                std::string_view flop_prefix, Block* block) {
  VLOG(2) << "Adding pipeline " << flop_prefix << " flops.";

  RDVNodeGroup left_buffers = slot.GetUpstreamBufferBank();

  std::string data_flop_name = absl::StrCat(flop_prefix, "_data");
  std::string valid_flop_name = absl::StrCat(flop_prefix, "_valid");
  std::string data_load_en_name = absl::StrCat(flop_prefix, "_data_load_en");
  std::string valid_load_en_name = absl::StrCat(flop_prefix, "_valid_load_en");

  // Options for pipeline flops.
  bool do_reset = options.reset().has_value();
  bool do_reset_datapath =
      options.reset().has_value() && options.reset()->reset_data_path();

  // 1. Add flops for the data and valid paths.
  // Add a node for load_enables (will be replaced later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  std::optional<Value> datapath_reset_value =
      do_reset_datapath
          ? std::make_optional(ZeroOfType(left_buffers.data->GetType()))
          : std::nullopt;

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_flop_read,
      AddRegisterAfterNode(data_flop_name, /*load_enable=*/literal_1,
                           /*reset_value=*/datapath_reset_value,
                           left_buffers.data));

  std::optional<Value> valid_reset_value =
      do_reset ? std::make_optional(ZeroOfType(left_buffers.valid->GetType()))
               : std::nullopt;

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * valid_flop_read,
      AddRegisterAfterNode(valid_flop_name, /*load_enable=*/literal_1,
                           /*reset_value=*/valid_reset_value,
                           left_buffers.valid));

  // 2. Construct and update the ready signal.
  Node* right_buffers_ready = slot.GetDownstreamBufferBank().ready;

  // valid_load_enable = valid && ready || ! ready = valid || ! ready.
  // data_load_enable = valid_load_enable && valid.
  XLS_ASSIGN_OR_RETURN(
      Node * not_valid,
      block->MakeNode<UnOp>(valid_flop_read->loc(), valid_flop_read, Op::kNot));
  XLS_ASSIGN_OR_RETURN(Node * valid_load_en,
                       block->MakeNodeWithName<NaryOp>(
                           not_valid->loc(),
                           std::vector<Node*>({right_buffers_ready, not_valid}),
                           Op::kOr, valid_load_en_name));

  XLS_ASSIGN_OR_RETURN(
      Node * data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>({left_buffers.valid, valid_load_en}), Op::kAnd,
          data_load_en_name));

  // 3. Update load enables for the data and valid registers.
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(
      valid_load_en, valid_flop_read->GetRegister(), block));
  XLS_RETURN_IF_ERROR(
      UpdateRegisterLoadEn(data_load_en, data_flop_read->GetRegister(), block));

  XLS_RET_CHECK_OK(left_buffers.ready->ReplaceOperandNumber(0, data_load_en));

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Block*> InsertPipelineIntoBlock(
    const CodegenOptions& options, BlockMetadata& top_block_metadata) {
  Block* top_block = top_block_metadata.block();

  // Get a list of all send channel refs between stages.
  for (std::unique_ptr<BlockChannelMetadata>& send_channel :
       top_block_metadata.outputs()) {
    if (send_channel->IsInternalChannel()) {
      VLOG(2) << "Adding pipeline flop for internal channel "
              << send_channel->channel_interface()->name();

      absl::Span<BlockRDVSlot> slots = send_channel->slots();
      // Based on sequencing of passes, there should only be one slot right
      // now.
      XLS_RET_CHECK(slots.size() == 1);

      XLS_RETURN_IF_ERROR(InsertPipelineFlop(
          options, slots[0], send_channel->channel_interface()->name(),
          top_block_metadata.block()));
    }
  }

  return top_block;
}

}  // namespace xls::verilog

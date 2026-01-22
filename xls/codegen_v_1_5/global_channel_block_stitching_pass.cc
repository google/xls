// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/global_channel_block_stitching_pass.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_utils.h"
#include "xls/codegen_v_1_5/global_channel_map.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

// Instantiate all blocks in the package in the container block.
// Returns a map that identifies each block with its instantiation in the
// container block.
absl::Status InstantiateBlocksInContainer(
    Block* container, const verilog::CodegenOptions& options) {
  std::vector<Block*> blocks_to_instantiate;
  for (std::unique_ptr<Block>& block : container->package()->blocks()) {
    if (block.get() == container) {
      continue;
    }
    blocks_to_instantiate.push_back(block.get());
  }
  absl::c_sort(blocks_to_instantiate, &FunctionBase::NameLessThan);
  std::vector<Node*> idle_outputs;
  if (options.add_idle_output()) {
    idle_outputs.reserve(blocks_to_instantiate.size());
  }
  // Instantiate all blocks in the container block.
  for (int64_t i = 0; i < blocks_to_instantiate.size(); i++) {
    Block* block = blocks_to_instantiate[i];
    std::string inst_name = absl::StrCat(block->name(), "_inst", i);
    XLS_ASSIGN_OR_RETURN(
        Instantiation * instantiation,
        container->AddInstantiation(
            inst_name,
            std::make_unique<xls::BlockInstantiation>(inst_name, block)));
    // Pass down reset.
    if (std::optional<Node*> block_reset = block->GetResetPort()) {
      std::optional<InputPort*> container_reset = container->GetResetPort();
      XLS_RET_CHECK(container_reset.has_value());
      XLS_RETURN_IF_ERROR(container
                              ->MakeNode<xls::InstantiationInput>(
                                  SourceInfo(), container_reset.value(),
                                  instantiation, (*block_reset)->GetName())
                              .status());
    } else {
      XLS_RET_CHECK(!options.reset().has_value());
    }
    // Connect idle output if it exists.
    if (options.add_idle_output()) {
      XLS_ASSIGN_OR_RETURN(Node * block_idle_output,
                           container->MakeNode<xls::InstantiationOutput>(
                               SourceInfo(), instantiation, "idle"));
      idle_outputs.push_back(block_idle_output);
    }
  }

  if (options.add_idle_output()) {
    XLS_ASSIGN_OR_RETURN(
        Node * container_idle_output,
        container->MakeNode<xls::NaryOp>(SourceInfo(), idle_outputs, Op::kAnd));
    XLS_RETURN_IF_ERROR(
        container->AddOutputPort("idle", container_idle_output).status());
  }
  return absl::OkStatus();
}

// Stitch a streaming output (i.e. a send on a streaming channel) to its
// associated FIFO.
absl::Status StitchStreamingOutputToFifo(
    Block* caller, xls::BlockInstantiation* callee_instantiation,
    xls::Instantiation* fifo, const ChannelPortMetadata& output) {
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_data,
      caller->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), callee_instantiation, *output.data_port));
  XLS_RET_CHECK(output.valid_port.has_value());
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_valid,
      caller->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), callee_instantiation, *output.valid_port));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * fifo_ready,
      caller->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), fifo, FifoInstantiation::kPushReadyPortName));
  XLS_RETURN_IF_ERROR(caller
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_data, fifo,
                              FifoInstantiation::kPushDataPortName)
                          .status());
  XLS_RETURN_IF_ERROR(caller
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_valid, fifo,
                              FifoInstantiation::kPushValidPortName)
                          .status());
  XLS_RET_CHECK(output.ready_port.has_value());
  XLS_RETURN_IF_ERROR(caller
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), fifo_ready, callee_instantiation,
                              *output.ready_port)
                          .status());
  return absl::OkStatus();
}

// Stitch a streaming input (i.e. a receive on a streaming channel) to its
// associated FIFO.
absl::Status StitchStreamingInputToFifo(
    Block* caller, xls::BlockInstantiation* callee_instantiation,
    xls::Instantiation* fifo, const ChannelPortMetadata& input) {
  XLS_ASSIGN_OR_RETURN(
      xls::Node * fifo_data,
      caller->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), fifo, FifoInstantiation::kPopDataPortName));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * fifo_valid,
      caller->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), fifo, FifoInstantiation::kPopValidPortName));
  XLS_RET_CHECK(input.ready_port.has_value());

  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_ready,
      caller->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), callee_instantiation, *input.ready_port));

  XLS_RETURN_IF_ERROR(
      caller
          ->MakeNode<xls::InstantiationInput>(
              SourceInfo(), fifo_data, callee_instantiation, *input.data_port)
          .status());
  XLS_RET_CHECK(input.valid_port.has_value());
  XLS_RETURN_IF_ERROR(
      caller
          ->MakeNode<xls::InstantiationInput>(
              SourceInfo(), fifo_valid, callee_instantiation, *input.valid_port)
          .status());
  XLS_RETURN_IF_ERROR(caller
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_ready, fifo, "pop_ready")
                          .status());
  return absl::OkStatus();
}

// Punch a streaming output through the container block. This happens when a
// block has an external send on a streaming channel.
absl::Status ExposeStreamingOutput(
    StreamingChannel* channel, const ChannelPortMetadata& output, Block* block,
    xls::BlockInstantiation* block_instantiation) {
  auto name_or_none = [](std::optional<std::string> port) {
    return port.has_value() ? *port : "<none>";
  };
  VLOG(5) << "Exposing output port: " << name_or_none(output.data_port);
  VLOG(5) << "Exposing output port valid: " << name_or_none(output.valid_port);
  VLOG(5) << "Exposing output port ready: " << name_or_none(output.ready_port);

  XLS_RET_CHECK(output.data_port.has_value());
  XLS_RET_CHECK(output.ready_port.has_value());
  XLS_RET_CHECK(output.valid_port.has_value());

  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_instantiation, *output.data_port));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_valid,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_instantiation, *output.valid_port));

  XLS_ASSIGN_OR_RETURN(InputPort * ready_port,
                       block_instantiation->instantiated_block()->GetInputPort(
                           *output.ready_port));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * ext_ready,
      block->AddInputPort(*output.ready_port, ready_port->port_type()));
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(*output.data_port, block_data).status());
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(*output.valid_port, block_valid).status());
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(
              SourceInfo(), ext_ready, block_instantiation, *output.ready_port)
          .status());

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      channel, ChannelDirection::kSend, *output.data_port, *output.valid_port,
      *output.ready_port, /*stage=*/std::nullopt));

  return absl::OkStatus();
}

// Punch a streaming input from the container block down. This happens when a
// block has an external receive on a streaming channel.
absl::Status ExposeStreamingInput(
    StreamingChannel* channel, const ChannelPortMetadata& input, Block* block,
    ::xls::BlockInstantiation* block_instantiation) {
  XLS_RET_CHECK(input.data_port.has_value());
  XLS_RET_CHECK(input.ready_port.has_value());
  XLS_RET_CHECK(input.valid_port.has_value());

  Block* instantiated_block = block_instantiation->instantiated_block();
  XLS_ASSIGN_OR_RETURN(InputPort * data_port,
                       instantiated_block->GetInputPort(*input.data_port));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * ext_data,
      block->AddInputPort(*input.data_port, data_port->port_type()));

  XLS_ASSIGN_OR_RETURN(InputPort * valid_port,
                       instantiated_block->GetInputPort(*input.valid_port));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * ext_valid,
      block->AddInputPort(*input.valid_port, valid_port->port_type()));

  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_ready,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_instantiation, *input.ready_port));
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(
              SourceInfo(), ext_data, block_instantiation, *input.data_port)
          .status());
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(
              SourceInfo(), ext_valid, block_instantiation, *input.valid_port)
          .status());
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(*input.ready_port, block_ready).status());

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      channel, ChannelDirection::kReceive, *input.data_port, *input.valid_port,
      *input.ready_port, /*stage=*/std::nullopt));

  return absl::OkStatus();
}

// Stitch two ends of a single value channels together.
absl::Status StitchSingleValueChannel(Block* container,
                                      SingleValueChannel* channel,
                                      const GlobalChannelMap& channel_map) {
  std::optional<BlockInstantiation*> input_instantiation =
      channel_map.GetInputInstantiation(channel);
  std::optional<BlockInstantiation*> output_instantiation =
      channel_map.GetOutputInstantiation(channel);

  if (!input_instantiation.has_value() && !output_instantiation.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not find an input or an output for channel %s.",
                        channel->name()));
  }

  Node* input_node = nullptr;
  if (output_instantiation.has_value()) {
    const ChannelPortMetadata& output =
        channel_map.single_value_output_channels().at(channel);
    XLS_RET_CHECK(output.data_port.has_value());
    XLS_ASSIGN_OR_RETURN(input_node, container->MakeNode<InstantiationOutput>(
                                         SourceInfo(), *output_instantiation,
                                         *output.data_port));
  } else {
    const ChannelPortMetadata& input =
        channel_map.single_value_input_channels().at(channel);
    XLS_RET_CHECK(input.data_port.has_value());
    XLS_ASSIGN_OR_RETURN(input_node,
                         container->AddInputPort(*input.data_port, input.type));
    XLS_RETURN_IF_ERROR(container->AddChannelPortMetadata(
        channel, ChannelDirection::kReceive, *input.data_port,
        /*valid_port=*/std::nullopt, /*ready_port=*/std::nullopt,
        /*stage=*/std::nullopt));
  }

  if (input_instantiation.has_value()) {
    const ChannelPortMetadata& input =
        channel_map.single_value_input_channels().at(channel);
    return container
        ->MakeNode<InstantiationInput>(SourceInfo(), input_node,
                                       *input_instantiation, *input.data_port)
        .status();
  }

  const ChannelPortMetadata& output =
      channel_map.single_value_output_channels().at(channel);
  XLS_RET_CHECK(output.data_port.has_value());
  XLS_RETURN_IF_ERROR(
      container->AddOutputPort(*output.data_port, input_node).status());
  return container->AddChannelPortMetadata(
      channel, ChannelDirection::kSend, *output.data_port,
      /*valid_port=*/std::nullopt, /*ready_port=*/std::nullopt,
      /*stage=*/std::nullopt);
}

// Stitch two ends of a streaming channel together.
absl::Status StitchStreamingChannel(Block* container, StreamingChannel* channel,
                                    const GlobalChannelMap& channel_map) {
  std::optional<BlockInstantiation*> input_instantiation =
      channel_map.GetInputInstantiation(channel);
  std::optional<BlockInstantiation*> output_instantiation =
      channel_map.GetOutputInstantiation(channel);

  if (!input_instantiation.has_value() && !output_instantiation.has_value()) {
    VLOG(3) << "Saw loopback channel " << channel->name()
            << ", no stitching required.";
    return absl::OkStatus();
  }

  // Instantiate FIFO and the two blocks.
  if (input_instantiation.has_value() && output_instantiation.has_value()) {
    std::string inst_name = absl::StrCat("fifo_", channel->name());
    XLS_RET_CHECK(channel->channel_config().fifo_config().has_value())
        << absl::StreamFormat("Channel %s has no fifo config.",
                              channel->name());
    XLS_ASSIGN_OR_RETURN(
        xls::FifoInstantiation * fifo,
        container->AddFifoInstantiation(
            inst_name, *channel->channel_config().fifo_config(),
            channel->type(), channel->name()));
    XLS_RET_CHECK(container->GetResetPort().has_value());
    XLS_RETURN_IF_ERROR(container
                            ->MakeNode<xls::InstantiationInput>(
                                SourceInfo(), *container->GetResetPort(), fifo,
                                FifoInstantiation::kResetPortName)
                            .status());
    XLS_RETURN_IF_ERROR(StitchStreamingOutputToFifo(
        container, *output_instantiation, fifo,
        channel_map.streaming_output_channels().at(channel)));
    XLS_RETURN_IF_ERROR(StitchStreamingInputToFifo(
        container, *input_instantiation, fifo,
        channel_map.streaming_input_channels().at(channel)));
  } else if (input_instantiation.has_value() &&
             !output_instantiation.has_value()) {
    XLS_RETURN_IF_ERROR(ExposeStreamingInput(
        channel, channel_map.streaming_input_channels().at(channel), container,
        *input_instantiation));
  } else if (!input_instantiation.has_value() &&
             output_instantiation.has_value()) {
    XLS_RETURN_IF_ERROR(ExposeStreamingOutput(
        channel, channel_map.streaming_output_channels().at(channel), container,
        *output_instantiation));
  }
  return absl::OkStatus();
}

absl::Status StitchChannel(Block* container, Channel* channel,
                           const GlobalChannelMap& channel_map) {
  switch (channel->kind()) {
    case ChannelKind::kStreaming:
      return StitchStreamingChannel(
          container, down_cast<StreamingChannel*>(channel), channel_map);
    case ChannelKind::kSingleValue:
      return StitchSingleValueChannel(
          container, down_cast<SingleValueChannel*>(channel), channel_map);
  }
}

// Stitch all blocks in the container block together, punching external
// sends/receives through the container.
absl::Status StitchBlocks(Package* package,
                          const verilog::CodegenOptions& options) {
  XLS_ASSIGN_OR_RETURN(Block * top_block, package->GetTopAsBlock());
  XLS_RETURN_IF_ERROR(InstantiateBlocksInContainer(top_block, options));

  XLS_ASSIGN_OR_RETURN(GlobalChannelMap channel_map,
                       GlobalChannelMap::Create(package));
  std::vector<Channel*> channels_sorted_by_name;
  channels_sorted_by_name.reserve(package->channels().size());
  for (Channel* channel : package->channels()) {
    channels_sorted_by_name.push_back(channel);
  }
  absl::c_sort(channels_sorted_by_name, Channel::NameLessThan);

  for (Channel* channel : channels_sorted_by_name) {
    XLS_RETURN_IF_ERROR(StitchChannel(top_block, channel, channel_map));
  }

  return absl::OkStatus();
}

// Adds a block with the given name to the package, renaming an existing block
// that has the same name if one exists. Also, update the metadata for the
// existing block if it is renamed.
absl::StatusOr<Block*> AddBlockWithName(Package* package,
                                        std::string_view name) {
  NameUniquer uniquer(/*separator=*/"__", /*reserved_names=*/{});
  Block* block_with_name = nullptr;
  std::vector<Block*> blocks_sorted_by_name;
  blocks_sorted_by_name.reserve(package->blocks().size());
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    blocks_sorted_by_name.push_back(block.get());
  }
  absl::c_sort(blocks_sorted_by_name, FunctionBase::NameLessThan);

  for (Block* block : blocks_sorted_by_name) {
    std::string new_name = uniquer.GetSanitizedUniqueName(block->name());
    XLS_RET_CHECK_EQ(block->name(), new_name);
    if (new_name == name) {
      XLS_RET_CHECK_EQ(block_with_name, nullptr);
      block_with_name = block;
    }
  }

  if (block_with_name != nullptr) {
    std::string new_name = uniquer.GetSanitizedUniqueName(name);
    XLS_RET_CHECK_NE(block_with_name->name(), new_name);
    block_with_name->SetName(new_name);
    XLS_RET_CHECK(block_with_name->IsScheduled());
  }

  auto new_block = std::make_unique<ScheduledBlock>(name, package);
  return package->AddBlock(std::move(new_block));
}

}  // namespace

absl::StatusOr<bool> GlobalChannelBlockStitchingPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  // No need to stitch blocks when we don't have 2+ blocks, channels are
  // proc-scoped, or the top is not a proc.
  std::vector<std::pair<ScheduledBlock*, Proc*>> blocks_with_proc_sources =
      GetScheduledBlocksWithProcSources(package);
  const bool proc_scoped_channels = absl::c_any_of(
      blocks_with_proc_sources,
      [](const auto& pair) { return pair.second->is_new_style_proc(); });
  const bool proc_top = absl::c_any_of(
      blocks_with_proc_sources,
      [&](const auto& pair) { return package->IsTop(pair.first); });
  if (package->blocks().size() < 2 || proc_scoped_channels || !proc_top) {
    return false;
  }

  // TODO: google/xls#1336 - support this codegen option.
  if (options.codegen_options.split_outputs()) {
    return absl::UnimplementedError(
        "Splitting outputs is not supported by block stitching.");
  }

  XLS_ASSIGN_OR_RETURN(Block * original_top_block, package->GetTopAsBlock());
  std::string top_block_name(options.codegen_options.module_name().value_or(
      original_top_block->name()));

  XLS_ASSIGN_OR_RETURN(Block * top_block,
                       AddBlockWithName(package, top_block_name));

  XLS_RETURN_IF_ERROR(package->SetTop(top_block));
  XLS_RETURN_IF_ERROR(top_block->AddClockPort("clk"));
  XLS_RETURN_IF_ERROR(MaybeAddResetPort(top_block, options.codegen_options));

  VLOG(2) << "Stitching blocks for " << top_block->name();
  XLS_RETURN_IF_ERROR(StitchBlocks(package, options.codegen_options));

  return true;
}

}  // namespace xls::codegen

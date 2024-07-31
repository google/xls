// Copyright 2024 The XLS Authors
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

#include "xls/codegen/block_stitching_pass.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"

namespace xls::verilog {
namespace {

// Instantiate all blocks in the package in the container block.
// Returns a map that identifies each block with its instantiation in the
// container block.
absl::StatusOr<absl::flat_hash_map<Block*, ::xls::Instantiation*>>
InstantiateBlocksInContainer(
    Block* container,
    const std::optional<::xls::verilog::ResetProto>& reset_proto) {
  std::vector<Block*> blocks_to_instantiate;
  for (std::unique_ptr<Block>& block : container->package()->blocks()) {
    if (block.get() == container) {
      continue;
    }
    blocks_to_instantiate.push_back(block.get());
  }
  absl::c_sort(blocks_to_instantiate, &FunctionBase::NameLessThan);
  // Instantiate all blocks in the container block.
  absl::flat_hash_map<Block*, ::xls::Instantiation*> instantiations;
  for (Block* block : blocks_to_instantiate) {
    std::string inst_name =
        absl::StrCat(block->name(), "_inst", instantiations.size());
    XLS_ASSIGN_OR_RETURN(
        ::xls::Instantiation * instantiation,
        container->AddInstantiation(
            inst_name,
            std::make_unique<xls::BlockInstantiation>(inst_name, block)));
    instantiations.insert({block, instantiation});
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
      XLS_RET_CHECK(!reset_proto.has_value());
    }
  }
  return instantiations;
}

// Stitch a streaming output (i.e. a send on a streaming channel) to its
// associated FIFO.
absl::Status StitchStreamingOutputToFifo(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, xls::Instantiation* fifo_inst,
    const StreamingOutput* output) {
  Block* instantiated_block =
      output->port.value()->function_base()->AsBlockOrDie();
  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, output->port.value()->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_valid,
      block->MakeNode<xls::InstantiationOutput>(SourceInfo(), block_inst,
                                                output->port_valid->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * fifo_ready,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), fifo_inst, FifoInstantiation::kPushReadyPortName));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_data, fifo_inst,
                              FifoInstantiation::kPushDataPortName)
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_valid, fifo_inst,
                              FifoInstantiation::kPushValidPortName)
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), fifo_ready, block_inst,
                              output->port_ready->GetName())
                          .status());
  return absl::OkStatus();
}

// Stitch a streaming input (i.e. a receive on a streaming channel) to its
// associated FIFO.
absl::Status StitchStreamingInputToFifo(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, xls::Instantiation* fifo_inst, const StreamingInput* input) {
  Block* instantiated_block =
      input->port.value()->function_base()->AsBlockOrDie();
  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * fifo_data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), fifo_inst, FifoInstantiation::kPopDataPortName));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * fifo_valid,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), fifo_inst, FifoInstantiation::kPopValidPortName));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_ready,
      block->MakeNode<xls::InstantiationOutput>(SourceInfo(), block_inst,
                                                input->port_ready->GetName()));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), fifo_data, block_inst,
                              input->port.value()->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), fifo_valid, block_inst,
                              input->port_valid->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_ready, fifo_inst, "pop_ready")
                          .status());
  return absl::OkStatus();
}

// Directly stitch a streaming output (i.e. a send on a streaming channel) to
// its associated input (i.e. a receive on a streaming channel). This is used
// when the FIFO configuration has depth 0.
absl::Status StitchStreamingOutputToInput(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, const StreamingInput* input, const StreamingOutput* output) {
  Block* in_block = input->port.value()->function_base()->AsBlockOrDie();
  Block* out_block = output->port.value()->function_base()->AsBlockOrDie();
  auto in_itr = instantiations.find(in_block);
  if (in_itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *in_block));
  }
  auto out_itr = instantiations.find(out_block);
  if (out_itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *out_block));
  }
  xls::Instantiation* in_inst = in_itr->second;
  xls::Instantiation* out_inst = out_itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), out_inst, output->port.value()->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * valid,
      block->MakeNode<xls::InstantiationOutput>(SourceInfo(), out_inst,
                                                output->port_valid->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * ready,
      block->MakeNode<xls::InstantiationOutput>(SourceInfo(), in_inst,
                                                input->port_ready->GetName()));
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(SourceInfo(), data, in_inst,
                                              input->port.value()->GetName())
          .status());
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(SourceInfo(), valid, in_inst,
                                              input->port_valid->GetName())
          .status());
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(SourceInfo(), ready, out_inst,
                                              output->port_ready->GetName())
          .status());
  return absl::OkStatus();
}

// Punch a streaming output through the container block. This happens when a
// block has an external send on a streaming channel.
absl::Status ExposeStreamingOutput(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, const StreamingOutput* output) {
  VLOG(5) << "Exposing output port: " << output->port.value()->GetName();
  VLOG(5) << "Exposing output port valid: " << output->port_valid->GetName();
  VLOG(5) << "Exposing output port ready: " << output->port_ready->GetName();
  output->channel->SetBlockName(block->name());

  output->channel->SetDataPortName(output->port.value()->GetName());
  output->channel->SetValidPortName(output->port_valid->GetName());
  output->channel->SetReadyPortName(output->port_ready->GetName());

  Block* instantiated_block =
      output->port.value()->function_base()->AsBlockOrDie();
  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, output->port.value()->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_valid,
      block->MakeNode<xls::InstantiationOutput>(SourceInfo(), block_inst,
                                                output->port_valid->GetName()));
  XLS_ASSIGN_OR_RETURN(xls::Node * ext_ready,
                       block->AddInputPort(output->port_ready->GetName(),
                                           output->port_ready->GetType()));
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(output->port.value()->GetName(), block_data)
          .status());
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(output->port_valid->GetName(), block_valid)
          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), ext_ready, block_inst,
                              output->port_ready->GetName())
                          .status());
  return absl::OkStatus();
}

// Punch a streaming input from the container block down. This happens when a
// block has an external receive on a streaming channel.
absl::Status ExposeStreamingInput(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, const StreamingInput* input) {
  input->channel->SetDataPortName(input->port.value()->GetName());
  input->channel->SetValidPortName(input->port_valid->GetName());
  input->channel->SetReadyPortName(input->port_ready->GetName());
  input->channel->SetBlockName(block->name());

  Block* instantiated_block =
      input->port.value()->function_base()->AsBlockOrDie();

  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(xls::Node * ext_data,
                       block->AddInputPort(input->port.value()->GetName(),
                                           input->port.value()->GetType()));
  XLS_ASSIGN_OR_RETURN(xls::Node * ext_valid,
                       block->AddInputPort(input->port_valid->GetName(),
                                           input->port_valid->GetType()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_ready,
      block->MakeNode<xls::InstantiationOutput>(SourceInfo(), block_inst,
                                                input->port_ready->GetName()));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), ext_data, block_inst,
                              input->port.value()->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<xls::InstantiationInput>(
              SourceInfo(), ext_valid, block_inst, input->port_valid->GetName())
          .status());
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(input->port_ready->GetName(), block_ready).status());
  return absl::OkStatus();
}

// Stitch two ends of a single value channels together.
absl::Status StitchSingleValueChannel(
    Block* container, SingleValueChannel* channel,
    const ChannelMap& channel_map,
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations) {
  auto input_iter = channel_map.channel_to_single_value_input().find(channel);
  auto output_iter = channel_map.channel_to_single_value_output().find(channel);
  bool has_input =
      input_iter != channel_map.channel_to_single_value_input().end();
  bool has_output =
      output_iter != channel_map.channel_to_single_value_output().end();
  const SingleValueInput* subblock_input =
      has_input ? input_iter->second : nullptr;
  const SingleValueOutput* subblock_output =
      has_output ? output_iter->second : nullptr;

  if (!has_input && !has_output) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not find an input or an output for channel %s.",
                        channel->name()));
  }

  Node* input_node = nullptr;
  if (has_output) {
    auto subblock_inst_iter = instantiations.find(
        subblock_output->port->function_base()->AsBlockOrDie());
    if (subblock_inst_iter == instantiations.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not find inst for ", *subblock_output->port->function_base()));
    }
    XLS_ASSIGN_OR_RETURN(input_node,
                         container->MakeNode<InstantiationOutput>(
                             SourceInfo(), subblock_inst_iter->second,
                             subblock_output->port->GetName()));
  } else {
    XLS_ASSIGN_OR_RETURN(
        input_node, container->AddInputPort(subblock_input->port->GetName(),
                                            subblock_input->port->GetType()));
  }
  if (has_input) {
    auto subblock_inst_iter = instantiations.find(
        subblock_input->port->function_base()->AsBlockOrDie());
    if (subblock_inst_iter == instantiations.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not find inst for ", *subblock_input->port->function_base()));
    }
    return container
        ->MakeNode<InstantiationInput>(SourceInfo(), input_node,
                                       subblock_inst_iter->second,
                                       subblock_input->port->GetName())
        .status();
  }
  return container->AddOutputPort(subblock_output->port->GetName(), input_node)
      .status();
}

// Stitch two ends of a streaming channel together.
absl::Status StitchStreamingChannel(
    Block* container, StreamingChannel* channel, const ChannelMap& channel_map,
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations) {
  auto input_iter = channel_map.channel_to_streaming_input().find(channel);
  auto output_iter = channel_map.channel_to_streaming_output().find(channel);
  bool has_input = input_iter != channel_map.channel_to_streaming_input().end();
  bool has_output =
      output_iter != channel_map.channel_to_streaming_output().end();
  const StreamingInput* input = has_input ? input_iter->second : nullptr;
  const StreamingOutput* output = has_output ? output_iter->second : nullptr;

  if (!has_input && !has_output) {
    VLOG(3) << "Saw loopback channel " << channel->name()
            << ", no stitching required.";
    return absl::OkStatus();
  }

  // Instantiate FIFO and the two blocks.
  if (has_input && has_output) {
    std::string inst_name = absl::StrCat("fifo_", channel->name());
    XLS_RET_CHECK(channel->fifo_config().has_value()) << absl::StreamFormat(
        "Channel %s has no fifo config.", channel->name());
    if (channel->fifo_config()->depth() > 0) {
      XLS_ASSIGN_OR_RETURN(
          xls::Instantiation * instantiation,
          container->AddInstantiation(
              inst_name,
              std::make_unique<xls::FifoInstantiation>(
                  inst_name, *channel->fifo_config(), channel->type(),
                  channel->name(), container->package())));
      XLS_RET_CHECK(container->GetResetPort().has_value());
      XLS_RETURN_IF_ERROR(container
                              ->MakeNode<xls::InstantiationInput>(
                                  SourceInfo(), *container->GetResetPort(),
                                  instantiation,
                                  FifoInstantiation::kResetPortName)
                              .status());
      XLS_RETURN_IF_ERROR(StitchStreamingOutputToFifo(instantiations, container,
                                                      instantiation, output));
      XLS_RETURN_IF_ERROR(StitchStreamingInputToFifo(instantiations, container,
                                                     instantiation, input));
    } else {
      XLS_RETURN_IF_ERROR(StitchStreamingOutputToInput(
          instantiations, container, input, output));
    }
  } else if (has_input && !has_output) {
    XLS_RETURN_IF_ERROR(ExposeStreamingInput(instantiations, container, input));
  } else if (!has_input && has_output) {
    XLS_RETURN_IF_ERROR(
        ExposeStreamingOutput(instantiations, container, output));
  }
  return absl::OkStatus();
}

absl::Status StitchChannel(
    Block* container, Channel* channel, const ChannelMap& channel_map,
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations) {
  switch (channel->kind()) {
    case ChannelKind::kStreaming:
      return StitchStreamingChannel(container,
                                    down_cast<StreamingChannel*>(channel),
                                    channel_map, instantiations);
    case ChannelKind::kSingleValue:
      return StitchSingleValueChannel(container,
                                      down_cast<SingleValueChannel*>(channel),
                                      channel_map, instantiations);
  }
}

// Stitch all blocks in the container block together, punching external
// sends/receives through the container.
absl::Status StitchBlocks(CodegenPassUnit& unit,
                          const CodegenOptions& options) {
  VLOG(2) << "Stitching blocks for " << unit.top_block->name();
  auto channel_map = ChannelMap::Create(unit);
  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<Block*, ::xls::Instantiation*> instantiations),
      InstantiateBlocksInContainer(unit.top_block, options.reset()));
  std::vector<Channel*> channels_sorted_by_name;
  channels_sorted_by_name.reserve(unit.package->channels().size());
  for (Channel* channel : unit.package->channels()) {
    channels_sorted_by_name.push_back(channel);
  }
  absl::c_sort(channels_sorted_by_name, Channel::NameLessThan);

  for (Channel* channel : channels_sorted_by_name) {
    XLS_RETURN_IF_ERROR(
        StitchChannel(unit.top_block, channel, channel_map, instantiations));
  }

  return absl::OkStatus();
}

// Adds a block with the given name to the package, renaming an existing block
// that has the same name if one exists. Also, update the metadata for the
// existing block if it is renamed.
absl::StatusOr<Block*> AddBlockWithName(
    Package* package, std::string_view name,
    CodegenPassUnit::MetadataMap& metadata_map) {
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
    auto metadata = metadata_map.extract(block_with_name);
    block_with_name->SetName(new_name);
    if (metadata) {
      metadata_map.insert({block_with_name, std::move(metadata.mapped())});
    }
  }
  return package->AddBlock(std::make_unique<Block>(name, package));
}
}  // namespace

absl::StatusOr<bool> BlockStitchingPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  // No need to stitch blocks when we don't have 2+ blocks.
  if (unit->package->blocks().size() < 2) {
    return false;
  }

  // TODO: google/xls#1336 - support these codegen options.
  if (options.codegen_options.add_idle_output()) {
    return absl::UnimplementedError(
        "Idle output is not supported by block stitching.");
  }
  if (options.codegen_options.split_outputs()) {
    return absl::UnimplementedError(
        "Splitting outputs is not supported by block stitching.");
  }
  std::string top_block_name(options.codegen_options.module_name().value_or(
      SanitizeIdentifier(unit->name())));

  XLS_ASSIGN_OR_RETURN(
      unit->top_block,
      AddBlockWithName(unit->package, top_block_name, unit->metadata));

  // Insert metadata for the new container block.
  unit->metadata.insert({unit->top_block, CodegenMetadata{}});

  XLS_RETURN_IF_ERROR(unit->top_block->AddClockPort("clk"));
  if (options.codegen_options.reset().has_value()) {
    XLS_RET_CHECK_OK(
        unit->top_block->AddResetPort(options.codegen_options.reset()->name()));
  }

  XLS_RETURN_IF_ERROR(StitchBlocks(*unit, options.codegen_options));
  return true;
}

}  // namespace xls::verilog

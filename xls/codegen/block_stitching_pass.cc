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
#include "xls/codegen/conversion_utils.h"
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
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

// Map from channel to block inputs/outputs.
class ChannelMap {
 public:
  using StreamingInputMap =
      absl::flat_hash_map<Channel*, const StreamingInput*>;
  using StreamingOutputMap =
      absl::flat_hash_map<Channel*, const StreamingOutput*>;
  using SingleValueInputMap =
      absl::flat_hash_map<Channel*, const SingleValueInput*>;
  using SingleValueOutputMap =
      absl::flat_hash_map<Channel*, const SingleValueOutput*>;

  // Populate mapping from channel to block inputs/outputs for all blocks.
  static ChannelMap Create(const CodegenContext& context) {
    ChannelMap::StreamingInputMap channel_to_streaming_input;
    ChannelMap::StreamingOutputMap channel_to_streaming_output;
    ChannelMap::SingleValueInputMap channel_to_single_value_input;
    ChannelMap::SingleValueOutputMap channel_to_single_value_output;

    for (auto& [block, metadata] : context.metadata()) {
      for (const std::vector<StreamingInput>& inputs :
           metadata.streaming_io_and_pipeline.inputs) {
        for (const StreamingInput& input : inputs) {
          VLOG(5) << absl::StreamFormat("Input found on %v for %s", *block,
                                        input.GetChannelName());
          if (!input.IsExternal()) {
            VLOG(5) << absl::StreamFormat("Skipping internal input %s",
                                          input.GetChannelName());
            continue;
          }
          Channel* channel =
              block->package()->GetChannel(input.GetChannelName()).value();
          channel_to_streaming_input[channel] = &input;
        }
      }
      for (const SingleValueInput& input :
           metadata.streaming_io_and_pipeline.single_value_inputs) {
        VLOG(5) << absl::StreamFormat("Input found on %v for %s", *block,
                                      ChannelRefName(input.GetChannel()));
        channel_to_single_value_input[std::get<Channel*>(input.GetChannel())] =
            &input;
      }
      for (const std::vector<StreamingOutput>& outputs :
           metadata.streaming_io_and_pipeline.outputs) {
        for (const StreamingOutput& output : outputs) {
          VLOG(5) << absl::StreamFormat("Output found on %v for %s.", *block,
                                        output.GetChannelName());
          if (!output.IsExternal()) {
            VLOG(5) << absl::StreamFormat("Skipping internal output %s",
                                          output.GetChannelName());
            continue;
          }
          Channel* channel =
              block->package()->GetChannel(output.GetChannelName()).value();
          channel_to_streaming_output[channel] = &output;
        }
      }
      for (const SingleValueOutput& output :
           metadata.streaming_io_and_pipeline.single_value_outputs) {
        VLOG(5) << absl::StreamFormat("Output found on %v for %s.", *block,
                                      ChannelRefName(output.GetChannel()));
        channel_to_single_value_output[std::get<Channel*>(
            output.GetChannel())] = &output;
      }
    }
    return ChannelMap(std::move(channel_to_streaming_input),
                      std::move(channel_to_streaming_output),
                      std::move(channel_to_single_value_input),
                      std::move(channel_to_single_value_output));
  }

  const StreamingInputMap& channel_to_streaming_input() const {
    return channel_to_streaming_input_;
  }
  const StreamingOutputMap& channel_to_streaming_output() const {
    return channel_to_streaming_output_;
  }
  const SingleValueInputMap& channel_to_single_value_input() const {
    return channel_to_single_value_input_;
  }
  const SingleValueOutputMap& channel_to_single_value_output() const {
    return channel_to_single_value_output_;
  }

 private:
  ChannelMap(StreamingInputMap&& channel_to_streaming_input,
             StreamingOutputMap&& channel_to_streaming_output,
             SingleValueInputMap&& channel_to_single_value_input,
             SingleValueOutputMap&& channel_to_single_value_output)
      : channel_to_streaming_input_(std::move(channel_to_streaming_input)),
        channel_to_streaming_output_(std::move(channel_to_streaming_output)),
        channel_to_single_value_input_(
            std::move(channel_to_single_value_input)),
        channel_to_single_value_output_(
            std::move(channel_to_single_value_output)) {}
  StreamingInputMap channel_to_streaming_input_;
  StreamingOutputMap channel_to_streaming_output_;
  SingleValueInputMap channel_to_single_value_input_;
  SingleValueOutputMap channel_to_single_value_output_;
};

// Instantiate all blocks in the package in the container block.
// Returns a map that identifies each block with its instantiation in the
// container block.
absl::StatusOr<absl::flat_hash_map<Block*, ::xls::Instantiation*>>
InstantiateBlocksInContainer(Block* container, const CodegenOptions& options) {
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
  return instantiations;
}

// Stitch a streaming output (i.e. a send on a streaming channel) to its
// associated FIFO.
absl::Status StitchStreamingOutputToFifo(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, xls::Instantiation* fifo_inst,
    const StreamingOutput* output) {
  Block* instantiated_block =
      output->GetDataPort().value()->function_base()->AsBlockOrDie();
  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, output->GetDataPort().value()->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_valid,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, output->GetValidPort()->GetName()));
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
                              output->GetReadyPort()->GetName())
                          .status());
  return absl::OkStatus();
}

// Stitch a streaming input (i.e. a receive on a streaming channel) to its
// associated FIFO.
absl::Status StitchStreamingInputToFifo(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, xls::Instantiation* fifo_inst, const StreamingInput* input) {
  Block* instantiated_block =
      input->GetDataPort().value()->function_base()->AsBlockOrDie();
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
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, input->GetReadyPort()->GetName()));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), fifo_data, block_inst,
                              input->GetDataPort().value()->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), fifo_valid, block_inst,
                              input->GetValidPort()->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block_ready, fifo_inst, "pop_ready")
                          .status());
  return absl::OkStatus();
}

// Punch a streaming output through the container block. This happens when a
// block has an external send on a streaming channel.
absl::Status ExposeStreamingOutput(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, const StreamingOutput* output) {
  VLOG(5) << "Exposing output port: "
          << output->GetDataPort().value()->GetName();
  VLOG(5) << "Exposing output port valid: "
          << output->GetValidPort()->GetName();
  VLOG(5) << "Exposing output port ready: "
          << output->GetReadyPort()->GetName();

  Block* instantiated_block =
      output->GetDataPort().value()->function_base()->AsBlockOrDie();
  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_data,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, output->GetDataPort().value()->GetName()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_valid,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, output->GetValidPort()->GetName()));
  XLS_ASSIGN_OR_RETURN(xls::Node * ext_ready,
                       block->AddInputPort(output->GetReadyPort()->GetName(),
                                           output->GetReadyPort()->GetType()));
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(output->GetDataPort().value()->GetName(), block_data)
          .status());
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(output->GetValidPort()->GetName(), block_valid)
          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), ext_ready, block_inst,
                              output->GetReadyPort()->GetName())
                          .status());

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      output->GetChannel(), ChannelDirection::kSend,
      output->GetDataPort().value()->GetName(),
      output->GetValidPort()->GetName(), output->GetReadyPort()->GetName()));

  return absl::OkStatus();
}

// Punch a streaming input from the container block down. This happens when a
// block has an external receive on a streaming channel.
absl::Status ExposeStreamingInput(
    const absl::flat_hash_map<Block*, ::xls::Instantiation*>& instantiations,
    Block* block, const StreamingInput* input) {
  Block* instantiated_block =
      input->GetDataPort().value()->function_base()->AsBlockOrDie();

  auto itr = instantiations.find(instantiated_block);
  if (itr == instantiations.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find inst for ", *instantiated_block));
  }
  xls::Instantiation* block_inst = itr->second;
  XLS_ASSIGN_OR_RETURN(
      xls::Node * ext_data,
      block->AddInputPort(input->GetDataPort().value()->GetName(),
                          input->GetDataPort().value()->GetType()));
  XLS_ASSIGN_OR_RETURN(xls::Node * ext_valid,
                       block->AddInputPort(input->GetValidPort()->GetName(),
                                           input->GetValidPort()->GetType()));
  XLS_ASSIGN_OR_RETURN(
      xls::Node * block_ready,
      block->MakeNode<xls::InstantiationOutput>(
          SourceInfo(), block_inst, input->GetReadyPort()->GetName()));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), ext_data, block_inst,
                              input->GetDataPort().value()->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), ext_valid, block_inst,
                              input->GetValidPort()->GetName())
                          .status());
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(input->GetReadyPort()->GetName(), block_ready)
          .status());

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      input->GetChannel(), ChannelDirection::kReceive,
      input->GetDataPort().value()->GetName(), input->GetValidPort()->GetName(),
      input->GetReadyPort()->GetName()));

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
    std::optional<OutputPort*> output_port = subblock_output->GetDataPort();
    XLS_RET_CHECK(output_port.has_value());
    auto subblock_inst_iter = instantiations.find(
        output_port.value()->function_base()->AsBlockOrDie());
    if (subblock_inst_iter == instantiations.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not find inst for ", *output_port.value()->function_base()));
    }
    XLS_ASSIGN_OR_RETURN(input_node,
                         container->MakeNode<InstantiationOutput>(
                             SourceInfo(), subblock_inst_iter->second,
                             output_port.value()->GetName()));
  } else {
    std::optional<InputPort*> input_port = subblock_input->GetDataPort();
    XLS_RET_CHECK(input_port.has_value());
    XLS_ASSIGN_OR_RETURN(
        input_node, container->AddInputPort(input_port.value()->GetName(),
                                            input_port.value()->GetType()));
    XLS_RETURN_IF_ERROR(container->AddChannelPortMetadata(
        channel, ChannelDirection::kReceive, input_port.value()->GetName(),
        /*valid_port=*/std::nullopt, /*ready_port=*/std::nullopt));
  }
  if (has_input) {
    std::optional<InputPort*> input_port = subblock_input->GetDataPort();
    XLS_RET_CHECK(input_port.has_value());
    auto subblock_inst_iter = instantiations.find(
        input_port.value()->function_base()->AsBlockOrDie());
    if (subblock_inst_iter == instantiations.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not find inst for ", *input_port.value()->function_base()));
    }
    return container
        ->MakeNode<InstantiationInput>(SourceInfo(), input_node,
                                       subblock_inst_iter->second,
                                       input_port.value()->GetName())
        .status();
  }
  std::optional<OutputPort*> output_port = subblock_output->GetDataPort();
  XLS_RET_CHECK(output_port.has_value());
  XLS_RETURN_IF_ERROR(
      container->AddOutputPort(output_port.value()->GetName(), input_node)
          .status());
  return container->AddChannelPortMetadata(
      channel, ChannelDirection::kSend, output_port.value()->GetName(),
      /*valid_port=*/std::nullopt, /*ready_port=*/std::nullopt);
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
    XLS_RET_CHECK(channel->channel_config().fifo_config().has_value())
        << absl::StreamFormat("Channel %s has no fifo config.",
                              channel->name());
    XLS_ASSIGN_OR_RETURN(
        xls::FifoInstantiation * instantiation,
        container->AddFifoInstantiation(
            inst_name, *channel->channel_config().fifo_config(),
            channel->type(), channel->name()));
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
absl::Status StitchBlocks(Package* package, CodegenContext& unit,
                          const CodegenOptions& options) {
  VLOG(2) << "Stitching blocks for " << unit.top_block()->name();
  auto channel_map = ChannelMap::Create(unit);
  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<Block*, ::xls::Instantiation*> instantiations),
      InstantiateBlocksInContainer(unit.top_block(), options));
  std::vector<Channel*> channels_sorted_by_name;
  channels_sorted_by_name.reserve(package->channels().size());
  for (Channel* channel : package->channels()) {
    channels_sorted_by_name.push_back(channel);
  }
  absl::c_sort(channels_sorted_by_name, Channel::NameLessThan);

  for (Channel* channel : channels_sorted_by_name) {
    XLS_RETURN_IF_ERROR(
        StitchChannel(unit.top_block(), channel, channel_map, instantiations));
  }

  return absl::OkStatus();
}

// Adds a block with the given name to the package, renaming an existing block
// that has the same name if one exists. Also, update the metadata for the
// existing block if it is renamed.
absl::StatusOr<Block*> AddBlockWithName(
    Package* package, std::string_view name,
    CodegenContext::MetadataMap& metadata_map) {
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
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  // No need to stitch blocks when we don't have 2+ blocks or if channels are
  // proc-scoped.
  if (package->blocks().size() < 2 || package->ChannelsAreProcScoped()) {
    return false;
  }

  // TODO: google/xls#1336 - support this codegen option.
  if (options.codegen_options.split_outputs()) {
    return absl::UnimplementedError(
        "Splitting outputs is not supported by block stitching.");
  }
  std::string top_block_name(options.codegen_options.module_name().value_or(
      context.top_block()->name()));

  XLS_ASSIGN_OR_RETURN(
      Block * top_block,
      AddBlockWithName(package, top_block_name, context.metadata()));
  context.SetTopBlock(top_block);

  // Insert metadata for the new container block.
  context.metadata().insert({context.top_block(), CodegenMetadata{}});

  XLS_RETURN_IF_ERROR(context.top_block()->AddClockPort("clk"));
  XLS_RETURN_IF_ERROR(
      MaybeAddResetPort(context.top_block(), options.codegen_options));

  XLS_RETURN_IF_ERROR(StitchBlocks(package, context, options.codegen_options));
  return true;
}

}  // namespace xls::verilog

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

#include "xls/codegen_v_1_5/global_channel_map.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen_v_1_5/block_conversion_utils.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/package.h"

namespace xls::codegen {

absl::StatusOr<GlobalChannelMap> GlobalChannelMap::Create(Package* package) {
  GlobalChannelMap map;
  absl::flat_hash_map<Block*, BlockInstantiation*> instantiations =
      GetInstantiatedBlocks(package);
  for (std::unique_ptr<Block>& block : package->blocks()) {
    if (!instantiations.contains(block.get())) {
      // Block lowered from passthrough proc.
      continue;
    }
    for (const auto& [key, metadata] : block->GetAllChannelPortMetadata()) {
      const auto& [channel_name, channel_direction] = key;
      XLS_ASSIGN_OR_RETURN(Channel * channel,
                           package->GetChannel(channel_name));
      map.block_to_channels_[block.get()].emplace(channel->name(), channel);
      if (channel_direction == ChannelDirection::kReceive) {
        XLS_RETURN_IF_ERROR(map.PopulateReceiveDirection(
            block.get(), instantiations.at(block.get()), channel, metadata));
      } else {
        XLS_RETURN_IF_ERROR(map.PopulateSendDirection(
            block.get(), instantiations.at(block.get()), channel, metadata));
      }
    }
  }
  return std::move(map);
}

absl::Status GlobalChannelMap::PopulateReceiveDirection(
    Block* block, BlockInstantiation* instantiation, Channel* channel,
    const ChannelPortMetadata& metadata) {
  if (channel->kind() == ChannelKind::kStreaming) {
    VLOG(5) << absl::StreamFormat("Input found on %v for %s", *block,
                                  channel->name());

    if (block
            ->GetChannelPortMetadata(metadata.channel_name,
                                     ChannelDirection::kSend)
            .ok()) {
      VLOG(5) << absl::StreamFormat("Skipping internal input %s",
                                    channel->name());
      return absl::OkStatus();
    }
    streaming_input_channels_[channel] = std::move(metadata);
  } else {
    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kSingleValue);
    VLOG(5) << absl::StreamFormat("Input found on %v for %s", *block,
                                  channel->name());
    single_value_input_channels_[channel] = metadata;
  }
  XLS_RET_CHECK(input_instantiations_.emplace(channel, instantiation).second);
  return absl::OkStatus();
}

absl::Status GlobalChannelMap::PopulateSendDirection(
    Block* block, BlockInstantiation* instantiation, Channel* channel,
    const ChannelPortMetadata& metadata) {
  if (channel->kind() == ChannelKind::kStreaming) {
    VLOG(5) << absl::StreamFormat("Output found on %v for %s.", *block,
                                  channel->name());
    if (block
            ->GetChannelPortMetadata(metadata.channel_name,
                                     ChannelDirection::kReceive)
            .ok()) {
      VLOG(5) << absl::StreamFormat("Skipping internal output %s",
                                    channel->name());
      return absl::OkStatus();
    }
    streaming_output_channels_[channel] = std::move(metadata);
  } else {
    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kSingleValue);

    VLOG(5) << absl::StreamFormat("Output found on %v for %s.", *block,
                                  channel->name());
    single_value_output_channels_[channel] = metadata;
  }
  XLS_RET_CHECK(output_instantiations_.emplace(channel, instantiation).second);
  return absl::OkStatus();
}

GlobalChannelMap GlobalChannelMap::GetBlockLevelMap(Block* block) {
  GlobalChannelMap block_level_map;
  const auto it = block_to_channels_.find(block);
  if (it == block_to_channels_.end()) {
    return block_level_map;
  }
  block_level_map.block_to_channels_.emplace(block, it->second);
  for (const auto& [_, channel] : it->second) {
    CopyChannelData(channel, streaming_input_channels_,
                    block_level_map.streaming_input_channels_);
    CopyChannelData(channel, streaming_output_channels_,
                    block_level_map.streaming_output_channels_);
    CopyChannelData(channel, single_value_input_channels_,
                    block_level_map.single_value_input_channels_);
    CopyChannelData(channel, single_value_output_channels_,
                    block_level_map.single_value_output_channels_);
    CopyChannelData(channel, input_instantiations_,
                    block_level_map.input_instantiations_);
    CopyChannelData(channel, output_instantiations_,
                    block_level_map.output_instantiations_);
  }
  return block_level_map;
}

}  // namespace xls::codegen

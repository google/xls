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

#ifndef XLS_CODEGEN_V_1_5_GLOBAL_CHANNEL_MAP_H_
#define XLS_CODEGEN_V_1_5_GLOBAL_CHANNEL_MAP_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"

namespace xls::codegen {

// A map of global channel information, for a package not using proc-scoped
// channels.
class GlobalChannelMap {
 public:
  using StreamingInputMap = absl::flat_hash_map<Channel*, ChannelPortMetadata>;
  using StreamingOutputMap = absl::flat_hash_map<Channel*, ChannelPortMetadata>;
  using SingleValueInputMap =
      absl::flat_hash_map<Channel*, ChannelPortMetadata>;
  using SingleValueOutputMap =
      absl::flat_hash_map<Channel*, ChannelPortMetadata>;
  using BlockInstantiationMap =
      absl::flat_hash_map<Channel*, BlockInstantiation*>;

  // Populates a mapping for all blocks in `package`.
  static absl::StatusOr<GlobalChannelMap> Create(Package* package);

  const StreamingInputMap& streaming_input_channels() const {
    return streaming_input_channels_;
  }
  const StreamingOutputMap& streaming_output_channels() const {
    return streaming_output_channels_;
  }
  const SingleValueInputMap& single_value_input_channels() const {
    return single_value_input_channels_;
  }
  const SingleValueOutputMap& single_value_output_channels() const {
    return single_value_output_channels_;
  }

  std::optional<BlockInstantiation*> GetInputInstantiation(
      Channel* channel) const {
    const auto it = input_instantiations_.find(channel);
    return it == input_instantiations_.end() ? std::nullopt
                                             : std::make_optional(it->second);
  }

  std::optional<BlockInstantiation*> GetOutputInstantiation(
      Channel* channel) const {
    const auto it = output_instantiations_.find(channel);
    return it == output_instantiations_.end() ? std::nullopt
                                              : std::make_optional(it->second);
  }

 private:
  absl::Status PopulateReceiveDirection(Block* block,
                                        BlockInstantiation* instantiation,
                                        Channel* channel,
                                        const ChannelPortMetadata& metadata);

  absl::Status PopulateSendDirection(Block* block,
                                     BlockInstantiation* instantiation,
                                     Channel* channel,
                                     const ChannelPortMetadata& metadata);

  StreamingInputMap streaming_input_channels_;
  StreamingOutputMap streaming_output_channels_;
  SingleValueInputMap single_value_input_channels_;
  SingleValueOutputMap single_value_output_channels_;
  BlockInstantiationMap input_instantiations_;
  BlockInstantiationMap output_instantiations_;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_GLOBAL_CHANNEL_MAP_H_

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

#include "xls/codegen_v_1_5/global_channel_removal_pass.h"

#include <iterator>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

absl::StatusOr<absl::flat_hash_set<Channel*>> GetLivePackageScopedChannels(
    Package* package) {
  absl::flat_hash_set<Channel*> channels;
  for (FunctionBase* fb : package->GetFunctionBases()) {
    for (Node* node : fb->nodes()) {
      if (!node->Is<ChannelNode>()) {
        continue;
      }
      ChannelNode* channel_node = node->As<ChannelNode>();
      XLS_ASSIGN_OR_RETURN(ChannelRef channel, channel_node->GetChannelRef());
      if (std::holds_alternative<Channel*>(channel)) {
        channels.insert(std::get<Channel*>(channel));
      }
    }
  }
  return channels;
}

}  // namespace

absl::StatusOr<bool> GlobalChannelRemovalPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Channel*> live_channels,
                       GetLivePackageScopedChannels(package));
  std::vector<Channel*> channels_to_remove;
  absl::c_copy_if(
      package->channels(), std::back_inserter(channels_to_remove),
      [&](Channel* channel) { return !live_channels.contains(channel); });
  for (Channel* channel : channels_to_remove) {
    XLS_RETURN_IF_ERROR(package->RemoveChannel(channel));
  }
  return !channels_to_remove.empty();
}

}  // namespace xls::codegen

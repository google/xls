// Copyright 2020 The XLS Authors
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

#include "xls/jit/jit_channel_queue.h"

#include "absl/memory/memory.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"

namespace xls {

absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
JitChannelQueueManager::Create(Package* package) {
  auto queue_mgr = absl::WrapUnique(new JitChannelQueueManager(package));
  XLS_RETURN_IF_ERROR(queue_mgr->Init());
  return queue_mgr;
}

JitChannelQueueManager::JitChannelQueueManager(Package* package)
    : package_(package) {}

absl::Status JitChannelQueueManager::Init() {
  for (const auto& chan : package_->channels()) {
    queues_.insert({chan->id(), std::make_unique<JitChannelQueue>(chan->id())});
  }
  return absl::OkStatus();
}

}  // namespace xls

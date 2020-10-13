// Copyright 2020 Google LLC
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
#ifndef XLS_JIT_JIT_CHANNEL_QUEUE_H_
#define XLS_JIT_JIT_CHANNEL_QUEUE_H_

#include "absl/status/statusor.h"
#include "xls/common/integral_types.h"

namespace xls {

// Interface defining the operations necessary for a queue used to back Proc
// communication.
// Very similiar to interpreter/channel_queue.h, as they perform similar
// functions, but for performance, we can't depend on passing XLS Values
// (there's a high cost in marshaling LLVM data into a XLS Value).
// TODO(rspringer): Could templates be used here for some type safety?
class JitChannelQueue {
 public:
  virtual ~JitChannelQueue() {}

  // Called to push data onto this queue/FIFO.
  virtual void Send(uint8* data) = 0;

  // Called to pull data off of this queue/FIFO.
  virtual void Recv(uint8* buffer) = 0;
};

// JitChannelQueue respository. Holds the set of queues known by a given proc.
class JitChannelQueueManager {
 public:
  virtual ~JitChannelQueueManager() {}
  virtual absl::StatusOr<JitChannelQueue*> GetQueueById(int64 id) = 0;
};

}  // namespace xls

#endif  // XLS_JIT_JIT_CHANNEL_QUEUE_H_

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

#include <cstdint>
#include <deque>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/package.h"

namespace xls {

// Interface defining the operations necessary for a queue used to back Proc
// communication.
// Very similiar to interpreter/channel_queue.h, as they perform similar
// functions, but for performance, we can't depend on passing XLS Values
// (there's a high cost in marshaling LLVM data into a XLS Value).
// If the need arises for custom queue implementations, this can be made
// abstract.
// TODO(rspringer): Add data pools to avoid extra memcpy and heap alloc.
class JitChannelQueue {
 public:
  JitChannelQueue(int64_t channel_id) : channel_id_(channel_id) {}

  // Called to push data onto this queue/FIFO.
  void Send(uint8_t* data, int64_t num_bytes) {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
    __msan_unpoison(data, num_bytes);
#endif
    std::unique_ptr<uint8_t[]> buffer;
    if (buffer_pool_.empty()) {
      buffer = std::make_unique<uint8_t[]>(num_bytes);
    } else {
      buffer = std::move(buffer_pool_.back());
      buffer_pool_.pop_back();
    }
    memcpy(buffer.get(), data, num_bytes);
    absl::MutexLock lock(&mutex_);
    the_queue_.push_back(std::move(buffer));
  }

  // Called to pull data off of this queue/FIFO.
  void Recv(uint8_t* buffer, int64_t num_bytes) {
    absl::MutexLock lock(&mutex_);
    memcpy(buffer, the_queue_.front().get(), num_bytes);
    buffer_pool_.push_back(std::move(the_queue_.front()));
    the_queue_.pop_front();
  }

  bool Empty() {
    absl::MutexLock lock(&mutex_);
    return the_queue_.empty();
  }

  int64_t channel_id() { return channel_id_; }

 protected:
  int64_t channel_id_;
  absl::Mutex mutex_;
  std::deque<std::unique_ptr<uint8_t[]>> the_queue_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::unique_ptr<uint8_t[]>> buffer_pool_;
};

// JitChannelQueue respository. Holds the set of queues known by a given proc.
class JitChannelQueueManager {
 public:
  // Returns a JitChannelQueueManager holding a JitChannelQueue for every proc
  // in the provided package.
  static absl::StatusOr<std::unique_ptr<JitChannelQueueManager>> Create(
      Package* package);

  absl::StatusOr<JitChannelQueue*> GetQueueById(int64_t channel_id) {
    XLS_RET_CHECK(queues_.contains(channel_id));
    return queues_.at(channel_id).get();
  }

 private:
  explicit JitChannelQueueManager(Package* package);
  absl::Status Init();

  Package* package_;
  absl::flat_hash_map<int64_t, std::unique_ptr<JitChannelQueue>> queues_;
};

}  // namespace xls

#endif  // XLS_JIT_JIT_CHANNEL_QUEUE_H_

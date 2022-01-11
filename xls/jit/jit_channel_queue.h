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
  explicit JitChannelQueue(int64_t channel_id) : channel_id_(channel_id) {}
  virtual ~JitChannelQueue() = default;

  // Called to push data onto this queue/FIFO.
  virtual void Send(uint8_t* data, int64_t num_bytes) = 0;

  // Called to pull data off of this queue/FIFO.
  virtual void Recv(uint8_t* buffer, int64_t num_bytes) = 0;

  virtual bool Empty() = 0;

  int64_t channel_id() { return channel_id_; }

 protected:
  int64_t channel_id_;
};

// Queue for streaming channels. This queue behaves as an infinite depth FIFO.
class FifoJitChannelQueue : public JitChannelQueue {
 public:
  explicit FifoJitChannelQueue(int64_t channel_id)
      : JitChannelQueue(channel_id) {}

  void Send(uint8_t* data, int64_t num_bytes) override {
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

  void Recv(uint8_t* buffer, int64_t num_bytes) override {
    absl::MutexLock lock(&mutex_);
    memcpy(buffer, the_queue_.front().get(), num_bytes);
    buffer_pool_.push_back(std::move(the_queue_.front()));
    the_queue_.pop_front();
  }

  bool Empty() override {
    absl::MutexLock lock(&mutex_);
    return the_queue_.empty();
  }

  int64_t channel_id() { return channel_id_; }

 protected:
  absl::Mutex mutex_;
  std::deque<std::unique_ptr<uint8_t[]>> the_queue_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::unique_ptr<uint8_t[]>> buffer_pool_;
};

// Queue for single value channels. Unsurprisingly, this queue holds a single
// value. The value is read non-destructively with the Recv method and is
// overwritten via the Send method.
class SingleValueJitChannelQueue : public JitChannelQueue {
 public:
  explicit SingleValueJitChannelQueue(int64_t channel_id)
      : JitChannelQueue(channel_id), buffer_size_(0) {}

  virtual void Send(uint8_t* data, int64_t num_bytes) {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
    __msan_unpoison(data, num_bytes);
#endif
    absl::MutexLock lock(&mutex_);
    if (buffer_ == nullptr) {
      buffer_ = std::make_unique<uint8_t[]>(num_bytes);
      buffer_size_ = num_bytes;
    } else {
      XLS_CHECK_EQ(num_bytes, buffer_size_);
    }
    memcpy(buffer_.get(), data, num_bytes);
  }

  virtual void Recv(uint8_t* buffer, int64_t num_bytes) {
    absl::MutexLock lock(&mutex_);
    XLS_CHECK_NE(buffer_.get(), nullptr);
    XLS_CHECK_EQ(buffer_size_, num_bytes);
    memcpy(buffer, buffer_.get(), num_bytes);
  }

  virtual bool Empty() {
    absl::MutexLock lock(&mutex_);
    return buffer_ != nullptr;
  }

 protected:
  absl::Mutex mutex_;
  int64_t buffer_size_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<uint8_t[]> buffer_ ABSL_GUARDED_BY(mutex_);
};

// JitChannelQueue respository. Holds the set of queues known by a given proc.
class JitChannelQueueManager {
 public:
  // Returns a JitChannelQueueManager holding a JitChannelQueue for every
  // proc in the provided package.
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

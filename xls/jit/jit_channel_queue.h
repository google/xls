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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/channel.h"
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
  virtual void Send(const uint8_t* data, int64_t num_bytes) = 0;

  // Called to pull data off of this queue/FIFO.
  //  Returns
  //   true : The queue is non-empty, and the data at queue front is copied
  //          into buffer.
  //   false: The queue is empty, and the buffer is untouched.
  virtual bool Recv(uint8_t* buffer, int64_t num_bytes) = 0;

  virtual bool Empty() = 0;

  int64_t channel_id() { return channel_id_; }

 protected:
  int64_t channel_id_;
};

// Queue for streaming channels. This queue behaves as an infinite depth FIFO.
// It is preallocated with storage.
class FifoJitChannelQueue : public JitChannelQueue {
 public:
  FifoJitChannelQueue(int64_t channel_id)
      : JitChannelQueue(channel_id),
        channel_element_size_(0),
        allocated_element_size_(0),
        bytes_used_(0),
        enqueue_index_(0),
        dequeue_index_(0) {}

  void Send(const uint8_t* data, int64_t num_bytes) override {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
    __msan_unpoison(data, num_bytes);
#endif
    bool first_call = false;
    if (channel_element_size_ == 0) {
      channel_element_size_ = num_bytes;
      allocated_element_size_ =
          RoundUpToNearest(channel_element_size_,
                           static_cast<int64_t>(alignof(std::max_align_t)));
      first_call = true;
    }
    XLS_CHECK_EQ(num_bytes, channel_element_size_) << absl::StrFormat(
        "Invalid number of bytes given to Send Function of "
        "FifoJitChannelQueue: expected (%s), got (%s).",
        std::to_string(channel_element_size_), std::to_string(num_bytes));

    absl::MutexLock lock(&mutex_);
    // TODO(vmirian): 8-09-2022 Provide the element size at the constructor and
    // remove the following if statement.
    if (first_call) {
      // Align the vector allocation to a power of 2 for efficient utilization
      // of the memory.
      int64_t element_size_2 = 1 << CeilOfLog2(allocated_element_size_);
      if (element_size_2 > kInitBufferSize) {
        circular_buffer_.resize(element_size_2);
      } else {
        circular_buffer_.resize(kInitBufferSize);
      }
      max_byte_count_ =
          FloorOfRatio(static_cast<int64_t>(circular_buffer_.size()),
                       allocated_element_size_) *
          allocated_element_size_;
    }
    // Resize the circular buffer for a new entry when there is insufficient
    // space in the queue.
    if (bytes_used_ == max_byte_count_) {
      circular_buffer_.resize(circular_buffer_.size() * 2);
      max_byte_count_ =
          FloorOfRatio(static_cast<int64_t>(circular_buffer_.size()),
                       allocated_element_size_) *
          allocated_element_size_;
      // The content of the circular buffer must be rearranged when the dequeue
      // index is not at the beginning of the circular buffer to ensure correct
      // ordering.
      if (dequeue_index_ != 0) {
        std::move(circular_buffer_.begin(),
                  circular_buffer_.begin() + dequeue_index_,
                  circular_buffer_.begin() + bytes_used_);
      }
      // Realign the enqueue index to the next available slot.
      enqueue_index_ = bytes_used_ + dequeue_index_;
      if (enqueue_index_ == max_byte_count_) {
        enqueue_index_ = 0;
      }
    }
    memcpy(circular_buffer_.data() + enqueue_index_, data,
           channel_element_size_);
    bytes_used_ += allocated_element_size_;
    enqueue_index_ = enqueue_index_ + allocated_element_size_;
    if (enqueue_index_ == max_byte_count_) {
      enqueue_index_ = 0;
    }
  }

  bool Recv(uint8_t* buffer, int64_t num_bytes) override {
    absl::MutexLock lock(&mutex_);

    if (bytes_used_ == 0) {
      return false;
    }

    XLS_CHECK_EQ(num_bytes, channel_element_size_) << absl::StrFormat(
        "Invalid number of bytes given to Recv Function of "
        "FifoJitChannelQueue: expected (%s), got (%s).",
        std::to_string(channel_element_size_), std::to_string(num_bytes));

    memcpy(buffer, circular_buffer_.data() + dequeue_index_,
           channel_element_size_);
    bytes_used_ -= allocated_element_size_;
    dequeue_index_ = dequeue_index_ + allocated_element_size_;
    if (dequeue_index_ == max_byte_count_) {
      dequeue_index_ = 0;
    }
    return true;
  }

  bool Empty() override {
    absl::MutexLock lock(&mutex_);
    return bytes_used_ == 0;
  }

 protected:
  static constexpr int64_t kInitBufferSize = 128;
  absl::Mutex mutex_;
  // Size of an element in the channel in units of bytes. The producers and
  // consumers must sent buffer length equivalent to this size. It is
  // initialized on the first call to send.
  int64_t channel_element_size_;
  // Allocated size of an element in the circular buffer in units of bytes. The
  // elements are aligned to the largest scalar type.
  int64_t allocated_element_size_;
  // TODO(vmirian): 8-09-2022 Place the following guarded members on a single
  // cache line for optimal performance.
  // The maximum number of bytes that can hold elements in the circular buffer.
  int64_t max_byte_count_ ABSL_GUARDED_BY(mutex_);
  // The number of bytes used in the circular buffer.
  int64_t bytes_used_ ABSL_GUARDED_BY(mutex_);
  // Index in the circular buffer to enqueue send requests.
  int64_t enqueue_index_ ABSL_GUARDED_BY(mutex_);
  // Index in the circular buffer to dequeue receive requests.
  int64_t dequeue_index_ ABSL_GUARDED_BY(mutex_);
  // A circular buffer to store the elements. It is preallocated with storage.
  absl::InlinedVector<uint8_t, kInitBufferSize> circular_buffer_
      ABSL_GUARDED_BY(mutex_);
};

// Queue for streaming channels. This queue behaves as an infinite depth FIFO.
// It is preallocated with storage. Compared to xls::FifoJitChannelQueue, the
// implementation is lockless, thus thread unsafe. However, the implementation
// is more efficient than xls::FifoJitChannelQueue in scenarios where the
// producer and consumer do not access the channel simultaneously. An example of
// a scenario is a producer sends to the channel and does not send until a
// consumer receives from the channel.
class LocklessFifoJitChannelQueue : public JitChannelQueue {
 public:
  LocklessFifoJitChannelQueue(int64_t channel_id)
      : JitChannelQueue(channel_id),
        channel_element_size_(0),
        allocated_element_size_(0),
        bytes_used_(0),
        enqueue_index_(0),
        dequeue_index_(0) {}

  void Send(const uint8_t* data, int64_t num_bytes) override {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
    __msan_unpoison(data, num_bytes);
#endif
    bool first_call = false;
    if (channel_element_size_ == 0) {
      channel_element_size_ = num_bytes;
      allocated_element_size_ =
          RoundUpToNearest(channel_element_size_,
                           static_cast<int64_t>(alignof(std::max_align_t)));
      first_call = true;
    }
    XLS_CHECK_EQ(num_bytes, channel_element_size_) << absl::StrFormat(
        "Invalid number of bytes given to Send Function of "
        "LocklessFifoJitChannelQueue: expected (%s), got (%s).",
        std::to_string(channel_element_size_), std::to_string(num_bytes));

    // TODO(vmirian): 8-09-2022 Provide the element size at the constructor and
    // remove the following if statement.
    if (first_call) {
      // Align the vector allocation to a power of 2 for efficient utilization
      // of the memory.
      int64_t element_size_2 = 1 << CeilOfLog2(allocated_element_size_);
      if (element_size_2 > kInitBufferSize) {
        circular_buffer_.resize(element_size_2);
      } else {
        circular_buffer_.resize(kInitBufferSize);
      }
      max_byte_count_ =
          FloorOfRatio(static_cast<int64_t>(circular_buffer_.size()),
                       allocated_element_size_) *
          allocated_element_size_;
    }
    // Resize the circular buffer for a new entry when there is insufficient
    // space in the queue.
    if (bytes_used_ == max_byte_count_) {
      circular_buffer_.resize(circular_buffer_.size() * 2);
      max_byte_count_ =
          FloorOfRatio(static_cast<int64_t>(circular_buffer_.size()),
                       allocated_element_size_) *
          allocated_element_size_;
      // The content of the circular buffer must be rearranged when the dequeue
      // index is not at the beginning of the circular buffer to ensure correct
      // ordering.
      if (dequeue_index_ != 0) {
        std::move(circular_buffer_.begin(),
                  circular_buffer_.begin() + dequeue_index_,
                  circular_buffer_.begin() + bytes_used_);
      }
      // Realign the enqueue index to the next available slot.
      enqueue_index_ = bytes_used_ + dequeue_index_;
      if (enqueue_index_ == max_byte_count_) {
        enqueue_index_ = 0;
      }
    }
    memcpy(circular_buffer_.data() + enqueue_index_, data,
           channel_element_size_);
    bytes_used_ += allocated_element_size_;
    enqueue_index_ = enqueue_index_ + allocated_element_size_;
    if (enqueue_index_ == max_byte_count_) {
      enqueue_index_ = 0;
    }
  }

  bool Recv(uint8_t* buffer, int64_t num_bytes) override {
    if (bytes_used_ == 0) {
      return false;
    }

    XLS_CHECK_EQ(num_bytes, channel_element_size_) << absl::StrFormat(
        "Invalid number of bytes given to Recv Function of "
        "LocklessFifoJitChannelQueue: expected (%s), got (%s).",
        std::to_string(channel_element_size_), std::to_string(num_bytes));

    memcpy(buffer, circular_buffer_.data() + dequeue_index_,
           channel_element_size_);
    bytes_used_ -= allocated_element_size_;
    dequeue_index_ = dequeue_index_ + allocated_element_size_;
    if (dequeue_index_ == max_byte_count_) {
      dequeue_index_ = 0;
    }
    return true;
  }

  bool Empty() override { return bytes_used_ == 0; }

 protected:
  static constexpr int64_t kInitBufferSize = 128;
  // Size of an element in the channel in units of bytes. The producers and
  // consumers must sent buffer length equivalent to this size. It is
  // initialized on the first call to send.
  int64_t channel_element_size_;
  // Allocated size of an element in the circular buffer in units of bytes. The
  // elements are aligned to the largest scalar type.
  int64_t allocated_element_size_;
  // The maximum number of bytes that can hold elements in the circular buffer.
  int64_t max_byte_count_;
  // The number of bytes used in the circular buffer.
  int64_t bytes_used_;
  // Index in the circular buffer to enqueue send requests.
  int64_t enqueue_index_;
  // Index in the circular buffer to dequeue receive requests.
  int64_t dequeue_index_;
  // A circular buffer to store the elements. It is preallocated with storage.
  absl::InlinedVector<uint8_t, kInitBufferSize> circular_buffer_;
};

// Queue for single value channels. Unsurprisingly, this queue holds a single
// value. The value is read non-destructively with the Recv method and is
// overwritten via the Send method.
class SingleValueJitChannelQueue : public JitChannelQueue {
 public:
  explicit SingleValueJitChannelQueue(int64_t channel_id)
      : JitChannelQueue(channel_id), buffer_size_(0) {}

  virtual void Send(const uint8_t* data, int64_t num_bytes) {
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

  virtual bool Recv(uint8_t* buffer, int64_t num_bytes) {
    absl::MutexLock lock(&mutex_);
    XLS_CHECK(buffer_ != nullptr);
    XLS_CHECK_EQ(buffer_size_, num_bytes);
    memcpy(buffer, buffer_.get(), num_bytes);
    return true;
  }

  virtual bool Empty() {
    absl::MutexLock lock(&mutex_);
    return buffer_ == nullptr;
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

  // TODO(vmirian) : 8-11-2022 Merge the following function and
  // xls::JitChannelQueueManager::Create into a single function. Returns a
  // JitChannelQueueManager holding a JitChannelQueue for every proc in the
  // provided package. The behavior is identical to
  // xls::JitChannelQueueManager::Create with the exception that a streaming
  // channel is mapped to xls::LocklessFifoJitChannelQueue.
  static absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
  CreateThreadUnsafe(Package* package);

  absl::StatusOr<JitChannelQueue*> GetQueueById(int64_t channel_id) {
    XLS_RET_CHECK(queues_.contains(channel_id));
    return queues_.at(channel_id).get();
  }

 private:
  explicit JitChannelQueueManager(Package* package);
  template <typename QueueT>
  absl::Status Init() {
    for (Channel* chan : package_->channels()) {
      if (chan->kind() == ChannelKind::kStreaming) {
        queues_.insert({chan->id(), std::make_unique<QueueT>(chan->id())});
      } else {
        XLS_RET_CHECK_EQ(chan->kind(), ChannelKind::kSingleValue);
        queues_.insert(
            {chan->id(),
             std::make_unique<SingleValueJitChannelQueue>(chan->id())});
      }
    }
    return absl::OkStatus();
  }

  Package* package_;
  absl::flat_hash_map<int64_t, std::unique_ptr<JitChannelQueue>> queues_;
};

}  // namespace xls

#endif  // XLS_JIT_JIT_CHANNEL_QUEUE_H_

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
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

// A queue from which raw bytes may be written or read.
class ByteQueue {
 public:
  // `channel_element_size` is the granuality of the queue access. Each read or
  // write to the queue handles this many bytes at a time. `is_single_value`
  // indicates whether this queue follows single-value channel semantics where
  // the queue only holds a single value; writes overwrite the value in the
  // queue and reads are non-destructive. If `is_single_value` is false then the
  // queue has FIFO semantics.
  ByteQueue(int64_t channel_element_size, bool is_single_value);

  int64_t element_size() const { return channel_element_size_; }

  // Doubles the size of the queue.
  void Resize();

  void Write(const uint8_t* data) {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
    __msan_unpoison(data, channel_element_size_);
#endif
    if (bytes_used_ == max_byte_count_ && !is_single_value_) {
      Resize();
    }
    memcpy(circular_buffer_.data() + write_index_, data, channel_element_size_);
    if (is_single_value_) {
      bytes_used_ = allocated_element_size_;
    } else {
      bytes_used_ += allocated_element_size_;
      write_index_ = write_index_ + allocated_element_size_;
      if (write_index_ == max_byte_count_) {
        write_index_ = 0;
      }
    }
  }

  bool Read(uint8_t* buffer) {
    if (bytes_used_ == 0) {
      return false;
    }
    memcpy(buffer, circular_buffer_.data() + read_index_,
           channel_element_size_);
    if (!is_single_value_) {
      // Reads are destructive for non single-value channels.
      bytes_used_ -= allocated_element_size_;
      read_index_ = read_index_ + allocated_element_size_;
      if (read_index_ == max_byte_count_) {
        read_index_ = 0;
      }
    }
    return true;
  }

  int64_t size() const { return bytes_used_ / allocated_element_size_; }

  static constexpr int64_t kInitBufferSize = 128;

 private:
  // Size of an element in the channel in units of bytes.
  int64_t channel_element_size_ = 0;
  // Allocated size of an element in the circular buffer in units of bytes. The
  // elements are aligned to the largest scalar type.
  int64_t allocated_element_size_ = 0;
  // TODO(vmirian): 8-09-2022 Place the following guarded members on a single
  // cache line for optimal performance.
  // The maximum number of bytes that can hold elements in the circular buffer.
  int64_t max_byte_count_ = 0;
  // The number of bytes used in the circular buffer.
  int64_t bytes_used_ = 0;
  // Index in the circular buffer to write values to.
  int64_t read_index_ = 0;
  // Index in the circular buffer to read values from.
  int64_t write_index_ = 0;
  // A circular buffer to store the elements. It is preallocated with storage.
  absl::InlinedVector<uint8_t, kInitBufferSize> circular_buffer_;
  // Whether this queue follows single-value channel semantics.
  bool is_single_value_;
};

// Abstract base class for channel queues which may be used by the JIT. These
// queues support reading and writing raw bytes to the queue rather the just
// xls::Values.
class JitChannelQueue : public ChannelQueue {
 public:
  JitChannelQueue(Channel* channel, JitRuntime* jit_runtime)
      : ChannelQueue(channel), jit_runtime_(jit_runtime) {}
  virtual ~JitChannelQueue() = default;

  virtual void WriteRaw(const uint8_t* data) = 0;
  virtual bool ReadRaw(uint8_t* buffer) = 0;

 protected:
  JitRuntime* jit_runtime_;
};

// A thread-safe version of the JIT channel queue. All accesses are guarded by a
// mutex.
class ThreadSafeJitChannelQueue : public JitChannelQueue {
 public:
  ThreadSafeJitChannelQueue(Channel* channel, JitRuntime* jit_runtime)
      : JitChannelQueue(channel, jit_runtime),
        byte_queue_(jit_runtime->GetTypeByteSize(channel->type()),
                    channel->kind() == ChannelKind::kSingleValue) {}
  virtual ~ThreadSafeJitChannelQueue() = default;

  // Write raw bytes representing a value in LLVM's native format.
  void WriteRaw(const uint8_t* data) override {
    absl::MutexLock lock(&mutex_);
    byte_queue_.Write(data);
  }

  // Reads raw bytes representing a value in LLVM's native format. Returns
  // true if queue was not empty and data was read.
  bool ReadRaw(uint8_t* buffer) override {
    absl::MutexLock lock(&mutex_);
    if (generator_.has_value()) {
      std::optional<Value> generated_value = (*generator_)();
      if (generated_value.has_value()) {
        WriteInternal(generated_value.value());
      }
    }
    return byte_queue_.Read(buffer);
  }

 protected:
  int64_t GetSizeInternal() const ABSL_SHARED_LOCKS_REQUIRED(mutex_) override;
  void WriteInternal(const Value& value)
      ABSL_SHARED_LOCKS_REQUIRED(mutex_) override;
  std::optional<Value> ReadInternal()
      ABSL_SHARED_LOCKS_REQUIRED(mutex_) override;

  ByteQueue byte_queue_ ABSL_GUARDED_BY(mutex_);
};

// A thread-unsafe version of the JIT channel queue.
class ThreadUnsafeJitChannelQueue : public JitChannelQueue {
 public:
  ThreadUnsafeJitChannelQueue(Channel* channel, JitRuntime* jit_runtime)
      : JitChannelQueue(channel, jit_runtime),
        byte_queue_(jit_runtime->GetTypeByteSize(channel->type()),
                    channel->kind() == ChannelKind::kSingleValue) {}
  virtual ~ThreadUnsafeJitChannelQueue() = default;

  void WriteRaw(const uint8_t* data) override { byte_queue_.Write(data); }
  bool ReadRaw(uint8_t* buffer) override {
    if (generator_.has_value()) {
      std::optional<Value> generated_value = (*generator_)();
      if (generated_value.has_value()) {
        WriteInternal(generated_value.value());
      }
    }
    return byte_queue_.Read(buffer);
  }

 protected:
  int64_t GetSizeInternal() const ABSL_SHARED_LOCKS_REQUIRED(mutex_) override;
  void WriteInternal(const Value& value) override;
  std::optional<Value> ReadInternal() override;

  ByteQueue byte_queue_;
};

// A Channel manager which holds exclusively JitChannelQueues.
class JitChannelQueueManager : public ChannelQueueManager {
 public:
  virtual ~JitChannelQueueManager() = default;

  // Factories which create a queue manager with exclusively ThreadSafe/Unsafe
  // queues.
  static absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
  CreateThreadSafe(Package* package);
  static absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
  CreateThreadUnsafe(Package* package);

  JitChannelQueue& GetJitQueue(Channel* channel);

  JitRuntime& runtime() { return *runtime_; }

 protected:
  JitChannelQueueManager(Package* package,
                         std::vector<std::unique_ptr<ChannelQueue>>&& queues,
                         std::unique_ptr<JitRuntime> runtime)
      : ChannelQueueManager(package, std::move(queues)),
        runtime_(std::move(runtime)) {}

  std::unique_ptr<JitRuntime> runtime_;
};

}  // namespace xls

#endif  // XLS_JIT_JIT_CHANNEL_QUEUE_H_

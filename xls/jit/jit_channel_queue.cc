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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_runtime.h"

namespace xls {
namespace {

void WriteValueOnQueue(const Value& value, Type* type, JitRuntime& runtime,
                       ByteQueue& queue) {
  absl::InlinedVector<uint8_t, ByteQueue::kInitBufferSize> buffer(
      queue.element_size());
  runtime.BlitValueToBuffer(value, type, absl::MakeSpan(buffer));
  queue.Write(buffer.data());
}

std::optional<Value> ReadValueFromQueue(Type* type, JitRuntime& runtime,
                                        ByteQueue& queue) {
  std::vector<uint8_t> buffer(queue.element_size());
  if (!queue.Read(buffer.data())) {
    return std::nullopt;
  }
  return runtime.UnpackBuffer(buffer.data(), type);
}

}  // namespace

ByteQueue::ByteQueue(int64_t channel_element_size, bool is_single_value)
    : channel_element_size_(channel_element_size),
      allocated_element_size_(
          RoundUpToNearest(channel_element_size,
                           static_cast<int64_t>(alignof(std::max_align_t)))),
      is_single_value_(is_single_value) {
  // Special case to handle empty tuples. Assigning the allocated element size
  // to one serves as a tangible instance for the number of elements within the
  // queue.
  if (allocated_element_size_ == 0) {
    allocated_element_size_ = 1;
  }
  // Align the vector allocation to a power of 2 for efficient utilization
  // of the memory.
  int64_t element_size_2 = 1 << CeilOfLog2(allocated_element_size_);
  if (element_size_2 > kInitBufferSize) {
    circular_buffer_.resize(element_size_2);
  } else {
    circular_buffer_.resize(kInitBufferSize);
  }
  max_byte_count_ = FloorOfRatio(static_cast<int64_t>(circular_buffer_.size()),
                                 allocated_element_size_) *
                    allocated_element_size_;
}

void ByteQueue::Resize() {
  circular_buffer_.resize(circular_buffer_.size() * 2);
  max_byte_count_ = FloorOfRatio(static_cast<int64_t>(circular_buffer_.size()),
                                 allocated_element_size_) *
                    allocated_element_size_;
  // The content of the circular buffer must be rearranged when the read
  // index is not at the beginning of the circular buffer to ensure correct
  // ordering.
  if (read_index_ != 0) {
    std::move(circular_buffer_.begin(), circular_buffer_.begin() + read_index_,
              circular_buffer_.begin() + bytes_used_);
  }
  // Realign the write index to the next available slot.
  write_index_ = bytes_used_ + read_index_;
  if (write_index_ == max_byte_count_) {
    write_index_ = 0;
  }
}

int64_t ThreadSafeJitChannelQueue::GetSizeInternal() const {
  return byte_queue_.size();
}

void ThreadSafeJitChannelQueue::WriteInternal(const Value& value) {
  WriteValueOnQueue(value, channel()->type(), *jit_runtime_, byte_queue_);
}

std::optional<Value> ThreadSafeJitChannelQueue::ReadInternal() {
  return ReadValueFromQueue(channel()->type(), *jit_runtime_, byte_queue_);
}

int64_t ThreadUnsafeJitChannelQueue::GetSizeInternal() const {
  return byte_queue_.size();
}

void ThreadUnsafeJitChannelQueue::WriteInternal(const Value& value) {
  WriteValueOnQueue(value, channel()->type(), *jit_runtime_, byte_queue_);
}

std::optional<Value> ThreadUnsafeJitChannelQueue::ReadInternal() {
  return ReadValueFromQueue(channel()->type(), *jit_runtime_, byte_queue_);
}

/* static */ absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
JitChannelQueueManager::CreateThreadSafe(Package* package) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateThreadSafe(std::move(elaboration));
}

/* static */ absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
JitChannelQueueManager::CreateThreadSafe(ProcElaboration&& elaboration) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<JitRuntime> runtime,
                       JitRuntime::Create());
  std::vector<std::unique_ptr<ChannelQueue>> queues;
  for (ChannelInstance* channel_instance : elaboration.channel_instances()) {
    queues.push_back(std::make_unique<ThreadSafeJitChannelQueue>(
        channel_instance, runtime.get()));
  }
  return absl::WrapUnique(new JitChannelQueueManager(
      std::move(elaboration), std::move(queues), std::move(runtime)));
}

/* static */ absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
JitChannelQueueManager::CreateThreadUnsafe(Package* package) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateThreadUnsafe(std::move(elaboration));
}

/* static */ absl::StatusOr<std::unique_ptr<JitChannelQueueManager>>
JitChannelQueueManager::CreateThreadUnsafe(ProcElaboration&& elaboration) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<JitRuntime> runtime,
                       JitRuntime::Create());
  std::vector<std::unique_ptr<ChannelQueue>> queues;
  for (ChannelInstance* channel_instance : elaboration.channel_instances()) {
    queues.push_back(std::make_unique<ThreadUnsafeJitChannelQueue>(
        channel_instance, runtime.get()));
  }
  return absl::WrapUnique(new JitChannelQueueManager(
      std::move(elaboration), std::move(queues), std::move(runtime)));
}

JitChannelQueue& JitChannelQueueManager::GetJitQueue(Channel* channel) {
  JitChannelQueue* queue = dynamic_cast<JitChannelQueue*>(&GetQueue(channel));
  CHECK_NE(queue, nullptr);
  return *queue;
}

JitChannelQueue& JitChannelQueueManager::GetJitQueue(
    ChannelInstance* channel_instance) {
  JitChannelQueue* queue =
      dynamic_cast<JitChannelQueue*>(&GetQueue(channel_instance));
  CHECK_NE(queue, nullptr);
  return *queue;
}

}  // namespace xls

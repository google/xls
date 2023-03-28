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

#ifndef XLS_INTERPRETER_CHANNEL_QUEUE_H_
#define XLS_INTERPRETER_CHANNEL_QUEUE_H_

#include <deque>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

// Abstract base class for queues which represent channels during IR
// interpretation. During interpretation of a network of procs each channel is
// backed by exactly one ChannelQueue. ChannelQueues are thread-safe.
class ChannelQueue {
 public:
  explicit ChannelQueue(Channel* channel) : channel_(channel) {}

  // Channel queues should not be copyable. There should be no reason to as
  // there is a one-to-one correspondence between channels (which are not
  // copyable) and queues, and accidentally creating a copy of a queue could
  // produce a hard-to-debug problem.
  ChannelQueue(const ChannelQueue&) = delete;

  virtual ~ChannelQueue() = default;

  // Returns the channel associated with this queue.
  Channel* channel() const { return channel_; }

  // Returns the number of elements currently in the channel queue.
  int64_t GetSize() const {
    absl::MutexLock lock(&mutex_);
    return GetSizeInternal();
  }

  // Returns whether the channel queue is empty.
  bool IsEmpty() const { return GetSize() == 0; }

  // Writes the given value on to the channel.
  absl::Status Write(const Value& value);

  // Reads and returns a value from the channel. Returns an std::nullopt if
  // the channel is empty.
  std::optional<Value> Read();

  // Attaches a function which generates values for the channel. The generator
  // is called when a value is needed for reading. If a generator is attached
  // then calling `Write` returns an error.
  using GeneratorFn = std::function<std::optional<Value>()>;
  absl::Status AttachGenerator(GeneratorFn generator);

 protected:
  mutable absl::Mutex mutex_;

  virtual int64_t GetSizeInternal() const ABSL_SHARED_LOCKS_REQUIRED(mutex_);
  virtual void WriteInternal(const Value& value)
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);
  virtual std::optional<Value> ReadInternal()
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);
  Channel* channel_;

  std::deque<Value> queue_ ABSL_GUARDED_BY(mutex_);
  // The ThreadUnsafeJitChannelQueue reads this value without a lock.
  // TODO(meheff): 2022/09/27 Fix this, potentially by obviating the need for
  // the thread-unsafe version of the queue.
  std::optional<GeneratorFn> generator_ ABSL_GUARDED_BY_FIXME(mutex_);
};

// A functor which returns a sequence of Values when called. Maybe be attached
// to a ChannelQueue as a generator.
class FixedValueGenerator {
 public:
  explicit FixedValueGenerator(absl::Span<const Value> values)
      : values_(values.begin(), values.end()) {}

  std::optional<Value> operator()() {
    if (values_.empty()) {
      return std::nullopt;
    }
    Value value = values_.front();
    values_.pop_front();
    return std::move(value);
  }

 private:
  std::deque<Value> values_;
};

// An abstraction holding a collection of channel queues for interpreting the
// procs within a single package. Essentially a map of channel queues with some
// convenience methods.
class ChannelQueueManager {
 public:
  virtual ~ChannelQueueManager() = default;

  // Creates and returns a queue manager for the given package.
  static absl::StatusOr<std::unique_ptr<ChannelQueueManager>> Create(
      Package* package);

  static absl::StatusOr<std::unique_ptr<ChannelQueueManager>> Create(
      std::vector<std::unique_ptr<ChannelQueue>>&& queues, Package* package);

  // Get the channel queue associated with the channel with the given id/name.
  ChannelQueue& GetQueue(Channel* channel) { return *queues_.at(channel); }

  // Returns the vector of all queues sorted by channel ID.
  absl::Span<ChannelQueue* const> queues() { return queue_vec_; }

  // Returns the queue associated with the channel with the given
  // ID/name. Returns an error if no such channel exists.
  absl::StatusOr<ChannelQueue*> GetQueueById(int64_t channel_id);
  absl::StatusOr<ChannelQueue*> GetQueueByName(std::string_view name);

 protected:
  ChannelQueueManager(Package* package,
                      std::vector<std::unique_ptr<ChannelQueue>>&& queues);

  Package* package_;

  // Channel queues indexed by the associated channel pointer.
  absl::flat_hash_map<Channel*, std::unique_ptr<ChannelQueue>> queues_;

  // Vector containing pointers to the channel queues held in queues_.
  std::vector<ChannelQueue*> queue_vec_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_CHANNEL_QUEUE_H_

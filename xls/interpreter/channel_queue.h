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
// backed by exactly one ChannelQueue. ChannelQueues are thread-compatible, but
// not thread-safe.
class ChannelQueue {
 public:
  ChannelQueue(Channel* channel)
      : channel_(channel) {}

  // Channel queues should not be copyable. There should be no reason to as
  // there is a one-to-one correspondence between channels (which are not
  // copyable) and queues, and accidentally creating a copy of a queue could
  // produce a hard-to-debug problem.
  ChannelQueue(const ChannelQueue&) = delete;

  virtual ~ChannelQueue() = default;

  // Returns the channel associated with this queue.
  Channel* channel() const { return channel_; }

  // Returns the number of elements currently in the channel queue.
  virtual int64_t size() const = 0;

  // Returns whether the channel queue is empty.
  virtual bool empty() const = 0;

  // Enqueues the given value on to the channel.
  virtual absl::Status Enqueue(const Value& value) = 0;

  // Dequeues and returns a value from the channel. Returns an error if the
  // channel is empty.
  virtual absl::StatusOr<Value> Dequeue() = 0;

 protected:
  Channel* channel_;
};

// A queue representing an arbitrary-depth FIFO. This matches the abstract
// semantics of streaming channels. FifoChannelQueues are thread--safe.
class FifoChannelQueue : public ChannelQueue {
 public:
  FifoChannelQueue(Channel* channel) : ChannelQueue(channel) {}
  virtual ~FifoChannelQueue() = default;

  // Returns the number of elements currently in the channel queue.
  int64_t size() const override {
    absl::MutexLock lock(&mutex_);
    return queue_.size();
  }

  // Returns whether the channel queue is empty.
  bool empty() const override {
    absl::MutexLock lock(&mutex_);
    return queue_.empty();
  }

  // Enqueues the given value on to the channel.
  virtual absl::Status Enqueue(const Value& value);

  // Dequeues and returns a value from the channel. Returns an error if the
  // channel is empty.
  virtual absl::StatusOr<Value> Dequeue();

 protected:
  // Values are enqueued to the back, and dequeued from the front.
  std::deque<Value> queue_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex mutex_;
};

// A queue backing a receive-only channel. Receive-only channels provide inputs
// to a network of procs and are enqueued by components outside of XLS.
class GeneratedChannelQueue : public ChannelQueue {
 public:
  // generator_func is a function which returns the next value to enqueue on to
  // the channel. The generator_func may be called an arbitrary number of times
  // depending upon how many times the proc interpreter is ticked. The generator
  // function should return an error to terminate the interpreter session.
  GeneratedChannelQueue(Channel* channel, Package* package,
                        std::function<absl::StatusOr<Value>()> generator_func)
      : ChannelQueue(channel), generator_func_(std::move(generator_func)) {}
  virtual ~GeneratedChannelQueue() = default;

  // GeneratedChannelQueue::Enqueue returns an error unconditionally. Values in
  // the queue is generated from the generator function rather than being
  // enqueued.
  absl::Status Enqueue(const Value& value) override;

  // Calls the generator function and returns the result.
  absl::StatusOr<Value> Dequeue() override;

  // The number of elements is considered infinite as the generator function may
  // be called an arbitrary number of times.
  int64_t size() const override { return std::numeric_limits<int64_t>::max(); }
  bool empty() const override { return false; }

 protected:
  std::function<absl::StatusOr<Value>()> generator_func_;
};

// An input channel queue which produces a fixed sequence of values. Once the
// sequence is exhausted, any further calls to Dequeue return an errored.
class FixedChannelQueue : public GeneratedChannelQueue {
 public:
  FixedChannelQueue(Channel* channel, Package* package,
                    absl::Span<const Value> values)
      : GeneratedChannelQueue(
            channel, package,
            [this]() -> absl::StatusOr<Value> { return GenerateValue(); }),
        values_(values.begin(), values.end()) {}
  virtual ~FixedChannelQueue() = default;

  int64_t size() const override { return values_.size(); }
  bool empty() const override { return values_.empty(); }

 protected:
  // Pops and returns the next element out of the deque.
  absl::StatusOr<Value> GenerateValue();

  std::deque<Value> values_;
};

// A ChannelQueue with single-value channel semantics. The data structure holds
// a single value which is written (or overwritten) via Enqueue. Dequeue
// non-destructively returns the held value.
class SingleValueChannelQueue : public ChannelQueue {
 public:
  explicit SingleValueChannelQueue(Channel* channel) : ChannelQueue(channel) {}
  virtual ~SingleValueChannelQueue() = default;

  absl::Status Enqueue(const Value& value) override;
  absl::StatusOr<Value> Dequeue() override;

  int64_t size() const override {
    absl::MutexLock lock(&mutex_);
    return value_.has_value() ? 1 : 0;
  }

  bool empty() const override {
    absl::MutexLock lock(&mutex_);
    return !value_.has_value();
  }

 protected:
  absl::optional<Value> value_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex mutex_;
};

// An abstraction holding a collection of channel queues for interpreting the
// procs within a single package. Essentially a map of channel queues with some
// convenience methods.
class ChannelQueueManager {
 public:
  // Creates and returns a queue manager for the given package.
  // user_defined_queues can optionally contain user-constructed queues to use
  // with some or all of the receive-only channels. This is useful for testing
  // where the test may want to use a GeneratedChannelQueue or FixedChannelQueue
  // to feed inputs to the test.
  static absl::StatusOr<std::unique_ptr<ChannelQueueManager>> Create(
      std::vector<std::unique_ptr<ChannelQueue>>&& user_defined_queues,
      Package* package);

  // Get the channel queue associated with the channel with the given id/name.
  ChannelQueue& GetQueue(Channel* channel) { return *queues_.at(channel); }

  // Returns the vector of all queues sorted by channel ID.
  absl::Span<ChannelQueue* const> queues() { return queue_vec_; }

  // Returns the queue associated with the channel with the given
  // ID/name. Returns an error if no such channel exists.
  absl::StatusOr<ChannelQueue*> GetQueueById(int64_t channel_id);
  absl::StatusOr<ChannelQueue*> GetQueueByName(absl::string_view name);

 protected:
  explicit ChannelQueueManager(Package* package) : package_(package) {}

  Package* package_;

  // Channel queues indexed by the associated channel pointer.
  absl::flat_hash_map<Channel*, std::unique_ptr<ChannelQueue>> queues_;

  // Vector containing pointers to the channel queues held in queues_.
  std::vector<ChannelQueue*> queue_vec_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_CHANNEL_QUEUE_H_

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

#include "xls/interpreter/channel_queue.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"

namespace xls {

absl::Status FifoChannelQueue::Enqueue(const Value& value) {
  XLS_VLOG(4) << absl::StreamFormat("Enqueuing value on channel %s: { %s }",
                                    channel_->name(), value.ToString());
  if (!ValueConformsToType(value, channel_->type())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Channel %s expects values to have type %s, got: %s", channel_->name(),
        channel_->type()->ToString(), value.ToString()));
  }

  absl::MutexLock lock(&mutex_);
  queue_.push_back(value);
  XLS_VLOG(4) << absl::StreamFormat("Channel now has %d elements", size());
  return absl::OkStatus();
}

absl::StatusOr<Value> FifoChannelQueue::Dequeue() {
  if (empty()) {
    return absl::NotFoundError(
        absl::StrFormat("Attempting to dequeue data from empty channel %s (%d)",
                        channel_->name(), channel_->id()));
  }
  absl::MutexLock lock(&mutex_);
  Value value = queue_.front();
  queue_.pop_front();
  XLS_VLOG(4) << absl::StreamFormat("Dequeuing data on channel %s: %s",
                                    channel_->name(), value.ToString());
  XLS_VLOG(4) << absl::StreamFormat("Channel now has %d elements", size());
  return std::move(value);
}

absl::Status GeneratedChannelQueue::Enqueue(const Value& value) {
  return absl::UnimplementedError(
      absl::StrFormat("Cannot enqueue to GeneratedChannelQueue on channel %s.",
                      channel()->name()));
}

absl::StatusOr<Value> GeneratedChannelQueue::Dequeue() {
  XLS_ASSIGN_OR_RETURN(Value value, generator_func_());
  XLS_VLOG(4) << absl::StreamFormat("Dequeuing data on channel %s: %s",
                                    channel()->name(), value.ToString());
  return std::move(value);
}

absl::StatusOr<Value> FixedChannelQueue::GenerateValue() {
  if (values_.empty()) {
    return absl::ResourceExhaustedError(
        absl::StrFormat("FixedInputChannel for channel %s (%d) is empty.",
                        channel()->name(), channel()->id()));
  }
  Value value = std::move(values_.front());
  values_.pop_front();
  return std::move(value);
}

absl::Status SingleValueChannelQueue::Enqueue(const Value& value) {
  absl::MutexLock lock(&mutex_);
  XLS_CHECK(ValueConformsToType(value, channel_->type()));
  value_ = value;
  return absl::OkStatus();
}

absl::StatusOr<Value> SingleValueChannelQueue::Dequeue() {
  absl::MutexLock lock(&mutex_);
  if (!value_.has_value()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Value has not been written to single-value queue for channel %s",
        channel()->name()));
  }
  return value_.value();
}

static bool IsSingleValueChannelQueue(ChannelQueue* queue) {
  return dynamic_cast<SingleValueChannelQueue*>(queue) != nullptr;
}

/* static */
absl::StatusOr<std::unique_ptr<ChannelQueueManager>>
ChannelQueueManager::Create(
    std::vector<std::unique_ptr<ChannelQueue>>&& user_defined_queues,
    Package* package) {
  auto manager = absl::WrapUnique(new ChannelQueueManager(package));

  // Verify there is an receive-only queue for every ReceiveOnly channel in the
  // package.
  for (auto& queue : user_defined_queues) {
    if (queue->channel()->supported_ops() != ChannelOps::kReceiveOnly) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "User-defined queues can only be used with receive_only "
          "channels, used with %s channel %s",
          ChannelOpsToString(queue->channel()->supported_ops()),
          queue->channel()->name()));
    }

    if (manager->queues_.contains(queue->channel())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "More than one receive-only queue given for channel %s",
          queue->channel()->name()));
    }

    if (queue->channel()->kind() != ChannelKind::kSingleValue &&
        IsSingleValueChannelQueue(queue.get())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Single-value channel queue cannot be used for "
                          "non-single-value channel %s",
                          queue->channel()->name()));
    }
    if (queue->channel()->kind() == ChannelKind::kSingleValue &&
        !IsSingleValueChannelQueue(queue.get())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Non-single-value channel queue cannot be used for "
                          "single-value channel %s",
                          queue->channel()->name()));
    }
    manager->queues_[queue->channel()] = std::move(queue);
  }

  // Verify that every receive-only channel has an receive-only queue and create
  // queues for the remaining non-receive-only channels.
  for (Channel* channel : package->channels()) {
    if (channel->kind() != ChannelKind::kStreaming &&
        channel->kind() != ChannelKind::kSingleValue) {
      return absl::UnimplementedError(
          "Only streaming and single-value channels are supported.");
    }
    if (manager->queues_.contains(channel)) {
      continue;
    }
    if (channel->kind() == ChannelKind::kSingleValue) {
      manager->queues_[channel] =
          absl::make_unique<SingleValueChannelQueue>(channel);
    } else {
      manager->queues_[channel] = absl::make_unique<FifoChannelQueue>(channel);
    }
  }

  // Create a sorted vector of channel queues in the manager for easy iteration
  // through the queues.
  for (auto& [channel, queue] : manager->queues_) {
    manager->queue_vec_.push_back(queue.get());
  }
  std::sort(manager->queue_vec_.begin(), manager->queue_vec_.end(),
            [](ChannelQueue* a, ChannelQueue* b) {
              return a->channel()->id() < b->channel()->id();
            });
  return std::move(manager);
}

absl::StatusOr<ChannelQueue*> ChannelQueueManager::GetQueueById(
    int64_t channel_id) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package_->GetChannel(channel_id));
  return queues_.at(channel).get();
}

absl::StatusOr<ChannelQueue*> ChannelQueueManager::GetQueueByName(
    absl::string_view name) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package_->GetChannel(name));
  return queues_.at(channel).get();
}

}  // namespace xls

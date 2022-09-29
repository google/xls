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
#include "xls/common/status/status_macros.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"

namespace xls {

absl::Status ChannelQueue::AttachGenerator(GeneratorFn generator) {
  absl::MutexLock lock(&mutex_);
  if (generator_.has_value()) {
    return absl::InternalError("ChannelQueue already has a generator attached");
  }
  if (channel_->kind() == ChannelKind::kSingleValue) {
    return absl::InternalError(
        absl::StrFormat("ChannelQueues for single-value channels cannot have a "
                        "generator. Channel: %s",
                        channel()->name()));
  }
  generator_ = std::move(generator);
  return absl::OkStatus();
}

absl::Status ChannelQueue::Write(const Value& value) {
  XLS_VLOG(4) << absl::StreamFormat("Writing value to channel %s: { %s }",
                                    channel_->name(), value.ToString());
  absl::MutexLock lock(&mutex_);
  if (generator_.has_value()) {
    return absl::InternalError(
        "Cannot write to ChannelQueue because it has a generator function.");
  }
  if (!ValueConformsToType(value, channel_->type())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Channel %s expects values to have type %s, got: %s", channel_->name(),
        channel_->type()->ToString(), value.ToString()));
  }

  WriteInternal(value);
  XLS_VLOG(4) << absl::StreamFormat("Channel now has %d elements",
                                    queue_.size());
  return absl::OkStatus();
}

void ChannelQueue::WriteInternal(const Value& value) {
  if (channel()->kind() == ChannelKind::kSingleValue) {
    if (queue_.empty()) {
      queue_.push_back(value);
    } else {
      queue_.front() = value;
    }
    return;
  }

  XLS_CHECK_EQ(channel()->kind(), ChannelKind::kStreaming);
  queue_.push_back(value);
}

std::optional<Value> ChannelQueue::Read() {
  absl::MutexLock lock(&mutex_);
  if (generator_.has_value()) {
    // Write/ReadInternal are virtual and may have other side-effects so rather
    // than directly returning the generated value, write then read it.
    std::optional<Value> generated_value = (*generator_)();
    if (generated_value.has_value()) {
      WriteInternal(generated_value.value());
    }
  }
  std::optional<Value> value = ReadInternal();
  XLS_VLOG(4) << absl::StreamFormat(
      "Reading data from channel %s: %s", channel_->name(),
      value.has_value() ? value->ToString() : "(none)");
  XLS_VLOG(4) << absl::StreamFormat("Channel now has %d elements",
                                    queue_.size());
  return value;
}

int64_t ChannelQueue::GetSizeInternal() const { return queue_.size(); }

std::optional<Value> ChannelQueue::ReadInternal() {
  if (queue_.empty()) {
    return std::nullopt;
  }
  Value value = queue_.front();
  if (channel()->kind() != ChannelKind::kSingleValue) {
    queue_.pop_front();
  }
  return std::move(value);
}

/* static */
absl::StatusOr<std::unique_ptr<ChannelQueueManager>>
ChannelQueueManager::Create(std::vector<std::unique_ptr<ChannelQueue>>&& queues,
                            Package* package) {
  // Verify there is exactly one queue per channel.
  absl::flat_hash_set<Channel*> proc_channels(package->channels().begin(),
                                              package->channels().end());
  absl::flat_hash_set<Channel*> queue_channels;
  for (const std::unique_ptr<ChannelQueue>& queue : queues) {
    if (!proc_channels.contains(queue->channel())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel `%s` for queue does not exist in package `%s`",
          queue->channel()->name(), package->name()));
    }
    auto [ir, inserted] = queue_channels.insert(queue->channel());
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Multiple queues specified for channel `%s`",
                          queue->channel()->name()));
    }
  }
  for (Channel* channel : package->channels()) {
    if (!queue_channels.contains(channel)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No queue specified for channel `%s`", channel->name()));
    }
  }

  return absl::WrapUnique(new ChannelQueueManager(package, std::move(queues)));
}

/* static */
absl::StatusOr<std::unique_ptr<ChannelQueueManager>>
ChannelQueueManager::Create(Package* package) {
  std::vector<std::unique_ptr<ChannelQueue>> queues;

  // Create a queue per channel in the package.
  for (Channel* channel : package->channels()) {
    if (channel->kind() != ChannelKind::kStreaming &&
        channel->kind() != ChannelKind::kSingleValue) {
      return absl::UnimplementedError(
          "Only streaming and single-value channels are supported.");
    }
    queues.push_back(std::make_unique<ChannelQueue>(channel));
  }

  return absl::WrapUnique(new ChannelQueueManager(package, std::move(queues)));
}

ChannelQueueManager::ChannelQueueManager(
    Package* package, std::vector<std::unique_ptr<ChannelQueue>>&& queues)
    : package_(package) {
  for (std::unique_ptr<ChannelQueue>& queue : queues) {
    Channel* channel = queue->channel();
    queues_[channel] = std::move(queue);
    queue_vec_.push_back(queues_[channel].get());
  }
  // Stably sort the queues by channel ID.
  std::sort(queue_vec_.begin(), queue_vec_.end(),
            [](ChannelQueue* a, ChannelQueue* b) {
              return a->channel()->id() < b->channel()->id();
            });
}

absl::StatusOr<ChannelQueue*> ChannelQueueManager::GetQueueById(
    int64_t channel_id) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package_->GetChannel(channel_id));
  return queues_.at(channel).get();
}

absl::StatusOr<ChannelQueue*> ChannelQueueManager::GetQueueByName(
    std::string_view name) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package_->GetChannel(name));
  return queues_.at(channel).get();
}

}  // namespace xls

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

absl::Status ChannelQueue::AttachGenerator(GeneratorFn generator) {
  absl::MutexLock lock(&mutex_);
  if (generator_.has_value()) {
    return absl::InternalError("ChannelQueue already has a generator attached");
  }
  if (channel()->kind() == ChannelKind::kSingleValue) {
    return absl::InternalError(
        absl::StrFormat("ChannelQueues for single-value channels cannot have a "
                        "generator. Channel: %s",
                        channel()->name()));
  }
  generator_ = std::move(generator);
  return absl::OkStatus();
}

absl::Status ChannelQueue::Write(const Value& value) {
  VLOG(4) << absl::StreamFormat(
      "Writing value to channel instance `%s`: { %s }",
      channel_instance()->ToString(), value.ToString());
  absl::MutexLock lock(&mutex_);
  if (generator_.has_value()) {
    return absl::InternalError(
        "Cannot write to ChannelQueue because it has a generator function.");
  }
  if (!ValueConformsToType(value, channel()->type())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Channel `%s` expects values to have type %s, got: %s",
        channel()->name(), channel()->type()->ToString(), value.ToString()));
  }

  WriteInternal(value);
  VLOG(4) << absl::StreamFormat("Channel now has %d elements", queue_.size());
  return absl::OkStatus();
}

void ChannelQueue::WriteInternal(const Value& value) {
  CallWriteCallbacks(value);
  if (channel()->kind() == ChannelKind::kSingleValue) {
    if (queue_.empty()) {
      queue_.push_back(value);
    } else {
      queue_.front() = value;
    }
    return;
  }

  CHECK_EQ(channel()->kind(), ChannelKind::kStreaming);
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
  VLOG(4) << absl::StreamFormat(
      "Reading data from channel instance %s: %s",
      channel_instance()->ToString(),
      value.has_value() ? value->ToString() : "(none)");
  VLOG(4) << absl::StreamFormat("Channel now has %d elements", queue_.size());
  return value;
}

int64_t ChannelQueue::GetSizeInternal() const { return queue_.size(); }

std::optional<Value> ChannelQueue::ReadInternal() {
  if (queue_.empty()) {
    return std::nullopt;
  }
  Value value = queue_.front();
  CallReadCallbacks(value);
  if (channel()->kind() != ChannelKind::kSingleValue) {
    queue_.pop_front();
  }
  return std::move(value);
}

/* static */ absl::StatusOr<std::unique_ptr<ChannelQueueManager>>
ChannelQueueManager::Create(Package* package) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return Create(std::move(elaboration));
}

/* static */ absl::StatusOr<std::unique_ptr<ChannelQueueManager>>
ChannelQueueManager::Create(std::vector<std::unique_ptr<ChannelQueue>>&& queues,
                            ProcElaboration elaboration) {
  // Verify there is exactly one queue per channel.
  absl::flat_hash_set<ChannelInstance*> channel_instances(
      elaboration.channel_instances().begin(),
      elaboration.channel_instances().end());
  absl::flat_hash_set<ChannelInstance*> queue_chan_instances;
  for (const std::unique_ptr<ChannelQueue>& queue : queues) {
    if (!channel_instances.contains(queue->channel_instance())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel instance `%s` for queue does not exist in package `%s`",
          queue->channel_instance()->ToString(),
          elaboration.package()->name()));
    }
    auto [ir, inserted] =
        queue_chan_instances.insert(queue->channel_instance());
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Multiple queues specified for channel instance `%s`",
                          queue->channel_instance()->ToString()));
    }
  }
  for (ChannelInstance* instance : elaboration.channel_instances()) {
    if (!queue_chan_instances.contains(instance)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("No queue specified for channel instance `%s`",
                          instance->ToString()));
    }
  }

  return absl::WrapUnique(
      new ChannelQueueManager(std::move(elaboration), std::move(queues)));
}

/* static */ absl::StatusOr<std::unique_ptr<ChannelQueueManager>>
ChannelQueueManager::Create(ProcElaboration elaboration) {
  std::vector<std::unique_ptr<ChannelQueue>> queues;

  // Create a queue per channel instance in the elaboration.
  for (ChannelInstance* channel_instance : elaboration.channel_instances()) {
    if (channel_instance->channel->kind() != ChannelKind::kStreaming &&
        channel_instance->channel->kind() != ChannelKind::kSingleValue) {
      return absl::UnimplementedError(
          "Only streaming and single-value channels are supported.");
    }
    queues.push_back(std::make_unique<ChannelQueue>(channel_instance));
  }

  return absl::WrapUnique(
      new ChannelQueueManager(std::move(elaboration), std::move(queues)));
}

ChannelQueueManager::ChannelQueueManager(
    ProcElaboration elaboration,
    std::vector<std::unique_ptr<ChannelQueue>>&& queues)
    : elaboration_(std::move(elaboration)) {
  for (std::unique_ptr<ChannelQueue>& queue : queues) {
    ChannelInstance* instance = queue->channel_instance();
    queues_[instance] = std::move(queue);
    queue_vec_.push_back(queues_[instance].get());
  }
  // Stably sort the queues by channel ID.
  std::sort(queue_vec_.begin(), queue_vec_.end(),
            [](ChannelQueue* a, ChannelQueue* b) {
              return a->channel()->id() < b->channel()->id();
            });
}

absl::StatusOr<ChannelQueue*> ChannelQueueManager::GetQueueById(
    int64_t channel_id) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package()->GetChannel(channel_id));
  XLS_ASSIGN_OR_RETURN(ChannelInstance * instance,
                       elaboration().GetUniqueInstance(channel));
  return queues_.at(instance).get();
}

absl::StatusOr<ChannelQueue*> ChannelQueueManager::GetQueueByName(
    std::string_view name) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package()->GetChannel(name));
  XLS_ASSIGN_OR_RETURN(ChannelInstance * instance,
                       elaboration().GetUniqueInstance(channel));
  return queues_.at(instance).get();
}

}  // namespace xls

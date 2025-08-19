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

#include "xls/interpreter/proc_runtime.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/observer.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {

// Functor for recording channel activity as trace messages.
class ChannelTraceRecorder : public ChannelQueueCallback {
 public:
  ChannelTraceRecorder(ProcRuntime* runtime, FormatPreference format_preference)
      : runtime_(runtime), format_preference_(format_preference) {}
  ~ChannelTraceRecorder() override = default;

  void ReadValue(ChannelInstance* channel_instance,
                 const Value& value) override {
    std::string message = absl::StrFormat("Received data on channel `%s`: %s",
                                          channel_instance->ToString(),
                                          value.ToString(format_preference_));
    VLOG(3) << message;
    runtime_->AddTraceMessage(
        TraceMessage{.message = std::move(message), .verbosity = 0});
  }

  void WriteValue(ChannelInstance* channel_instance,
                  const Value& value) override {
    std::string message = absl::StrFormat("Sent data on channel `%s`: %s",
                                          channel_instance->ToString(),
                                          value.ToString(format_preference_));
    VLOG(3) << message;
    runtime_->AddTraceMessage(
        TraceMessage{.message = std::move(message), .verbosity = 0});
  }

 private:
  ProcRuntime* runtime_;
  FormatPreference format_preference_;
};

void ProcRuntime::ClearObserver() {
  if (!observer_) {
    return;
  }
  for (const auto& [_, cont] : continuations_) {
    cont->ClearObserver();
  }
  observer_.reset();
}
absl::Status ProcRuntime::SetObserver(EvaluationObserver* obs) {
  for (const auto& [inst, cont] : continuations_) {
    XLS_RETURN_IF_ERROR(cont->SetObserver(obs));
  }
  observer_ = obs;
  return absl::OkStatus();
}

bool ProcRuntime::SupportsObservers() const {
  for (const auto& [inst, cont] : continuations_) {
    if (!cont->SupportsObservers()) {
      return false;
    }
  }
  return true;
}

ProcRuntime::ProcRuntime(
    absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>>&& evaluators,
    std::unique_ptr<ChannelQueueManager>&& queue_manager,
    const EvaluatorOptions& options)
    : queue_manager_(std::move(queue_manager)),
      evaluators_(std::move(evaluators)) {
  for (ProcInstance* instance : elaboration().proc_instances()) {
    std::unique_ptr<ProcContinuation> continuation =
        evaluators_.at(instance->proc())->NewContinuation(instance);
    continuations_[instance] = std::move(continuation);
  }
  if (options.trace_channels()) {
    for (ChannelQueue* queue : queue_manager_->queues()) {
      queue->AddCallback(std::make_unique<ChannelTraceRecorder>(
          this, options.format_preference()));
    }
  }
}

absl::Status ProcRuntime::Tick() {
  std::vector<Channel*> blocked_channels;
  XLS_ASSIGN_OR_RETURN(NetworkTickResult result, TickInternal());
  if (!result.progress_made) {
    // Not a single instruction executed on any proc. This is necessarily a
    // deadlock.
    return absl::InternalError(absl::StrFormat(
        "Proc network is deadlocked. Blocked channel instances: %s",
        absl::StrJoin(result.blocked_channel_instances, ", ",
                      [](std::string* s, ChannelInstance* c) {
                        absl::StrAppend(s, c->ToString());
                      })));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> ProcRuntime::TickUntilOutput(
    const absl::flat_hash_map<Channel*, int64_t>& output_counts,
    std::optional<int64_t> max_ticks) {
  absl::flat_hash_map<ChannelInstance*, int64_t> instance_output_counts;
  for (const auto& [channel, count] : output_counts) {
    XLS_ASSIGN_OR_RETURN(ChannelInstance * channel_instance,
                         elaboration().GetUniqueInstance(channel));
    instance_output_counts[channel_instance] = count;
  }
  return TickUntilOutput(instance_output_counts, max_ticks);
}

absl::StatusOr<int64_t> ProcRuntime::TickUntilOutput(
    const absl::flat_hash_map<ChannelInstance*, int64_t>& output_counts,
    std::optional<int64_t> max_ticks) {
  VLOG(3) << absl::StreamFormat("TickUntilOutput on package %s",
                                package()->name());
  // Create a deterministically sorted vector of the output channels for
  // deterministic behavior and error messages.
  std::vector<ChannelInstance*> output_channels;
  for (ChannelInstance* channel_instance : elaboration().channel_instances()) {
    if (output_counts.contains(channel_instance)) {
      output_channels.push_back(channel_instance);
    }
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Expected outputs produced for each channel instance:";
    for (ChannelInstance* channel_instance : output_channels) {
      VLOG(3) << absl::StreamFormat("  %s : %d", channel_instance->ToString(),
                                    output_counts.at(channel_instance));
    }
  }

  for (ChannelInstance* channel_instance : output_channels) {
    Channel* channel = channel_instance->channel;
    if (channel->supported_ops() != ChannelOps::kSendOnly) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel `%s` is not a send-only channel", channel->name()));
    }
    if (channel->kind() == ChannelKind::kSingleValue &&
        output_counts.at(channel_instance) > 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel `%s` is single-value, expected number of "
                          "elements must be one or less, is %d",
                          channel->name(), output_counts.at(channel_instance)));
    }
  }
  int64_t ticks = 0;
  auto needs_more_output = [&]() {
    for (ChannelInstance* ch : output_channels) {
      if (queue_manager().GetQueue(ch).GetSize() < output_counts.at(ch)) {
        return true;
      }
    }
    return false;
  };
  while (needs_more_output()) {
    if (max_ticks.has_value() && ticks >= max_ticks.value()) {
      return absl::DeadlineExceededError(
          absl::StrFormat("Exceeded limit of %d ticks of the proc network "
                          "before expected output produced",
                          max_ticks.value()));
    }
    XLS_RETURN_IF_ERROR(Tick());
    ticks++;
  }
  return ticks;
}

absl::StatusOr<int64_t> ProcRuntime::TickUntilBlocked(
    std::optional<int64_t> max_ticks) {
  VLOG(3) << absl::StreamFormat("TickUntilBlocked on package %s",
                                package()->name());
  int64_t ticks = 0;
  while (!max_ticks.has_value() || ticks < max_ticks.value()) {
    XLS_ASSIGN_OR_RETURN(NetworkTickResult result, TickInternal());
    if (!result.progress_made_on_io_procs) {
      return ticks;
    }
    ticks++;
  }
  return absl::DeadlineExceededError(absl::StrFormat(
      "Exceeded limit of %d ticks of the proc network before blocking",
      max_ticks.value()));
}

void ProcRuntime::ResetState() {
  for (ProcInstance* instance : elaboration().proc_instances()) {
    continuations_[instance] =
        evaluators_.at(instance->proc())->NewContinuation(instance);
    if (observer_) {
      // We must have called this successfully at least once.
      CHECK_OK(continuations_[instance]->SetObserver(*observer_));
    }
  }
}

absl::StatusOr<JitChannelQueueManager*>
ProcRuntime::GetJitChannelQueueManager() {
  auto* jit_qm = dynamic_cast<JitChannelQueueManager*>(queue_manager_.get());
  if (jit_qm == nullptr) {
    return absl::InternalError("Queue manager is not a JitChannelQueueManager");
  }
  return jit_qm;
}

InterpreterEvents ProcRuntime::GetGlobalEvents() const {
  absl::MutexLock lock(global_events_mutex_);
  return global_events_;
}

void ProcRuntime::ClearInterpreterEvents() {
  for (const auto& [_, continuation] : continuations_) {
    continuation->ClearEvents();
  }
  {
    absl::MutexLock lock(global_events_mutex_);
    global_events_.Clear();
  }
}

void ProcRuntime::AddTraceMessage(TraceMessage message) {
  absl::MutexLock lock(global_events_mutex_);
  global_events_.trace_msgs.push_back(std::move(message));
}

}  // namespace xls

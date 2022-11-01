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

#include <deque>

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"

namespace xls {

ProcRuntime::ProcRuntime(
    Package* package,
    absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>>&& evaluators,
    std::unique_ptr<ChannelQueueManager>&& queue_manager)
    : package_(package), queue_manager_(std::move(queue_manager)) {
  for (const std::unique_ptr<Proc>& proc : package_->procs()) {
    std::unique_ptr<ProcContinuation> continuation =
        evaluators.at(proc.get())->NewContinuation();
    evaluator_contexts_[proc.get()] =
        EvaluatorContext{.evaluator = std::move(evaluators.at(proc.get())),
                         .continuation = std::move(continuation)};
  }
}

absl::Status ProcRuntime::Tick() {
  std::vector<Channel*> blocked_channels;
  XLS_ASSIGN_OR_RETURN(NetworkTickResult result, TickInternal());
  if (!result.progress_made) {
    // Not a single instruction executed on any proc. This is necessarily a
    // deadlock.
    return absl::InternalError(absl::StrFormat(
        "Proc network is deadlocked. Blocked channels: %s",
        absl::StrJoin(result.blocked_channels, ", ", ChannelFormatter)));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> ProcRuntime::TickUntilOutput(
    absl::flat_hash_map<Channel*, int64_t> output_counts,
    std::optional<int64_t> max_ticks) {
  XLS_VLOG(3) << absl::StreamFormat("TickUntilOutput on package %s",
                                    package_->name());
  // Create a deterministically sorted vector of the output channels for
  // determistic behavior and error messages.
  std::vector<Channel*> output_channels;
  for (auto [channel, _] : output_counts) {
    output_channels.push_back(channel);
  }
  std::sort(output_channels.begin(), output_channels.end(),
            [](Channel* a, Channel* b) { return a->name() < b->name(); });

  if (XLS_VLOG_IS_ON(3)) {
    XLS_VLOG(3) << "Expected outputs produced for each channel:";
    for (Channel* channel : output_channels) {
      XLS_VLOG(3) << absl::StreamFormat("  %s : %d", channel->name(),
                                        output_counts.at(channel));
    }
  }

  for (Channel* channel : output_channels) {
    if (channel->supported_ops() != ChannelOps::kSendOnly) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel `%s` is not a send-only channel", channel->name()));
    }
    if (channel->kind() == ChannelKind::kSingleValue &&
        output_counts.at(channel) > 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel `%s` is single-value, expected number of "
                          "elements must be one or less, is %d",
                          channel->name(), output_counts.at(channel)));
    }
  }
  int64_t ticks = 0;
  auto needs_more_output = [&]() {
    for (Channel* ch : output_channels) {
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
  XLS_VLOG(3) << absl::StreamFormat("TickUntilBlocked on package %s",
                                    package_->name());
  int64_t ticks = 0;
  while (!max_ticks.has_value() || ticks < max_ticks.value()) {
    XLS_ASSIGN_OR_RETURN(NetworkTickResult result, TickInternal());
    if (!result.progress_made) {
      return ticks;
    }
    ticks++;
  }
  return absl::DeadlineExceededError(absl::StrFormat(
      "Exceeded limit of %d ticks of the proc network before blocking",
      max_ticks.value()));
}

void ProcRuntime::ResetState() {
  for (const std::unique_ptr<Proc>& proc : package_->procs()) {
    EvaluatorContext& context = evaluator_contexts_[proc.get()];
    context.continuation = context.evaluator->NewContinuation();
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

}  // namespace xls

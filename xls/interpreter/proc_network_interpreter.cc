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

#include "xls/interpreter/proc_network_interpreter.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"

namespace xls {

/* static */
absl::StatusOr<std::unique_ptr<ProcNetworkInterpreter>>
ProcNetworkInterpreter::Create(
    Package* package,
    std::vector<std::unique_ptr<ChannelQueue>>&& user_defined_queues) {
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(std::move(user_defined_queues), package));

  // Create a network interpreter.
  auto interpreter = absl::WrapUnique(
      new ProcNetworkInterpreter(package, std::move(queue_manager)));

  for (auto& proc : package->procs()) {
    interpreter->proc_interpreters_.push_back(std::make_unique<ProcInterpreter>(
        proc.get(), &interpreter->queue_manager()));
  }

  // Inject initial values into channels.
  for (Channel* channel : package->channels()) {
    ChannelQueue& queue = interpreter->queue_manager().GetQueue(channel);
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(queue.Enqueue(value));
    }
  }

  // Create a ProcInterpreter for each Proc.
  return std::move(interpreter);
}

absl::Status ProcNetworkInterpreter::Tick() {
  std::vector<Channel*> blocked_channels;
  XLS_ASSIGN_OR_RETURN(bool progress_made, TickInternal(&blocked_channels));
  if (!progress_made) {
    // Not a single instruction executed on any proc. This is necessarily a
    // deadlock.
    return absl::InternalError(absl::StrFormat(
        "Proc network is deadlocked. Blocked channels: %s",
        absl::StrJoin(blocked_channels, ", ", ChannelFormatter)));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> ProcNetworkInterpreter::TickUntilOutput(
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
      if (queue_manager().GetQueue(ch).size() < output_counts.at(ch)) {
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

absl::StatusOr<int64_t> ProcNetworkInterpreter::TickUntilBlocked(
    std::optional<int64_t> max_ticks) {
  XLS_VLOG(3) << absl::StreamFormat("TickUntilBlocked on package %s",
                                    package_->name());
  int64_t ticks = 0;
  while (!max_ticks.has_value() || ticks < max_ticks.value()) {
    std::vector<Channel*> blocked_channels;
    XLS_ASSIGN_OR_RETURN(bool progress_made, TickInternal(&blocked_channels));
    if (!progress_made) {
      return ticks;
    }
    ticks++;
  }
  return absl::DeadlineExceededError(absl::StrFormat(
      "Exceeded limit of %d ticks of the proc network before blocking",
      max_ticks.value()));
}

absl::StatusOr<bool> ProcNetworkInterpreter::TickInternal(
    std::vector<Channel*>* blocked_channels) {
  XLS_VLOG(3) << absl::StreamFormat("TickInternal on package %s",
                                    package_->name());
  absl::flat_hash_set<ProcInterpreter*> completed_procs;
  absl::flat_hash_set<Channel*> blocked_channel_set;
  bool global_progress_made = false;
  bool progress_made_this_loop = true;
  while (progress_made_this_loop) {
    progress_made_this_loop = false;
    blocked_channel_set.clear();
    for (auto& interpreter : proc_interpreters_) {
      if (completed_procs.contains(interpreter.get())) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(ProcInterpreter::RunResult result,
                           interpreter->RunIterationUntilCompleteOrBlocked());

      progress_made_this_loop |= result.progress_made;
      if (result.iteration_complete) {
        completed_procs.insert(interpreter.get());
      }
      blocked_channel_set.insert(result.blocked_channels.begin(),
                                 result.blocked_channels.end());
    }
    global_progress_made |= progress_made_this_loop;
  }
  if (!global_progress_made) {
    // Not a single instruction executed on any proc. This is necessarily a
    // deadlock. Sort blocked channels by channel id so the channels order is
    // deterministic.
    blocked_channels->clear();
    blocked_channels->insert(blocked_channels->begin(),
                             blocked_channel_set.begin(),
                             blocked_channel_set.end());
    std::sort(blocked_channels->begin(), blocked_channels->end(),
              [](Channel* a, Channel* b) { return a->id() < b->id(); });
    XLS_VLOG(3) << absl::StreamFormat(
        "TickInternal: no progress made, blocked channels: %s",
        absl::StrJoin(*blocked_channels, ", ", ChannelFormatter));
    return false;
  }
  XLS_VLOG(3) << "TickInternal: Progress made";
  return true;
}

absl::flat_hash_map<Proc*, absl::StatusOr<std::vector<Value>>>
ProcNetworkInterpreter::ResolveState() const {
  absl::flat_hash_map<Proc*, absl::StatusOr<std::vector<Value>>> states;
  for (const auto& interpreter : proc_interpreters_) {
    states[interpreter->proc()] = interpreter->ResolveState();
  }
  return states;
}

void ProcNetworkInterpreter::ResetState() {
  for (const auto& interpreter : proc_interpreters_) {
    interpreter->ResetState();
  }
}

}  // namespace xls

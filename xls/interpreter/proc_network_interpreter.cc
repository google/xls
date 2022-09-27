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

#include <deque>

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"

namespace xls {

/* static */
absl::StatusOr<std::unique_ptr<ProcNetworkInterpreter>>
ProcNetworkInterpreter::Create(
    Package* package, std::vector<std::unique_ptr<ProcEvaluator>>&& evaluators,
    std::unique_ptr<ChannelQueueManager>&& queue_manager) {
  // Verify there exists exactly one evaluator per proc in the package.
  absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>> evaluator_map;
  for (std::unique_ptr<ProcEvaluator>& evaluator : evaluators) {
    Proc* proc = evaluator->proc();
    auto [it, inserted] = evaluator_map.insert({proc, std::move(evaluator)});
    XLS_RET_CHECK(inserted) << absl::StreamFormat(
        "More than one evaluator given for proc `%s`", proc->name());
  }
  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    XLS_RET_CHECK(evaluator_map.contains(proc.get()))
        << absl::StreamFormat("No evaluator given for proc `%s`", proc->name());
  }
  XLS_RET_CHECK_EQ(evaluator_map.size(), package->procs().size())
      << "More evaluators than procs given.";
  auto network_interpreter = absl::WrapUnique(new ProcNetworkInterpreter(
      package, std::move(evaluator_map), std::move(queue_manager)));

  return std::move(network_interpreter);
}

absl::Status ProcNetworkInterpreter::Tick() {
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

absl::StatusOr<ProcNetworkInterpreter::NetworkTickResult>
ProcNetworkInterpreter::TickInternal() {
  XLS_VLOG(3) << absl::StreamFormat("TickInternal on package %s",
                                    package_->name());
  // Map containing any blocked procs and the channels they are blocked on.
  absl::flat_hash_map<Channel*, Proc*> blocked_procs;

  std::deque<Proc*> ready_procs;

  // Put all procs on the ready list.
  for (const std::unique_ptr<Proc>& proc : package_->procs()) {
    XLS_VLOG(3) << absl::StreamFormat("Proc `%s` added to ready list",
                                      proc->name());
    ready_procs.push_back(proc.get());
  }

  bool progress_made = false;
  while (!ready_procs.empty()) {
    Proc* proc = ready_procs.front();
    EvaluatorContext& context = evaluator_contexts_.at(proc);
    ready_procs.pop_front();

    XLS_VLOG(3) << absl::StreamFormat("Ticking proc `%s`", proc->name());
    XLS_ASSIGN_OR_RETURN(TickResult tick_result,
                         context.evaluator->Tick(*context.continuation));
    XLS_VLOG(3) << "Tick result: " << tick_result;

    progress_made |= tick_result.progress_made;
    for (Channel* channel : tick_result.sent_channels) {
      if (blocked_procs.contains(channel)) {
        XLS_VLOG(3) << absl::StreamFormat(
            "Unblocking proc `%s` and adding to ready list",
            blocked_procs.at(channel)->name());
        ready_procs.push_back(blocked_procs.at(channel));
        blocked_procs.erase(channel);
      }
    }
    if (tick_result.blocked_channel.has_value()) {
      XLS_VLOG(3) << absl::StreamFormat(
          "Proc `%s` is now blocked on channel `%s`", proc->name(),
          tick_result.blocked_channel.value()->ToString());
      blocked_procs[tick_result.blocked_channel.value()] = proc;
    }
  }
  auto get_blocked_channels = [&]() {
    std::vector<Channel*> channels;
    for (auto [channel, proc] : blocked_procs) {
      channels.push_back(channel);
    }
    std::sort(channels.begin(), channels.end(),
              [](Channel* a, Channel* b) { return a->id() < b->id(); });
    return channels;
  };
  return NetworkTickResult{
      .progress_made = progress_made,
      .blocked_channels = get_blocked_channels(),
  };
}

absl::StatusOr<std::unique_ptr<ProcNetworkInterpreter>>
CreateProcNetworkInterpreter(
    Package* package,
    std::vector<std::unique_ptr<ChannelQueue>>&& user_defined_queues) {
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(std::move(user_defined_queues), package));

  // Create a ProcInterpreter for each Proc.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_interpreters;
  for (auto& proc : package->procs()) {
    proc_interpreters.push_back(
        std::make_unique<ProcInterpreter>(proc.get(), queue_manager.get()));
  }
  // Create a network interpreter.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ProcNetworkInterpreter> network_interpreter,
      ProcNetworkInterpreter::Create(package, std::move(proc_interpreters),
                                     std::move(queue_manager)));

  // Inject initial values into channels.
  for (Channel* channel : package->channels()) {
    ChannelQueue& queue =
        network_interpreter->queue_manager().GetQueue(channel);
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(queue.Enqueue(value));
    }
  }

  return std::move(network_interpreter);
}

absl::flat_hash_map<Proc*, std::vector<Value>>
ProcNetworkInterpreter::ResolveState() const {
  absl::flat_hash_map<Proc*, std::vector<Value>> states;
  for (const auto& [proc, context] : evaluator_contexts_) {
    states[proc] = context.continuation->GetState();
  }
  return states;
}

void ProcNetworkInterpreter::ResetState() {
  for (const std::unique_ptr<Proc>& proc : package_->procs()) {
    EvaluatorContext& context = evaluator_contexts_[proc.get()];
    context.continuation = context.evaluator->NewContinuation();
  }
}

}  // namespace xls

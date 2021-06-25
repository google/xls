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
      new ProcNetworkInterpreter(std::move(queue_manager)));

  for (auto& proc : package->procs()) {
    interpreter->proc_interpreters_.push_back(
        absl::make_unique<ProcInterpreter>(proc.get(),
                                           &interpreter->queue_manager()));
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
  absl::flat_hash_set<ProcInterpreter*> completed_procs;
  absl::flat_hash_set<Channel*> blocked_channels;
  bool global_progress_made = false;
  bool progress_made_this_loop = true;
  while (progress_made_this_loop) {
    progress_made_this_loop = false;
    blocked_channels.clear();
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
      blocked_channels.insert(result.blocked_channels.begin(),
                              result.blocked_channels.end());
    }
    global_progress_made |= progress_made_this_loop;
  }
  if (!global_progress_made) {
    // Not a single instruction executed on any proc. This is necessarily a
    // deadlock. Sort blocked channels by channel id so the return message is
    // stable.
    std::vector<Channel*> blocked_vec(blocked_channels.begin(),
                                      blocked_channels.end());
    std::sort(blocked_vec.begin(), blocked_vec.end(),
              [](Channel* a, Channel* b) { return a->id() < b->id(); });
    return absl::InternalError(absl::StrFormat(
        "Proc network is deadlocked. Blocked channels: %s",
        absl::StrJoin(blocked_vec, ", ", [](std::string* out, Channel* ch) {
          return absl::StrAppend(out, ch->name());
        })));
  }
  return absl::OkStatus();
}

}  // namespace xls

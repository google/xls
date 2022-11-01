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

#include "xls/interpreter/serial_proc_runtime.h"

#include <deque>

#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"

namespace xls {

/* static */
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> SerialProcRuntime::Create(
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
  auto network_interpreter = absl::WrapUnique(new SerialProcRuntime(
      package, std::move(evaluator_map), std::move(queue_manager)));
  return std::move(network_interpreter);
}

absl::StatusOr<SerialProcRuntime::NetworkTickResult>
SerialProcRuntime::TickInternal() {
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
    if (tick_result.execution_state == TickExecutionState::kSentOnChannel) {
      Channel* channel = tick_result.channel.value();
      if (blocked_procs.contains(channel)) {
        XLS_VLOG(3) << absl::StreamFormat(
            "Unblocking proc `%s` and adding to ready list",
            blocked_procs.at(channel)->name());
        ready_procs.push_back(blocked_procs.at(channel));
        blocked_procs.erase(channel);
      }
      // This proc can go back on the ready queue.
      ready_procs.push_back(proc);
    } else if (tick_result.execution_state ==
               TickExecutionState::kBlockedOnReceive) {
      Channel* channel = tick_result.channel.value();
      XLS_VLOG(3) << absl::StreamFormat(
          "Proc `%s` is now blocked on channel `%s`", proc->name(),
          channel->ToString());
      blocked_procs[channel] = proc;
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

}  // namespace xls

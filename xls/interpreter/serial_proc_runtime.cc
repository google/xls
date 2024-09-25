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
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {

/* static */ absl::StatusOr<std::unique_ptr<SerialProcRuntime>>
SerialProcRuntime::Create(
    std::vector<std::unique_ptr<ProcEvaluator>>&& evaluators,
    std::unique_ptr<ChannelQueueManager>&& queue_manager,
    const EvaluatorOptions& options) {
  // Verify there exists exactly one evaluator per proc in the package.
  absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>> evaluator_map;
  for (std::unique_ptr<ProcEvaluator>& evaluator : evaluators) {
    Proc* proc = evaluator->proc();
    auto [it, inserted] = evaluator_map.insert({proc, std::move(evaluator)});
    XLS_RET_CHECK(inserted) << absl::StreamFormat(
        "More than one evaluator given for proc `%s`", proc->name());
  }
  for (Proc* proc : queue_manager->elaboration().procs()) {
    XLS_RET_CHECK(evaluator_map.contains(proc))
        << absl::StreamFormat("No evaluator given for proc `%s`", proc->name());
  }
  XLS_RET_CHECK_EQ(evaluator_map.size(),
                   queue_manager->elaboration().procs().size())
      << "More evaluators than procs given.";
  auto network_interpreter = absl::WrapUnique(new SerialProcRuntime(
      std::move(evaluator_map), std::move(queue_manager), options));
  return std::move(network_interpreter);
}

absl::StatusOr<SerialProcRuntime::NetworkTickResult>
SerialProcRuntime::TickInternal() {
  VLOG(3) << absl::StreamFormat("TickInternal on package %s",
                                package()->name());
  // Map containing any blocked proc instances and the channels they are blocked
  // on.
  absl::flat_hash_map<ChannelInstance*, ProcInstance*> blocked_instances;

  struct QueueElement {
    ProcInstance* instance;
    ProcEvaluator* evaluator;
    ProcContinuation* continuation;
  };
  std::deque<QueueElement> ready_instances;

  // Put all proc instances on the ready list.
  for (ProcInstance* instance : elaboration().proc_instances()) {
    VLOG(3) << absl::StreamFormat("Proc instance `%s` added to ready list",
                                  instance->GetName());
    ready_instances.push_back(
        QueueElement{.instance = instance,
                     .evaluator = evaluators_.at(instance->proc()).get(),
                     .continuation = continuations_.at(instance).get()});
  }

  bool progress_made = false;
  bool progress_made_on_io_procs = false;
  while (!ready_instances.empty()) {
    const QueueElement element = ready_instances.front();
    ready_instances.pop_front();

    VLOG(3) << absl::StreamFormat("Ticking proc instance `%s`",
                                  element.instance->GetName());
    XLS_ASSIGN_OR_RETURN(TickResult tick_result,
                         element.evaluator->Tick(*element.continuation));
    const InterpreterEvents& events =
        this->GetInterpreterEvents(element.instance);
    XLS_RETURN_IF_ERROR(InterpreterEventsToStatus(events));
    VLOG(3) << "Tick result: " << tick_result;

    progress_made |= tick_result.progress_made;
    progress_made_on_io_procs |=
        (tick_result.progress_made && element.evaluator->ProcHasIoOperations());
    if (tick_result.execution_state == TickExecutionState::kSentOnChannel) {
      ChannelInstance* channel_instance = tick_result.channel_instance.value();
      if (blocked_instances.contains(channel_instance)) {
        VLOG(3) << absl::StreamFormat(
            "Unblocking proc instance `%s` and adding to ready list",
            blocked_instances.at(channel_instance)->GetName());
        ProcInstance* instance = blocked_instances.at(channel_instance);
        ready_instances.push_back(
            QueueElement{.instance = instance,
                         .evaluator = evaluators_.at(instance->proc()).get(),
                         .continuation = continuations_.at(instance).get()});
        blocked_instances.erase(channel_instance);
      }
      // This proc instance can go back on the ready queue.
      ready_instances.push_back(element);
    } else if (tick_result.execution_state ==
               TickExecutionState::kBlockedOnReceive) {
      ChannelInstance* channel_instance = tick_result.channel_instance.value();
      VLOG(3) << absl::StreamFormat(
          "Proc instance `%s` is now blocked on channel instance `%s`",
          element.instance->GetName(), channel_instance->ToString());
      blocked_instances[channel_instance] = element.instance;
    }
  }
  auto get_blocked_channel_instances = [&]() {
    std::vector<ChannelInstance*> instances;
    for (ChannelInstance* instance : elaboration().channel_instances()) {
      if (blocked_instances.contains(instance)) {
        instances.push_back(instance);
      }
    }
    return instances;
  };
  return NetworkTickResult{
      .progress_made = progress_made,
      .progress_made_on_io_procs = progress_made_on_io_procs,
      .blocked_channel_instances = get_blocked_channel_instances(),
  };
}

}  // namespace xls

// Copyright 2022 The XLS Authors
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

#include "xls/interpreter/interpreter_proc_runtime.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateRuntime(
    ProcElaboration elaboration) {
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ChannelQueueManager> queue_manager,
                       ChannelQueueManager::Create(std::move(elaboration)));

  // Create a ProcInterpreter for each Proc.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_interpreters;
  for (Proc* proc : queue_manager->elaboration().procs()) {
    proc_interpreters.push_back(
        std::make_unique<ProcInterpreter>(proc, queue_manager.get()));
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> proc_runtime,
                       SerialProcRuntime::Create(std::move(proc_interpreters),
                                                 std::move(queue_manager)));

  // Inject initial values into channel queues.
  for (ChannelInstance* channel_instance :
       proc_runtime->elaboration().channel_instances()) {
    Channel* channel = channel_instance->channel;
    ChannelQueue& queue =
        proc_runtime->queue_manager().GetQueue(channel_instance);
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(queue.Write(value));
    }
  }

  return std::move(proc_runtime);
}

}  // namespace

absl::StatusOr<std::unique_ptr<SerialProcRuntime>>
CreateInterpreterSerialProcRuntime(Package* package) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateRuntime(std::move(elaboration));
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>>
CreateInterpreterSerialProcRuntime(Proc* top) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::Elaborate(top));
  return CreateRuntime(std::move(elaboration));
}

}  // namespace xls

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

#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/ir/proc.h"

namespace xls {

absl::StatusOr<std::unique_ptr<SerialProcRuntime>>
CreateInterpreterSerialProcRuntime(Package* package) {
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ChannelQueueManager> queue_manager,
                       ChannelQueueManager::Create(package));

  // Create a ProcInterpreter for each Proc.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_interpreters;
  for (auto& proc : package->procs()) {
    proc_interpreters.push_back(
        std::make_unique<ProcInterpreter>(proc.get(), queue_manager.get()));
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<SerialProcRuntime> proc_runtime,
      SerialProcRuntime::Create(package, std::move(proc_interpreters),
                                std::move(queue_manager)));

  // Inject initial values into channels.
  for (Channel* channel : package->channels()) {
    ChannelQueue& queue = proc_runtime->queue_manager().GetQueue(channel);
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(queue.Write(value));
    }
  }

  return std::move(proc_runtime);
}

}  // namespace xls

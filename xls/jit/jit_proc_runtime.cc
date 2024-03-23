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

#include "xls/jit/jit_proc_runtime.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/channel.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_jit.h"

namespace xls {
namespace {

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateRuntime(
    ProcElaboration elaboration) {
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<JitChannelQueueManager> queue_manager,
      JitChannelQueueManager::CreateThreadSafe(std::move(elaboration)));

  // Create a ProcJit for each Proc.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_jits;
  for (Proc* proc : queue_manager->elaboration().procs()) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<ProcJit> proc_jit,
        ProcJit::Create(proc, &queue_manager->runtime(), queue_manager.get()));
    proc_jits.push_back(std::move(proc_jit));
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> proc_runtime,
                       SerialProcRuntime::Create(std::move(proc_jits),
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

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateJitSerialProcRuntime(
    Package* package) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateRuntime(std::move(elaboration));
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateJitSerialProcRuntime(
    Proc* top) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::Elaborate(top));
  return CreateRuntime(std::move(elaboration));
}

}  // namespace xls

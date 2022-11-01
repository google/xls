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

#include "xls/interpreter/serial_proc_runtime.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/interpreter/proc_runtime_test_base.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/proc_jit.h"

namespace xls {
namespace {

// Create a SerialProcRuntime composed of a mix of ProcInterpreters and
// ProcJits.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateMixedSerialProcRuntime(
    Package* package) {
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<JitChannelQueueManager> queue_manager,
                       JitChannelQueueManager::CreateThreadSafe(package));

  // Create a ProcJit or a ProcInterpreter for each Proc. Alternate between the
  // two options.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_evaluators;
  bool use_jit = true;
  for (auto& proc : package->procs()) {
    if (use_jit) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ProcJit> proc_jit,
          ProcJit::Create(proc.get(), &queue_manager->runtime(),
                          queue_manager.get()));
      proc_evaluators.push_back(std::move(proc_jit));
    } else {
      proc_evaluators.push_back(
          std::make_unique<ProcInterpreter>(proc.get(), queue_manager.get()));
    }
    use_jit = !use_jit;
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<SerialProcRuntime> proc_runtime,
      SerialProcRuntime::Create(package, std::move(proc_evaluators),
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

// Instantiate and run all the tests in proc_runtime_test_base.cc using
// proc interpreters.
INSTANTIATE_TEST_SUITE_P(
    ProcRuntimeTest, ProcRuntimeTestBase,
    testing::Values(
        ProcRuntimeTestParam(
            "interpreter",
            [](Package* package) -> std::unique_ptr<ProcRuntime> {
              return CreateInterpreterSerialProcRuntime(package).value();
            }),
        ProcRuntimeTestParam(
            "jit",
            [](Package* package) -> std::unique_ptr<ProcRuntime> {
              return CreateJitSerialProcRuntime(package).value();
            }),
        ProcRuntimeTestParam(
            "mixed",
            [](Package* package) -> std::unique_ptr<ProcRuntime> {
              return CreateMixedSerialProcRuntime(package).value();
            })),
    [](const testing::TestParamInfo<ProcRuntimeTestBase::ParamType>& info) {
      return info.param.name();
    });

}  // namespace
}  // namespace xls

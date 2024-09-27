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

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/interpreter/proc_runtime_test_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"
#include "xls/jit/proc_jit.h"

namespace xls {
namespace {

constexpr const char kIrAssertPath[] = "xls/interpreter/force_assert.ir";

// Create a SerialProcRuntime composed of a mix of ProcInterpreters and
// ProcJits.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateMixedSerialProcRuntime(
    ProcElaboration elaboration, const EvaluatorOptions& options) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc, OrcJit::Create());
  XLS_ASSIGN_OR_RETURN(auto data_layout, orc->CreateDataLayout());
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<JitChannelQueueManager> queue_manager,
      JitChannelQueueManager::CreateThreadSafe(
          std::move(elaboration), std::make_unique<JitRuntime>(data_layout)));

  // Create a ProcJit or a ProcInterpreter for each Proc. Alternate between the
  // two options.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_evaluators;
  bool use_jit = true;
  for (Proc* proc : queue_manager->elaboration().procs()) {
    if (use_jit) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProcJit> proc_jit,
                           ProcJit::Create(proc, &queue_manager->runtime(),
                                           queue_manager.get()));
      proc_evaluators.push_back(std::move(proc_jit));
    } else {
      proc_evaluators.push_back(
          std::make_unique<ProcInterpreter>(proc, queue_manager.get()));
    }
    use_jit = !use_jit;
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<SerialProcRuntime> proc_runtime,
      SerialProcRuntime::Create(std::move(proc_evaluators),
                                std::move(queue_manager), options));

  // Inject initial values into channels.
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

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateMixedSerialProcRuntime(
    Package* package, const EvaluatorOptions& options) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateMixedSerialProcRuntime(std::move(elaboration), options);
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateMixedSerialProcRuntime(
    Proc* top, const EvaluatorOptions& options) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::Elaborate(top));
  return CreateMixedSerialProcRuntime(std::move(elaboration), options);
}

// Negative test - can we handle asserts
// when using JitSerialProcRuntime?
TEST(SerialProcRuntimeTest, JitAsserts) {
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                           GetXlsRunfilePath(kIrAssertPath));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto interpreter,
                           CreateJitSerialProcRuntime(package.get()));

  EXPECT_THAT(interpreter->Tick(),
              status_testing::StatusIs(
                  absl::StatusCode::kAborted,
                  ::testing::HasSubstr("Assertion failure via fail!")));
}

// Negative test - can we handle asserts
// when using InterpreterSerialProcRuntime?
TEST(SerialProcRuntimeTest, InterpreterAsserts) {
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                           GetXlsRunfilePath(kIrAssertPath));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto interpreter,
                           CreateInterpreterSerialProcRuntime(package.get()));

  EXPECT_THAT(interpreter->Tick(),
              status_testing::StatusIs(
                  absl::StatusCode::kAborted,
                  ::testing::HasSubstr("Assertion failure via fail!")));
}

// Instantiate and run all the tests in proc_runtime_test_base.cc using
// proc interpreters.
INSTANTIATE_TEST_SUITE_P(
    ProcRuntimeTest, ProcRuntimeTestBase,
    testing::Values(
        ProcRuntimeTestParam(
            "interpreter",
            [](Package* package, const EvaluatorOptions& options)
                -> std::unique_ptr<ProcRuntime> {
              return CreateInterpreterSerialProcRuntime(package, options)
                  .value();
            },
            [](Proc* top, const EvaluatorOptions& options)
                -> std::unique_ptr<ProcRuntime> {
              return CreateInterpreterSerialProcRuntime(top, options).value();
            },
            /*supports_observers=*/true),
        ProcRuntimeTestParam(
            "jit",
            [](Package* package, const EvaluatorOptions& options)
                -> std::unique_ptr<ProcRuntime> {
              return CreateJitSerialProcRuntime(package, options).value();
            },
            [](Proc* top, const EvaluatorOptions& options)
                -> std::unique_ptr<ProcRuntime> {
              return CreateJitSerialProcRuntime(top, options).value();
            },
            /*supports_observers=*/false),
        ProcRuntimeTestParam(
            "mixed",
            [](Package* package, const EvaluatorOptions& options)
                -> std::unique_ptr<ProcRuntime> {
              return CreateMixedSerialProcRuntime(package, options).value();
            },
            [](Proc* top, const EvaluatorOptions& options)
                -> std::unique_ptr<ProcRuntime> {
              return CreateMixedSerialProcRuntime(top, options).value();
            },
            /*supports_observers=*/false)),
    [](const testing::TestParamInfo<ProcRuntimeTestBase::ParamType>& info) {
      return info.param.name();
    });

}  // namespace
}  // namespace xls

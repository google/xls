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

#include "xls/jit/proc_jit.h"

#include <memory>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "xls/common/logging/logging.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_evaluator_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

JitRuntime* GetJitRuntime() {
  static auto jit_runtime = std::make_unique<JitRuntime>(
      OrcJit::CreateDataLayout(/*aot_specification=*/false).value());
  return jit_runtime.get();
}

// Instantiate and run all the tests in proc_evaluator_test_base.cc.
INSTANTIATE_TEST_SUITE_P(
    ProcJitTest, ProcEvaluatorTestBase,
    testing::Values(ProcEvaluatorTestParam(
        [](Proc* proc, ChannelQueueManager* queue_manager)
            -> std::unique_ptr<ProcEvaluator> {
          JitChannelQueueManager* jit_queue_manager =
              dynamic_cast<JitChannelQueueManager*>(queue_manager);
          CHECK(jit_queue_manager != nullptr);
          return ProcJit::Create(proc, GetJitRuntime(), jit_queue_manager)
              .value();
        },
        [](Package* package) -> std::unique_ptr<ChannelQueueManager> {
          return JitChannelQueueManager::CreateThreadSafe(package).value();
        })));

}  // namespace
}  // namespace xls

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

#include "absl/log/check.h"
#include "gtest/gtest.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_evaluator_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_evaluator_options.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

JitRuntime* GetJitRuntime() {
  static auto orc_jit = OrcJit::Create().value();
  static auto jit_runtime =
      std::make_unique<JitRuntime>(orc_jit->CreateDataLayout().value());
  return jit_runtime.get();
}

template <bool kWithObserver>
std::unique_ptr<ProcEvaluator> EvaluatorFromProc(
    Proc* proc, ChannelQueueManager* queue_manager) {
  JitChannelQueueManager* jit_queue_manager =
      dynamic_cast<JitChannelQueueManager*>(queue_manager);
  CHECK(jit_queue_manager != nullptr);
  return ProcJit::Create(proc, GetJitRuntime(), jit_queue_manager,
                         EvaluatorOptions(),
                         JitEvaluatorOptions().set_include_observer_callbacks(
                             kWithObserver))
      .value();
}

std::unique_ptr<ChannelQueueManager> QueueManagerForPackage(Package* package) {
  return JitChannelQueueManager::CreateThreadSafe(
             package,
             std::make_unique<JitRuntime>(GetJitRuntime()->data_layout()))
      .value();
}
// Instantiate and run all the tests in proc_evaluator_test_base.cc.
INSTANTIATE_TEST_SUITE_P(
    ProcJitTest, ProcEvaluatorTestBase,
    testing::Values(ProcEvaluatorTestParam(EvaluatorFromProc<false>,
                                           QueueManagerForPackage,
                                           /*supports_observers=*/false),
                    ProcEvaluatorTestParam(EvaluatorFromProc<true>,
                                           QueueManagerForPackage,
                                           /*supports_observers=*/true)));

}  // namespace
}  // namespace xls

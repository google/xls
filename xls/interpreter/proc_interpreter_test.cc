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

#include "xls/interpreter/proc_interpreter.h"

#include <memory>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_evaluator_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_conversion.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {
namespace {

// Instantiate and run all the tests in proc_evaluator_test_base.cc.
INSTANTIATE_TEST_SUITE_P(
    ProcInterpreterTest, ProcEvaluatorTestBase,
    testing::Values(
        ProcEvaluatorTestParam(
            [](Proc* proc, ChannelQueueManager* queue_manager)
                -> std::unique_ptr<ProcEvaluator> {
              return std::make_unique<ProcInterpreter>(proc, queue_manager);
            },
            [](Package* package) -> std::unique_ptr<ChannelQueueManager> {
              return ChannelQueueManager::Create(package).value();
            }),
        // Convert to Proc-scoped channels.
        ProcEvaluatorTestParam(
            [](Proc* proc, ChannelQueueManager* queue_manager)
                -> std::unique_ptr<ProcEvaluator> {
              return std::make_unique<ProcInterpreter>(proc, queue_manager);
            },
            [](Package* package) -> std::unique_ptr<ChannelQueueManager> {
              CHECK(!package->ChannelsAreProcScoped())
                  << "If channels are proc-scoped by default, remove this "
                     "parameter value";
              CHECK_OK(ConvertPackageToNewStyleProcs(package));
              Proc* top = package->GetTopAsProc().value();
              return ChannelQueueManager::Create(
                         ProcElaboration::Elaborate(top).value())
                  .value();
            })));

}  // namespace
}  // namespace xls

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

#ifndef XLS_INTERPRETER_PROC_EVALUATOR_TEST_BASE_H_
#define XLS_INTERPRETER_PROC_EVALUATOR_TEST_BASE_H_

#include <functional>
#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {

class ProcEvaluatorTestParam {
 public:
  ProcEvaluatorTestParam(
      std::function<std::unique_ptr<ProcEvaluator>(Proc*, ChannelQueueManager*)>
          evaluator_factory,
      std::function<std::unique_ptr<ChannelQueueManager>(Package*)>
          queue_manager_factory,
      bool supports_observers = true)
      : evaluator_factory_(std::move(evaluator_factory)),
        queue_manager_factory_(std::move(queue_manager_factory)),
        supports_observers_(supports_observers) {}
  ProcEvaluatorTestParam() = default;

  std::unique_ptr<ChannelQueueManager> CreateQueueManager(
      Package* package) const {
    return queue_manager_factory_(package);
  }

  std::unique_ptr<ProcEvaluator> CreateEvaluator(
      Proc* proc, ChannelQueueManager* queue_manager) const {
    return evaluator_factory_(proc, queue_manager);
  }

  bool supports_observers() const { return supports_observers_; }

 private:
  std::function<std::unique_ptr<ProcEvaluator>(Proc*, ChannelQueueManager*)>
      evaluator_factory_;
  std::function<std::unique_ptr<ChannelQueueManager>(Package*)>
      queue_manager_factory_;
  bool supports_observers_;
};

// A suite of test which can be run against arbitrary ProcEvaluator
// implementations. Users should instantiate with an INSTANTIATE_TEST_SUITE_P
// macro.
class ProcEvaluatorTestBase
    : public IrTestBase,
      public testing::WithParamInterface<ProcEvaluatorTestParam> {};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_EVALUATOR_TEST_BASE_H_

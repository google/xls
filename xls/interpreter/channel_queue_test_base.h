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

#ifndef XLS_INTERPRETER_CHANNEL_QUEUE_TEST_BASE_H_
#define XLS_INTERPRETER_CHANNEL_QUEUE_TEST_BASE_H_

#include <functional>
#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {

class ChannelQueueTestParam {
 public:
  // `new_queue` is a factory which creates a channel queue to test.
  explicit ChannelQueueTestParam(
      std::function<std::unique_ptr<ChannelQueue>(ChannelInstance*)> new_queue)
      : new_queue_(std::move(new_queue)) {}

  std::unique_ptr<ChannelQueue> CreateQueue(
      ChannelInstance* channel_instance) const {
    return new_queue_(channel_instance);
  }

 private:
  std::function<std::unique_ptr<ChannelQueue>(ChannelInstance*)> new_queue_;
};

// A suite of test which can be run against arbitrary ChannelQueue
// implementations. Users should instantiate with an INSTANTIATE_TEST_SUITE_P
// macro.
class ChannelQueueTestBase
    : public IrTestBase,
      public testing::WithParamInterface<ChannelQueueTestParam> {};

}  // namespace xls

#endif  // XLS_INTERPRETER_CHANNEL_QUEUE_TEST_BASE_H_

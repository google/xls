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
#include "xls/common/pointer_utils.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {

// Helper which deals with destruction of the queue in a type-erased way.
struct ChannelQueueTestData {
  // Opaque holder that must keep 'queue' alive and do any cleanup in the
  // destructor.
  TypeErasedUniquePtr data_holder_;
  ChannelQueue* queue;

  // Transparent forwarding.
  ChannelQueue* operator->() { return queue; }
  const ChannelQueue* operator->() const { return queue; }
};

class ChannelQueueTestParam {
 public:
  // `new_queue` is a factory which creates a channel queue to test.
  explicit ChannelQueueTestParam(
      std::function<ChannelQueueTestData(ChannelInstance*)> new_queue)
      : new_queue_(std::move(new_queue)) {}

  // Create a param that doesn't need any specific destructor.
  static ChannelQueueTestParam Basic(
      std::function<std::unique_ptr<ChannelQueue>(ChannelInstance*)>
          new_queue) {
    return ChannelQueueTestParam(
        [new_queue = std::move(new_queue)](
            ChannelInstance* channel_instance) -> ChannelQueueTestData {
          auto queue = new_queue(channel_instance);
          auto* ptr = queue.get();
          return ChannelQueueTestData{
              .data_holder_ = EraseType(std::move(queue)),
              .queue = ptr,
          };
        });
  }

  ChannelQueueTestData CreateQueue(ChannelInstance* channel_instance) const {
    return new_queue_(channel_instance);
  }

 private:
  std::function<ChannelQueueTestData(ChannelInstance*)> new_queue_;
};

// A suite of test which can be run against arbitrary ChannelQueue
// implementations. Users should instantiate with an INSTANTIATE_TEST_SUITE_P
// macro.
class ChannelQueueTestBase
    : public IrTestBase,
      public testing::WithParamInterface<ChannelQueueTestParam> {};

}  // namespace xls

#endif  // XLS_INTERPRETER_CHANNEL_QUEUE_TEST_BASE_H_

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

#include "xls/jit/jit_channel_queue.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/pointer_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/channel_queue_test_base.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

template <typename JitQueue>
struct JitRuntimeInfo {
  std::unique_ptr<OrcJit> orc_jit;
  std::unique_ptr<JitRuntime> jit_runtime;
  std::unique_ptr<JitQueue> jit_queue;

  JitRuntimeInfo(const JitRuntimeInfo&) = delete;
  JitRuntimeInfo& operator=(const JitRuntimeInfo&) = delete;
  JitRuntimeInfo(JitRuntimeInfo&&) = default;
  JitRuntimeInfo& operator=(JitRuntimeInfo&&) = default;

  explicit JitRuntimeInfo(ChannelInstance* instance)
      : orc_jit(OrcJit::Create().value()),
        jit_runtime(
            std::make_unique<JitRuntime>(orc_jit->CreateDataLayout().value())),
        jit_queue(std::make_unique<JitQueue>(instance, jit_runtime.get())) {}
};

template <typename JitQueue>
ChannelQueueTestData CreateTestData(ChannelInstance* channel_instance) {
  auto info = std::make_unique<JitRuntimeInfo<JitQueue>>(channel_instance);
  ChannelQueue* queue = info->jit_queue.get();
  return ChannelQueueTestData{
      .data_holder_ = EraseType(std::move(info)),
      .queue = queue,
  };
}

INSTANTIATE_TEST_SUITE_P(ThreadSafeJitChannelQueueTest, ChannelQueueTestBase,
                         testing::Values(ChannelQueueTestParam(
                             CreateTestData<ThreadSafeJitChannelQueue>)));

INSTANTIATE_TEST_SUITE_P(LockLessJitChannelQueueTest, ChannelQueueTestBase,
                         testing::Values(ChannelQueueTestParam(
                             CreateTestData<ThreadUnsafeJitChannelQueue>)));

template <typename QueueT>
class JitChannelQueueTest : public ::testing::Test {};

using QueueTypes =
    ::testing::Types<ThreadSafeJitChannelQueue, ThreadUnsafeJitChannelQueue>;
TYPED_TEST_SUITE(JitChannelQueueTest, QueueTypes);

// An empty tuple represents a zero width.
TYPED_TEST(JitChannelQueueTest, ChannelWithEmptyTuple) {
  Package package("test");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetTupleType({})));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));

  JitRuntimeInfo<TypeParam> info(
      elaboration.GetUniqueInstance(channel).value());
  TypeParam& queue = *info.jit_queue;

  EXPECT_TRUE(queue.IsEmpty());
  std::vector<uint8_t> send_buffer(0);
  std::vector<uint8_t> recv_buffer(0);
  // Send and receive immediately.
  for (int64_t i = 0; i < 10; i++) {
    queue.WriteRaw(send_buffer.data());
    EXPECT_FALSE(queue.IsEmpty());
    EXPECT_TRUE(queue.ReadRaw(recv_buffer.data()));
    EXPECT_TRUE(queue.IsEmpty());
  }

  EXPECT_FALSE(queue.ReadRaw(recv_buffer.data()));

  // Send then receive.
  for (int64_t i = 0; i < 10; i++) {
    queue.WriteRaw(send_buffer.data());
  }
  for (int64_t i = 0; i < 10; i++) {
    EXPECT_TRUE(queue.ReadRaw(recv_buffer.data()));
  }
  EXPECT_TRUE(queue.IsEmpty());
}

TYPED_TEST(JitChannelQueueTest, BasicAccess) {
  Package package("test");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));

  JitRuntimeInfo<TypeParam> info(
      elaboration.GetUniqueInstance(channel).value());
  TypeParam& queue = *info.jit_queue;

  EXPECT_TRUE(queue.IsEmpty());
  std::vector<uint8_t> send_buffer(4);
  std::vector<uint8_t> recv_buffer(4);
  // Send and receive immediately.
  for (int64_t i = 0; i < 10; i++) {
    send_buffer[0] = i;
    queue.WriteRaw(send_buffer.data());
    EXPECT_FALSE(queue.IsEmpty());
    EXPECT_TRUE(queue.ReadRaw(recv_buffer.data()));
    EXPECT_THAT(recv_buffer[0], i);
    EXPECT_TRUE(queue.IsEmpty());
  }

  EXPECT_FALSE(queue.ReadRaw(recv_buffer.data()));

  // Send then receive.
  for (int64_t i = 0; i < 10; i++) {
    send_buffer[0] = i;
    queue.WriteRaw(send_buffer.data());
  }
  for (int64_t i = 0; i < 10; i++) {
    EXPECT_TRUE(queue.ReadRaw(recv_buffer.data()));
    EXPECT_THAT(recv_buffer[0], i);
  }
  EXPECT_TRUE(queue.IsEmpty());
}

TYPED_TEST(JitChannelQueueTest, IotaGeneratorWithRawApi) {
  Package package("test");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));

  JitRuntimeInfo<TypeParam> info(
      elaboration.GetUniqueInstance(channel).value());
  TypeParam& queue = *info.jit_queue;

  int64_t counter = 42;
  XLS_ASSERT_OK(queue.AttachGenerator(
      [&]() -> std::optional<Value> { return Value(UBits(counter++, 32)); }));

  auto read_u32 = [&]() {
    std::vector<uint8_t> recv_buffer(4);
    EXPECT_TRUE(queue.ReadRaw(recv_buffer.data()));
    uint32_t result;
    memcpy(&result, recv_buffer.data(), 4);
    return result;
  };

  EXPECT_EQ(read_u32(), 42);
  EXPECT_EQ(read_u32(), 43);
  EXPECT_EQ(read_u32(), 44);
  EXPECT_EQ(read_u32(), 45);

  EXPECT_THAT(queue.Write(Value(UBits(22, 32))),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Cannot write to ChannelQueue because it has "
                                 "a generator function")));
}

}  // namespace
}  // namespace xls

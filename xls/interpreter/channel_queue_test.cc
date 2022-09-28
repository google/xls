// Copyright 2020 The XLS Authors
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

#include "xls/interpreter/channel_queue.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

class ChannelQueueTest : public IrTestBase {};

TEST_F(ChannelQueueTest, FifoChannelQueueTest) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  ChannelQueue queue(channel);
  EXPECT_EQ(queue.channel(), channel);
  EXPECT_EQ(queue.GetSize(), 0);
  EXPECT_TRUE(queue.IsEmpty());

  XLS_ASSERT_OK(queue.Enqueue(Value(UBits(42, 32))));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_FALSE(queue.IsEmpty());

  XLS_ASSERT_OK(queue.Enqueue(Value(UBits(123, 32))));
  EXPECT_EQ(queue.GetSize(), 2);
  EXPECT_FALSE(queue.IsEmpty());

  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(42, 32))));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_FALSE(queue.IsEmpty());

  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(123, 32))));
  EXPECT_EQ(queue.GetSize(), 0);
  EXPECT_TRUE(queue.IsEmpty());

  EXPECT_EQ(queue.Dequeue(), std::nullopt);
}

TEST_F(ChannelQueueTest, ErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(1)));

  ChannelQueue queue(channel);

  EXPECT_EQ(queue.Dequeue(), std::nullopt);

  EXPECT_THAT(queue.Enqueue(Value(UBits(44, 123))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Channel my_channel expects values to have "
                                 "type bits[1], got: bits[123]:0x2c")));
}

TEST_F(ChannelQueueTest, IotaGenerator) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  ChannelQueue queue(channel);
  int64_t counter = 42;
  XLS_ASSERT_OK(queue.AttachGenerator(
      [&]() -> std::optional<Value> { return Value(UBits(counter++, 32)); }));

  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(42, 32))));
  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(43, 32))));
  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(44, 32))));

  EXPECT_THAT(queue.Enqueue(Value(UBits(22, 32))),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Cannot write to ChannelQueue because it has "
                                 "a generator function")));
}

TEST_F(ChannelQueueTest, FixedValueGenerator) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  ChannelQueue queue(channel);
  XLS_ASSERT_OK(queue.AttachGenerator(FixedValueGenerator(
      {Value(UBits(22, 32)), Value(UBits(44, 32)), Value(UBits(55, 32))})));

  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(22, 32))));
  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(44, 32))));
  EXPECT_THAT(queue.Dequeue(), Optional(Value(UBits(55, 32))));

  EXPECT_EQ(queue.Dequeue(), std::nullopt);
}

TEST_F(ChannelQueueTest, EmptyGenerator) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  ChannelQueue queue(channel);
  XLS_ASSERT_OK(queue.AttachGenerator(
      []() -> std::optional<Value> { return std::nullopt; }));

  EXPECT_EQ(queue.Dequeue(), std::nullopt);
}

TEST_F(ChannelQueueTest, SimpleChannelQueueManager) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateStreamingChannel("c", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ChannelQueueManager> manager,
                           ChannelQueueManager::Create(&package));
  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_EQ(manager->GetQueue(channel_a).channel(), channel_a);
  EXPECT_EQ(manager->GetQueue(channel_b).channel(), channel_b);
  EXPECT_EQ(manager->GetQueue(channel_c).channel(), channel_c);
}

TEST_F(ChannelQueueTest, ChannelQueueManagerNoChannels) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ChannelQueueManager> manager,
                           ChannelQueueManager::Create(&package));
  EXPECT_EQ(manager->queues().size(), 0);
}

TEST_F(ChannelQueueTest, ChannelQueueManagerCustomQueues) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateStreamingChannel("c", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));

  std::vector<std::unique_ptr<ChannelQueue>> queues;
  queues.push_back(std::make_unique<ChannelQueue>(channel_a));
  ChannelQueue* queue_a = queues.back().get();
  queues.push_back(std::make_unique<ChannelQueue>(channel_b));
  ChannelQueue* queue_b = queues.back().get();
  queues.push_back(std::make_unique<ChannelQueue>(channel_c));
  ChannelQueue* queue_c = queues.back().get();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> manager,
      ChannelQueueManager::Create(std::move(queues), &package));

  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_EQ(&manager->GetQueue(channel_a), queue_a);
  EXPECT_EQ(&manager->GetQueue(channel_b), queue_b);
  EXPECT_EQ(&manager->GetQueue(channel_c), queue_c);
}

TEST_F(ChannelQueueTest, ChannelQueueManagerErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateStreamingChannel("c", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));

  {
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    queues.push_back(std::make_unique<ChannelQueue>(channel_a));
    queues.push_back(std::make_unique<ChannelQueue>(channel_b));
    queues.push_back(std::make_unique<ChannelQueue>(channel_c));
    queues.push_back(std::make_unique<ChannelQueue>(channel_a));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(queues), &package).status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Multiple queues specified for channel `a`")));
  }

  {
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    queues.push_back(std::make_unique<ChannelQueue>(channel_a));
    queues.push_back(std::make_unique<ChannelQueue>(channel_b));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(queues), &package).status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("No queue specified for channel `c`")));
  }
}

}  // namespace
}  // namespace xls

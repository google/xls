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

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

class ChannelQueueTest : public IrTestBase {};

TEST_F(ChannelQueueTest, SingleDataElementEnqueueDequeue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  ChannelQueue queue(channel);
  EXPECT_EQ(queue.channel(), channel);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());

  XLS_ASSERT_OK(queue.Enqueue(Value(UBits(42, 32))));
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.empty());

  XLS_ASSERT_OK(queue.Enqueue(Value(UBits(123, 32))));
  EXPECT_EQ(queue.size(), 2);
  EXPECT_FALSE(queue.empty());

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.empty());

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(123, 32))));
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
}

TEST_F(ChannelQueueTest, ErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(1)));

  ChannelQueue queue(channel);

  EXPECT_THAT(
      queue.Dequeue(),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr(
              "Attempting to dequeue data from empty channel my_channel")));

  EXPECT_THAT(queue.Enqueue(Value(UBits(44, 123))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Channel my_channel expects values to have "
                                 "type bits[1], got: bits[123]:0x2c")));
}

TEST_F(ChannelQueueTest, InputQueue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  int64_t counter = 42;
  RxOnlyChannelQueue queue(channel, &package, [&]() -> absl::StatusOr<Value> {
    return Value(UBits(counter++, 32));
  });
  EXPECT_FALSE(queue.empty());
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(43, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(44, 32))));
  EXPECT_FALSE(queue.empty());

  EXPECT_THAT(
      queue.Enqueue(Value(UBits(22, 32))),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr(
              "Cannot enqueue to RxOnlyChannelQueue on channel my_channel")));
}

TEST_F(ChannelQueueTest, FixedInputQueue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  FixedRxOnlyChannelQueue queue(
      channel, &package,
      {Value(UBits(22, 32)), Value(UBits(44, 32)), Value(UBits(55, 32))});
  EXPECT_EQ(queue.size(), 3);
  EXPECT_FALSE(queue.empty());
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(22, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(44, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(55, 32))));
  EXPECT_TRUE(queue.empty());
}

TEST_F(ChannelQueueTest, EmptyFixedInputQueue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  FixedRxOnlyChannelQueue queue(channel, &package, {});
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
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

  std::vector<std::unique_ptr<RxOnlyChannelQueue>> rx_only_queues;
  rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
      channel_a, &package,
      []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
  std::vector<Value> fixed_input = {Value(UBits(1, 32)), Value(UBits(2, 32)),
                                    Value(UBits(3, 32))};
  rx_only_queues.push_back(absl::make_unique<FixedRxOnlyChannelQueue>(
      channel_b, &package, fixed_input));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> manager,
      ChannelQueueManager::Create(std::move(rx_only_queues), &package));
  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_THAT(manager->GetQueue(channel_a).Dequeue(),
              IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_THAT(manager->GetQueue(channel_a).Dequeue(),
              IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_THAT(manager->GetQueue(channel_b).Dequeue(),
              IsOkAndHolds(Value(UBits(1, 32))));
  EXPECT_THAT(manager->GetQueue(channel_b).Dequeue(),
              IsOkAndHolds(Value(UBits(2, 32))));
  EXPECT_THAT(manager->GetQueue(channel_b).Dequeue(),
              IsOkAndHolds(Value(UBits(3, 32))));
  EXPECT_TRUE(manager->GetQueue(channel_c).empty());
}

TEST_F(ChannelQueueTest, ChannelQueueManagerNoChannels) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> manager,
      ChannelQueueManager::Create(/*rx_only_queues=*/{}, &package));
  EXPECT_EQ(manager->queues().size(), 0);
}

TEST_F(ChannelQueueTest, ChannelQueueManagerPortChannel) {
  Package package(TestName());
  XLS_ASSERT_OK(package
                    .CreatePortChannel("ch", ChannelOps::kSendReceive,
                                       package.GetBitsType(32))
                    .status());
  EXPECT_THAT(
      ChannelQueueManager::Create(/*rx_only_queues=*/{}, &package).status(),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Only streaming channels are supported.")));
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

  EXPECT_THAT(
      ChannelQueueManager::Create(/*rx_only_queues=*/{}, &package).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "No receive-only queue specified for receive_only channel a")));

  EXPECT_THAT(
      ChannelQueueManager::Create(/*rx_only_queues=*/{}, &package).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "No receive-only queue specified for receive_only channel a")));
  {
    std::vector<std::unique_ptr<RxOnlyChannelQueue>> rx_only_queues;
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_a, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(rx_only_queues), &package)
            .status(),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr(
                "No receive-only queue specified for receive_only channel b")));
  }

  {
    std::vector<std::unique_ptr<RxOnlyChannelQueue>> rx_only_queues;
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_a, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_a, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_b, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(rx_only_queues), &package)
            .status(),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("More than one receive-only queue given for channel a")));
  }

  {
    std::vector<std::unique_ptr<RxOnlyChannelQueue>> rx_only_queues;
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_a, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_b, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_c, &package,
        []() -> absl::StatusOr<Value> { return Value(UBits(42, 32)); }));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(rx_only_queues), &package)
            .status(),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("receive-only queues only can be used with receive_only "
                      "channels, used with send_receive channel c")));
  }
}

}  // namespace
}  // namespace xls

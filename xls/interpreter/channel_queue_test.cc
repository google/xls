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
#include "xls/common/status/matchers.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class ChannelQueueTest : public IrTestBase {};

TEST_F(ChannelQueueTest, SingleDataElementEnqueueDequeue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateChannel("my_channel", ChannelKind::kSendReceive,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  ChannelQueue queue(channel, &package);
  EXPECT_EQ(queue.channel(), channel);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());

  XLS_ASSERT_OK(queue.Enqueue({Value(UBits(42, 32))}));
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.empty());

  XLS_ASSERT_OK(queue.Enqueue({Value(UBits(123, 32))}));
  EXPECT_EQ(queue.size(), 2);
  EXPECT_FALSE(queue.empty());

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(42, 32)))));
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.empty());

  EXPECT_THAT(queue.Dequeue(),
              IsOkAndHolds(ElementsAre(Value(UBits(123, 32)))));
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
}

TEST_F(ChannelQueueTest, MultipleDataElementEnqueueDequeue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateChannel("my_channel", ChannelKind::kSendReceive,
                            {DataElement{"bool", package.GetBitsType(1)},
                             DataElement{"int", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  ChannelQueue queue(channel, &package);
  XLS_ASSERT_OK(queue.Enqueue({Value(UBits(0, 1)), Value(UBits(42, 32))}));
  XLS_ASSERT_OK(queue.Enqueue({Value(UBits(1, 1)), Value(UBits(123, 32))}));
  XLS_ASSERT_OK(queue.Enqueue({Value(UBits(1, 1)), Value(UBits(555, 32))}));

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(0, 1)),
                                                        Value(UBits(42, 32)))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(
                                   Value(UBits(1, 1)), Value(UBits(123, 32)))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(
                                   Value(UBits(1, 1)), Value(UBits(555, 32)))));
}

TEST_F(ChannelQueueTest, ErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateChannel("my_channel", ChannelKind::kSendReceive,
                            {DataElement{"bool", package.GetBitsType(1)},
                             DataElement{"int", package.GetBitsType(32)}},
                            ChannelMetadataProto()));

  ChannelQueue queue(channel, &package);

  EXPECT_THAT(
      queue.Dequeue(),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr(
              "Attempting to dequeue data from empty channel my_channel")));

  EXPECT_THAT(
      queue.Enqueue({Value(UBits(1, 1))}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Channel my_channel expects 2 data elements, got 1")));

  EXPECT_THAT(queue.Enqueue({Value(UBits(1, 1)), Value(UBits(123, 555))}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Channel my_channel expects data element 1 to "
                                 "have type bits[32], got bits[555]")));
}

TEST_F(ChannelQueueTest, InputQueue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateChannel("my_channel", ChannelKind::kReceiveOnly,
                            {DataElement{"int", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  int64 counter = 42;
  RxOnlyChannelQueue queue(channel, &package,
                           [&]() -> absl::StatusOr<ChannelData> {
                             ChannelData data({Value(UBits(counter, 32))});
                             ++counter;
                             return std::move(data);
                           });
  EXPECT_FALSE(queue.empty());
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(42, 32)))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(43, 32)))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(44, 32)))));
  EXPECT_FALSE(queue.empty());

  EXPECT_THAT(
      queue.Enqueue(ChannelData({Value(UBits(22, 32))})),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr(
              "Cannot enqueue to RxOnlyChannelQueue on channel my_channel")));
}

TEST_F(ChannelQueueTest, FixedInputQueue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateChannel("my_channel", ChannelKind::kReceiveOnly,
                            {DataElement{"int", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  FixedRxOnlyChannelQueue queue(
      channel, &package,
      {ChannelData({Value(UBits(22, 32))}), ChannelData({Value(UBits(44, 32))}),
       ChannelData({Value(UBits(55, 32))})});
  EXPECT_EQ(queue.size(), 3);
  EXPECT_FALSE(queue.empty());
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(22, 32)))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(44, 32)))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(ElementsAre(Value(UBits(55, 32)))));
  EXPECT_TRUE(queue.empty());
}

TEST_F(ChannelQueueTest, EmptyFixedInputQueue) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateChannel("my_channel", ChannelKind::kReceiveOnly,
                            {DataElement{"int", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  FixedRxOnlyChannelQueue queue(channel, &package, {});
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
}

TEST_F(ChannelQueueTest, SimpleChannelQueueManager) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateChannel("a", ChannelKind::kReceiveOnly,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateChannel("b", ChannelKind::kReceiveOnly,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateChannel("c", ChannelKind::kSendReceive,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));

  std::vector<std::unique_ptr<RxOnlyChannelQueue>> rx_only_queues;
  rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
      channel_a, &package, []() -> absl::StatusOr<ChannelData> {
        return ChannelData({Value(UBits(42, 32))});
      }));
  std::vector<ChannelData> fixed_input = {ChannelData({Value(UBits(1, 32))}),
                                          ChannelData({Value(UBits(2, 32))}),
                                          ChannelData({Value(UBits(3, 32))})};
  rx_only_queues.push_back(absl::make_unique<FixedRxOnlyChannelQueue>(
      channel_b, &package, fixed_input));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> manager,
      ChannelQueueManager::Create(std::move(rx_only_queues), &package));
  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_THAT(manager->GetQueue(channel_a).Dequeue(),
              IsOkAndHolds(ElementsAre(Value(UBits(42, 32)))));
  EXPECT_THAT(manager->GetQueue(channel_a).Dequeue(),
              IsOkAndHolds(ElementsAre(Value(UBits(42, 32)))));
  EXPECT_THAT(manager->GetQueue(channel_b).Dequeue(),
              IsOkAndHolds(ElementsAre(Value(UBits(1, 32)))));
  EXPECT_THAT(manager->GetQueue(channel_b).Dequeue(),
              IsOkAndHolds(ElementsAre(Value(UBits(2, 32)))));
  EXPECT_THAT(manager->GetQueue(channel_b).Dequeue(),
              IsOkAndHolds(ElementsAre(Value(UBits(3, 32)))));
  EXPECT_TRUE(manager->GetQueue(channel_c).empty());
}

TEST_F(ChannelQueueTest, ChannelQueueManagerNoChannels) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> manager,
      ChannelQueueManager::Create(/*rx_only_queues=*/{}, &package));
  EXPECT_EQ(manager->queues().size(), 0);
}

TEST_F(ChannelQueueTest, ChannelQueueManagerErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateChannel("a", ChannelKind::kReceiveOnly,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateChannel("b", ChannelKind::kReceiveOnly,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateChannel("c", ChannelKind::kSendReceive,
                            {DataElement{"data", package.GetBitsType(32)}},
                            ChannelMetadataProto()));

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
        channel_a, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
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
        channel_a, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_a, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_b, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
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
        channel_a, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_b, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
    rx_only_queues.push_back(absl::make_unique<RxOnlyChannelQueue>(
        channel_c, &package, []() -> absl::StatusOr<ChannelData> {
          return ChannelData({Value(UBits(42, 32))});
        }));
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

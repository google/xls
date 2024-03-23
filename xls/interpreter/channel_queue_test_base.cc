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

#include "xls/interpreter/channel_queue_test_base.h"

#include <cstdint>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

TEST_P(ChannelQueueTestBase, FifoChannelQueueTest) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());

  EXPECT_EQ(queue->channel(), channel);
  EXPECT_EQ(queue->GetSize(), 0);
  EXPECT_TRUE(queue->IsEmpty());

  XLS_ASSERT_OK(queue->Write(Value(UBits(42, 32))));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_FALSE(queue->IsEmpty());

  XLS_ASSERT_OK(queue->Write(Value(UBits(123, 32))));
  EXPECT_EQ(queue->GetSize(), 2);
  EXPECT_FALSE(queue->IsEmpty());

  EXPECT_THAT(queue->Read(), Optional(Value(UBits(42, 32))));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_FALSE(queue->IsEmpty());

  EXPECT_THAT(queue->Read(), Optional(Value(UBits(123, 32))));
  EXPECT_EQ(queue->GetSize(), 0);
  EXPECT_TRUE(queue->IsEmpty());

  EXPECT_EQ(queue->Read(), std::nullopt);
}

TEST_P(ChannelQueueTestBase, SingleValueChannelQueueTest) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateSingleValueChannel("my_channel", ChannelOps::kSendReceive,
                                       package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());

  EXPECT_EQ(queue->channel(), channel);
  EXPECT_EQ(queue->GetSize(), 0);
  EXPECT_TRUE(queue->IsEmpty());
  EXPECT_EQ(queue->Read(), std::nullopt);

  XLS_ASSERT_OK(queue->Write(Value(UBits(42, 32))));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(42, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(42, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(42, 32))));
  EXPECT_EQ(queue->GetSize(), 1);

  XLS_ASSERT_OK(queue->Write(Value(UBits(123, 32))));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(123, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(123, 32))));
  EXPECT_EQ(queue->GetSize(), 1);

  XLS_ASSERT_OK(queue->Write(Value(UBits(10, 32))));
  XLS_ASSERT_OK(queue->Write(Value(UBits(20, 32))));
  XLS_ASSERT_OK(queue->Write(Value(UBits(30, 32))));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(30, 32))));
}

TEST_P(ChannelQueueTestBase, ErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());

  EXPECT_EQ(queue->Read(), std::nullopt);

  EXPECT_THAT(queue->Write(Value(UBits(44, 123))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Channel `my_channel` expects values to have "
                                 "type bits[1], got: bits[123]:0x2c")));
}

TEST_P(ChannelQueueTestBase, IotaGenerator) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());

  int64_t counter = 42;
  XLS_ASSERT_OK(queue->AttachGenerator(
      [&]() -> std::optional<Value> { return Value(UBits(counter++, 32)); }));

  EXPECT_THAT(queue->Read(), Optional(Value(UBits(42, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(43, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(44, 32))));

  EXPECT_THAT(queue->Write(Value(UBits(22, 32))),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Cannot write to ChannelQueue because it has "
                                 "a generator function")));
}

TEST_P(ChannelQueueTestBase, FixedValueGenerator) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());

  XLS_ASSERT_OK(queue->AttachGenerator(FixedValueGenerator(
      {Value(UBits(22, 32)), Value(UBits(44, 32)), Value(UBits(55, 32))})));

  EXPECT_THAT(queue->Read(), Optional(Value(UBits(22, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(44, 32))));
  EXPECT_THAT(queue->Read(), Optional(Value(UBits(55, 32))));

  EXPECT_EQ(queue->Read(), std::nullopt);
}

TEST_P(ChannelQueueTestBase, EmptyGenerator) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());
  XLS_ASSERT_OK(queue->AttachGenerator(
      []() -> std::optional<Value> { return std::nullopt; }));

  EXPECT_EQ(queue->Read(), std::nullopt);
}

TEST_P(ChannelQueueTestBase, ChannelWithEmptyTuple) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                     package.GetTupleType({})));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elaboration,
                           ProcElaboration::ElaborateOldStylePackage(&package));
  auto queue =
      GetParam().CreateQueue(elaboration.GetUniqueInstance(channel).value());

  EXPECT_EQ(queue->channel(), channel);
  EXPECT_EQ(queue->GetSize(), 0);
  EXPECT_TRUE(queue->IsEmpty());

  XLS_ASSERT_OK(queue->Write(Value::Tuple({})));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_FALSE(queue->IsEmpty());

  XLS_ASSERT_OK(queue->Write(Value::Tuple({})));
  EXPECT_EQ(queue->GetSize(), 2);
  EXPECT_FALSE(queue->IsEmpty());

  EXPECT_THAT(queue->Read(), Optional(Value::Tuple({})));
  EXPECT_EQ(queue->GetSize(), 1);
  EXPECT_FALSE(queue->IsEmpty());

  EXPECT_THAT(queue->Read(), Optional(Value::Tuple({})));
  EXPECT_EQ(queue->GetSize(), 0);
  EXPECT_TRUE(queue->IsEmpty());

  EXPECT_EQ(queue->Read(), std::nullopt);
  EXPECT_EQ(queue->channel(), channel);
  EXPECT_EQ(queue->GetSize(), 0);
  EXPECT_TRUE(queue->IsEmpty());
  EXPECT_EQ(queue->Read(), std::nullopt);
}

}  // namespace
}  // namespace xls

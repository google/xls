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

#include "xls/ir/channel.h"

#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(ChannelTest, ChannelOpsToString) {
  EXPECT_EQ(ChannelOpsToString(ChannelOps::kSendOnly), "send_only");
  EXPECT_EQ(ChannelOpsToString(ChannelOps::kReceiveOnly), "receive_only");
  EXPECT_EQ(ChannelOpsToString(ChannelOps::kSendReceive), "send_receive");

  EXPECT_THAT(StringToChannelOps("send_only"),
              IsOkAndHolds(ChannelOps::kSendOnly));
  EXPECT_THAT(StringToChannelOps("receive_only"),
              IsOkAndHolds(ChannelOps::kReceiveOnly));
  EXPECT_THAT(StringToChannelOps("send_receive"),
              IsOkAndHolds(ChannelOps::kSendReceive));

  EXPECT_THAT(StringToChannelOps("send"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown channel ops")));
}

TEST(ChannelTest, ChannelStrictnessStringConversions) {
  EXPECT_EQ(
      ChannelStrictnessToString(ChannelStrictness::kProvenMutuallyExclusive),
      "proven_mutually_exclusive");
  EXPECT_EQ(
      ChannelStrictnessToString(ChannelStrictness::kRuntimeMutuallyExclusive),
      "runtime_mutually_exclusive");
  EXPECT_EQ(ChannelStrictnessToString(ChannelStrictness::kRuntimeOrdered),
            "runtime_ordered");
  EXPECT_EQ(
      ChannelStrictnessToString(ChannelStrictness::kProvenMutuallyExclusive),
      "proven_mutually_exclusive");
  EXPECT_EQ(ChannelStrictnessToString(ChannelStrictness::kTotalOrder),
            "total_order");

  EXPECT_THAT(ChannelStrictnessFromString("proven_mutually_exclusive"),
              IsOkAndHolds(ChannelStrictness::kProvenMutuallyExclusive));
  EXPECT_THAT(ChannelStrictnessFromString("runtime_mutually_exclusive"),
              IsOkAndHolds(ChannelStrictness::kRuntimeMutuallyExclusive));
  EXPECT_THAT(ChannelStrictnessFromString("runtime_ordered"),
              IsOkAndHolds(ChannelStrictness::kRuntimeOrdered));
  EXPECT_THAT(ChannelStrictnessFromString("proven_mutually_exclusive"),
              IsOkAndHolds(ChannelStrictness::kProvenMutuallyExclusive));
  EXPECT_THAT(ChannelStrictnessFromString("total_order"),
              IsOkAndHolds(ChannelStrictness::kTotalOrder));
}

TEST(ChannelTest, ConstructStreamingChannel) {
  Package p("my_package");
  StreamingChannel ch(
      "my_channel", 42, ChannelOps::kReceiveOnly, p.GetBitsType(32),
      /*initial_values=*/{}, ChannelConfig(), FlowControl::kReadyValid,
      ChannelStrictness::kProvenMutuallyExclusive);

  EXPECT_EQ(ch.name(), "my_channel");
  EXPECT_EQ(ch.kind(), ChannelKind::kStreaming);
  EXPECT_EQ(ch.id(), 42);
  EXPECT_EQ(ch.supported_ops(), ChannelOps::kReceiveOnly);
  EXPECT_EQ(ch.type(), p.GetBitsType(32));
  EXPECT_TRUE(ch.initial_values().empty());
  EXPECT_FALSE(ch.GetFifoDepth().has_value());
  EXPECT_EQ(ch.GetFlowControl(), FlowControl::kReadyValid);
  EXPECT_EQ(ch.GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
}

TEST(ChannelTest, ConstructSingleValueChannel) {
  Package p("my_package");
  SingleValueChannel ch("foo", 42, ChannelOps::kSendOnly, p.GetBitsType(123));

  EXPECT_EQ(ch.name(), "foo");
  EXPECT_EQ(ch.kind(), ChannelKind::kSingleValue);
}

TEST(ChannelTest, StreamingChannelWithInitialValues) {
  Package p("my_package");
  StreamingChannel ch(
      "my_channel", 42, ChannelOps::kSendReceive, p.GetBitsType(32),
      {Value(UBits(11, 32)), Value(UBits(22, 32))}, ChannelConfig(),
      FlowControl::kNone, ChannelStrictness::kProvenMutuallyExclusive);

  EXPECT_EQ(ch.name(), "my_channel");
  EXPECT_EQ(ch.id(), 42);
  EXPECT_EQ(ch.supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch.type(), p.GetBitsType(32));
  EXPECT_THAT(ch.initial_values(),
              ElementsAre(Value(UBits(11, 32)), Value(UBits(22, 32))));
  EXPECT_EQ(ch.GetFlowControl(), FlowControl::kNone);
  EXPECT_EQ(ch.GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
}

TEST(ChannelTest, StreamingChannelWithFifoDepth) {
  Package p("my_package");
  StreamingChannel ch(
      "my_channel", 42, ChannelOps::kSendReceive, p.GetBitsType(32), {},
      ChannelConfig(FifoConfig(/*depth=*/123, /*bypass=*/true,
                               /*register_push_outputs=*/true,
                               /*register_pop_outputs=*/false)),
      FlowControl::kNone, ChannelStrictness::kProvenMutuallyExclusive);

  EXPECT_EQ(ch.name(), "my_channel");
  EXPECT_EQ(ch.id(), 42);
  EXPECT_EQ(ch.supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch.type(), p.GetBitsType(32));
  EXPECT_TRUE(ch.initial_values().empty());
  EXPECT_EQ(ch.GetFifoDepth(), 123);
  EXPECT_EQ(ch.GetFlowControl(), FlowControl::kNone);
  EXPECT_EQ(ch.GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
}

TEST(ChannelTest, StreamingChannelWithFifoConfigSerializesFifoConfigCorrectly) {
  Package p("my_package");

  auto ch_with_fifo_config = [&](FifoConfig fifo_config) {
    return StreamingChannel(
        "my_channel", 42, ChannelOps::kSendReceive, p.GetBitsType(32), {},
        /*channel_config=*/ChannelConfig(fifo_config), FlowControl::kNone,
        ChannelStrictness::kProvenMutuallyExclusive);
  };
  EXPECT_THAT(ch_with_fifo_config(FifoConfig(/*depth=*/123, /*bypass=*/false,
                                             /*register_push_outputs=*/false,
                                             /*register_pop_outputs=*/false))
                  .ToString(),
              AllOf(HasSubstr("fifo_depth=123"), HasSubstr("bypass=false"),
                    HasSubstr("register_push_outputs=false"),
                    HasSubstr("register_pop_outputs=false")));
  EXPECT_THAT(ch_with_fifo_config(FifoConfig(/*depth=*/123, /*bypass=*/true,
                                             /*register_push_outputs=*/false,
                                             /*register_pop_outputs=*/false))
                  .ToString(),
              AllOf(HasSubstr("fifo_depth=123"), HasSubstr("bypass=true"),
                    HasSubstr("register_push_outputs=false"),
                    HasSubstr("register_pop_outputs=false")));
  EXPECT_THAT(ch_with_fifo_config(FifoConfig(/*depth=*/123, /*bypass=*/false,
                                             /*register_push_outputs=*/true,
                                             /*register_pop_outputs=*/false))
                  .ToString(),
              AllOf(HasSubstr("fifo_depth=123"), HasSubstr("bypass=false"),
                    HasSubstr("register_push_outputs=true"),
                    HasSubstr("register_pop_outputs=false")));
  EXPECT_THAT(ch_with_fifo_config(FifoConfig(/*depth=*/123, /*bypass=*/false,
                                             /*register_push_outputs=*/false,
                                             /*register_pop_outputs=*/true))
                  .ToString(),
              AllOf(HasSubstr("fifo_depth=123"), HasSubstr("bypass=false"),
                    HasSubstr("register_push_outputs=false"),
                    HasSubstr("register_pop_outputs=true")));
}

TEST(ChannelTest, StreamingToStringParses) {
  Package p("my_package");
  std::vector<Value> initial_values = {
      Value::Tuple({Value(UBits(1234, 32)), Value(UBits(33, 23))}),
      Value::Tuple({Value(UBits(2222, 32)), Value(UBits(444, 23))})};
  StreamingChannel ch("my_channel", 42, ChannelOps::kReceiveOnly,
                      p.GetTypeForValue(initial_values.front()), initial_values,
                      ChannelConfig(), FlowControl::kReadyValid,
                      ChannelStrictness::kProvenMutuallyExclusive);
  std::string channel_str = ch.ToString();
  EXPECT_EQ(channel_str,
            "chan my_channel((bits[32], bits[23]), initial_values={(1234, 33), "
            "(2222, 444)}, id=42, kind=streaming, ops=receive_only, "
            "flow_control=ready_valid, strictness=proven_mutually_exclusive)");

  // Create another package and try to parse the channel into the other
  // package. We can't use the existing package because adding the channel will
  // fail because the id already exists.
  Package other_p("other_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * parsed_ch,
                           Parser::ParseChannel(channel_str, &other_p));
  EXPECT_EQ(parsed_ch->name(), "my_channel");
  EXPECT_EQ(parsed_ch->id(), 42);
}

TEST(ChannelTest, SingleValueToStringParses) {
  Package p("my_package");
  SingleValueChannel ch("my_channel", 42, ChannelOps::kReceiveOnly,
                        p.GetBitsType(32));
  std::string channel_str = ch.ToString();
  EXPECT_EQ(channel_str,
            "chan my_channel(bits[32], id=42, kind=single_value, "
            "ops=receive_only)");

  // Create another package and try to parse the channel into the other
  // package. We can't use the existing package because adding the channel will
  // fail because the id already exists.
  Package other_p("other_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * parsed_ch,
                           Parser::ParseChannel(channel_str, &other_p));
  EXPECT_EQ(parsed_ch->name(), "my_channel");
  EXPECT_EQ(parsed_ch->id(), 42);
}

TEST(ChannelTest, NameLessThan) {
  Package p("my_package");
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK(
      p.CreateStreamingChannel("c", ChannelOps::kSendOnly, u32).status());
  XLS_ASSERT_OK(
      p.CreateStreamingChannel("b", ChannelOps::kSendOnly, u32).status());
  XLS_ASSERT_OK(
      p.CreateSingleValueChannel("a", ChannelOps::kSendOnly, u32).status());
  absl::btree_set<Channel*, struct Channel::NameLessThan> channel_set(
      p.channels().begin(), p.channels().end());
  EXPECT_THAT(channel_set,
              ElementsAre(m::Channel("a"), m::Channel("b"), m::Channel("c")));
  std::vector<Channel*> channel_vector(p.channels().begin(),
                                       p.channels().end());
  absl::c_sort(channel_vector, Channel::NameLessThan);
  EXPECT_THAT(channel_vector,
              ElementsAre(m::Channel("a"), m::Channel("b"), m::Channel("c")));
}

TEST(ChannelTest, ChannelDirectionTest) {
  EXPECT_EQ(ChannelDirectionToString(ChannelDirection::kSend), "send");
  EXPECT_EQ(ChannelDirectionToString(ChannelDirection::kReceive), "receive");

  EXPECT_THAT(ChannelDirectionFromString("send"),
              IsOkAndHolds(ChannelDirection::kSend));
  EXPECT_THAT(ChannelDirectionFromString("receive"),
              IsOkAndHolds(ChannelDirection::kReceive));
  EXPECT_THAT(ChannelDirectionFromString("foo"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xls

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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(ChannelTest, SupportedOpsToString) {
  EXPECT_EQ(SupportedOpsToString(Channel::SupportedOps::kSendOnly),
            "send_only");
  EXPECT_EQ(SupportedOpsToString(Channel::SupportedOps::kReceiveOnly),
            "receive_only");
  EXPECT_EQ(SupportedOpsToString(Channel::SupportedOps::kSendReceive),
            "send_receive");

  EXPECT_THAT(StringToSupportedOps("send_only"),
              IsOkAndHolds(Channel::SupportedOps::kSendOnly));
  EXPECT_THAT(StringToSupportedOps("receive_only"),
              IsOkAndHolds(Channel::SupportedOps::kReceiveOnly));
  EXPECT_THAT(StringToSupportedOps("send_receive"),
              IsOkAndHolds(Channel::SupportedOps::kSendReceive));

  EXPECT_THAT(StringToSupportedOps("send"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown channel kind")));
}

TEST(ChannelTest, ConstructStreamingChannel) {
  Package p("my_package");
  ChannelMetadataProto metadata;
  metadata.mutable_module_port()->set_flopped(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamingChannel> ch,
      StreamingChannel::Create("my_channel", 42,
                               Channel::SupportedOps::kReceiveOnly,
                               {DataElement{"foo", p.GetBitsType(32)},
                                DataElement{"bar", p.GetBitsType(123)}},
                               metadata));

  EXPECT_EQ(ch->name(), "my_channel");
  EXPECT_TRUE(ch->IsStreaming());
  EXPECT_FALSE(ch->IsPort());
  EXPECT_FALSE(ch->IsRegister());
  EXPECT_FALSE(ch->IsLogical());
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kReceiveOnly);
  EXPECT_EQ(ch->data_elements().size(), 2);
  EXPECT_EQ(ch->data_element(0).name, "foo");
  EXPECT_EQ(ch->data_element(0).type, p.GetBitsType(32));
  EXPECT_EQ(ch->data_element(1).name, "bar");
  EXPECT_EQ(ch->data_element(1).type, p.GetBitsType(123));
  EXPECT_TRUE(ch->initial_values().empty());
}

TEST(ChannelTest, ConstructPortChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PortChannel> ch,
      PortChannel::Create("my_channel", 42, Channel::SupportedOps::kSendOnly,
                          {DataElement{"foo", p.GetBitsType(32)}}));

  EXPECT_EQ(ch->name(), "my_channel");
  EXPECT_FALSE(ch->IsStreaming());
  EXPECT_TRUE(ch->IsPort());
  EXPECT_FALSE(ch->IsRegister());
  EXPECT_FALSE(ch->IsLogical());
}

TEST(ChannelTest, ConstructRegisterChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RegisterChannel> ch,
      RegisterChannel::Create("my_channel", 42,
                              {DataElement{"foo", p.GetBitsType(32)}}));

  EXPECT_EQ(ch->name(), "my_channel");
  EXPECT_FALSE(ch->IsStreaming());
  EXPECT_FALSE(ch->IsPort());
  EXPECT_TRUE(ch->IsRegister());
  EXPECT_FALSE(ch->IsLogical());
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kSendReceive);
}

TEST(ChannelTest, ConstructLogicalChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PortChannel> rdy_ch,
      PortChannel::Create("ready", 42, Channel::SupportedOps::kSendOnly,
                          {DataElement{"rdy", p.GetBitsType(1)}}));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PortChannel> vld_ch,
      PortChannel::Create("valid", 42, Channel::SupportedOps::kSendOnly,
                          {DataElement{"vld", p.GetBitsType(1)}}));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PortChannel> data_ch,
      PortChannel::Create("data", 42, Channel::SupportedOps::kSendOnly,
                          {DataElement{"data", p.GetBitsType(1)}}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LogicalChannel> ch,
      LogicalChannel::Create("my_channel", 42, rdy_ch.get(), vld_ch.get(),
                             data_ch.get()));

  EXPECT_EQ(ch->name(), "my_channel");
  EXPECT_FALSE(ch->IsStreaming());
  EXPECT_FALSE(ch->IsPort());
  EXPECT_FALSE(ch->IsRegister());
  EXPECT_TRUE(ch->IsLogical());
}

TEST(ChannelTest, StreamingChannelWithInitialValues) {
  Package p("my_package");
  std::vector<Value> bar_initial_values = {Value(UBits(11, 32)),
                                           Value(UBits(22, 32))};
  std::vector<Value> foo_initial_values = {Value(UBits(44, 123)),
                                           Value(UBits(55, 123))};
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamingChannel> ch,
      StreamingChannel::Create(
          "my_channel", 42, Channel::SupportedOps::kSendReceive,
          {DataElement{"foo", p.GetBitsType(32), bar_initial_values},
           DataElement{"bar", p.GetBitsType(123), foo_initial_values}}));

  EXPECT_EQ(ch->name(), "my_channel");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kSendReceive);
  EXPECT_EQ(ch->data_elements().size(), 2);
  EXPECT_EQ(ch->data_element(0).name, "foo");
  EXPECT_EQ(ch->data_element(0).type, p.GetBitsType(32));
  EXPECT_EQ(ch->data_element(1).name, "bar");
  EXPECT_EQ(ch->data_element(1).type, p.GetBitsType(123));
  ASSERT_EQ(ch->initial_values().size(), 2);
  EXPECT_THAT(ch->initial_values()[0],
              ElementsAre(Value(UBits(11, 32)), Value(UBits(44, 123))));
  EXPECT_THAT(ch->initial_values()[1],
              ElementsAre(Value(UBits(22, 32)), Value(UBits(55, 123))));
}

TEST(ChannelTest, StreamingToStringParses) {
  Package p("my_package");
  ChannelMetadataProto metadata;
  metadata.mutable_module_port()->set_flopped(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamingChannel> ch,
      StreamingChannel::Create("my_channel", 42,
                               Channel::SupportedOps::kReceiveOnly,
                               {DataElement{"foo", p.GetBitsType(32)},
                                DataElement{"bar", p.GetBitsType(123)}},
                               metadata));
  std::string channel_str = ch->ToString();
  EXPECT_EQ(channel_str,
            "chan my_channel(foo: bits[32], bar: bits[123], id=42, "
            "kind=streaming, ops=receive_only, "
            "metadata=\"\"\"module_port { flopped: true }\"\"\")");

  // Create another package and try to parse the channel into the other
  // package. We can't use the existing package because adding the channel will
  // fail because the id already exists.
  Package other_p("other_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * parsed_ch,
                           Parser::ParseChannel(channel_str, &other_p));
  EXPECT_EQ(parsed_ch->name(), "my_channel");
  EXPECT_EQ(parsed_ch->id(), 42);
}

TEST(ChannelTest, PortToStringParses) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PortChannel> ch,
      PortChannel::Create("my_channel", 42, Channel::SupportedOps::kReceiveOnly,
                          {DataElement{"foo", p.GetBitsType(32)},
                           DataElement{"bar", p.GetBitsType(123)}}));
  std::string channel_str = ch->ToString();
  EXPECT_EQ(channel_str,
            "chan my_channel(foo: bits[32], bar: bits[123], id=42, "
            "kind=port, ops=receive_only, "
            "metadata=\"\"\"\"\"\")");

  // Create another package and try to parse the channel into the other
  // package. We can't use the existing package because adding the channel will
  // fail because the id already exists.
  Package other_p("other_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * parsed_ch,
                           Parser::ParseChannel(channel_str, &other_p));
  EXPECT_EQ(parsed_ch->name(), "my_channel");
  EXPECT_EQ(parsed_ch->id(), 42);
}

}  // namespace
}  // namespace xls

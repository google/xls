// Copyright 2020 Google LLC
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
using ::testing::HasSubstr;

TEST(ChannelTest, ChannelKindToString) {
  EXPECT_EQ(ChannelKindToString(ChannelKind::kSendOnly), "send_only");
  EXPECT_EQ(ChannelKindToString(ChannelKind::kReceiveOnly), "receive_only");
  EXPECT_EQ(ChannelKindToString(ChannelKind::kSendReceive), "send_receive");

  EXPECT_THAT(StringToChannelKind("send_only"),
              IsOkAndHolds(ChannelKind::kSendOnly));
  EXPECT_THAT(StringToChannelKind("receive_only"),
              IsOkAndHolds(ChannelKind::kReceiveOnly));
  EXPECT_THAT(StringToChannelKind("send_receive"),
              IsOkAndHolds(ChannelKind::kSendReceive));

  EXPECT_THAT(StringToChannelKind("send"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown channel kind")));
}

TEST(ChannelTest, ConstructChannel) {
  Package p("my_package");
  ChannelMetadataProto metadata;
  metadata.mutable_module_port()->set_flopped(true);
  Channel ch("my_channel", 42, ChannelKind::kReceiveOnly,
             {DataElement{"foo", p.GetBitsType(32)},
              DataElement{"bar", p.GetBitsType(123)}},
             metadata);

  EXPECT_EQ(ch.name(), "my_channel");
  EXPECT_EQ(ch.id(), 42);
  EXPECT_EQ(ch.kind(), ChannelKind::kReceiveOnly);
  EXPECT_EQ(ch.data_elements().size(), 2);
  EXPECT_EQ(ch.data_element(0).name, "foo");
  EXPECT_EQ(ch.data_element(0).type, p.GetBitsType(32));
  EXPECT_EQ(ch.data_element(1).name, "bar");
  EXPECT_EQ(ch.data_element(1).type, p.GetBitsType(123));
}

TEST(ChannelTest, ToStringParses) {
  Package p("my_package");
  ChannelMetadataProto metadata;
  metadata.mutable_module_port()->set_flopped(true);
  Channel ch("my_channel", 42, ChannelKind::kReceiveOnly,
             {DataElement{"foo", p.GetBitsType(32)},
              DataElement{"bar", p.GetBitsType(123)}},
             metadata);
  std::string ch_to_string = ch.ToString();
  EXPECT_EQ(
      ch_to_string,
      "chan my_channel(foo: bits[32], bar: bits[123], id=42, "
      "kind=receive_only, metadata=\"\"\"module_port { flopped: true }\"\"\")");

  // Create another package and try to parse the channel into the other
  // package. We can't use the existing package because adding the channel will
  // fail because the id already exists.
  Package other_p("other_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * parsed_ch,
                           Parser::ParseChannel(ch_to_string, &other_p));
  EXPECT_EQ(parsed_ch->name(), "my_channel");
  EXPECT_EQ(parsed_ch->id(), 42);
}

}  // namespace
}  // namespace xls

// Copyright 2021 The XLS Authors
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

#include "xls/noc/config_ng/channel_options_proto_builder.h"

#include <string_view>

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(ChannelOptionsProtoBuilderTest, FieldValuesForPtr) {
  constexpr std::string_view kSource = "Source";
  constexpr std::string_view kSink = "Sink";
  ChannelOptionsProto proto;
  ChannelOptionsProtoBuilder builder(&proto);
  builder.SetSourceNodeName(kSource);
  EXPECT_EQ(proto.source_node_name(), kSource);
  builder.SetSinkNodeName(kSink);
  EXPECT_EQ(proto.sink_node_name(), kSink);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(ChannelOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  constexpr std::string_view kSource = "Source";
  constexpr std::string_view kSink = "Sink";
  ChannelOptionsProto default_proto;
  ChannelOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetSourceNodeName(kSource);
  builder_default.SetSinkNodeName(kSink);
  ChannelOptionsProto proto;
  ChannelOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.source_node_name(), kSource);
  EXPECT_EQ(proto.sink_node_name(), kSink);
}

// Test field values of the builder when copied from another builder.
TEST(ChannelOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  constexpr std::string_view kSource = "Source";
  constexpr std::string_view kSink = "Sink";
  ChannelOptionsProto default_proto;
  ChannelOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetSourceNodeName(kSource);
  builder_default.SetSinkNodeName(kSink);
  ChannelOptionsProto proto;
  ChannelOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.source_node_name(), kSource);
  EXPECT_EQ(proto.sink_node_name(), kSink);
}

}  // namespace
}  // namespace xls::noc

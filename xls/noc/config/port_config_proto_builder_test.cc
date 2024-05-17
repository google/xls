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

#include "xls/noc/config/port_config_proto_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

// Test field values for an input port
TEST(PortConfigBuilderTest, InputPortFieldValues) {
  const char* kName = "Test";
  const char* kVirtualChannelName = "VC0";
  PortConfigProto proto;
  PortConfigProtoBuilder builder(&proto);
  builder.WithName(kName);
  builder.AsInputDirection();
  builder.WithVirtualChannel(kVirtualChannelName);

  EXPECT_TRUE(proto.has_name());
  EXPECT_TRUE(proto.has_direction());
  EXPECT_EQ(proto.virtual_channels_size(), 1);
  EXPECT_THAT(proto.name(), kName);
  EXPECT_EQ(proto.direction(), PortConfigProto::INPUT);
  EXPECT_THAT(proto.virtual_channels(0), kVirtualChannelName);
}

// Test field values for an output port
TEST(PortConfigBuilderTest, OutputPortFieldValues) {
  const char* kName = "Test";
  const char* kVirtualChannelName = "VC0";
  PortConfigProto proto;
  PortConfigProtoBuilder builder(&proto);
  builder.WithName(kName);
  builder.AsOutputDirection();
  builder.WithVirtualChannel(kVirtualChannelName);

  EXPECT_TRUE(proto.has_name());
  EXPECT_TRUE(proto.has_direction());
  EXPECT_EQ(proto.virtual_channels_size(), 1);
  EXPECT_THAT(proto.name(), kName);
  EXPECT_EQ(proto.direction(), PortConfigProto::OUTPUT);
  EXPECT_THAT(proto.virtual_channels(0), kVirtualChannelName);
}

// Test last enabled direction of a port.
TEST(PortConfigBuilderTest, LastEnabledPortDirection) {
  PortConfigProto proto;
  PortConfigProtoBuilder builder(&proto);
  EXPECT_FALSE(proto.has_direction());
  builder.AsOutputDirection();
  EXPECT_EQ(proto.direction(), PortConfigProto::OUTPUT);
  builder.AsInputDirection();
  EXPECT_EQ(proto.direction(), PortConfigProto::INPUT);
}

}  // namespace xls::noc

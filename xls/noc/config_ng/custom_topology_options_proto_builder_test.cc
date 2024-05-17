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

#include "xls/noc/config_ng/custom_topology_options_proto_builder.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/channel_options_proto_builder.h"
#include "xls/noc/config_ng/node_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(CustomTopologyOptionsProtoBuilderTest, FieldValuesForPtr) {
  NodeOptionsProto node_proto;
  ChannelOptionsProto channel_proto;
  CustomTopologyOptionsProto proto;
  CustomTopologyOptionsProtoBuilder builder(&proto);
  builder.AddNode(NodeOptionsProtoBuilder(&node_proto));
  EXPECT_EQ(proto.nodes_size(), 1);
  builder.AddChannel(ChannelOptionsProtoBuilder(&channel_proto));
  EXPECT_EQ(proto.channels_size(), 1);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(CustomTopologyOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  CustomTopologyOptionsProto default_proto;
  CustomTopologyOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AddNode();
  builder_default.AddChannel();
  CustomTopologyOptionsProto proto;
  CustomTopologyOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.nodes_size(), 1);
  EXPECT_EQ(proto.channels_size(), 1);
}

// Test field values of the builder when copied from another builder.
TEST(CustomTopologyOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  CustomTopologyOptionsProto default_proto;
  CustomTopologyOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AddNode();
  builder_default.AddChannel();
  CustomTopologyOptionsProto proto;
  CustomTopologyOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.nodes_size(), 1);
  EXPECT_EQ(proto.channels_size(), 1);
}

}  // namespace
}  // namespace xls::noc

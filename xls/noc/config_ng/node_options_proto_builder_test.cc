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

#include "xls/noc/config_ng/node_options_proto_builder.h"

#include <string_view>

#include "gtest/gtest.h"
#include "xls/noc/config_ng/coordinate_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(NodeOptionsProtoBuilderTest, FieldValuesForPtr) {
  constexpr std::string_view kName = "Test";
  CoordinateOptionsProto coordinate_proto;
  NodeOptionsProto proto;
  NodeOptionsProtoBuilder builder(&proto);
  builder.SetName(kName);
  EXPECT_EQ(proto.name(), kName);
  builder.SetSendPortCount(42);
  EXPECT_EQ(proto.send_port_count(), 42);
  builder.SetRecvPortCount(1337);
  EXPECT_EQ(proto.recv_port_count(), 1337);
  builder.AddCoordinate(CoordinateOptionsProtoBuilder(&coordinate_proto));
  EXPECT_EQ(proto.coordinate_size(), 1);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(NodeOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  constexpr std::string_view kName = "Test";
  NodeOptionsProto default_proto;
  NodeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetName(kName);
  builder_default.SetSendPortCount(42);
  builder_default.SetRecvPortCount(1337);
  builder_default.AddCoordinate();
  NodeOptionsProto proto;
  NodeOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.name(), kName);
  EXPECT_EQ(proto.send_port_count(), 42);
  EXPECT_EQ(proto.recv_port_count(), 1337);
  EXPECT_EQ(proto.coordinate_size(), 1);
}

// Test field values of the builder when copied from another builder.
TEST(NodeOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  constexpr std::string_view kName = "Test";
  NodeOptionsProto default_proto;
  NodeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetName(kName);
  builder_default.SetSendPortCount(42);
  builder_default.SetRecvPortCount(1337);
  builder_default.AddCoordinate();
  NodeOptionsProto proto;
  NodeOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.name(), kName);
  EXPECT_EQ(proto.send_port_count(), 42);
  EXPECT_EQ(proto.recv_port_count(), 1337);
  EXPECT_EQ(proto.coordinate_size(), 1);
}

}  // namespace
}  // namespace xls::noc

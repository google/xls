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

#include "xls/noc/config_ng/topology_endpoint_options_proto_builder.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(TopologyEndpointOptionsProtoBuilderTest, FieldValuesForPtr) {
  constexpr int64_t kSendPortCount = 42;
  constexpr int64_t kRecvPortCount = 1337;
  TopologyEndpointOptionsProto proto;
  TopologyEndpointOptionsProtoBuilder builder(&proto);
  builder.SetSendPortCount(kSendPortCount);
  EXPECT_EQ(proto.send_port_count(), kSendPortCount);
  builder.SetRecvPortCount(kRecvPortCount);
  EXPECT_EQ(proto.recv_port_count(), kRecvPortCount);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(TopologyEndpointOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  constexpr int64_t kSendPortCount = 42;
  constexpr int64_t kRecvPortCount = 1337;
  TopologyEndpointOptionsProto default_proto;
  TopologyEndpointOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetSendPortCount(kSendPortCount);
  builder_default.SetRecvPortCount(kRecvPortCount);
  TopologyEndpointOptionsProto proto;
  TopologyEndpointOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.send_port_count(), kSendPortCount);
  EXPECT_EQ(proto.recv_port_count(), kRecvPortCount);
}

// Test field values of the builder when copied from another builder.
TEST(TopologyEndpointOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  constexpr int64_t kSendPortCount = 42;
  constexpr int64_t kRecvPortCount = 1337;
  TopologyEndpointOptionsProto default_proto;
  TopologyEndpointOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetSendPortCount(kSendPortCount);
  builder_default.SetRecvPortCount(kRecvPortCount);
  TopologyEndpointOptionsProto proto;
  TopologyEndpointOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.send_port_count(), kSendPortCount);
  EXPECT_EQ(proto.recv_port_count(), kRecvPortCount);
}

}  // namespace
}  // namespace xls::noc

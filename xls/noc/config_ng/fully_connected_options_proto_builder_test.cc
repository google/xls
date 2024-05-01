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

#include "xls/noc/config_ng/fully_connected_options_proto_builder.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/topology_endpoint_options_proto_builder.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(FullyConnectedOptionsProtoBuilderTest, FieldValuesForPtr) {
  TopologyEndpointOptionsProto topology_proto;
  FullyConnectedOptionsProto proto;
  FullyConnectedOptionsProtoBuilder builder(&proto);
  builder.SetRouterCount(42);
  EXPECT_EQ(proto.router_count(), 42);
  builder.SetEndpointOptions(
      TopologyEndpointOptionsProtoBuilder(&topology_proto));
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(FullyConnectedOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  FullyConnectedOptionsProto default_proto;
  FullyConnectedOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetRouterCount(42);
  builder_default.GetEndpointOptions();
  FullyConnectedOptionsProto proto;
  FullyConnectedOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.router_count(), 42);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the builder when copied from another builder.
TEST(FullyConnectedOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  FullyConnectedOptionsProto default_proto;
  FullyConnectedOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetRouterCount(42);
  builder_default.GetEndpointOptions();
  FullyConnectedOptionsProto proto;
  FullyConnectedOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.router_count(), 42);
  EXPECT_TRUE(proto.has_endpoint_options());
}

}  // namespace
}  // namespace xls::noc

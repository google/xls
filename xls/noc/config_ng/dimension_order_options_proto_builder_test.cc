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

#include "xls/noc/config_ng/dimension_order_options_proto_builder.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/topology_endpoint_options_proto_builder.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(DimensionOrderOptionsProtoBuilderTest, FieldValuesForPtr) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderEntryOptionsProto dim_entry_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.AddDimension();
  EXPECT_EQ(proto.dimensions_size(), 1);
  builder.AddDimension(
      DimensionOrderEntryOptionsProtoBuilder(&dim_entry_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  builder.SetEndpointOptions(
      TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(DimensionOrderOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderEntryOptionsProto dim_entry_proto;
  DimensionOrderOptionsProto default_proto;
  DimensionOrderOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AddDimension();
  builder_default.AddDimension(
      DimensionOrderEntryOptionsProtoBuilder(&dim_entry_proto));
  builder_default.SetEndpointOptions(
      TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  DimensionOrderOptionsProto proto;
  DimensionOrderOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the builder when copied from another builder.
TEST(DimensionOrderOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  DimensionOrderEntryOptionsProto dim_entry_proto;
  DimensionOrderOptionsProto default_proto;
  DimensionOrderOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AddDimension();
  builder_default.AddDimension(
      DimensionOrderEntryOptionsProtoBuilder(&dim_entry_proto));
  builder_default.GetEndpointOptions();
  DimensionOrderOptionsProto proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the LineTopology function.
TEST(DimensionOrderOptionsProtoBuilderTest, LineTopology) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.LineTopology(
      42, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the RingTopology function.
TEST(DimensionOrderOptionsProtoBuilderTest, RingTopology) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.RingTopology(
      42, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the SymmetricTorusTopology function.
TEST(DimensionOrderOptionsProtoBuilderTest, SymmetricTorusTopology) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.SymmetricTorusTopology(
      42, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the SymmetricMeshTopology function.
TEST(DimensionOrderOptionsProtoBuilderTest, SymmetricMeshTopology) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.SymmetricMeshTopology(
      42, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the TorusTopology function.
TEST(DimensionOrderOptionsProtoBuilderTest, TorusTopology) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.TorusTopology(
      42, 1337, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the MeshTopology function.
TEST(DimensionOrderOptionsProtoBuilderTest, MeshTopology) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.MeshTopology(
      42, 1337, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

// Test field values of the GridTopologyWithDimension0Loopback function.
TEST(DimensionOrderOptionsProtoBuilderTest,
     GridTopologyWithDimension0Loopback) {
  DimensionOrderOptionsProto proto;
  TopologyEndpointOptionsProto topology_endpoint_proto;
  DimensionOrderOptionsProtoBuilder builder(&proto);
  builder.GridTopologyWithDimension0Loopback(
      42, 1337, TopologyEndpointOptionsProtoBuilder(&topology_endpoint_proto));
  EXPECT_EQ(proto.dimensions_size(), 2);
  EXPECT_TRUE(proto.has_endpoint_options());
}

}  // namespace
}  // namespace xls::noc

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

#include "xls/noc/config_ng/unidirectional_tree_options_proto_builder.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(UnidirectionalTreeOptionsProtoBuilderTest, FieldValuesForPtr) {
  UnidirectionalTreeOptionsProto proto;
  UnidirectionalTreeOptionsProtoBuilder builder(&proto);
  builder.SetType(UnidirectionalTreeOptionsProto::DISTRIBUTION);
  EXPECT_EQ(proto.type(), UnidirectionalTreeOptionsProto::DISTRIBUTION);
  builder.AsAggregation();
  EXPECT_EQ(proto.type(), UnidirectionalTreeOptionsProto::AGGREGATION);
  builder.AsDistribution();
  EXPECT_EQ(proto.type(), UnidirectionalTreeOptionsProto::DISTRIBUTION);
  builder.SetPortCountAtRoot(42);
  EXPECT_EQ(proto.port_count_at_root(), 42);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(UnidirectionalTreeOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  UnidirectionalTreeOptionsProto default_proto;
  UnidirectionalTreeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AsAggregation();
  builder_default.SetPortCountAtRoot(42);
  UnidirectionalTreeOptionsProto proto;
  UnidirectionalTreeOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.type(), UnidirectionalTreeOptionsProto::AGGREGATION);
  EXPECT_EQ(proto.port_count_at_root(), 42);
}

// Test field values of the builder when copied from another builder.
TEST(UnidirectionalTreeOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  UnidirectionalTreeOptionsProto default_proto;
  UnidirectionalTreeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AsAggregation();
  builder_default.SetPortCountAtRoot(42);
  UnidirectionalTreeOptionsProto proto;
  UnidirectionalTreeOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.type(), UnidirectionalTreeOptionsProto::AGGREGATION);
  EXPECT_EQ(proto.port_count_at_root(), 42);
}

}  // namespace
}  // namespace xls::noc

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

#include "xls/noc/config_ng/tree_options_proto_builder.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "xls/noc/config_ng/bidirectional_tree_options_proto_builder.h"
#include "xls/noc/config_ng/level_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"
#include "xls/noc/config_ng/unidirectional_tree_options_proto_builder.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(TreeOptionsProtoBuilderTest, FieldValuesForPtr) {
  LevelOptionsProto level_proto;
  UnidirectionalTreeOptionsProto uni_proto;
  BidirectionalTreeOptionsProto bi_proto;
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto);
  builder.AddLevel();
  EXPECT_EQ(proto.levels_size(), 1);
  builder.AddLevel(LevelOptionsProtoBuilder(&level_proto));
  EXPECT_EQ(proto.levels_size(), 2);
  builder.AddLevels({0, 0, 0});
  EXPECT_EQ(proto.levels_size(), 5);
  builder.EnableUnidirectional(
      UnidirectionalTreeOptionsProtoBuilder(&uni_proto));
  EXPECT_TRUE(proto.has_unidirectional_tree());
  builder.EnableBidirectional(BidirectionalTreeOptionsProtoBuilder(&bi_proto));
  EXPECT_TRUE(proto.has_bidirectional_tree());
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(TreeOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  TreeOptionsProto default_proto;
  TreeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AddLevel();
  builder_default.AddLevels({0, 0, 0, 0});
  builder_default.EnableUnidirectional();
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.levels_size(), 5);
  EXPECT_TRUE(proto.has_unidirectional_tree());
}

// Test field values of the builder when copied from another builder.
TEST(TreeOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  TreeOptionsProto default_proto;
  TreeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.AddLevel();
  builder_default.AddLevels({0, 0, 0, 0});
  builder_default.EnableUnidirectional();
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.levels_size(), 5);
  EXPECT_TRUE(proto.has_unidirectional_tree());
}

// Test field values of the BidirectionalBinaryTreeTopology function.
TEST(TreeOptionsProtoBuilderTest, BidirectionalBinaryTreeTopology) {
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto);
  builder.BidirectionalBinaryTreeTopology(42, 42, 1337);
  EXPECT_EQ(proto.levels_size(), 42);
  EXPECT_TRUE(proto.has_bidirectional_tree());
}

// Test field values of the UnidirectionalBinaryTreeTopology function.
TEST(TreeOptionsProtoBuilderTest, UnidirectionalBinaryTreeTopology) {
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto);
  builder.UnidirectionalBinaryTreeTopology(
      42, 1337, UnidirectionalTreeOptionsProto::AGGREGATION);
  EXPECT_EQ(proto.levels_size(), 42);
  EXPECT_TRUE(proto.has_unidirectional_tree());
}

// Test field values of the AggregationTreeTopology function.
TEST(TreeOptionsProtoBuilderTest, AggregationTreeTopology) {
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto);
  builder.AggregationTreeTopology(std::vector<int64_t>(42), 1337);
  EXPECT_EQ(proto.levels_size(), 42);
  EXPECT_TRUE(proto.has_unidirectional_tree());
}

// Test field values of the DistributionTreeTopology function.
TEST(TreeOptionsProtoBuilderTest, DistributionTreeTopology) {
  TreeOptionsProto proto;
  TreeOptionsProtoBuilder builder(&proto);
  builder.DistributionTreeTopology(std::vector<int64_t>(42), 1337);
  EXPECT_EQ(proto.levels_size(), 42);
  EXPECT_TRUE(proto.has_unidirectional_tree());
}

}  // namespace
}  // namespace xls::noc

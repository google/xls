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

#include "xls/noc/config_ng/butterfly_options_proto_builder.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/bidirectional_butterfly_options_proto_builder.h"
#include "xls/noc/config_ng/unidirectional_butterfly_options_proto_builder.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(ButterflyOptionsProtoBuilderTest, FieldValuesForPtr) {
  UnidirectionalButterflyOptionsProto uni_proto;
  BidirectionalButterflyOptionsProto bi_proto;
  ButterflyOptionsProto proto;
  ButterflyOptionsProtoBuilder builder(&proto);
  builder.SetRadixPerStage(42);
  EXPECT_EQ(proto.radix_per_stage(), 42);
  builder.SetStageCount(1337);
  EXPECT_EQ(proto.stage_count(), 1337);
  builder.SetFlatten(true);
  EXPECT_TRUE(proto.flatten());
  builder.EnableUnidirectional(
      UnidirectionalButterflyOptionsProtoBuilder(&uni_proto));
  EXPECT_TRUE(proto.has_unidirectional_butterfly());
  builder.EnableBidirectional(
      BidirectionalButterflyOptionsProtoBuilder(&bi_proto));
  EXPECT_TRUE(proto.has_bidirectional_butterfly());
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(TopologyEndpointOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  ButterflyOptionsProto default_proto;
  ButterflyOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetRadixPerStage(42);
  builder_default.SetStageCount(1337);
  builder_default.SetFlatten(true);
  builder_default.EnableUnidirectional();
  ButterflyOptionsProto proto;
  ButterflyOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.radix_per_stage(), 42);
  EXPECT_EQ(proto.stage_count(), 1337);
  EXPECT_TRUE(proto.flatten());
  EXPECT_TRUE(proto.has_unidirectional_butterfly());
}

// Test field values of the builder when copied from another builder.
TEST(ButterflyOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  ButterflyOptionsProto default_proto;
  ButterflyOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetRadixPerStage(42);
  builder_default.SetStageCount(1337);
  builder_default.SetFlatten(true);
  builder_default.EnableUnidirectional();
  ButterflyOptionsProto proto;
  ButterflyOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.radix_per_stage(), 42);
  EXPECT_EQ(proto.stage_count(), 1337);
  EXPECT_TRUE(proto.flatten());
  EXPECT_TRUE(proto.has_unidirectional_butterfly());
}

// Test field values of the UnidirectionalButterflyTopology function.
TEST(ButterflyOptionsProtoBuilderTest, UnidirectionalButterflyTopology) {
  ButterflyOptionsProto proto;
  ButterflyOptionsProtoBuilder builder(&proto);
  builder.UnidirectionalButterflyTopology(
      1337, 42,
      UnidirectionalButterflyOptionsProto::FROM_FIRST_STAGE_TO_LAST_STAGE,
      true);
  EXPECT_EQ(proto.stage_count(), 1337);
  EXPECT_EQ(proto.radix_per_stage(), 42);
  EXPECT_TRUE(proto.flatten());
  EXPECT_TRUE(proto.has_unidirectional_butterfly());
}

// Test field values of the BidirectionalButterflyTopology function.
TEST(ButterflyOptionsProtoBuilderTest, BidirectionalButterflyTopology) {
  ButterflyOptionsProto proto;
  ButterflyOptionsProtoBuilder builder(&proto);
  builder.BidirectionalButterflyTopology(
      1337, 42, BidirectionalButterflyOptionsProto::CONNECT_TO_FIRST_STAGE,
      true);
  EXPECT_EQ(proto.stage_count(), 1337);
  EXPECT_EQ(proto.radix_per_stage(), 42);
  EXPECT_TRUE(proto.flatten());
  EXPECT_TRUE(proto.has_bidirectional_butterfly());
}

}  // namespace
}  // namespace xls::noc

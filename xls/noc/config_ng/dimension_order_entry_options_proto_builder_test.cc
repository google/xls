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

#include "xls/noc/config_ng/dimension_order_entry_options_proto_builder.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(DimensionOrderEntryOptionsProtoBuilderTest, FieldValuesForPtr) {
  constexpr int64_t kDimensionIndex = 42;
  constexpr int64_t kEntityCount = 1337;
  DimensionOrderEntryOptionsProto proto;
  DimensionOrderEntryOptionsProtoBuilder builder(&proto);
  builder.SetDimensionIndex(kDimensionIndex);
  EXPECT_EQ(proto.index(), kDimensionIndex);
  builder.SetEntityCount(kEntityCount);
  EXPECT_EQ(proto.entity_count(), kEntityCount);
  builder.SetLoopback(true);
  EXPECT_TRUE(proto.loopback());
  builder.SetLoopback(false);
  EXPECT_FALSE(proto.loopback());
  builder.SetLoopbackToTrue();
  EXPECT_TRUE(proto.loopback());
  builder.SetLoopbackToFalse();
  EXPECT_FALSE(proto.loopback());
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(DimensionOrderEntryOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  constexpr int64_t kDimensionIndex = 42;
  constexpr int64_t kEntityCount = 1337;
  DimensionOrderEntryOptionsProto default_proto;
  DimensionOrderEntryOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetDimensionIndex(kDimensionIndex);
  builder_default.SetEntityCount(kEntityCount);
  builder_default.SetLoopback(true);
  DimensionOrderEntryOptionsProto proto;
  DimensionOrderEntryOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.index(), kDimensionIndex);
  EXPECT_EQ(proto.entity_count(), kEntityCount);
  EXPECT_TRUE(proto.loopback());
}

// Test field values of the builder when copied from another builder.
TEST(DimensionOrderEntryOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  constexpr int64_t kDimensionIndex = 42;
  constexpr int64_t kEntityCount = 1337;
  DimensionOrderEntryOptionsProto default_proto;
  DimensionOrderEntryOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetDimensionIndex(kDimensionIndex);
  builder_default.SetEntityCount(kEntityCount);
  builder_default.SetLoopback(true);
  DimensionOrderEntryOptionsProto proto;
  DimensionOrderEntryOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.index(), kDimensionIndex);
  EXPECT_EQ(proto.entity_count(), kEntityCount);
  EXPECT_TRUE(proto.loopback());
}

}  // namespace
}  // namespace xls::noc

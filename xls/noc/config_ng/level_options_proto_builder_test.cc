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

#include "xls/noc/config_ng/level_options_proto_builder.h"

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(LevelOptionsProtoBuilderTest, FieldValuesForPtr) {
  LevelOptionsProto proto;
  LevelOptionsProtoBuilder builder(&proto);
  builder.SetIndex(42);
  EXPECT_EQ(proto.index(), 42);
  builder.SetNodeCount(1337);
  EXPECT_EQ(proto.node_count(), 1337);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(LevelOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  LevelOptionsProto default_proto;
  LevelOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetIndex(42);
  builder_default.SetNodeCount(1337);
  LevelOptionsProto proto;
  LevelOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.index(), 42);
  EXPECT_EQ(proto.node_count(), 1337);
}

// Test field values of the builder when copied from another builder.
TEST(LevelOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  LevelOptionsProto default_proto;
  LevelOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetIndex(42);
  builder_default.SetNodeCount(1337);
  LevelOptionsProto proto;
  LevelOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.index(), 42);
  EXPECT_EQ(proto.node_count(), 1337);
}

}  // namespace
}  // namespace xls::noc

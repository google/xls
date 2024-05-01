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

#include "xls/noc/config_ng/bidirectional_tree_options_proto_builder.h"

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(BidirectionalTreeOptionsProtoBuilderTest, FieldValuesForPtr) {
  BidirectionalTreeOptionsProto proto;
  BidirectionalTreeOptionsProtoBuilder builder(&proto);
  builder.SetSendPortCountAtRoot(42);
  EXPECT_EQ(proto.send_port_count_at_root(), 42);
  builder.SetRecvPortCountAtRoot(1337);
  EXPECT_EQ(proto.recv_port_count_at_root(), 1337);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(BidirectionalTreeOptionsProtoBuilderTest, FieldValuesForPtrWithDefault) {
  BidirectionalTreeOptionsProto default_proto;
  BidirectionalTreeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetSendPortCountAtRoot(42);
  builder_default.SetRecvPortCountAtRoot(1337);
  BidirectionalTreeOptionsProto proto;
  BidirectionalTreeOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.send_port_count_at_root(), 42);
  EXPECT_EQ(proto.recv_port_count_at_root(), 1337);
}

// Test field values of the builder when copied from another builder.
TEST(BidirectionalTreeOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  BidirectionalTreeOptionsProto default_proto;
  BidirectionalTreeOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetSendPortCountAtRoot(42);
  builder_default.SetRecvPortCountAtRoot(1337);
  BidirectionalTreeOptionsProto proto;
  BidirectionalTreeOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.send_port_count_at_root(), 42);
  EXPECT_EQ(proto.recv_port_count_at_root(), 1337);
}

}  // namespace
}  // namespace xls::noc

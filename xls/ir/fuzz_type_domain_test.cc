// Copyright 2024 The XLS Authors
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

#include "xls/ir/fuzz_type_domain.h"

#include "gtest/gtest.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
namespace {

TEST(FuzzTypeDomainTest, TypeProtoBitsSizeInRange) {
  // 55-bit bits type.
  TypeProto proto;
  proto.set_type_enum(TypeProto::BITS);
  proto.set_bit_count(55);
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/60));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/55));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/54));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/56, /*max_size=*/60));
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/55, /*max_size=*/60));
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/50, /*max_size=*/60));
}

TEST(FuzzTypeDomainTest, TypeProtoArraySizeInRange) {
  // array of 11 5-bit bits elements for a total size of 55 bits.
  TypeProto proto;
  proto.set_type_enum(TypeProto::ARRAY);
  proto.mutable_array_element()->set_type_enum(TypeProto::BITS);
  proto.mutable_array_element()->set_bit_count(5);
  proto.set_array_size(11);
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/60));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/55));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/54));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/56, /*max_size=*/60));
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/55, /*max_size=*/60));
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/50, /*max_size=*/60));
}

TEST(FuzzTypeDomainTest, TypeProtoTupleSizeInRange) {
  // Tuple with types (10, 20, 15, 10) for a total size of 55 bits.
  TypeProto proto;
  proto.set_type_enum(TypeProto::TUPLE);
  TypeProto* element0 = proto.add_tuple_elements();
  element0->set_type_enum(TypeProto::BITS);
  element0->set_bit_count(10);
  TypeProto* element1 = proto.add_tuple_elements();
  element1->set_type_enum(TypeProto::BITS);
  element1->set_bit_count(20);
  TypeProto* element2 = proto.add_tuple_elements();
  element2->set_type_enum(TypeProto::BITS);
  element2->set_bit_count(15);
  TypeProto* element3 = proto.add_tuple_elements();
  element3->set_type_enum(TypeProto::BITS);
  element3->set_bit_count(10);
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/60));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/55));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/0, /*max_size=*/54));
  EXPECT_FALSE(TypeProtoSizeInRange(proto, /*min_size=*/56, /*max_size=*/60));
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/55, /*max_size=*/60));
  EXPECT_TRUE(TypeProtoSizeInRange(proto, /*min_size=*/50, /*max_size=*/60));
}

}  // namespace
}  // namespace xls

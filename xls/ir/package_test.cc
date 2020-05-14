// Copyright 2020 Google LLC
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

#include "xls/ir/package.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/type.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class PackageTest : public IrTestBase {};

TEST_F(PackageTest, GetBitsTypes) {
  Package p("my_package");
  EXPECT_FALSE(p.IsOwnedType(nullptr));

  BitsType* bits42 = p.GetBitsType(42);
  EXPECT_TRUE(bits42->IsBits());
  EXPECT_EQ(bits42->bit_count(), 42);
  EXPECT_TRUE(p.IsOwnedType(bits42));
  EXPECT_EQ(bits42, p.GetBitsType(42));
  EXPECT_EQ("bits[42]", bits42->ToString());

  BitsType imposter(42);
  EXPECT_FALSE(p.IsOwnedType(&imposter));

  BitsType* bits77 = p.GetBitsType(77);
  EXPECT_TRUE(p.IsOwnedType(bits77));
  EXPECT_NE(bits77, bits42);

  TypeProto bits77_proto = bits77->ToProto();
  EXPECT_EQ(bits77_proto.type_enum(), TypeProto::BITS);
  EXPECT_EQ(bits77_proto.bit_count(), 77);
  EXPECT_THAT(p.GetTypeFromProto(bits77_proto), IsOkAndHolds(bits77));
}

TEST_F(PackageTest, GetArrayTypes) {
  Package p("my_package");
  BitsType* bits42 = p.GetBitsType(42);

  ArrayType* array_bits42 = p.GetArrayType(123, bits42);
  EXPECT_TRUE(array_bits42->IsArray());
  EXPECT_EQ(array_bits42->size(), 123);
  EXPECT_EQ(array_bits42, p.GetArrayType(123, bits42));
  EXPECT_EQ(array_bits42->element_type(), p.GetBitsType(42));
  EXPECT_EQ("bits[42][123]", array_bits42->ToString());

  ArrayType* array_array_bits42 = p.GetArrayType(444, array_bits42);
  EXPECT_TRUE(array_array_bits42->IsArray());
  EXPECT_EQ(array_array_bits42->size(), 444);
  EXPECT_EQ(array_array_bits42->element_type(), array_bits42);
  EXPECT_EQ(array_array_bits42, p.GetArrayType(444, array_bits42));

  EXPECT_EQ("bits[42][123][444]", array_array_bits42->ToString());

  EXPECT_THAT(p.GetTypeFromProto(array_array_bits42->ToProto()),
              IsOkAndHolds(array_array_bits42));
}

TEST_F(PackageTest, GetTupleTypes) {
  Package p("my_package");
  BitsType* bits42 = p.GetBitsType(42);
  BitsType* bits86 = p.GetBitsType(86);

  TupleType* tuple1 = p.GetTupleType({bits42});
  EXPECT_EQ("(bits[42])", tuple1->ToString());
  EXPECT_TRUE(p.IsOwnedType(tuple1));

  TupleType* tuple2 = p.GetTupleType({bits42, bits86});
  EXPECT_EQ("(bits[42], bits[86])", tuple2->ToString());
  EXPECT_TRUE(p.IsOwnedType(tuple2));

  TupleType* empty_tuple = p.GetTupleType({});
  EXPECT_EQ("()", empty_tuple->ToString());
  EXPECT_TRUE(p.IsOwnedType(empty_tuple));

  TupleType* tuple_of_arrays =
      p.GetTupleType({p.GetArrayType(2, bits42), p.GetArrayType(4, bits86)});
  EXPECT_EQ("(bits[42][2], bits[86][4])", tuple_of_arrays->ToString());
  EXPECT_TRUE(p.IsOwnedType(tuple_of_arrays));

  TupleType* nested_tuple = p.GetTupleType({empty_tuple, tuple2, bits86});
  EXPECT_EQ("((), (bits[42], bits[86]), bits[86])", nested_tuple->ToString());
  EXPECT_TRUE(p.IsOwnedType(nested_tuple));

  EXPECT_THAT(p.GetTypeFromProto(nested_tuple->ToProto()),
              IsOkAndHolds(nested_tuple));
}

TEST_F(PackageTest, IsDefinitelyEqualTo) {
  const char text1[] = R"(
package package1

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn main(a: bits[32]) -> bits[32] {
  ret invoke.5: bits[32] = invoke(a, a, to_apply=f)
}
)";

  // This package uses subtract instead of add.
  const char text2[] = R"(
package package1

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret sub.1: bits[32] = sub(x, y)
}

fn main(a: bits[32]) -> bits[32] {
  ret invoke.5: bits[32] = invoke(a, a, to_apply=f)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto p1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto p1_2, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto p2, ParsePackage(text2));

  EXPECT_TRUE(p1->IsDefinitelyEqualTo(p1.get()));
  EXPECT_TRUE(p1->IsDefinitelyEqualTo(p1_2.get()));
  EXPECT_FALSE(p1->IsDefinitelyEqualTo(p2.get()));
}

}  // namespace
}  // namespace xls

// Copyright 2020 The XLS Authors
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

#include "xls/ir/value_utils.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

TEST(ValueHelperTest, GenerateValues) {
  Package p("test_package");
  EXPECT_EQ(ZeroOfType(p.GetBitsType(32)), Value(Bits(32)));
  EXPECT_EQ(ZeroOfType(p.GetBitsType(0)), Value(Bits(0)));
  EXPECT_EQ(ZeroOfType(p.GetTokenType()), Value::Token());

  EXPECT_EQ(AllOnesOfType(p.GetBitsType(32)), Value(Bits::AllOnes(32)));
  EXPECT_EQ(AllOnesOfType(p.GetBitsType(0)), Value(Bits(0)));
  EXPECT_EQ(AllOnesOfType(p.GetTokenType()), Value::Token());

  Type* tuple_type =
      p.GetTupleType({p.GetBitsType(0), p.GetBitsType(16),
                      p.GetTupleType({p.GetBitsType(12), p.GetBitsType(8)})});
  EXPECT_EQ(ZeroOfType(tuple_type),
            Parser::ParseTypedValue(
                "(bits[0]:0, bits[16]:0, (bits[12]:0, bits[8]:0))")
                .value());
  EXPECT_EQ(AllOnesOfType(tuple_type),
            Parser::ParseTypedValue(
                "(bits[0]:0, bits[16]:0xffff, (bits[12]:0xfff, bits[8]:0xff))")
                .value());

  Type* array_type = p.GetArrayType(3, p.GetBitsType(8));
  EXPECT_EQ(
      ZeroOfType(array_type),
      Parser::ParseTypedValue("[bits[8]:0, bits[8]:0, bits[8]:0]]]]").value());
  EXPECT_EQ(
      AllOnesOfType(array_type),
      Parser::ParseTypedValue("[bits[8]:0xff, bits[8]:0xff, bits[8]:0xff]]]]")
          .value());
}

TEST(ValueHelperTest, ValueConformsToType) {
  Package p("test_package");
  EXPECT_TRUE(ValueConformsToType(Value(UBits(1234, 32)), p.GetBitsType(32)));
  EXPECT_FALSE(ValueConformsToType(Value(UBits(1234, 32)), p.GetBitsType(31)));
  EXPECT_FALSE(ValueConformsToType(Value(UBits(1234, 32)), p.GetBitsType(0)));
  EXPECT_FALSE(ValueConformsToType(Value(UBits(1234, 32)), p.GetTupleType({})));
  EXPECT_FALSE(ValueConformsToType(Value(UBits(1234, 32)),
                                   p.GetArrayType(42, p.GetBitsType(32))));

  Value tuple_value =
      Value::Tuple({Value(Bits(16)), Value(Bits(1234)),
                    Value::Tuple({Value(Bits(1)), Value(Bits(10))})});
  EXPECT_TRUE(ValueConformsToType(
      tuple_value,
      p.GetTupleType({p.GetBitsType(16), p.GetBitsType(1234),
                      p.GetTupleType({p.GetBitsType(1), p.GetBitsType(10)})})));
  EXPECT_FALSE(ValueConformsToType(
      tuple_value,
      p.GetTupleType({p.GetBitsType(16), p.GetBitsType(1234),
                      p.GetTupleType({p.GetBitsType(2), p.GetBitsType(10)})})));
  EXPECT_FALSE(ValueConformsToType(
      tuple_value,
      p.GetTupleType({p.GetBitsType(1234),
                      p.GetTupleType({p.GetBitsType(2), p.GetBitsType(10)})})));
  EXPECT_FALSE(ValueConformsToType(tuple_value,
                                   p.GetArrayType(12, p.GetBitsType(1234))));
  EXPECT_FALSE(ValueConformsToType(tuple_value, p.GetBitsType(32)));

  Value array_value = Value::UBitsArray({1, 2, 3, 4}, 333).value();
  EXPECT_TRUE(
      ValueConformsToType(array_value, p.GetArrayType(4, p.GetBitsType(333))));
  EXPECT_FALSE(
      ValueConformsToType(array_value, p.GetArrayType(3, p.GetBitsType(333))));
}

TEST(ValueHelperTest, ValueToLeafTypeTree) {
  Package p("test_package");
  auto run_test = [](Type* type) {
    XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<Value> ltt,
                             ValueToLeafTypeTree(ZeroOfType(type), type));
    EXPECT_TRUE(ltt.type()->IsEqualTo(type));
    XLS_EXPECT_OK(leaf_type_tree::ForEachIndex(
        ltt.AsView(),
        [](Type* type, const Value& value, absl::Span<const int64_t> index) {
          EXPECT_EQ(type->GetFlatBitCount(), value.GetFlatBitCount());
          EXPECT_EQ(type->IsBits(), value.IsBits());
          EXPECT_EQ(type->IsToken(), value.IsToken());
          EXPECT_TRUE(value.IsBits() || value.IsToken());
          return absl::OkStatus();
        }));
    XLS_ASSERT_OK_AND_ASSIGN(Value roundtrip,
                             LeafTypeTreeToValue(ltt.AsView()));
    EXPECT_EQ(roundtrip, ZeroOfType(type));
  };
  run_test(p.GetBitsType(10));
  run_test(p.GetTokenType());
  run_test(p.GetTupleType({p.GetBitsType(10), p.GetBitsType(11)}));
  run_test(p.GetTupleType(
      {p.GetBitsType(10), p.GetBitsType(11),
       p.GetArrayType(
           5, p.GetTupleType(
                  {p.GetBitsType(12), p.GetTokenType(),
                   p.GetTupleType({p.GetBitsType(13), p.GetBitsType(14)}),
                   p.GetTupleType({p.GetBitsType(15), p.GetBitsType(16)})}))}));
}

}  // namespace
}  // namespace xls

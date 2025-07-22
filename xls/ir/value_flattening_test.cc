// Copyright 2025 The XLS Authors
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

#include "xls/ir/value_flattening.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class FlatteningTest : public IrTestBase {};

TEST_F(FlatteningTest, FlatIndexing) {
  Package p(TestName());
  Type* b0 = p.GetBitsType(0);
  Type* b42 = p.GetBitsType(42);
  BitsType* b5 = p.GetBitsType(5);

  TupleType* t_empty = p.GetTupleType({});
  EXPECT_EQ(t_empty->GetFlatBitCount(), 0);

  TupleType* t1 = p.GetTupleType({b42});
  EXPECT_EQ(t1->GetFlatBitCount(), 42);
  EXPECT_EQ(GetFlatBitIndexOfElement(t1, 0), 0);

  TupleType* t2 = p.GetTupleType({b42, b0, b5, b5, b0});
  EXPECT_EQ(t2->GetFlatBitCount(), 52);
  EXPECT_EQ(GetFlatBitIndexOfElement(t2, 0), 10);
  EXPECT_EQ(GetFlatBitIndexOfElement(t2, 1), 10);
  EXPECT_EQ(GetFlatBitIndexOfElement(t2, 2), 5);
  EXPECT_EQ(GetFlatBitIndexOfElement(t2, 3), 0);
  EXPECT_EQ(GetFlatBitIndexOfElement(t2, 4), 0);

  ArrayType* a_of_b5 = p.GetArrayType(32, b5);
  EXPECT_EQ(a_of_b5->GetFlatBitCount(), 160);
  EXPECT_EQ(GetFlatBitIndexOfElement(a_of_b5, 0), 0);
  EXPECT_EQ(GetFlatBitIndexOfElement(a_of_b5, 15), 75);
  EXPECT_EQ(GetFlatBitIndexOfElement(a_of_b5, 31), 155);

  ArrayType* array_2d = p.GetArrayType(4, a_of_b5);
  EXPECT_EQ(array_2d->GetFlatBitCount(), 640);
  EXPECT_EQ(GetFlatBitIndexOfElement(array_2d, 0), 0);
  EXPECT_EQ(GetFlatBitIndexOfElement(array_2d, 2), 320);

  // Nested tuple with a nested array.
  TupleType* t3 = p.GetTupleType({t2, b5, b42, array_2d});
  EXPECT_EQ(t3->GetFlatBitCount(), 739);
  EXPECT_EQ(GetFlatBitIndexOfElement(t3, 0), 687);
  EXPECT_EQ(GetFlatBitIndexOfElement(t3, 1), 682);
  EXPECT_EQ(GetFlatBitIndexOfElement(t3, 2), 640);
}

TEST_F(FlatteningTest, FlattenValues) {
  Package p(TestName());

  Bits empty_bits;
  EXPECT_EQ(empty_bits, FlattenValueToBits(Value(empty_bits)));
  EXPECT_THAT(UnflattenBitsToValue(empty_bits, p.GetBitsType(0)),
              IsOkAndHolds(Value(empty_bits)));

  Bits forty_two = UBits(42, 123);
  EXPECT_EQ(forty_two, FlattenValueToBits(Value(forty_two)));
  EXPECT_THAT(UnflattenBitsToValue(forty_two, p.GetBitsType(123)),
              IsOkAndHolds(Value(forty_two)));

  // Empty tuple should flatten to a zero-bit Bits object.
  EXPECT_EQ(empty_bits, FlattenValueToBits(Value::Tuple({})));
  EXPECT_THAT(UnflattenBitsToValue(empty_bits, p.GetTupleType({})),
              IsOkAndHolds(Value::Tuple({})));

  Bits abc = UBits(0xabcdef, 24);
  Value tuple_abc = Value::Tuple(
      {Value(UBits(0xab, 8)), Value(UBits(0xc, 4)), Value(UBits(0xdef, 12))});
  EXPECT_EQ(abc, FlattenValueToBits(tuple_abc));
  EXPECT_THAT(UnflattenBitsToValue(abc, p.GetTypeForValue(tuple_abc)),
              IsOkAndHolds(tuple_abc));

  XLS_ASSERT_OK_AND_ASSIGN(
      Value arr, Value::Array({Value(UBits(0x12, 8)), Value(UBits(0x34, 8))}));
  EXPECT_EQ(UBits(0x3412, 16), FlattenValueToBits(arr));
  EXPECT_THAT(UnflattenBitsToValue(UBits(0x3412, 16), p.GetTypeForValue(arr)),
              IsOkAndHolds(arr));

  // Two-element array of tuples.
  Bits abc123 = UBits(0x123456abcdefULL, 48);
  Value tuple_123 = Value::Tuple(
      {Value(UBits(0x12, 8)), Value(UBits(0x3, 4)), Value(UBits(0x456, 12))});
  XLS_ASSERT_OK_AND_ASSIGN(Value abc_array,
                           Value::Array({tuple_abc, tuple_123}));
  EXPECT_EQ(abc123, FlattenValueToBits(abc_array));
  EXPECT_THAT(UnflattenBitsToValue(abc123, p.GetTypeForValue(abc_array)),
              IsOkAndHolds(abc_array));
}

}  // namespace
}  // namespace xls

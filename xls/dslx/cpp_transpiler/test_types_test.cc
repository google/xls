// Copyright 2023 The XLS Authors
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/cpp_transpiler/test_types_lib.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TestTypesTest, EnumToString) {
  EXPECT_EQ(MyEnumToString(test::MyEnum::kA), "MyEnum::kA (0)");
  EXPECT_EQ(MyEnumToString(test::MyEnum::kB), "MyEnum::kB (1)");
  EXPECT_EQ(MyEnumToString(test::MyEnum::kC), "MyEnum::kC (42)");
  EXPECT_EQ(MyEnumToString(test::MyEnum(123)), "<unknown> (123)");
  // 1234 overflows the uint8_t making it 210.
  EXPECT_EQ(MyEnumToString(test::MyEnum(1234)), "<unknown> (210)");

  EXPECT_EQ(MyEnumToDslxString(test::MyEnum::kA), "MyEnum::kA (0)");
  EXPECT_EQ(MyEnumToDslxString(test::MyEnum::kB), "MyEnum::kB (1)");
  EXPECT_EQ(MyEnumToDslxString(test::MyEnum::kC), "MyEnum::kC (42)");
  EXPECT_EQ(MyEnumToDslxString(test::MyEnum(123)), "<unknown> (123)");
  // 1234 overflows the uint8_t making it 210.
  EXPECT_EQ(MyEnumToDslxString(test::MyEnum(1234)), "<unknown> (210)");
}

TEST(TestTypesTest, VerifyEnum) {
  XLS_EXPECT_OK(test::VerifyMyEnum(test::MyEnum::kA));
  XLS_EXPECT_OK(test::VerifyMyEnum(test::MyEnum::kB));
  XLS_EXPECT_OK(test::VerifyMyEnum(test::MyEnum::kC));
  EXPECT_THAT(test::VerifyMyEnum(static_cast<test::MyEnum>(33)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value for MyEnum enum: 33")));
  EXPECT_THAT(test::VerifyMyEnum(static_cast<test::MyEnum>(250)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("MyEnum value does not fit in 7 bits: 0xfa")));
}

TEST(TestTypesTest, EnumToValue) {
  EXPECT_THAT(test::MyEnumToValue(test::MyEnum::kA),
              IsOkAndHolds(Value(UBits(0, 7))));
  EXPECT_THAT(test::MyEnumToValue(test::MyEnum::kB),
              IsOkAndHolds(Value(UBits(1, 7))));
  EXPECT_THAT(test::MyEnumToValue(test::MyEnum::kC),
              IsOkAndHolds(Value(UBits(42, 7))));
  EXPECT_THAT(test::MyEnumToValue(static_cast<test::MyEnum>(33)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value for MyEnum enum: 33")));
}

TEST(TestTypesTest, EnumFromValue) {
  EXPECT_THAT(test::MyEnumFromValue(Value(UBits(0, 7))),
              IsOkAndHolds(test::MyEnum::kA));
  EXPECT_THAT(test::MyEnumFromValue(Value(UBits(1, 7))),
              IsOkAndHolds(test::MyEnum::kB));
  EXPECT_THAT(test::MyEnumFromValue(Value(UBits(42, 7))),
              IsOkAndHolds(test::MyEnum::kC));
  EXPECT_THAT(test::MyEnumFromValue(Value(UBits(100, 7))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value for MyEnum enum: 100")));
  EXPECT_THAT(test::MyEnumFromValue(Value(UBits(42, 33))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not a bits type of 7 bits")));
  EXPECT_THAT(test::MyEnumFromValue(Value::Tuple({})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not a bits type of 7 bits")));
}

TEST(TestTypesTest, MyTypeToString) {
  test::MyType s = 1000;
  EXPECT_EQ(test::MyTypeToString(s), "bits[37]:0x3e8");
  EXPECT_EQ(test::kMyTypeWidth, 37);
  EXPECT_EQ(test::MyTypeToDslxString(s), "MyType:0x3e8");
}

TEST(TestTypesTest, MyTypeAliasToString) {
  test::MyType s = 1000;
  EXPECT_EQ(test::MyTypeAliasToString(s), "bits[37]:0x3e8");
  EXPECT_EQ(test::kMyTypeAliasWidth, 37);
  EXPECT_EQ(test::MyTypeAliasToDslxString(s), "MyType:0x3e8");
}

TEST(TestTypesTest, VerifyMyType) {
  XLS_EXPECT_OK(test::VerifyMyType(test::MyType{42}));
  EXPECT_THAT(
      test::VerifyMyType(test::MyType{0xffffaaaabbb}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("MyType value does not fit in 37 bits: 0xffffaaaabbb")));
}

TEST(TestTypesTest, MySignedTypeToString) {
  EXPECT_EQ(test::MySignedTypeToString(test::MySignedType{-1000}),
            "bits[20]:0xfffffc18");
  EXPECT_EQ(test::MySignedTypeToString(test::MySignedType{0xabcd}),
            "bits[20]:0xabcd");

  EXPECT_EQ(test::MySignedTypeToDslxString(test::MySignedType{-1000}),
            "MySignedType:-1000");
  EXPECT_EQ(test::MySignedTypeToDslxString(test::MySignedType{0xabcd}),
            "MySignedType:43981");
}

TEST(TestTypesTest, VerifyMySignedType) {
  XLS_EXPECT_OK(test::VerifyMySignedType(test::MySignedType{-42}));
  XLS_EXPECT_OK(test::VerifyMySignedType(test::MySignedType{42}));
  EXPECT_THAT(
      test::VerifyMySignedType(test::MySignedType{100000000}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "MySignedType value does not fit in signed 20 bits: 0x5f5e100")));
  EXPECT_THAT(test::VerifyMySignedType(test::MySignedType{-100000000}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("MySignedType value does not fit in signed 20 "
                                 "bits: 0xfa0a1f00")));
}

TEST(TestTypesTest, MyTypeToValue) {
  EXPECT_THAT(test::MyTypeToValue(test::MyType{42}),
              IsOkAndHolds(Value(UBits(42, 37))));
  EXPECT_THAT(
      test::MyTypeToValue(test::MyType{0xffffaaaabbb}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unsigned value 0xffffaaaabbb does not fit in 37 bits")));
}

TEST(TestTypesTest, MyTypeFromValue) {
  EXPECT_THAT(test::MyTypeFromValue(Value(UBits(42, 37))),
              IsOkAndHolds(test::MyType{42}));
  EXPECT_THAT(test::MyTypeFromValue(Value(UBits(42, 100))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not a bits type of 37 bits")));
  EXPECT_THAT(test::MyTypeFromValue(Value::Tuple({})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not a bits type of 37 bits")));
}

TEST(TestTypesTest, SimpleStructToString) {
  test::InnerStruct s{.x = 42, .y = test::MyEnum::kB};
  EXPECT_EQ(s.ToString(), R"(InnerStruct {
  x: bits[17]:0x2a,
  y: MyEnum::kB (1),
})");
  EXPECT_EQ(s.ToDslxString(), R"(InnerStruct {
  x: u17:0x2a,
  y: MyEnum::kB (1),
})");
}

TEST(TestTypesTest, VerifySimpleStruct) {
  test::InnerStruct s{.x = 42, .y = test::MyEnum::kB};
  XLS_EXPECT_OK(s.Verify());
  test::InnerStruct t{.x = 0xffffffff, .y = test::MyEnum::kB};
  EXPECT_THAT(
      t.Verify(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "InnerStruct.x value does not fit in 17 bits: 0xffffffff")));
  test::InnerStruct u{.x = 0x123, .y = static_cast<test::MyEnum>(250)};
  EXPECT_THAT(u.Verify(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("MyEnum value does not fit in 7 bits: 0xfa")));
  test::InnerStruct v{.x = 0x123, .y = static_cast<test::MyEnum>(7)};
  EXPECT_THAT(v.Verify(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value for MyEnum enum: 7")));
}

TEST(TestTypesTest, SimpleStructToValue) {
  test::InnerStruct s{.x = 42, .y = test::MyEnum::kB};
  EXPECT_THAT(
      s.ToValue(),
      IsOkAndHolds(Value::Tuple({Value(UBits(42, 17)), Value(UBits(1, 7))})));
  test::InnerStruct t{.x = 0xffffffff, .y = test::MyEnum::kB};
  EXPECT_THAT(
      t.ToValue(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsigned value 0xffffffff does not fit in 17 bits")));
  test::InnerStruct u{.x = 0x123, .y = static_cast<test::MyEnum>(250)};
  EXPECT_THAT(u.ToValue(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("MyEnum value does not fit in 7 bits: 0xfa")));
  test::InnerStruct v{.x = 0x123, .y = static_cast<test::MyEnum>(7)};
  EXPECT_THAT(v.ToValue(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value for MyEnum enum: 7")));
}

TEST(TestTypesTest, SimpleStructFromValue) {
  EXPECT_THAT(test::InnerStruct::FromValue(
                  Value::Tuple({Value(UBits(42, 17)), Value(UBits(1, 7))})),
              IsOkAndHolds(test::InnerStruct{.x = 42, .y = test::MyEnum::kB}));
  EXPECT_THAT(test::InnerStruct::FromValue(
                  Value::Tuple({Value(UBits(42, 47)), Value(UBits(1, 7))})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not a bits type of 17 bits")));
  EXPECT_THAT(test::InnerStruct::FromValue(
                  Value::Tuple({Value(UBits(42, 17)), Value(UBits(55, 7))})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value for MyEnum enum: 55")));
  EXPECT_THAT(test::InnerStruct::FromValue(Value(UBits(3, 4))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not a tuple of 2 elements")));
}

TEST(TestTypesTest, TupleToString) {
  test::MyTuple s{42, -3, 123, -1};
  EXPECT_EQ(test::MyTupleToString(s), R"((
  bits[35]:0x2a,
  bits[4]:0xfd,
  bits[37]:0x7b,
  bits[20]:0xffffffff,
))");
  EXPECT_EQ(test::MyTupleToDslxString(s), R"((
  u35:0x2a,
  s4:-3,
  MyType:0x7b,
  MySignedType:-1,
))");

  // Round trip through a value.
  XLS_ASSERT_OK_AND_ASSIGN(Value value, test::MyTupleToValue(s));
  XLS_ASSERT_OK_AND_ASSIGN(test::MyTuple s_copy, test::MyTupleFromValue(value));
  EXPECT_EQ(s, s_copy);

  EXPECT_EQ(test::MyTupleToString(s_copy), R"((
  bits[35]:0x2a,
  bits[4]:0xfd,
  bits[37]:0x7b,
  bits[20]:0xffffffff,
))");
}

TEST(TestTypesTest, TupleAliasToString) {
  test::MyTupleAlias s{42, -3, 123, -1};
  EXPECT_EQ(test::MyTupleAliasToString(s), R"((
  bits[35]:0x2a,
  bits[4]:0xfd,
  bits[37]:0x7b,
  bits[20]:0xffffffff,
))");
  EXPECT_EQ(test::MyTupleAliasToDslxString(s), R"((
  u35:0x2a,
  s4:-3,
  MyType:0x7b,
  MySignedType:-1,
))");

  // Round trip through a value.
  XLS_ASSERT_OK_AND_ASSIGN(Value value, test::MyTupleAliasToValue(s));
  XLS_ASSERT_OK_AND_ASSIGN(test::MyTupleAlias s_copy,
                           test::MyTupleAliasFromValue(value));
  EXPECT_EQ(s, s_copy);

  EXPECT_EQ(test::MyTupleAliasToString(s_copy), R"((
  bits[35]:0x2a,
  bits[4]:0xfd,
  bits[37]:0x7b,
  bits[20]:0xffffffff,
))");
}

TEST(TestTypesTest, TupleAliasAliasToString) {
  test::MyTupleAliasAlias s{42, -3, 123, -1};
  EXPECT_EQ(test::MyTupleAliasAliasToString(s), R"((
  bits[35]:0x2a,
  bits[4]:0xfd,
  bits[37]:0x7b,
  bits[20]:0xffffffff,
))");
}

TEST(TestTypesTest, TupleOfTupleToString) {
  test::MyTupleOfTuples s{2, {42, -3, 123, -1}};
  EXPECT_EQ(test::MyTupleOfTuplesToString(s), R"((
  bits[3]:0x2,
  (
    bits[35]:0x2a,
    bits[4]:0xfd,
    bits[37]:0x7b,
    bits[20]:0xffffffff,
  ),
))");

  // Round trip through a value.
  XLS_ASSERT_OK_AND_ASSIGN(Value value, test::MyTupleOfTuplesToValue(s));
  XLS_ASSERT_OK_AND_ASSIGN(test::MyTupleOfTuples s_copy,
                           test::MyTupleOfTuplesFromValue(value));
  EXPECT_EQ(s, s_copy);
  EXPECT_EQ(test::MyTupleOfTuplesToString(s_copy), R"((
  bits[3]:0x2,
  (
    bits[35]:0x2a,
    bits[4]:0xfd,
    bits[37]:0x7b,
    bits[20]:0xffffffff,
  ),
))");
}

TEST(TestTypesTest, EmptyTupleToString) {
  test::MyEmptyTuple s;
  EXPECT_EQ(test::MyEmptyTupleToString(s), R"((
))");
  EXPECT_EQ(test::MyEmptyTupleToDslxString(s), R"((
))");

  // Round trip through a value.
  XLS_ASSERT_OK_AND_ASSIGN(Value value, test::MyEmptyTupleToValue(s));
  XLS_ASSERT_OK_AND_ASSIGN(test::MyEmptyTuple s_copy,
                           test::MyEmptyTupleFromValue(value));

  EXPECT_EQ(s, s_copy);
  EXPECT_EQ(test::MyEmptyTupleToString(s_copy), R"((
))");
}

TEST(TestTypesTest, VerifyTuples) {
  XLS_EXPECT_OK(test::VerifyMyEmptyTuple(test::MyEmptyTuple()));

  test::TupleOfStructs s{test::InnerStruct{.x = 12, .y = test::MyEnum::kA},
                         test::InnerStruct{.x = 22, .y = test::MyEnum::kB}};
  XLS_EXPECT_OK(test::VerifyTupleOfStructs(s));
  test::TupleOfStructs t{
      test::InnerStruct{.x = 12, .y = test::MyEnum::kA},
      test::InnerStruct{.x = 0xfffff, .y = test::MyEnum::kB}};
  EXPECT_THAT(
      test::VerifyTupleOfStructs(t),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("InnerStruct.x value does not fit in 17 bits: 0xfffff")));
}

TEST(TestTypesTest, ArrayToString) {
  test::MyArray s{2, 22};
  EXPECT_EQ(test::MyArrayToString(s), R"([
  bits[17]:0x2,
  bits[17]:0x16,
])");
  EXPECT_EQ(test::MyArrayToDslxString(s), R"([
  u17:0x2,
  u17:0x16,
])");

  // Round trip through a value.
  XLS_ASSERT_OK_AND_ASSIGN(Value value, test::MyArrayToValue(s));
  XLS_ASSERT_OK_AND_ASSIGN(test::MyArray s_copy, test::MyArrayFromValue(value));

  EXPECT_EQ(s, s_copy);
  EXPECT_EQ(test::MyArrayToString(s_copy), R"([
  bits[17]:0x2,
  bits[17]:0x16,
])");
}

TEST(TestTypesTest, ArrayOfArraysToString) {
  test::MyArrayOfArrays s;
  s[0] = {2, 22};
  s[1] = {23, 1};
  s[2] = {12, 24};
  EXPECT_EQ(test::MyArrayOfArraysToString(s), R"([
  [
    bits[5]:0x2,
    bits[5]:0x16,
  ],
  [
    bits[5]:0x17,
    bits[5]:0x1,
  ],
  [
    bits[5]:0xc,
    bits[5]:0x18,
  ],
])");
  EXPECT_EQ(test::MyArrayOfArraysToDslxString(s), R"([
  [
    MyU5:0x2,
    MyU5:0x16,
  ],
  [
    MyU5:0x17,
    MyU5:0x1,
  ],
  [
    MyU5:0xc,
    MyU5:0x18,
  ],
])");

  // Round trip through a value.
  XLS_ASSERT_OK_AND_ASSIGN(Value value, test::MyArrayOfArraysToValue(s));
  XLS_ASSERT_OK_AND_ASSIGN(test::MyArrayOfArrays s_copy,
                           test::MyArrayOfArraysFromValue(value));

  EXPECT_EQ(s, s_copy);
  EXPECT_EQ(test::MyArrayOfArraysToString(s_copy), R"([
  [
    bits[5]:0x2,
    bits[5]:0x16,
  ],
  [
    bits[5]:0x17,
    bits[5]:0x1,
  ],
  [
    bits[5]:0xc,
    bits[5]:0x18,
  ],
])");
}

TEST(TestTypesTest, VerifyArrayOfArrays) {
  test::MyArrayOfArrays s;
  s[0] = {2, 22};
  s[1] = {23, 82};
  s[2] = {12, 24};
  EXPECT_THAT(test::VerifyMyArrayOfArrays(s),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("MyU5 value does not fit in 5 bits: 0x52")));
}

TEST(TestTypesTest, StructWithLotsOfTypes) {
  test::StructWithLotsOfTypes s{
      .v = false, .w = 1, .x = false, .y = 0x1234, .z = -3};
  EXPECT_EQ(s.ToString(), R"(StructWithLotsOfTypes {
  v: bits[1]:0x0,
  w: bits[3]:0x1,
  x: bits[1]:0x0,
  y: bits[44]:0x1234,
  z: bits[11]:0xfffd,
})");

  EXPECT_EQ(s.ToDslxString(), R"(StructWithLotsOfTypes {
  v: bool:false,
  w: bits[3]:0x1,
  x: u1:0x0,
  y: uN[44]:0x1234,
  z: sN[11]:-3,
})");
}

TEST(TestTypesTest, StructWithTuplesArrayToString) {
  test::StructWithTuplesArray s{{}, {4, 100}};
  EXPECT_EQ(s.ToString(), R"(StructWithTuplesArray {
  x: (
    ),
  y: (
      bits[2]:0x4,
      bits[4]:0x64,
    ),
})");
}

TEST(TestTypesTest, TupleOfStructsToString) {
  test::TupleOfStructs s{test::InnerStruct{.x = 12, .y = test::MyEnum::kA},
                         test::InnerStruct{.x = 22, .y = test::MyEnum::kB}};
  EXPECT_EQ(TupleOfStructsToString(s), R"((
  InnerStruct {
    x: bits[17]:0xc,
    y: MyEnum::kA (0),
  },
  InnerStruct {
    x: bits[17]:0x16,
    y: MyEnum::kB (1),
  },
))");
}

TEST(TestTypesTest, SimpleStructEq) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 123, .y = test::MyEnum::kB};
  test::InnerStruct c{.x = 42, .y = test::MyEnum::kC};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
}

TEST(TestTypesTest, EmptyStructToString) {
  test::EmptyStruct s;
  EXPECT_EQ(s.ToString(), R"(EmptyStruct {
})");
}

TEST(TestTypesTest, EmptyStructEq) {
  test::EmptyStruct s;
  test::EmptyStruct t;
  EXPECT_EQ(s, s);
  EXPECT_EQ(s, t);
}

TEST(TestTypesTest, NestedStructToString) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 123, .y = test::MyEnum::kC};
  test::OuterStruct s{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  EXPECT_EQ(s.ToString(), R"(OuterStruct {
  a: InnerStruct {
      x: bits[17]:0x2a,
      y: MyEnum::kB (1),
    },
  b: InnerStruct {
      x: bits[17]:0x7b,
      y: MyEnum::kC (42),
    },
  c: bits[37]:0xdead,
  v: MyEnum::kA (0),
})");

  EXPECT_EQ(test::OuterStruct::kCWidth, 37);
  EXPECT_EQ(test::OuterStruct::kVWidth, 7);
}

TEST(TestTypesTest, NestedStructEq) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 42, .y = test::MyEnum::kC};
  test::InnerStruct c{.x = 123, .y = test::MyEnum::kB};
  test::OuterStruct x{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterStruct y{.a = a, .b = c, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterStruct z{.a = a, .b = b, .c = 0x1111, .v = test::MyEnum::kA};

  EXPECT_EQ(x, x);
  EXPECT_NE(x, y);
  EXPECT_NE(x, z);
}

TEST(TestTypesTest, DoublyNestedStructToString) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 123, .y = test::MyEnum::kC};
  test::OuterStruct o{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterOuterStruct s{
      .q = test::EmptyStruct(), .some_array = {1, 2, 3}, .s = o};
  EXPECT_EQ(s.ToString(), R"(OuterOuterStruct {
  q: EmptyStruct {
    },
  some_array: [
      bits[5]:0x1,
      bits[5]:0x2,
      bits[5]:0x3,
    ],
  s: OuterStruct {
      a: InnerStruct {
          x: bits[17]:0x2a,
          y: MyEnum::kB (1),
        },
      b: InnerStruct {
          x: bits[17]:0x7b,
          y: MyEnum::kC (42),
        },
      c: bits[37]:0xdead,
      v: MyEnum::kA (0),
    },
})");
  EXPECT_EQ(s.ToDslxString(), R"(OuterOuterStruct {
  q: EmptyStruct {
    },
  some_array: [
      u5:0x1,
      u5:0x2,
      u5:0x3,
    ],
  s: OuterStruct {
      a: InnerStruct {
          x: u17:0x2a,
          y: MyEnum::kB (1),
        },
      b: InnerStruct {
          x: u17:0x7b,
          y: MyEnum::kC (42),
        },
      c: MyType:0xdead,
      v: MyEnum::kA (0),
    },
})");
}

TEST(TestTypesTest, VerifyDoublyNestedStruct) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 1234567, .y = test::MyEnum::kC};
  test::OuterStruct o{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterOuterStruct s{
      .q = test::EmptyStruct(), .some_array = {1, 2, 3}, .s = o};
  EXPECT_THAT(
      s.Verify(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("InnerStruct.x value does not fit in 17 bits: 0x12d687")));
}

TEST(TestTypesTest, SnakeCaseToString) {
  test::SnakeCaseStructT s{.some_field = 0x42,
                           .some_other_field = test::SnakeCaseEnumT::kA};
  EXPECT_EQ(s.ToString(), R"(SnakeCaseStructT {
  some_field: bits[13]:0x42,
  some_other_field: SnakeCaseEnumT::kA (0),
})");
  EXPECT_EQ(s.ToDslxString(), R"(snake_case_struct_t {
  some_field: snake_case_type_t:0x42,
  some_other_field: snake_case_enum_t::kA (0),
})");
}

TEST(TestTypesTest, StructWithKeywordFields) {
  test::StructWithKeywordFields a{._float = 42, ._int = 1};

  EXPECT_EQ(a.ToString(), R"(StructWithKeywordFields {
  _float: bits[32]:0x2a,
  _int: bits[42]:0x1,
})");
}

}  // namespace
}  // namespace xls

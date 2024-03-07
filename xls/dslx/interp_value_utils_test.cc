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
#include "xls/dslx/interp_value_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

TEST(InterpValueHelpersTest, CastBitsToArray) {
  InterpValue input(InterpValue::MakeU32(0xa5a5a5a5));

  ArrayType array_type(BitsType::MakeU8(), TypeDim::CreateU32(4));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue converted,
                           CastBitsToArray(input, array_type));
  ASSERT_TRUE(converted.IsArray());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t length, converted.GetLength());
  ASSERT_EQ(length, 4);
  for (int i = 0; i < 4; i++) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, converted.Index(i));
    ASSERT_TRUE(value.IsBits());
    XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueViaSign());
    ASSERT_EQ(int_value, 0xa5);
  }
}

TEST(InterpValueHelpersTest, CastBitsToEnumAndCreatZeroValue) {
  constexpr int kBitCount = 13;
  constexpr int kNumMembers = 16;
  Module module("my_test_module", /*fs_path=*/std::nullopt);

  std::vector<EnumMember> members;
  std::vector<InterpValue> member_values;
  BuiltinNameDef* builtin_name_def = module.GetOrCreateBuiltinNameDef("u13");
  TypeAnnotation* element_type = module.Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU13, builtin_name_def);
  for (int i = 0; i < kNumMembers; i++) {
    NameDef* name_def =
        module.Make<NameDef>(Span::Fake(), absl::StrCat("member_", i), nullptr);
    Number* number = module.Make<Number>(Span::Fake(), absl::StrCat(i),
                                         NumberKind::kOther, element_type);
    name_def->set_definer(number);
    members.push_back(EnumMember{name_def, number});
    member_values.push_back(InterpValue::MakeUBits(kBitCount, i));
  }

  NameDef* name_def =
      module.Make<NameDef>(Span::Fake(), "my_test_enum", nullptr);
  EnumDef* enum_def = module.Make<EnumDef>(Span::Fake(), name_def, element_type,
                                           members, /*is_public=*/true);

  EnumType enum_type(*enum_def, TypeDim::CreateU32(kBitCount),
                     /*is_signed=*/false, member_values);

  InterpValue bits_value(InterpValue::MakeUBits(kBitCount, 11));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue converted,
                           CastBitsToEnum(bits_value, enum_type));
  ASSERT_TRUE(converted.IsEnum());
  InterpValue::EnumData enum_data = converted.GetEnumData().value();
  ASSERT_EQ(enum_data.def, enum_def);
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_value, enum_data.value.ToUint64());
  ASSERT_EQ(int_value, 11);

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue enum_zero,
                           CreateZeroValueFromType(enum_type));
  EXPECT_TRUE(
      InterpValue::MakeEnum(Bits(kBitCount), /*is_signed=*/false, enum_def)
          .Eq(enum_zero));
}

TEST(InterpValueHelpersTest, CreateZeroBitsAndArrayValues) {
  // Create zero bits.
  std::unique_ptr<BitsType> u8 = BitsType::MakeU8();
  std::unique_ptr<BitsType> s32 = BitsType::MakeS32();

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue u8_zero, CreateZeroValueFromType(*u8));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue s32_zero, CreateZeroValueFromType(*s32));

  EXPECT_TRUE(InterpValue::MakeUBits(/*bit_count=*/8, 0).Eq(u8_zero));
  EXPECT_FALSE(u8_zero.IsSigned());

  EXPECT_TRUE(InterpValue::MakeSBits(/*bit_count=*/32, 0).Eq(s32_zero));
  EXPECT_TRUE(s32_zero.IsSigned());

  // Create a zero tuple.
  std::vector<std::unique_ptr<Type>> tuple_members;
  tuple_members.push_back(u8->CloneToUnique());
  tuple_members.push_back(s32->CloneToUnique());
  TupleType tuple(std::move(tuple_members));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue tuple_zero,
                           CreateZeroValueFromType(tuple));
  EXPECT_TRUE(InterpValue::MakeTuple({u8_zero, s32_zero}).Eq(tuple_zero));

  // Create a zero array of tuples.
  ArrayType array_type(tuple.CloneToUnique(), TypeDim::CreateU32(2));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue array_zero,
                           CreateZeroValueFromType(array_type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue array_zero_golden,
                           InterpValue::MakeArray({tuple_zero, tuple_zero}));
  EXPECT_TRUE(array_zero_golden.Eq(array_zero));
}

TEST(InterpValueHelpersTest, CreateZeroStructValue) {
  const Span kFakeSpan = Span::Fake();

  Module module("test", /*fs_path=*/std::nullopt);

  std::vector<StructMember> ast_members;
  ast_members.emplace_back(
      StructMember{kFakeSpan, "x",
                   module.Make<BuiltinTypeAnnotation>(
                       kFakeSpan, BuiltinType::kU8,
                       module.GetOrCreateBuiltinNameDef("u8"))});
  ast_members.emplace_back(
      StructMember{kFakeSpan, "y",
                   module.Make<BuiltinTypeAnnotation>(
                       kFakeSpan, BuiltinType::kU1,
                       module.GetOrCreateBuiltinNameDef("u1"))});

  auto* struct_def = module.Make<StructDef>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "S", nullptr),
      std::vector<ParametricBinding*>{}, ast_members, /*is_public=*/false);
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  members.push_back(BitsType::MakeU1());
  StructType s(std::move(members), *struct_def);

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue struct_zero, CreateZeroValueFromType(s));

  InterpValue u8_zero = InterpValue::MakeUBits(/*bit_count=*/8, 0);
  InterpValue u1_zero = InterpValue::MakeUBits(/*bit_count=*/1, 0);

  EXPECT_TRUE(InterpValue::MakeTuple({u8_zero, u1_zero}).Eq(struct_zero));
}

TEST(InterpValueHelpersTest, InterpValueAsStringWorks) {
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue hello_world_u8_array,
                           InterpValue::MakeArray({
                               InterpValue::MakeUBits(/*bit_count=*/8, 72),
                               InterpValue::MakeUBits(/*bit_count=*/8, 101),
                               InterpValue::MakeUBits(/*bit_count=*/8, 108),
                               InterpValue::MakeUBits(/*bit_count=*/8, 108),
                               InterpValue::MakeUBits(/*bit_count=*/8, 111),
                               InterpValue::MakeUBits(/*bit_count=*/8, 32),
                               InterpValue::MakeUBits(/*bit_count=*/8, 119),
                               InterpValue::MakeUBits(/*bit_count=*/8, 111),
                               InterpValue::MakeUBits(/*bit_count=*/8, 114),
                               InterpValue::MakeUBits(/*bit_count=*/8, 108),
                               InterpValue::MakeUBits(/*bit_count=*/8, 100),
                               InterpValue::MakeUBits(/*bit_count=*/8, 33),
                           }));
  EXPECT_THAT(InterpValueAsString(hello_world_u8_array),
              IsOkAndHolds("Hello world!"));

  EXPECT_THAT(InterpValueAsString(InterpValue::MakeUBits(/*bit_count=*/8, 72)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be an array")));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue u9_array,
      InterpValue::MakeArray({InterpValue::MakeUBits(/*bit_count=*/9, 257)}));
  EXPECT_THAT(InterpValueAsString(u9_array),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Array elements must be u8")));
}

TEST(InterpValueHelpersTest, ValueToInterpValue) {
  EXPECT_THAT(ValueToInterpValue(Value(UBits(3, 32))),
              IsOkAndHolds(Eq(InterpValue::MakeUBits(32, 3))));
  EXPECT_THAT(
      ValueToInterpValue(Value(UBits(3, 32)), BitsType::MakeU32().get()),
      IsOkAndHolds(Eq(InterpValue::MakeU32(3))));

  EXPECT_THAT(
      ValueToInterpValue(Value::UBitsArray({3, 4, 5}, 32).value()),
      IsOkAndHolds(Eq(InterpValue::MakeArray({
                                                 InterpValue::MakeU32(3),
                                                 InterpValue::MakeU32(4),
                                                 InterpValue::MakeU32(5),
                                             })
                          .value())));
  ArrayType array_type(BitsType::MakeU32(), TypeDim::CreateU32(3));
  EXPECT_THAT(
      ValueToInterpValue(Value::UBitsArray({3, 4, 5}, 32).value(), &array_type),
      IsOkAndHolds(Eq(InterpValue::MakeArray({
                                                 InterpValue::MakeU32(3),
                                                 InterpValue::MakeU32(4),
                                                 InterpValue::MakeU32(5),
                                             })
                          .value())));

  EXPECT_THAT(ValueToInterpValue(
                  Value::Tuple({Value(UBits(3, 32)), Value(UBits(4, 32))})),
              IsOkAndHolds(Eq(InterpValue::MakeTuple(
                  {InterpValue::MakeU32(3), InterpValue::MakeU32(4)}))));
  // Tuple values can either come from structs or tuples, try passing in a
  // compatible concrete type of both.
  EXPECT_THAT(
      ValueToInterpValue(
          Value::Tuple({Value(UBits(3, 32)), Value(UBits(4, 32))}),
          TupleType::Create2(BitsType::MakeU32(), BitsType::MakeU32()).get()),
      IsOkAndHolds(Eq(InterpValue::MakeTuple(
          {InterpValue::MakeU32(3), InterpValue::MakeU32(4)}))));
  NameDef struct_name_def(/*owner=*/nullptr, /*span=*/Span::Fake(), "my_struct",
                          /*definer=*/nullptr);

  StructDef struct_def(/*owner=*/nullptr, /*span=*/Span::Fake(),
                       /*name_def=*/&struct_name_def,
                       /*parametric_bindings=*/{},
                       // these members are unused, but need to have the same
                       // number of elements as members in 'struct_type'.
                       /*members=*/
                       std::vector<StructMember>{{Span::Fake(), "", nullptr},
                                                 {Span::Fake(), "", nullptr}},
                       /*is_public=*/false);
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  members.push_back(BitsType::MakeU1());
  StructType struct_type(std::move(members), struct_def);
  EXPECT_THAT(ValueToInterpValue(
                  Value::Tuple({Value(UBits(3, 32)), Value(UBits(4, 32))}),
                  &struct_type),
              IsOkAndHolds(Eq(InterpValue::MakeTuple(
                  {InterpValue::MakeU32(3), InterpValue::MakeU32(4)}))));
}

}  // namespace
}  // namespace xls::dslx

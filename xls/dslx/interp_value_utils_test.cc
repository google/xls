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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/make_value_format_descriptor.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

SumType MakeMixedPayloadSumType(Module& module) {
  const Span kFakeSpan = Span::Fake();

  auto* sum_name = module.Make<NameDef>(kFakeSpan, "Example", nullptr);
  auto* none_name = module.Make<NameDef>(kFakeSpan, "None", nullptr);
  auto* byte_name = module.Make<NameDef>(kFakeSpan, "Byte", nullptr);
  auto* wide_name = module.Make<NameDef>(kFakeSpan, "Wide", nullptr);

  auto* u8_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8));
  auto* u16_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU16,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU16));

  auto* none = module.Make<SumVariant>(
      kFakeSpan, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* byte = module.Make<SumVariant>(
      kFakeSpan, byte_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u8_type}, std::vector<StructMemberNode*>{});
  auto* wide = module.Make<SumVariant>(
      kFakeSpan, wide_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u16_type}, std::vector<StructMemberNode*>{});
  auto* sum_def = module.Make<SumDef>(
      kFakeSpan, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{none, byte, wide}, /*is_public=*/false);
  sum_name->set_definer(sum_def);

  std::vector<SumTypeVariant> variants;
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> byte_members;
  byte_members.push_back(BitsType::MakeU8());
  variants.push_back(SumTypeVariant::MakeTuple(*byte, std::move(byte_members)));
  std::vector<std::unique_ptr<Type>> wide_members;
  wide_members.push_back(std::make_unique<BitsType>(/*is_signed=*/false, 16));
  variants.push_back(SumTypeVariant::MakeTuple(*wide, std::move(wide_members)));
  return SumType(*sum_def, std::move(variants));
}

SumType MakeOuterSumWithInactiveEmptyPayloadType(Module& module) {
  const Span kFakeSpan = Span::Fake();

  auto* empty_name = module.Make<NameDef>(kFakeSpan, "Empty", nullptr);
  auto* empty_def = module.Make<SumDef>(
      kFakeSpan, empty_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{}, /*is_public=*/false);
  empty_name->set_definer(empty_def);
  SumType empty_type(*empty_def, std::vector<SumTypeVariant>{});

  auto* outer_name = module.Make<NameDef>(kFakeSpan, "Outer", nullptr);
  auto* wrapped_name = module.Make<NameDef>(kFakeSpan, "Wrapped", nullptr);
  auto* nothing_name = module.Make<NameDef>(kFakeSpan, "Nothing", nullptr);
  auto* wrapped = module.Make<SumVariant>(
      kFakeSpan, wrapped_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{module.Make<TypeRefTypeAnnotation>(
          kFakeSpan, module.Make<TypeRef>(kFakeSpan, empty_def),
          std::vector<ExprOrType>{})},
      std::vector<StructMemberNode*>{});
  auto* nothing = module.Make<SumVariant>(
      kFakeSpan, nothing_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* outer_def = module.Make<SumDef>(
      kFakeSpan, outer_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{wrapped, nothing}, /*is_public=*/false);
  outer_name->set_definer(outer_def);

  std::vector<SumTypeVariant> outer_variants;
  std::vector<std::unique_ptr<Type>> wrapped_members;
  wrapped_members.push_back(empty_type.CloneToUnique());
  outer_variants.push_back(
      SumTypeVariant::MakeTuple(*wrapped, std::move(wrapped_members)));
  outer_variants.push_back(SumTypeVariant::MakeUnit(*nothing));
  return SumType(*outer_def, std::move(outer_variants));
}

SumType MakeOuterSumWithInactiveEmptyEnumPayloadType(Module& module) {
  const Span kFakeSpan = Span::Fake();

  auto* enum_name = module.Make<NameDef>(kFakeSpan, "Empty", nullptr);
  TypeAnnotation* enum_element_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU2,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU2));
  EnumDef* enum_def = module.Make<EnumDef>(
      kFakeSpan, enum_name, enum_element_type, std::vector<EnumMember>{},
      /*is_public=*/false);
  enum_name->set_definer(enum_def);
  EnumType enum_type(*enum_def, TypeDim::CreateU32(2), /*is_signed=*/false, {});

  auto* outer_name =
      module.Make<NameDef>(kFakeSpan, "MaybeImpossible", nullptr);
  auto* unit_name = module.Make<NameDef>(kFakeSpan, "Unit", nullptr);
  auto* impossible_name =
      module.Make<NameDef>(kFakeSpan, "Impossible", nullptr);
  auto* unit = module.Make<SumVariant>(
      kFakeSpan, unit_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* impossible = module.Make<SumVariant>(
      kFakeSpan, impossible_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{module.Make<TypeRefTypeAnnotation>(
          kFakeSpan, module.Make<TypeRef>(kFakeSpan, enum_def),
          std::vector<ExprOrType>{})},
      std::vector<StructMemberNode*>{});
  auto* outer_def = module.Make<SumDef>(
      kFakeSpan, outer_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{unit, impossible}, /*is_public=*/false);
  outer_name->set_definer(outer_def);

  std::vector<SumTypeVariant> outer_variants;
  outer_variants.push_back(SumTypeVariant::MakeUnit(*unit));
  std::vector<std::unique_ptr<Type>> impossible_members;
  impossible_members.push_back(enum_type.CloneToUnique());
  outer_variants.push_back(
      SumTypeVariant::MakeTuple(*impossible, std::move(impossible_members)));
  return SumType(*outer_def, std::move(outer_variants));
}

SumType MakeEnumPayloadSumType(Module& module, EnumDef** enum_def_out) {
  const Span kFakeSpan = Span::Fake();

  auto* enum_name = module.Make<NameDef>(kFakeSpan, "Flavor", nullptr);
  TypeAnnotation* enum_element_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU2,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU2));
  auto* vanilla_name = module.Make<NameDef>(kFakeSpan, "Vanilla", nullptr);
  auto* vanilla_value = module.Make<Number>(kFakeSpan, "0", NumberKind::kOther,
                                            enum_element_type);
  vanilla_name->set_definer(vanilla_value);
  auto* mint_name = module.Make<NameDef>(kFakeSpan, "Mint", nullptr);
  auto* mint_value = module.Make<Number>(kFakeSpan, "1", NumberKind::kOther,
                                         enum_element_type);
  mint_name->set_definer(mint_value);
  auto* enum_def = module.Make<EnumDef>(
      kFakeSpan, enum_name, enum_element_type,
      std::vector<EnumMember>{
          EnumMember{.name_def = vanilla_name, .value = vanilla_value},
          EnumMember{.name_def = mint_name, .value = mint_value},
      },
      /*is_public=*/false);
  enum_name->set_definer(enum_def);
  EnumType enum_type(
      *enum_def, TypeDim::CreateU32(2), /*is_signed=*/false,
      std::vector<InterpValue>{
          InterpValue::MakeEnum(UBits(0, 2), /*is_signed=*/false, enum_def),
          InterpValue::MakeEnum(UBits(1, 2), /*is_signed=*/false, enum_def),
      });

  auto* sum_name = module.Make<NameDef>(kFakeSpan, "Choice", nullptr);
  auto* some_name = module.Make<NameDef>(kFakeSpan, "Some", nullptr);
  auto* none_name = module.Make<NameDef>(kFakeSpan, "None", nullptr);
  auto* some = module.Make<SumVariant>(
      kFakeSpan, some_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{module.Make<TypeRefTypeAnnotation>(
          kFakeSpan, module.Make<TypeRef>(kFakeSpan, enum_def),
          std::vector<ExprOrType>{})},
      std::vector<StructMemberNode*>{});
  auto* none = module.Make<SumVariant>(
      kFakeSpan, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* sum_def = module.Make<SumDef>(
      kFakeSpan, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{some, none}, /*is_public=*/false);
  sum_name->set_definer(sum_def);

  std::vector<SumTypeVariant> variants;
  std::vector<std::unique_ptr<Type>> some_members;
  some_members.push_back(enum_type.CloneToUnique());
  variants.push_back(SumTypeVariant::MakeTuple(*some, std::move(some_members)));
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  *enum_def_out = enum_def;
  return SumType(*sum_def, std::move(variants));
}

SumType MakeOptionalPayloadSumType(Module& module, TypeAnnotation* annotation,
                                   std::unique_ptr<Type> payload_type) {
  const Span span = Span::Fake();
  auto* sum_name = module.Make<NameDef>(span, "Option", nullptr);
  auto* none_name = module.Make<NameDef>(span, "None", nullptr);
  auto* some_name = module.Make<NameDef>(span, "Some", nullptr);
  auto* none = module.Make<SumVariant>(
      span, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* some =
      module.Make<SumVariant>(span, some_name, SumVariant::PayloadShape::kTuple,
                              std::vector<TypeAnnotation*>{annotation},
                              std::vector<StructMemberNode*>{});
  auto* sum_def = module.Make<SumDef>(
      span, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{none, some}, /*is_public=*/false);
  sum_name->set_definer(sum_def);

  std::vector<SumTypeVariant> variants;
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> payload_members;
  payload_members.push_back(std::move(payload_type));
  variants.push_back(
      SumTypeVariant::MakeTuple(*some, std::move(payload_members)));
  return SumType(*sum_def, std::move(variants));
}

SumType MakeOptionalPayloadSumType(Module& module, BuiltinType annotation_kind,
                                   std::unique_ptr<Type> payload_type) {
  auto* annotation = module.Make<BuiltinTypeAnnotation>(
      Span::Fake(), annotation_kind,
      module.GetOrCreateBuiltinNameDef(annotation_kind));
  return MakeOptionalPayloadSumType(module, annotation,
                                    std::move(payload_type));
}

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
  FileTable file_table;
  Module module("my_test_module", /*fs_path=*/std::nullopt, file_table);

  std::vector<EnumMember> members;
  std::vector<InterpValue> member_values;
  BuiltinNameDef* builtin_name_def =
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU13);
  TypeAnnotation* element_type = module.Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU13, builtin_name_def);
  for (int i = 0; i < kNumMembers; i++) {
    NameDef* name_def =
        module.Make<NameDef>(Span::Fake(), absl::StrCat("member_", i), nullptr);
    Number* number = module.Make<Number>(Span::Fake(), absl::StrCat(i),
                                         NumberKind::kOther, element_type);
    name_def->set_definer(number);
    members.push_back(EnumMember{.name_def = name_def, .value = number});
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

TEST(InterpValueHelpersTest, CastUnsignedBitsToSignedEnumPreservesEnumSign) {
  const Span kFakeSpan = Span::Fake();
  FileTable file_table;
  Module module("signed_enum_test", /*fs_path=*/std::nullopt, file_table);
  TypeAnnotation* element_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kS2,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kS2));
  NameDef* member_name = module.Make<NameDef>(kFakeSpan, "Neg", nullptr);
  Number* member_value =
      module.Make<Number>(kFakeSpan, "-1", NumberKind::kOther, element_type);
  member_name->set_definer(member_value);
  NameDef* enum_name = module.Make<NameDef>(kFakeSpan, "Signed", nullptr);
  EnumDef* enum_def =
      module.Make<EnumDef>(kFakeSpan, enum_name, element_type,
                           std::vector<EnumMember>{EnumMember{
                               .name_def = member_name, .value = member_value}},
                           /*is_public=*/false);
  enum_name->set_definer(enum_def);
  EnumType enum_type(*enum_def, TypeDim::CreateU32(2), /*is_signed=*/true,
                     std::vector<InterpValue>{InterpValue::MakeSBits(2, -1)});

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue directly_cast,
      CastBitsToEnum(InterpValue::MakeUBits(2, 3), enum_type));
  ASSERT_TRUE(directly_cast.IsEnum());
  EXPECT_TRUE(directly_cast.GetEnumData().value().is_signed);
  EXPECT_THAT(directly_cast.GetBitValueViaSign(), IsOkAndHolds(-1));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue converted,
      SignConvertValue(enum_type, InterpValue::MakeUBits(2, 3)));
  ASSERT_TRUE(converted.IsEnum());
  EXPECT_TRUE(converted.GetEnumData().value().is_signed);
  EXPECT_THAT(converted.GetBitValueViaSign(), IsOkAndHolds(-1));
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

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  std::vector<StructMemberNode*> ast_members;
  ast_members.emplace_back(module.Make<StructMemberNode>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "x", nullptr), kFakeSpan,
      module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU8,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8))));
  ast_members.emplace_back(module.Make<StructMemberNode>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "y", nullptr), kFakeSpan,
      module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU1,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU1))));

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

TEST(InterpValueHelpersTest, CreateZeroSumValueFails) {
  const Span kFakeSpan = Span::Fake();

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* inner_name = module.Make<NameDef>(kFakeSpan, "Inner", nullptr);
  auto* inner_none_name = module.Make<NameDef>(kFakeSpan, "None", nullptr);
  auto* inner_some_name = module.Make<NameDef>(kFakeSpan, "Some", nullptr);
  auto* u32_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU32,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  auto* inner_none = module.Make<SumVariant>(
      kFakeSpan, inner_none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* inner_some = module.Make<SumVariant>(
      kFakeSpan, inner_some_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u32_type}, std::vector<StructMemberNode*>{});
  auto* inner_def = module.Make<SumDef>(
      kFakeSpan, inner_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{inner_none, inner_some}, /*is_public=*/false);
  inner_name->set_definer(inner_def);

  std::vector<SumTypeVariant> inner_variants;
  inner_variants.push_back(SumTypeVariant::MakeUnit(*inner_none));
  std::vector<std::unique_ptr<Type>> inner_some_members;
  inner_some_members.push_back(BitsType::MakeU32());
  inner_variants.push_back(
      SumTypeVariant::MakeTuple(*inner_some, std::move(inner_some_members)));
  SumType inner_type(*inner_def, std::move(inner_variants));

  auto* outer_name = module.Make<NameDef>(kFakeSpan, "Outer", nullptr);
  auto* outer_wrap_name = module.Make<NameDef>(kFakeSpan, "Wrap", nullptr);
  auto* outer_none_name = module.Make<NameDef>(kFakeSpan, "Nothing", nullptr);
  auto* outer_wrap = module.Make<SumVariant>(
      kFakeSpan, outer_wrap_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u32_type}, std::vector<StructMemberNode*>{});
  auto* outer_none = module.Make<SumVariant>(
      kFakeSpan, outer_none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* outer_def = module.Make<SumDef>(
      kFakeSpan, outer_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{outer_wrap, outer_none}, /*is_public=*/false);
  outer_name->set_definer(outer_def);

  std::vector<SumTypeVariant> outer_variants;
  std::vector<std::unique_ptr<Type>> outer_wrap_members;
  outer_wrap_members.push_back(inner_type.CloneToUnique());
  outer_variants.push_back(
      SumTypeVariant::MakeTuple(*outer_wrap, std::move(outer_wrap_members)));
  outer_variants.push_back(SumTypeVariant::MakeUnit(*outer_none));
  SumType outer_type(*outer_def, std::move(outer_variants));

  EXPECT_THAT(CreateZeroValueFromType(outer_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("semantic sum type `Outer`")));
}

TEST(InterpValueHelpersTest, CreatesSumWithBitsConstructorPayload) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto payload_type = std::make_unique<ArrayType>(
      std::make_unique<BitsConstructorType>(TypeDim::CreateBool(false)),
      TypeDim::CreateU32(8));
  SumType sum_type = MakeOptionalPayloadSumType(module, BuiltinType::kU8,
                                                std::move(payload_type));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue result,
      CreateSumValue(sum_type, "Some", {InterpValue::MakeUBits(8, 7)}));
  const std::vector<InterpValue>& slots =
      result.GetValuesOrDie().at(1).GetValuesOrDie();
  ASSERT_EQ(slots.size(), 1);
  EXPECT_EQ(slots.at(0).GetBitValueUnsigned().value(), 7);
}

TEST(InterpValueHelpersTest, CreatesActiveAndInactiveTokenSumPayloads) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeOptionalPayloadSumType(module, BuiltinType::kToken,
                                                std::make_unique<TokenType>());

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue inactive,
                           CreateSumValue(sum_type, "None", {}));
  EXPECT_TRUE(inactive.GetValuesOrDie().at(1).GetValuesOrDie().at(0).IsToken());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue another_inactive,
                           CreateSumValue(sum_type, "None", {}));
  EXPECT_TRUE(inactive.Eq(another_inactive));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue active,
      CreateSumValue(sum_type, "Some", {InterpValue::MakeToken()}));
  EXPECT_TRUE(active.GetValuesOrDie().at(1).GetValuesOrDie().at(0).IsToken());
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {InterpValue::MakeU8(0)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token-typed value")));

  XLS_ASSERT_OK_AND_ASSIGN(Value raw, inactive.ConvertToIr());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue restored,
                           ValueToInterpValue(raw, &sum_type));
  EXPECT_TRUE(restored.GetValuesOrDie().at(1).GetValuesOrDie().at(0).IsToken());
  EXPECT_TRUE(inactive.Eq(restored));
}

TEST(InterpValueHelpersTest,
     RoundTripsInactivePayloadWithTokenAndUninhabitedSum) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* never_name = module.Make<NameDef>(span, "Never", nullptr);
  auto* never_def =
      module.Make<SumDef>(span, never_name, std::vector<ParametricBinding*>{},
                          std::vector<SumVariant*>{}, /*is_public=*/false);
  never_name->set_definer(never_def);
  auto* token_annotation = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kToken,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kToken));
  auto* never_annotation = module.Make<TypeRefTypeAnnotation>(
      span, module.Make<TypeRef>(span, never_def), std::vector<ExprOrType>{});
  auto* tuple_annotation = module.Make<TupleTypeAnnotation>(
      span, std::vector<TypeAnnotation*>{token_annotation, never_annotation});

  std::vector<std::unique_ptr<Type>> tuple_members;
  tuple_members.push_back(std::make_unique<TokenType>());
  tuple_members.push_back(
      std::make_unique<SumType>(*never_def, std::vector<SumTypeVariant>{}));
  SumType sum_type = MakeOptionalPayloadSumType(
      module, tuple_annotation,
      std::make_unique<TupleType>(std::move(tuple_members)));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue inactive,
                           CreateSumValue(sum_type, "None", {}));
  XLS_ASSERT_OK_AND_ASSIGN(Value raw, inactive.ConvertToIr());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue restored,
                           ValueToInterpValue(raw, &sum_type));
  EXPECT_TRUE(restored.GetValuesOrDie()
                  .at(1)
                  .GetValuesOrDie()
                  .at(0)
                  .GetValuesOrDie()
                  .at(0)
                  .IsToken());
}

TEST(InterpValueHelpersTest, RoundTripsActiveAndInactiveProcSumPayloads) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* proc_name = module.Make<NameDef>(span, "Worker", nullptr);
  auto* proc_def = module.Make<ProcDef>(
      span, proc_name, std::vector<ParametricBinding*>{},
      std::vector<StructMemberNode*>{}, /*is_public=*/false);
  proc_name->set_definer(proc_def);
  auto* proc_annotation = module.Make<TypeRefTypeAnnotation>(
      span, module.Make<TypeRef>(span, proc_def), std::vector<ExprOrType>{});
  SumType sum_type = MakeOptionalPayloadSumType(
      module, proc_annotation,
      std::make_unique<ProcType>(std::vector<std::unique_ptr<Type>>{},
                                 *proc_def));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue inactive,
                           CreateSumValue(sum_type, "None", {}));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue active,
      CreateSumValue(sum_type, "Some", {InterpValue::MakeTuple({})}));

  for (const InterpValue& value : {inactive, active}) {
    XLS_ASSERT_OK_AND_ASSIGN(Value raw, value.ConvertToIr());
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue restored,
                             ValueToInterpValue(raw, &sum_type));
    EXPECT_TRUE(restored.Eq(value));
  }
}

TEST(InterpValueHelpersTest, CreateZeroEmptySumValueFails) {
  const Span kFakeSpan = Span::Fake();

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* empty_name = module.Make<NameDef>(kFakeSpan, "Empty", nullptr);
  auto* empty_def = module.Make<SumDef>(
      kFakeSpan, empty_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{}, /*is_public=*/false);
  empty_name->set_definer(empty_def);
  SumType empty_type(*empty_def, std::vector<SumTypeVariant>{});

  EXPECT_THAT(CreateZeroValueFromType(empty_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("semantic sum type `Empty`")));
}

TEST(InterpValueHelpersTest,
     CreateInternalPlaceholderEmptySumValueUsesZeroTag) {
  const Span kFakeSpan = Span::Fake();

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* empty_name = module.Make<NameDef>(kFakeSpan, "Empty", nullptr);
  auto* empty_def = module.Make<SumDef>(
      kFakeSpan, empty_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{}, /*is_public=*/false);
  empty_name->set_definer(empty_def);
  SumType empty_type(*empty_def, std::vector<SumTypeVariant>{});

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue zero,
      internal::CreateInternalPlaceholderValueFromType(empty_type));
  EXPECT_TRUE(InterpValue::MakeTuple(
                  {InterpValue::MakeUBits(1, 0), InterpValue::MakeTuple({})})
                  .Eq(zero));
}

TEST(InterpValueHelpersTest,
     CreateInternalPlaceholderEmptyEnumValueUsesZeroBits) {
  const Span kFakeSpan = Span::Fake();

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* enum_name = module.Make<NameDef>(kFakeSpan, "Empty", nullptr);
  TypeAnnotation* enum_element_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU2,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU2));
  EnumDef* enum_def = module.Make<EnumDef>(
      kFakeSpan, enum_name, enum_element_type, std::vector<EnumMember>{},
      /*is_public=*/false);
  enum_name->set_definer(enum_def);
  EnumType enum_type(*enum_def, TypeDim::CreateU32(2), /*is_signed=*/false, {});

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue placeholder,
      internal::CreateInternalPlaceholderValueFromType(enum_type));
  EXPECT_TRUE(InterpValue::MakeEnum(Bits(2), /*is_signed=*/false, enum_def)
                  .Eq(placeholder));
  EXPECT_THAT(CreateZeroValueFromType(enum_type),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Cannot create zero value")));
}

TEST(InterpValueHelpersTest,
     CreateSumValueUsesInternalPlaceholderForInactiveEmptySumPayload) {
  const Span kFakeSpan = Span::Fake();

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* empty_name = module.Make<NameDef>(kFakeSpan, "Empty", nullptr);
  auto* empty_def = module.Make<SumDef>(
      kFakeSpan, empty_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{}, /*is_public=*/false);
  empty_name->set_definer(empty_def);
  SumType empty_type(*empty_def, std::vector<SumTypeVariant>{});

  auto* outer_name = module.Make<NameDef>(kFakeSpan, "Outer", nullptr);
  auto* wrapped_name = module.Make<NameDef>(kFakeSpan, "Wrapped", nullptr);
  auto* nothing_name = module.Make<NameDef>(kFakeSpan, "Nothing", nullptr);
  auto* wrapped = module.Make<SumVariant>(
      kFakeSpan, wrapped_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{module.Make<TypeRefTypeAnnotation>(
          kFakeSpan, module.Make<TypeRef>(kFakeSpan, empty_def),
          std::vector<ExprOrType>{})},
      std::vector<StructMemberNode*>{});
  auto* nothing = module.Make<SumVariant>(
      kFakeSpan, nothing_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* outer_def = module.Make<SumDef>(
      kFakeSpan, outer_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{wrapped, nothing}, /*is_public=*/false);
  outer_name->set_definer(outer_def);

  std::vector<SumTypeVariant> outer_variants;
  std::vector<std::unique_ptr<Type>> wrapped_members;
  wrapped_members.push_back(empty_type.CloneToUnique());
  outer_variants.push_back(
      SumTypeVariant::MakeTuple(*wrapped, std::move(wrapped_members)));
  outer_variants.push_back(SumTypeVariant::MakeUnit(*nothing));
  SumType outer_type(*outer_def, std::move(outer_variants));

  const std::vector<InterpValue> no_payload_values;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      CreateSumValue(outer_type, "Nothing", no_payload_values));
  EXPECT_TRUE(
      InterpValue::MakeTuple(
          {InterpValue::MakeUBits(1, 1),
           InterpValue::MakeTuple({InterpValue::MakeTuple(
               {InterpValue::MakeUBits(1, 0), InterpValue::MakeTuple({})})})})
          .Eq(value));
}

TEST(InterpValueHelpersTest, CreateSumValueRejectsPayloadTypeMismatch) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeMixedPayloadSumType(module);

  EXPECT_THAT(CreateSumValue(sum_type, "Byte", {InterpValue::MakeUBits(16, 1)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match")));
}

TEST(InterpValueHelpersTest, CreateSumValueRejectsMalformedTuplePayload) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* u8 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* tuple_annotation =
      module.Make<TupleTypeAnnotation>(span, std::vector<TypeAnnotation*>{u8});
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  SumType sum_type = MakeOptionalPayloadSumType(
      module, tuple_annotation,
      std::make_unique<TupleType>(std::move(members)));

  EXPECT_THAT(CreateSumValue(sum_type, "Some", {InterpValue::MakeU8(7)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected tuple-typed value")));
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {InterpValue::MakeTuple({})}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected 1 members; got 0")));
  EXPECT_THAT(
      CreateSumValue(sum_type, "Some",
                     {InterpValue::MakeTuple({InterpValue::MakeUBits(16, 7)})}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("does not match")));

  Value malformed_raw =
      Value::Tuple({Value(UBits(1, 1)), Value::Tuple({Value::Tuple({})})});
  EXPECT_THAT(ValueToInterpValue(malformed_raw, &sum_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected 1 elements; got 0")));
}

TEST(InterpValueHelpersTest, CreateSumValueRejectsMalformedStructPayload) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* u8 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* struct_name = module.Make<NameDef>(span, "Payload", nullptr);
  std::vector<StructMemberNode*> fields = {
      module.Make<StructMemberNode>(
          span, module.Make<NameDef>(span, "value", nullptr), span, u8),
  };
  auto* struct_def = module.Make<StructDef>(
      span, struct_name, std::vector<ParametricBinding*>{}, fields,
      /*is_public=*/false);
  struct_name->set_definer(struct_def);
  auto* annotation = module.Make<TypeRefTypeAnnotation>(
      span, module.Make<TypeRef>(span, struct_def), std::vector<ExprOrType>{});
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  SumType sum_type = MakeOptionalPayloadSumType(
      module, annotation,
      std::make_unique<StructType>(std::move(members), *struct_def));

  EXPECT_THAT(CreateSumValue(sum_type, "Some", {InterpValue::MakeU8(7)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected struct-typed value")));
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {InterpValue::MakeTuple({})}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected 1 members; got 0")));
  EXPECT_THAT(
      CreateSumValue(sum_type, "Some",
                     {InterpValue::MakeTuple({InterpValue::MakeUBits(16, 7)})}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("does not match")));
}

TEST(InterpValueHelpersTest, CreateSumValueRejectsMalformedArrayPayload) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* u8 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* u32 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU32,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU32));
  auto* dimension = module.Make<Number>(span, "2", NumberKind::kOther, u32);
  auto* annotation = module.Make<ArrayTypeAnnotation>(span, u8, dimension);
  SumType sum_type = MakeOptionalPayloadSumType(
      module, annotation,
      std::make_unique<ArrayType>(BitsType::MakeU8(), TypeDim::CreateU32(2)));

  EXPECT_THAT(CreateSumValue(sum_type, "Some", {InterpValue::MakeU8(7)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected array-typed value")));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue short_array,
                           InterpValue::MakeArray({InterpValue::MakeU8(7)}));
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {short_array}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected 2 elements; got 1")));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue wide_array,
      InterpValue::MakeArray(
          {InterpValue::MakeUBits(16, 7), InterpValue::MakeUBits(16, 8)}));
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {wide_array}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match")));
}

TEST(InterpValueHelpersTest,
     FormatsEverySumVariantFromItsCanonicalProductionTypeDescriptor) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* sum_name = module.Make<NameDef>(span, "Option", nullptr);
  auto* none_name = module.Make<NameDef>(span, "None", nullptr);
  auto* some_name = module.Make<NameDef>(span, "Some", nullptr);
  auto* pair_name = module.Make<NameDef>(span, "Pair", nullptr);
  auto* empty_tuple_name = module.Make<NameDef>(span, "EmptyTuple", nullptr);
  auto* empty_struct_name = module.Make<NameDef>(span, "EmptyStruct", nullptr);
  auto* u8 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* u16 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU16,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU16));
  auto* none = module.Make<SumVariant>(
      span, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* some = module.Make<SumVariant>(
      span, some_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u8}, std::vector<StructMemberNode*>{});
  std::vector<StructMemberNode*> fields = {
      module.Make<StructMemberNode>(
          span, module.Make<NameDef>(span, "left", nullptr), span, u8),
      module.Make<StructMemberNode>(
          span, module.Make<NameDef>(span, "right", nullptr), span, u16),
  };
  auto* pair = module.Make<SumVariant>(span, pair_name,
                                       SumVariant::PayloadShape::kStruct,
                                       std::vector<TypeAnnotation*>{}, fields);
  auto* empty_tuple = module.Make<SumVariant>(
      span, empty_tuple_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* empty_struct = module.Make<SumVariant>(
      span, empty_struct_name, SumVariant::PayloadShape::kStruct,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* sum_def = module.Make<SumDef>(
      span, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{none, some, pair, empty_tuple, empty_struct},
      /*is_public=*/false);
  sum_name->set_definer(sum_def);

  std::vector<SumTypeVariant> variants;
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> some_types;
  some_types.push_back(BitsType::MakeU8());
  variants.push_back(SumTypeVariant::MakeTuple(*some, std::move(some_types)));
  std::vector<std::unique_ptr<Type>> payload_types;
  payload_types.push_back(BitsType::MakeU8());
  payload_types.push_back(std::make_unique<BitsType>(false, 16));
  variants.push_back(
      SumTypeVariant::MakeStruct(*pair, std::move(payload_types)));
  variants.push_back(SumTypeVariant::MakeTuple(
      *empty_tuple, std::vector<std::unique_ptr<Type>>{}));
  variants.push_back(SumTypeVariant::MakeStruct(
      *empty_struct, std::vector<std::unique_ptr<Type>>{}));
  SumType sum_type(*sum_def, std::move(variants));

  XLS_ASSERT_OK_AND_ASSIGN(
      ValueFormatDescriptor descriptor,
      MakeValueFormatDescriptor(sum_type, FormatPreference::kDefault));
  ASSERT_EQ(descriptor.sum_variant_count(), 5);
  EXPECT_EQ(descriptor.sum_payload_slot_count(), 3);
  const ValueFormatSumVariantView none_view = descriptor.sum_variant(0);
  EXPECT_EQ(none_view.name(), "None");
  EXPECT_EQ(none_view.kind(), ValueFormatSumVariantKind::kUnit);
  EXPECT_EQ(none_view.payload_slot_count(), 0);
  EXPECT_THAT(none_view.field_names(), ::testing::IsEmpty());
  EXPECT_THAT(none_view.payload_formats(), ::testing::IsEmpty());

  const ValueFormatSumVariantView some_view = descriptor.sum_variant(1);
  EXPECT_EQ(some_view.name(), "Some");
  EXPECT_EQ(some_view.kind(), ValueFormatSumVariantKind::kTuple);
  EXPECT_EQ(some_view.payload_start(), 0);
  EXPECT_EQ(some_view.payload_slot_count(), 1);
  EXPECT_THAT(some_view.field_names(), ::testing::IsEmpty());
  ASSERT_EQ(some_view.payload_formats().size(), 1);
  EXPECT_TRUE(some_view.payload_formats().front().IsLeafValue());

  const ValueFormatSumVariantView pair_view = descriptor.sum_variant(2);
  EXPECT_EQ(pair_view.name(), "Pair");
  EXPECT_EQ(pair_view.kind(), ValueFormatSumVariantKind::kStruct);
  EXPECT_EQ(pair_view.payload_start(), 1);
  EXPECT_EQ(pair_view.payload_slot_count(), 2);
  EXPECT_THAT(pair_view.field_names(), ::testing::ElementsAre("left", "right"));
  ASSERT_EQ(pair_view.payload_formats().size(), 2);
  EXPECT_TRUE(pair_view.payload_formats().front().IsLeafValue());
  EXPECT_TRUE(pair_view.payload_formats().back().IsLeafValue());

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue none_value,
                           CreateSumValue(sum_type, "None", {}));
  EXPECT_THAT(none_value.ToFormattedString(descriptor,
                                           /*include_type_prefix=*/true),
              IsOkAndHolds("Option::None"));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue some_value,
      CreateSumValue(sum_type, "Some", {InterpValue::MakeUBits(8, 7)}));
  EXPECT_THAT(some_value.ToFormattedString(descriptor,
                                           /*include_type_prefix=*/true),
              IsOkAndHolds("Option::Some(u8:7)"));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           CreateSumValue(sum_type, "Pair",
                                          {InterpValue::MakeUBits(8, 3),
                                           InterpValue::MakeUBits(16, 4)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string formatted,
      value.ToFormattedString(descriptor, /*include_type_prefix=*/true));
  EXPECT_EQ(formatted, "Option::Pair {\n    left: u8:3,\n    right: u16:4\n}");

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue empty_tuple_value,
                           CreateSumValue(sum_type, "EmptyTuple", {}));
  EXPECT_THAT(empty_tuple_value.ToFormattedString(descriptor,
                                                  /*include_type_prefix=*/true),
              IsOkAndHolds("Option::EmptyTuple()"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue empty_struct_value,
                           CreateSumValue(sum_type, "EmptyStruct", {}));
  EXPECT_THAT(empty_struct_value.ToFormattedString(
                  descriptor, /*include_type_prefix=*/true),
              IsOkAndHolds("Option::EmptyStruct {}"));
}

TEST(InterpValueHelpersTest, ValidatesDeeplyNestedSemanticSums) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* base_name = module.Make<NameDef>(span, "Base", nullptr);
  auto* unit_name = module.Make<NameDef>(span, "Unit", nullptr);
  auto* unit = module.Make<SumVariant>(
      span, unit_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* base_def =
      module.Make<SumDef>(span, base_name, std::vector<ParametricBinding*>{},
                          std::vector<SumVariant*>{unit}, /*is_public=*/false);
  base_name->set_definer(base_def);
  SumDef* current_def = base_def;
  std::vector<SumTypeVariant> base_variants;
  base_variants.push_back(SumTypeVariant::MakeUnit(*unit));
  auto current = std::make_unique<SumType>(*base_def, std::move(base_variants));
  InterpValue value =
      internal::CreateEncodedSumTuple(InterpValue::MakeUBits(1, 0), {});
  InterpValue malformed =
      internal::CreateEncodedSumTuple(InterpValue::MakeUBits(1, 1), {});

  for (int64_t depth = 0; depth < 24; ++depth) {
    auto* outer_name =
        module.Make<NameDef>(span, absl::StrCat("Outer", depth), nullptr);
    auto* wrap_name =
        module.Make<NameDef>(span, absl::StrCat("Wrap", depth), nullptr);
    auto* annotation = module.Make<TypeRefTypeAnnotation>(
        span, module.Make<TypeRef>(span, current_def),
        std::vector<ExprOrType>{});
    auto* wrap = module.Make<SumVariant>(
        span, wrap_name, SumVariant::PayloadShape::kTuple,
        std::vector<TypeAnnotation*>{annotation},
        std::vector<StructMemberNode*>{});
    auto* outer_def = module.Make<SumDef>(
        span, outer_name, std::vector<ParametricBinding*>{},
        std::vector<SumVariant*>{wrap}, /*is_public=*/false);
    outer_name->set_definer(outer_def);
    std::vector<std::unique_ptr<Type>> members;
    members.push_back(current->CloneToUnique());
    std::vector<SumTypeVariant> outer_variants;
    outer_variants.push_back(
        SumTypeVariant::MakeTuple(*wrap, std::move(members)));
    current = std::make_unique<SumType>(*outer_def, std::move(outer_variants));
    current_def = outer_def;
    value = internal::CreateEncodedSumTuple(InterpValue::MakeUBits(1, 0),
                                            {std::move(value)});
    malformed = internal::CreateEncodedSumTuple(InterpValue::MakeUBits(1, 0),
                                                {std::move(malformed)});
  }

  XLS_ASSERT_OK_AND_ASSIGN(Value ir_value, value.ConvertToIr());
  EXPECT_THAT(ValueToInterpValue(ir_value, current.get()),
              ::absl_testing::IsOk());
  XLS_ASSERT_OK_AND_ASSIGN(Value malformed_ir_value, malformed.ConvertToIr());
  EXPECT_THAT(
      ValueToInterpValue(malformed_ir_value, current.get()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("invalid tag")));
}

TEST(InterpValueHelpersTest, SignConvertValuePreservesSumEnumPayload) {
  const Span kFakeSpan = Span::Fake();

  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);

  auto* enum_name = module.Make<NameDef>(kFakeSpan, "Tag", nullptr);
  auto* enum_member_name = module.Make<NameDef>(kFakeSpan, "One", nullptr);
  TypeAnnotation* enum_element_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU2,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU2));
  Number* enum_member_value = module.Make<Number>(
      kFakeSpan, "1", NumberKind::kOther, enum_element_type);
  enum_member_name->set_definer(enum_member_value);
  EnumDef* enum_def = module.Make<EnumDef>(
      kFakeSpan, enum_name, enum_element_type,
      std::vector<EnumMember>{
          EnumMember{.name_def = enum_member_name, .value = enum_member_value}},
      /*is_public=*/false);
  enum_name->set_definer(enum_def);
  EnumType enum_type(*enum_def, TypeDim::CreateU32(2),
                     /*is_signed=*/false,
                     {InterpValue::MakeUBits(/*bit_count=*/2, /*value=*/1)});

  auto* sum_name = module.Make<NameDef>(kFakeSpan, "Example", nullptr);
  auto* some_name = module.Make<NameDef>(kFakeSpan, "Some", nullptr);
  auto* none_name = module.Make<NameDef>(kFakeSpan, "None", nullptr);
  auto* some = module.Make<SumVariant>(
      kFakeSpan, some_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{module.Make<TypeRefTypeAnnotation>(
          kFakeSpan, module.Make<TypeRef>(kFakeSpan, enum_def),
          std::vector<ExprOrType>{})},
      std::vector<StructMemberNode*>{});
  auto* none = module.Make<SumVariant>(
      kFakeSpan, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* sum_def = module.Make<SumDef>(
      kFakeSpan, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{some, none}, /*is_public=*/false);
  sum_name->set_definer(sum_def);

  std::vector<SumTypeVariant> variants;
  std::vector<std::unique_ptr<Type>> some_members;
  some_members.push_back(enum_type.CloneToUnique());
  variants.push_back(SumTypeVariant::MakeTuple(*some, std::move(some_members)));
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  SumType sum_type(*sum_def, std::move(variants));

  const InterpValue enum_value =
      InterpValue::MakeEnum(UBits(/*value=*/1, /*bit_count=*/2),
                            /*is_signed=*/false, enum_def);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue sum_value,
                           CreateSumValue(sum_type, "Some", {enum_value}));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue converted,
                           SignConvertValue(sum_type, sum_value));
  EXPECT_TRUE(converted.Eq(sum_value));
}

TEST(InterpValueHelpersTest, SignConvertValueReifiesRawSumEnumPayload) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  EnumDef* enum_def = nullptr;
  SumType sum_type = MakeEnumPayloadSumType(module, &enum_def);

  InterpValue raw_value = InterpValue::MakeTuple(
      {InterpValue::MakeUBits(/*bit_count=*/1, /*value=*/0),
       InterpValue::MakeTuple(
           {InterpValue::MakeUBits(/*bit_count=*/2, /*value=*/1)})});
  InterpValue enum_value =
      InterpValue::MakeEnum(UBits(/*value=*/1, /*bit_count=*/2),
                            /*is_signed=*/false, enum_def);
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue expected,
      CreateSumValue(sum_type, "Some", std::vector<InterpValue>{enum_value}));

  EXPECT_THAT(SignConvertValue(sum_type, raw_value),
              IsOkAndHolds(Eq(expected)));
}

TEST(InterpValueHelpersTest,
     SignConvertValuePreservesInactiveEmptySumPlaceholder) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType outer_type = MakeOuterSumWithInactiveEmptyPayloadType(module);

  const std::vector<InterpValue> no_payload_values;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      CreateSumValue(outer_type, "Nothing", no_payload_values));
  EXPECT_THAT(SignConvertValue(outer_type, value), IsOkAndHolds(Eq(value)));
}

TEST(InterpValueHelpersTest,
     SignConvertValuePreservesInactiveEmptyEnumPlaceholder) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType outer_type = MakeOuterSumWithInactiveEmptyEnumPayloadType(module);

  const std::vector<InterpValue> no_payload_values;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value, CreateSumValue(outer_type, "Unit", no_payload_values));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue converted,
                           SignConvertValue(outer_type, value));
  EXPECT_TRUE(converted.Eq(value));
  EXPECT_EQ(converted.GetValuesOrDie().at(1).GetValuesOrDie().at(0).tag(),
            InterpValueTag::kEnum);
}

TEST(InterpValueHelpersTest, CreateSumValueRejectsActiveNonMemberEnumPayload) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  EnumDef* enum_def = nullptr;
  SumType sum_type = MakeEnumPayloadSumType(module, &enum_def);

  InterpValue invalid_enum_value =
      InterpValue::MakeEnum(UBits(/*value=*/3, /*bit_count=*/2),
                            /*is_signed=*/false, enum_def);
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {invalid_enum_value}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("declared member")));
}

TEST(InterpValueHelpersTest, CreateSumValueRejectsForeignNominalEnumPayload) {
  FileTable file_table;
  Module expected_module("expected", /*fs_path=*/std::nullopt, file_table);
  Module foreign_module("foreign", /*fs_path=*/std::nullopt, file_table);
  EnumDef* expected_enum = nullptr;
  EnumDef* foreign_enum = nullptr;
  SumType sum_type = MakeEnumPayloadSumType(expected_module, &expected_enum);
  MakeEnumPayloadSumType(foreign_module, &foreign_enum);
  ASSERT_NE(expected_enum, foreign_enum);

  InterpValue foreign_value = InterpValue::MakeEnum(
      UBits(/*value=*/1, /*bit_count=*/2), /*is_signed=*/false, foreign_enum);
  EXPECT_THAT(CreateSumValue(sum_type, "Some", {foreign_value}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match enum")));
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

  NameDef struct_member_name_def(/*owner=*/nullptr, /*span=*/Span::Fake(),
                                 "member",
                                 /*definer=*/nullptr);
  StructMemberNode member(/* owner= */ nullptr, Span::Fake(),
                          /*name_def= */ &struct_member_name_def,
                          /*colon_span=*/Span::Fake(), /*type=*/nullptr);
  StructDef struct_def(/*owner=*/nullptr, /*span=*/Span::Fake(),
                       /*name_def=*/&struct_name_def,
                       /*parametric_bindings=*/{},
                       // these members are unused, but need to have the same
                       // number of elements as members in 'struct_type'.
                       /*members=*/
                       std::vector<StructMemberNode*>{&member, &member},
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

TEST(InterpValueHelpersTest, ValueToInterpValueEnum) {
  EnumDef enum_def(/*owner=*/nullptr, /*span=*/Span::Fake(),
                   /*name_def=*/nullptr, /*type=*/{},
                   /*values=*/{}, /*is_public=*/false);
  EnumType enum_type(enum_def, TypeDim::CreateU32(32), /*is_signed=*/false, {});
  EXPECT_THAT(ValueToInterpValue(Value(UBits(3, 32)), &enum_type),
              IsOkAndHolds(Eq(InterpValue::MakeEnum(
                  UBits(3, 32), /*is_signed=*/false, &enum_def))));
}

TEST(InterpValueHelpersTest, GetLeafChannelReferences) {
  InterpValue ch0 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 0);
  InterpValue ch1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);
  InterpValue ch2 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 2);
  InterpValue ch3 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 3);

  InterpValue sub_arr0 = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 10, /*definer=*/nullptr, {ch0, ch1});
  InterpValue sub_arr1 = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 11, /*definer=*/nullptr, {ch2, ch3});
  InterpValue arr2d = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 12, /*definer=*/nullptr, {sub_arr0, sub_arr1});

  EXPECT_THAT(GetLeafChannelReferences(ch0), testing::ElementsAre(ch0));
  EXPECT_THAT(GetLeafChannelReferences(sub_arr0),
              testing::ElementsAre(ch0, ch1));
  EXPECT_THAT(GetLeafChannelReferences(arr2d),
              testing::ElementsAre(ch0, ch1, ch2, ch3));
}

TEST(InterpValueHelpersTest, ValueToInterpValueSumRejectsInvalidTag) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeMixedPayloadSumType(module);

  Value raw =
      Value::Tuple({Value(UBits(3, 2)),
                    Value::Tuple({Value(UBits(0, 8)), Value(UBits(0, 16))})});
  EXPECT_THAT(
      ValueToInterpValue(raw, &sum_type),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("invalid tag")));

  Value narrow_tag =
      Value::Tuple({Value(UBits(0, 1)),
                    Value::Tuple({Value(UBits(0, 8)), Value(UBits(0, 16))})});
  EXPECT_THAT(ValueToInterpValue(narrow_tag, &sum_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected a 2-bit tag")));

  Value non_bits_tag =
      Value::Tuple({Value::Tuple({}),
                    Value::Tuple({Value(UBits(0, 8)), Value(UBits(0, 16))})});
  EXPECT_THAT(
      ValueToInterpValue(non_bits_tag, &sum_type),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unsigned bits")));

  InterpValue signed_tag = internal::CreateEncodedSumTuple(
      InterpValue::MakeSBits(2, 0),
      {InterpValue::MakeUBits(8, 0), InterpValue::MakeUBits(16, 0)});
  EXPECT_THAT(
      SignConvertValue(sum_type, signed_tag),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unsigned bits")));
}

TEST(InterpValueHelpersTest, ValueToInterpValueSumRejectsMalformedRawShape) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeMixedPayloadSumType(module);

  EXPECT_THAT(ValueToInterpValue(Value(UBits(0, 2)), &sum_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tag and a payload tuple")));
  EXPECT_THAT(ValueToInterpValue(Value::Tuple({Value(UBits(0, 2))}), &sum_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tag and a payload tuple")));
  EXPECT_THAT(
      ValueToInterpValue(Value::Tuple({Value(UBits(0, 2)), Value(UBits(0, 8))}),
                         &sum_type),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("tag and a payload tuple")));
  EXPECT_THAT(
      ValueToInterpValue(Value::Tuple({Value(UBits(0, 2)),
                                       Value::Tuple({Value(UBits(0, 8))})}),
                         &sum_type),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must contain 2 payload slots")));

  Value malformed_active_payload = Value::Tuple(
      {Value(UBits(1, 2)),
       Value::Tuple({Value::Tuple({Value(UBits(0, 8)), Value(UBits(0, 8))}),
                     Value(UBits(0, 16))})});
  EXPECT_THAT(ValueToInterpValue(malformed_active_payload, &sum_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match expected type")));
}

TEST(InterpValueHelpersTest, ValueToInterpValueRejectsMalformedTypedAggregate) {
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  TupleType tuple_type(std::move(members));

  EXPECT_THAT(
      ValueToInterpValue(Value::Tuple({Value(UBits(0, 8)), Value(UBits(0, 8))}),
                         &tuple_type),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("expected 1 elements; got 2")));
  XLS_ASSERT_OK_AND_ASSIGN(Value raw_array, Value::Array({Value(UBits(0, 8))}));
  EXPECT_THAT(ValueToInterpValue(raw_array, &tuple_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match expected type")));
}

TEST(InterpValueHelpersTest,
     ValueToInterpValueSumRestoresNonzeroActivePayloadOffset) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeMixedPayloadSumType(module);

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue expected,
      CreateSumValue(sum_type, "Wide", {InterpValue::MakeUBits(16, 42)}));
  XLS_ASSERT_OK_AND_ASSIGN(Value raw, expected.ConvertToIr());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue actual,
                           ValueToInterpValue(raw, &sum_type));

  const std::vector<InterpValue>& payload_slots =
      actual.GetValuesOrDie().at(1).GetValuesOrDie();
  ASSERT_EQ(payload_slots.size(), 2);
  EXPECT_EQ(payload_slots.at(0).GetBitValueUnsigned().value(), 0);
  EXPECT_EQ(payload_slots.at(1).GetBitValueUnsigned().value(), 42);
  EXPECT_TRUE(actual.Eq(expected));
}

TEST(InterpValueHelpersTest,
     ValueToInterpValueSumAcceptsInactiveEmptySumPlaceholder) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType outer_type = MakeOuterSumWithInactiveEmptyPayloadType(module);

  Value raw = Value::Tuple(
      {Value(UBits(1, 1)),
       Value::Tuple({Value::Tuple({Value(UBits(0, 1)), Value::Tuple({})})})});
  const std::vector<InterpValue> no_payload_values;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue expected,
      CreateSumValue(outer_type, "Nothing", no_payload_values));
  EXPECT_THAT(ValueToInterpValue(raw, &outer_type), IsOkAndHolds(Eq(expected)));
}

TEST(InterpValueHelpersTest,
     ValueToInterpValueSumAcceptsInactiveEmptyEnumPlaceholder) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType outer_type = MakeOuterSumWithInactiveEmptyEnumPayloadType(module);

  Value raw =
      Value::Tuple({Value(UBits(0, 1)), Value::Tuple({Value(UBits(0, 2))})});
  const std::vector<InterpValue> no_payload_values;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue expected,
      CreateSumValue(outer_type, "Unit", no_payload_values));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue actual,
                           ValueToInterpValue(raw, &outer_type));
  EXPECT_TRUE(actual.Eq(expected));
  EXPECT_EQ(actual.GetValuesOrDie().at(1).GetValuesOrDie().at(0).tag(),
            InterpValueTag::kEnum);
}

TEST(InterpValueHelpersTest,
     ValueToInterpValueSumRejectsNoncanonicalInactivePayload) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeMixedPayloadSumType(module);

  Value raw =
      Value::Tuple({Value(UBits(0, 2)),
                    Value::Tuple({Value(UBits(1, 8)), Value(UBits(0, 16))})});
  EXPECT_THAT(ValueToInterpValue(raw, &sum_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("noncanonical inactive payload slot")));
}

}  // namespace
}  // namespace xls::dslx

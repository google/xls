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

#include "xls/dslx/type_system/type.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/ir/bits.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

const Pos kFakePos("<fake>", 0, 0);
const Span kFakeSpan(kFakePos, kFakePos);

TEST(TypeTest, TestU32) {
  BitsType t(false, 32);
  EXPECT_EQ("uN[32]", t.ToString());
  EXPECT_EQ("ubits", t.GetDebugTypeName());
  EXPECT_EQ(false, t.is_signed());
  EXPECT_EQ(false, t.HasEnum());
  EXPECT_EQ(std::vector<TypeDim>{TypeDim::CreateU32(32)}, t.GetAllDims());
  EXPECT_EQ(t, *t.ToUBits());
  EXPECT_TRUE(IsUBits(t));
}

TEST(TypeTest, TestUnit) {
  TupleType t({});
  EXPECT_EQ("()", t.ToString());
  EXPECT_EQ("tuple", t.GetDebugTypeName());
  EXPECT_EQ(false, t.HasEnum());
  EXPECT_TRUE(t.GetAllDims().empty());
  EXPECT_FALSE(IsUBits(t));
}

TEST(TypeTest, TestArrayOfU32) {
  ArrayType t(std::make_unique<BitsType>(false, 32), TypeDim::CreateU32(1));
  EXPECT_EQ("uN[32][1]", t.ToString());
  EXPECT_EQ("array", t.GetDebugTypeName());
  EXPECT_EQ(false, t.HasEnum());
  std::vector<TypeDim> want_dims = {TypeDim::CreateU32(1),
                                    TypeDim::CreateU32(32)};
  EXPECT_EQ(want_dims, t.GetAllDims());
  EXPECT_FALSE(IsUBits(t));
}

TEST(TypeTest, TestEnum) {
  Module m("test", /*fs_path=*/std::nullopt);
  Pos fake_pos("fake.x", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  auto* my_enum = m.Make<NameDef>(fake_span, "MyEnum", nullptr);
  auto* e = m.Make<EnumDef>(fake_span, my_enum, /*type=*/nullptr,
                            /*values=*/std::vector<EnumMember>{},
                            /*is_public=*/false);
  my_enum->set_definer(e);
  EnumType t(*e, /*bit_count=*/TypeDim::CreateU32(2),
             /*is_signed=*/false, {});
  EXPECT_TRUE(t.HasEnum());
  EXPECT_EQ(std::vector<TypeDim>{TypeDim::CreateU32(2)}, t.GetAllDims());
  EXPECT_EQ("MyEnum", t.ToString());
  EXPECT_EQ("fake.x:MyEnum", t.ToStringFullyQualified());
}

TEST(TypeTest, FunctionTypeU32ToS32) {
  std::vector<std::unique_ptr<Type>> params;
  params.push_back(std::make_unique<BitsType>(false, 32));
  FunctionType t(std::move(params), std::make_unique<BitsType>(true, 32));
  EXPECT_EQ(1, t.GetParams().size());
  EXPECT_EQ("uN[32]", t.GetParams()[0]->ToString());
  EXPECT_EQ("sN[32]", t.return_type().ToString());
}

TEST(TypeTest, FromInterpValueSbits) {
  auto s8_m1 = InterpValue::MakeSBits(8, -1);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> ct,
                           Type::FromInterpValue(s8_m1));
  EXPECT_EQ(ct->ToString(), "sN[8]");
}

TEST(TypeTest, FromInterpValueArrayU2) {
  auto v = InterpValue::MakeArray({
                                      InterpValue::MakeUBits(2, 0b10),
                                      InterpValue::MakeUBits(2, 0b01),
                                      InterpValue::MakeUBits(2, 0b11),
                                  })
               .value();
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> ct, Type::FromInterpValue(v));
  EXPECT_EQ(ct->ToString(), "uN[2][3]");
  EXPECT_THAT(ct->GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(6)));
}

TEST(TypeTest, FromInterpValueTupleEmpty) {
  auto v = InterpValue::MakeTuple({});
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> ct, Type::FromInterpValue(v));
  EXPECT_EQ(ct->ToString(), "()");
  EXPECT_TRUE(ct->IsUnit());
  EXPECT_THAT(ct->GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(0)));
}

TEST(TypeTest, FromInterpValueTupleOfTwoNumbers) {
  auto v = InterpValue::MakeTuple({
      InterpValue::MakeUBits(2, 0b10),
      InterpValue::MakeSBits(3, -1),
  });
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> ct, Type::FromInterpValue(v));
  EXPECT_EQ(ct->ToString(), "(uN[2], sN[3])");
  EXPECT_FALSE(ct->IsUnit());
  EXPECT_THAT(ct->GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(5)));
}

TEST(TypeTest, StructTypeGetTotalBitCount) {
  Module module("test", /*fs_path=*/std::nullopt);

  std::vector<StructMember> ast_members;
  ast_members.emplace_back(StructMember{
      .name_span = kFakeSpan,
      .name = "x",
      .type = module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU8,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8))});
  ast_members.emplace_back(StructMember{
      .name_span = kFakeSpan,
      .name = "y",
      .type = module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU1,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU1))});

  auto* struct_def = module.Make<StructDef>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "S", nullptr),
      std::vector<ParametricBinding*>{}, ast_members, /*is_public=*/false);
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  members.push_back(BitsType::MakeU1());
  StructType s(std::move(members), *struct_def);
  EXPECT_THAT(s.GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(9)));
  EXPECT_THAT(s.GetAllDims(),
              ElementsAre(TypeDim::CreateU32(8), TypeDim::CreateU32(1)));
  EXPECT_FALSE(s.HasEnum());
}

TEST(TypeTest, StructTypeNominalTypeDims) {
  Module module("test", /*fs_path=*/std::nullopt);

  std::vector<StructMember> ast_members;
  ast_members.emplace_back(StructMember{
      .name_span = kFakeSpan,
      .name = "x",
      .type = module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU8,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8))});
  ast_members.emplace_back(StructMember{
      .name_span = kFakeSpan,
      .name = "y",
      .type = module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU1,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU1))});

  std::vector<ParametricBinding*> bindings;
  NameDef* type_m_name_def = module.Make<NameDef>(Span::Fake(), "M", nullptr);
  BuiltinType u32_type = BuiltinTypeFromString("u32").value();
  TypeAnnotation* u32_type_annot = module.Make<BuiltinTypeAnnotation>(
      Span::Fake(), u32_type, module.GetOrCreateBuiltinNameDef(u32_type));
  bindings.push_back(
      module.Make<ParametricBinding>(type_m_name_def, u32_type_annot, nullptr));
  NameDef* type_n_name_def = module.Make<NameDef>(Span::Fake(), "N", nullptr);
  bindings.push_back(
      module.Make<ParametricBinding>(type_n_name_def, u32_type_annot, nullptr));

  auto* struct_def = module.Make<StructDef>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "S", nullptr), bindings,
      ast_members, /*is_public=*/false);
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  members.push_back(BitsType::MakeU1());
  StructType s(std::move(members), *struct_def);
  EXPECT_THAT(s.GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(9)));
  EXPECT_THAT(s.GetAllDims(),
              ElementsAre(TypeDim::CreateU32(8), TypeDim::CreateU32(1)));
  EXPECT_FALSE(s.HasEnum());
  EXPECT_THAT(s.nominal_type_dims_by_identifier(), IsEmpty());

  absl::flat_hash_map<std::string, TypeDim> nominal_dims = {
      {"M", TypeDim(InterpValue::MakeU32(5))},
      {"N", TypeDim(InterpValue::MakeU32(6))}};
  std::unique_ptr<Type> type_with_nominal_dims =
      s.AddNominalTypeDims(nominal_dims);
  EXPECT_THAT(
      type_with_nominal_dims->AsStruct().nominal_type_dims_by_identifier(),
      UnorderedElementsAre(Pair("M", TypeDim(InterpValue::MakeU32(5))),
                           Pair("N", TypeDim(InterpValue::MakeU32(6)))));
}

TEST(TypeTest, StructTypeNominalTypeDimsWithExpr) {
  Module module("test", /*fs_path=*/std::nullopt);

  std::vector<StructMember> ast_members;
  ast_members.emplace_back(StructMember{
      .name_span = kFakeSpan,
      .name = "x",
      .type = module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU8,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8))});
  ast_members.emplace_back(StructMember{
      .name_span = kFakeSpan,
      .name = "y",
      .type = module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU1,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU1))});

  std::vector<ParametricBinding*> bindings;
  NameDef* type_m_name_def = module.Make<NameDef>(Span::Fake(), "M", nullptr);
  BuiltinType u32_type = BuiltinTypeFromString("u32").value();
  TypeAnnotation* u32_type_annot = module.Make<BuiltinTypeAnnotation>(
      Span::Fake(), u32_type, module.GetOrCreateBuiltinNameDef(u32_type));
  bindings.push_back(
      module.Make<ParametricBinding>(type_m_name_def, u32_type_annot, nullptr));
  NameRef* type_m_name_ref =
      module.Make<NameRef>(Span::Fake(), "M", type_m_name_def);
  // Type N = M + M.
  NameDef* type_n_name_def = module.Make<NameDef>(
      Span::Fake(), "N",
      module.Make<Binop>(Span::Fake(), BinopKind::kAdd, type_m_name_ref,
                         type_m_name_ref));
  bindings.push_back(
      module.Make<ParametricBinding>(type_n_name_def, u32_type_annot, nullptr));

  auto* struct_def = module.Make<StructDef>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "S", nullptr), bindings,
      ast_members, /*is_public=*/false);
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(BitsType::MakeU8());
  members.push_back(BitsType::MakeU1());
  StructType s(std::move(members), *struct_def);
  EXPECT_THAT(s.GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(9)));
  EXPECT_THAT(s.GetAllDims(),
              ElementsAre(TypeDim::CreateU32(8), TypeDim::CreateU32(1)));
  EXPECT_FALSE(s.HasEnum());
  EXPECT_THAT(s.nominal_type_dims_by_identifier(), IsEmpty());

  absl::flat_hash_map<std::string, TypeDim> nominal_dims = {
      {"M", TypeDim(InterpValue::MakeU32(5))}};
  std::unique_ptr<Type> type_with_nominal_dims =
      s.AddNominalTypeDims(nominal_dims);
  EXPECT_THAT(
      type_with_nominal_dims->AsStruct().nominal_type_dims_by_identifier(),
      UnorderedElementsAre(Pair("M", TypeDim(InterpValue::MakeU32(5)))));
}

TEST(TypeTest, EmptyStructTypeIsNotUnit) {
  std::filesystem::path fs_path = "relpath/to/test.x";
  Module module("test", fs_path);
  Span fake_span(Pos(fs_path, 0, 0), Pos(fs_path, 0, 0));
  auto* struct_def = module.Make<StructDef>(
      fake_span, module.Make<NameDef>(fake_span, "S", nullptr),
      std::vector<ParametricBinding*>{}, std::vector<StructMember>{},
      /*is_public=*/false);
  std::vector<std::unique_ptr<Type>> members;
  StructType s(std::move(members), *struct_def);
  EXPECT_THAT(s.GetTotalBitCount(), IsOkAndHolds(TypeDim::CreateU32(0)));
  EXPECT_TRUE(s.GetAllDims().empty());
  EXPECT_FALSE(s.HasEnum());
  EXPECT_FALSE(s.IsUnit());
  EXPECT_EQ(s.ToString(), "S {}");
  EXPECT_EQ(s.ToStringFullyQualified(), "relpath/to/test.x:S {}");
}

// -- TypeDimTest

TEST(TypeDimTest, TestArithmetic) {
  auto two = TypeDim::CreateU32(2);
  auto three = TypeDim::CreateU32(3);
  auto five = TypeDim::CreateU32(5);
  auto six = TypeDim::CreateU32(6);
  EXPECT_THAT(two.Add(three), IsOkAndHolds(five));
  EXPECT_THAT(two.Mul(three), IsOkAndHolds(six));

  const Pos fake_pos;
  const Span fake_span(fake_pos, fake_pos);
  TypeDim m(std::make_unique<ParametricSymbol>("M", fake_span));
  TypeDim n(std::make_unique<ParametricSymbol>("N", fake_span));

  // M+N
  TypeDim mpn(std::make_unique<ParametricAdd>(m.parametric().Clone(),
                                              n.parametric().Clone()));

  // M*N
  TypeDim mtn(std::make_unique<ParametricMul>(m.parametric().Clone(),
                                              n.parametric().Clone()));

  EXPECT_THAT(m.Add(n), IsOkAndHolds(mpn));
  EXPECT_THAT(m.Mul(n), IsOkAndHolds(mtn));
}

TEST(TypeDimTest, TestGetAs64BitsU64) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      InterpValue::MakeUBits(/*bit_count=*/64, static_cast<uint64_t>(-1));
  EXPECT_THAT(TypeDim::GetAs64Bits(variant), IsOkAndHolds(int64_t{-1}));
}

TEST(TypeDimTest, TestGetAs64BitsU128) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      InterpValue::MakeBits(/*is_signed=*/false, Bits::AllOnes(128));
  EXPECT_THAT(
      TypeDim::GetAs64Bits(variant),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be represented as an unsigned 64-bit value")));
}

TEST(TypeDimTest, TestGetAs64BitsS128) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      InterpValue::MakeBits(/*is_signed=*/true, Bits::AllOnes(128));
  EXPECT_THAT(TypeDim::GetAs64Bits(variant), IsOkAndHolds(-1));
}

TEST(TypeDimTest, TestGetAs64BitsParametricSymbol) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      std::make_unique<ParametricSymbol>("N", kFakeSpan);
  EXPECT_THAT(TypeDim::GetAs64Bits(variant),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can't evaluate a ParametricExpression")));
}

}  // namespace
}  // namespace xls::dslx

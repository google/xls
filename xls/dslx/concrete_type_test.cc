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

#include "xls/dslx/concrete_type.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;

const Pos kFakePos("<fake>", 0, 0);
const Span kFakeSpan(kFakePos, kFakePos);

TEST(ConcreteTypeTest, TestU32) {
  BitsType t(false, 32);
  EXPECT_EQ("uN[32]", t.ToString());
  EXPECT_EQ("ubits", t.GetDebugTypeName());
  EXPECT_EQ(false, t.is_signed());
  EXPECT_EQ(false, t.HasEnum());
  EXPECT_EQ(std::vector<ConcreteTypeDim>{ConcreteTypeDim::CreateU32(32)},
            t.GetAllDims());
  EXPECT_EQ(t, *t.ToUBits());
  EXPECT_TRUE(IsUBits(t));
}

TEST(ConcreteTypeTest, TestUnit) {
  TupleType t({});
  EXPECT_EQ("()", t.ToString());
  EXPECT_EQ("tuple", t.GetDebugTypeName());
  EXPECT_EQ(false, t.HasEnum());
  EXPECT_TRUE(t.GetAllDims().empty());
  EXPECT_FALSE(IsUBits(t));
}

TEST(ConcreteTypeTest, TestArrayOfU32) {
  ArrayType t(std::make_unique<BitsType>(false, 32),
              ConcreteTypeDim::CreateU32(1));
  EXPECT_EQ("uN[32][1]", t.ToString());
  EXPECT_EQ("array", t.GetDebugTypeName());
  EXPECT_EQ(false, t.HasEnum());
  std::vector<ConcreteTypeDim> want_dims = {ConcreteTypeDim::CreateU32(1),
                                            ConcreteTypeDim::CreateU32(32)};
  EXPECT_EQ(want_dims, t.GetAllDims());
  EXPECT_FALSE(IsUBits(t));
}

TEST(ConcreteTypeTest, TestEnum) {
  Module m("test");
  Pos fake_pos("fake.x", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  auto* my_enum = m.Make<NameDef>(fake_span, "MyEnum", nullptr);
  auto* e = m.Make<EnumDef>(fake_span, my_enum, /*type=*/nullptr,
                            /*values=*/std::vector<EnumMember>{},
                            /*is_public=*/false);
  my_enum->set_definer(e);
  EnumType t(*e, /*bit_count=*/ConcreteTypeDim::CreateU32(2),
             /*is_signed=*/false, {});
  EXPECT_TRUE(t.HasEnum());
  EXPECT_EQ(std::vector<ConcreteTypeDim>{ConcreteTypeDim::CreateU32(2)},
            t.GetAllDims());
  EXPECT_EQ("MyEnum", t.ToString());
}

TEST(ConcreteTypeTest, FunctionTypeU32ToS32) {
  std::vector<std::unique_ptr<ConcreteType>> params;
  params.push_back(std::make_unique<BitsType>(false, 32));
  FunctionType t(std::move(params), std::make_unique<BitsType>(true, 32));
  EXPECT_EQ(1, t.GetParams().size());
  EXPECT_EQ("uN[32]", t.GetParams()[0]->ToString());
  EXPECT_EQ("sN[32]", t.return_type().ToString());
}

TEST(ConcreteTypeTest, FromInterpValueSbits) {
  auto s8_m1 = InterpValue::MakeSBits(8, -1);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConcreteType> ct,
                           ConcreteType::FromInterpValue(s8_m1));
  EXPECT_EQ(ct->ToString(), "sN[8]");
}

TEST(ConcreteTypeTest, FromInterpValueArrayU2) {
  auto v = InterpValue::MakeArray({
                                      InterpValue::MakeUBits(2, 0b10),
                                      InterpValue::MakeUBits(2, 0b01),
                                      InterpValue::MakeUBits(2, 0b11),
                                  })
               .value();
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConcreteType> ct,
                           ConcreteType::FromInterpValue(v));
  EXPECT_EQ(ct->ToString(), "uN[2][3]");
  EXPECT_THAT(ct->GetTotalBitCount(),
              IsOkAndHolds(ConcreteTypeDim::CreateU32(6)));
}

TEST(ConcreteTypeTest, FromInterpValueTupleEmpty) {
  auto v = InterpValue::MakeTuple({});
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConcreteType> ct,
                           ConcreteType::FromInterpValue(v));
  EXPECT_EQ(ct->ToString(), "()");
  EXPECT_TRUE(ct->IsUnit());
  EXPECT_THAT(ct->GetTotalBitCount(),
              IsOkAndHolds(ConcreteTypeDim::CreateU32(0)));
}

TEST(ConcreteTypeTest, FromInterpValueTupleOfTwoNumbers) {
  auto v = InterpValue::MakeTuple({
      InterpValue::MakeUBits(2, 0b10),
      InterpValue::MakeSBits(3, -1),
  });
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConcreteType> ct,
                           ConcreteType::FromInterpValue(v));
  EXPECT_EQ(ct->ToString(), "(uN[2], sN[3])");
  EXPECT_FALSE(ct->IsUnit());
  EXPECT_THAT(ct->GetTotalBitCount(),
              IsOkAndHolds(ConcreteTypeDim::CreateU32(5)));
}

TEST(ConcreteTypeTest, StructTypeGetTotalBitCount) {
  Module module("test");
  std::vector<std::pair<NameDef*, TypeAnnotation*>> ast_members;
  ast_members.emplace_back(
      module.Make<NameDef>(kFakeSpan, "x", nullptr),
      module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU8, module.GetOrCreateBuiltinNameDef("u8")));
  ast_members.emplace_back(
      module.Make<NameDef>(kFakeSpan, "y", nullptr),
      module.Make<BuiltinTypeAnnotation>(
          kFakeSpan, BuiltinType::kU1, module.GetOrCreateBuiltinNameDef("u1")));
  auto* struct_def = module.Make<StructDef>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "S", nullptr),
      std::vector<ParametricBinding*>{}, ast_members, /*is_public=*/false);
  std::vector<std::unique_ptr<ConcreteType>> members;
  members.push_back(BitsType::MakeU8());
  members.push_back(BitsType::MakeU1());
  StructType s(std::move(members), *struct_def);
  EXPECT_THAT(s.GetTotalBitCount(),
              IsOkAndHolds(ConcreteTypeDim::CreateU32(9)));
  EXPECT_THAT(s.GetAllDims(), ElementsAre(ConcreteTypeDim::CreateU32(8),
                                          ConcreteTypeDim::CreateU32(1)));
  EXPECT_FALSE(s.HasEnum());
}

TEST(ConcreteTypeTest, EmptyStructTypeIsNotUnit) {
  Module module("test");
  std::vector<std::pair<NameDef*, TypeAnnotation*>> ast_members;
  auto* struct_def = module.Make<StructDef>(
      kFakeSpan, module.Make<NameDef>(kFakeSpan, "S", nullptr),
      std::vector<ParametricBinding*>{}, ast_members, /*is_public=*/false);
  std::vector<std::unique_ptr<ConcreteType>> members;
  StructType s(std::move(members), *struct_def);
  EXPECT_THAT(s.GetTotalBitCount(),
              IsOkAndHolds(ConcreteTypeDim::CreateU32(0)));
  EXPECT_TRUE(s.GetAllDims().empty());
  EXPECT_FALSE(s.HasEnum());
  EXPECT_FALSE(s.IsUnit());
}

// -- ConcreteTypeDimTest

TEST(ConcreteTypeDimTest, TestArithmetic) {
  auto two = ConcreteTypeDim::CreateU32(2);
  auto three = ConcreteTypeDim::CreateU32(3);
  auto five = ConcreteTypeDim::CreateU32(5);
  auto six = ConcreteTypeDim::CreateU32(6);
  EXPECT_THAT(two.Add(three), IsOkAndHolds(five));
  EXPECT_THAT(two.Mul(three), IsOkAndHolds(six));

  const Pos fake_pos;
  const Span fake_span(fake_pos, fake_pos);
  ConcreteTypeDim m(std::make_unique<ParametricSymbol>("M", fake_span));
  ConcreteTypeDim n(std::make_unique<ParametricSymbol>("N", fake_span));

  // M+N
  ConcreteTypeDim mpn(std::make_unique<ParametricAdd>(m.parametric().Clone(),
                                                      n.parametric().Clone()));

  // M*N
  ConcreteTypeDim mtn(std::make_unique<ParametricMul>(m.parametric().Clone(),
                                                      n.parametric().Clone()));

  EXPECT_THAT(m.Add(n), IsOkAndHolds(mpn));
  EXPECT_THAT(m.Mul(n), IsOkAndHolds(mtn));
}

TEST(ConcreteTypeDimTest, TestGetAs64BitsU64) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      InterpValue::MakeUBits(/*bit_count=*/64, static_cast<uint64_t>(-1));
  EXPECT_THAT(ConcreteTypeDim::GetAs64Bits(variant), IsOkAndHolds(int64_t{-1}));
}

TEST(ConcreteTypeDimTest, TestGetAs64BitsU128) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      InterpValue::MakeBits(/*is_signed=*/false, Bits::AllOnes(128));
  EXPECT_THAT(
      ConcreteTypeDim::GetAs64Bits(variant),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be represented as an unsigned 64-bit value")));
}

TEST(ConcreteTypeDimTest, TestGetAs64BitsS128) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      InterpValue::MakeBits(/*is_signed=*/true, Bits::AllOnes(128));
  EXPECT_THAT(ConcreteTypeDim::GetAs64Bits(variant), IsOkAndHolds(-1));
}

TEST(ConcreteTypeDimTest, TestGetAs64BitsParametricSymbol) {
  std::variant<InterpValue, std::unique_ptr<ParametricExpression>> variant =
      std::make_unique<ParametricSymbol>("N", kFakeSpan);
  EXPECT_THAT(ConcreteTypeDim::GetAs64Bits(variant),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can't evaluate a ParametricExpression")));
}

}  // namespace
}  // namespace xls::dslx

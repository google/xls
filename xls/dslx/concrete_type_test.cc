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
  ArrayType t(absl::make_unique<BitsType>(false, 32),
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
  EnumType t(e, /*bit_count=*/ConcreteTypeDim::CreateU32(2));
  EXPECT_TRUE(t.HasEnum());
  EXPECT_EQ(std::vector<ConcreteTypeDim>{ConcreteTypeDim::CreateU32(2)},
            t.GetAllDims());
  EXPECT_EQ("MyEnum", t.ToString());
}

TEST(ConcreteTypeTest, FunctionTypeU32ToS32) {
  std::vector<std::unique_ptr<ConcreteType>> params;
  params.push_back(absl::make_unique<BitsType>(false, 32));
  FunctionType t(std::move(params), absl::make_unique<BitsType>(true, 32));
  EXPECT_EQ(1, t.GetParams().size());
  EXPECT_EQ("uN[32]", t.GetParams()[0]->ToString());
  EXPECT_EQ("sN[32]", t.return_type().ToString());
}

TEST(ConcreteTypeDimTest, TestArithmetic) {
  auto two = ConcreteTypeDim::CreateU32(2);
  auto three = ConcreteTypeDim::CreateU32(3);
  auto five = ConcreteTypeDim::CreateU32(5);
  auto six = ConcreteTypeDim::CreateU32(6);
  EXPECT_THAT(two.Add(three), IsOkAndHolds(five));
  EXPECT_THAT(two.Mul(three), IsOkAndHolds(six));

  const Pos fake_pos;
  const Span fake_span(fake_pos, fake_pos);
  ConcreteTypeDim m(absl::make_unique<ParametricSymbol>("M", fake_span));
  ConcreteTypeDim n(absl::make_unique<ParametricSymbol>("N", fake_span));

  // M+N
  ConcreteTypeDim mpn(absl::make_unique<ParametricAdd>(m.parametric().Clone(),
                                                       n.parametric().Clone()));

  // M*N
  ConcreteTypeDim mtn(absl::make_unique<ParametricMul>(m.parametric().Clone(),
                                                       n.parametric().Clone()));

  EXPECT_THAT(m.Add(n), IsOkAndHolds(mpn));
  EXPECT_THAT(m.Mul(n), IsOkAndHolds(mtn));
}

}  // namespace
}  // namespace xls::dslx

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
#include "xls/dslx/ir_convert/ir_conversion_utils.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls::dslx {

TEST(IrConversionUtilsTest, TypeToIr) {
  constexpr int kArraySize = 7;

  Package package("The Package");

  std::vector<std::unique_ptr<Type>> elements;
  elements.push_back(BitsType::MakeU32());
  elements.push_back(std::make_unique<ArrayType>(
      BitsType::MakeU8(), TypeDim::CreateU32(kArraySize)));
  auto dslx_tuple_type = std::make_unique<TupleType>(std::move(elements));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Type * type, TypeToIr(&package, *dslx_tuple_type, ParametricEnv{}));

  ASSERT_TRUE(type->IsTuple());
  xls::TupleType* tuple_type = type->AsTupleOrDie();
  ASSERT_EQ(tuple_type->size(), 2);
  ASSERT_TRUE(tuple_type->element_type(0)->IsBits());
  xls::BitsType* bits_type = tuple_type->element_type(0)->AsBitsOrDie();
  EXPECT_EQ(bits_type->GetFlatBitCount(), 32);

  ASSERT_TRUE(tuple_type->element_type(1)->IsArray());
  xls::ArrayType* array_type = tuple_type->element_type(1)->AsArrayOrDie();
  EXPECT_EQ(array_type->size(), kArraySize);

  ASSERT_TRUE(array_type->element_type()->IsBits());
  bits_type = array_type->element_type()->AsBitsOrDie();
  EXPECT_EQ(bits_type->GetFlatBitCount(), 8);
}

TEST(IrConversionUtilsTest, BitsConstructorTypeToIr) {
  Package package("p");
  const ParametricEnv bindings;

  TypeDim is_signed = TypeDim::CreateBool(true);
  TypeDim size = TypeDim::CreateU32(4);
  auto element_type = std::make_unique<BitsConstructorType>(is_signed);
  auto s4 = std::make_unique<ArrayType>(std::move(element_type), size);

  XLS_ASSERT_OK_AND_ASSIGN(xls::Type * type, TypeToIr(&package, *s4, bindings));
  EXPECT_EQ(type->ToString(), "bits[4]");
}

class IrConversionUtilsSemanticSumTest : public ::testing::Test {
 protected:
  IrConversionUtilsSemanticSumTest()
      : module_("test", /*fs_path=*/std::nullopt, file_table_) {
    const Span span = Span::Fake();
    auto* sum_name = module_.Make<NameDef>(span, "Example", nullptr);
    auto* variant_name = module_.Make<NameDef>(span, "X", nullptr);
    auto* variant = module_.Make<SumVariant>(
        span, variant_name, SumVariant::PayloadShape::kUnit,
        std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
    auto* sum_def = module_.Make<SumDef>(
        span, sum_name, std::vector<ParametricBinding*>{},
        std::vector<SumVariant*>{variant}, /*is_public=*/false);
    sum_name->set_definer(sum_def);

    std::vector<SumTypeVariant> variants;
    variants.push_back(SumTypeVariant::MakeUnit(*variant));
    sum_type_ = std::make_unique<SumType>(*sum_def, std::move(variants));
  }

  FileTable file_table_;
  Module module_;
  Package package_{"semantic_sum"};
  std::unique_ptr<SumType> sum_type_;
};

TEST_F(IrConversionUtilsSemanticSumTest, SemanticSumLoweringIsRejected) {
  EXPECT_THAT(
      TypeToIr(&package_, *sum_type_, ParametricEnv{}),
      ::absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          ::testing::HasSubstr("Semantic sum type lowering is not supported")));
}

TEST_F(IrConversionUtilsSemanticSumTest, AggregateContainingSumIsRejected) {
  std::unique_ptr<TupleType> aggregate =
      TupleType::Create2(BitsType::MakeU8(), sum_type_->CloneToUnique());
  EXPECT_THAT(
      TypeToIr(&package_, *aggregate, ParametricEnv{}),
      ::absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          ::testing::HasSubstr("Semantic sum type lowering is not supported")));
}

}  // namespace xls::dslx

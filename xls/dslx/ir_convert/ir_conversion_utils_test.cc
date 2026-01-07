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
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
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

}  // namespace xls::dslx

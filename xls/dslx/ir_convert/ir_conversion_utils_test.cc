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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/type.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls::dslx {

// Smoke test of ResolveDim.
TEST(IrConversionUtilsTest, ResolveDimSmoke) {
  constexpr int64_t kDimValue = 64;
  TypeDim dim(InterpValue::MakeUBits(/*bit_count=*/64, kDimValue));
  absl::flat_hash_map<std::string, InterpValue> items;
  ParametricEnv bindings(items);
  XLS_ASSERT_OK_AND_ASSIGN(TypeDim resolved, ResolveDim(dim, bindings));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t resolved_value, resolved.GetAsInt64());
  EXPECT_EQ(resolved_value, kDimValue);
}

TEST(IrConversionUtilsTest, ResolveSymbolicDim) {
  constexpr int64_t kDimValue = 64;
  const std::string kSymbolName = "my_symbol";

  auto symbol = std::make_unique<ParametricSymbol>(kSymbolName, Span::Fake());
  TypeDim dim(std::move(symbol));
  absl::flat_hash_map<std::string, InterpValue> items;
  items.insert(
      {kSymbolName, InterpValue::MakeUBits(/*bit_count=*/64, kDimValue)});
  ParametricEnv bindings(items);
  XLS_ASSERT_OK_AND_ASSIGN(TypeDim resolved, ResolveDim(dim, bindings));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t resolved_value, resolved.GetAsInt64());
  EXPECT_EQ(resolved_value, kDimValue);
}

TEST(IrConversionUtilsTest, ResolveSymbolicDimToInt) {
  constexpr int64_t kDimValue = 64;
  const std::string kSymbolName = "my_symbol";

  auto symbol = std::make_unique<ParametricSymbol>(kSymbolName, Span::Fake());
  TypeDim dim(std::move(symbol));
  absl::flat_hash_map<std::string, InterpValue> items;
  items.insert(
      {kSymbolName, InterpValue::MakeUBits(/*bit_count=*/64, kDimValue)});
  ParametricEnv bindings(items);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t resolved_value,
                           ResolveDimToInt(dim, bindings));
  EXPECT_EQ(resolved_value, kDimValue);
}

TEST(IrConversionUtilsTest, TypeToIr) {
  constexpr int kArraySize = 7;
  const std::string kSymbolName = "my_symbol";

  Package package("The Package");

  absl::flat_hash_map<std::string, InterpValue> items;
  auto symbol = std::make_unique<ParametricSymbol>(kSymbolName, Span::Fake());
  items.insert(
      {kSymbolName, InterpValue::MakeUBits(/*bit_count=*/64, kArraySize)});
  ParametricEnv bindings(items);

  std::vector<std::unique_ptr<Type>> elements;
  elements.push_back(BitsType::MakeU32());
  elements.push_back(std::make_unique<ArrayType>(
      BitsType::MakeU8(), TypeDim::CreateU32(kArraySize)));
  auto dslx_tuple_type = std::make_unique<TupleType>(std::move(elements));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Type * type,
                           TypeToIr(&package, *dslx_tuple_type, bindings));

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

// Copyright 2022 The XLS Authors
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
#include "xls/fuzzer/value_generator.h"

#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

TEST(ValueGeneratorTest, GenerateEmptyValues) {
  std::mt19937_64 rng;
  std::vector<const dslx::Type*> param_type_ptrs;
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> values,
                           GenerateInterpValues(rng, param_type_ptrs));
  ASSERT_TRUE(values.empty());
}

TEST(ValueGeneratorTest, GenerateSingleBitsArgument) {
  std::mt19937_64 rng;
  std::vector<std::unique_ptr<dslx::Type>> param_types;
  param_types.push_back(std::make_unique<dslx::BitsType>(
      /*signed=*/false,
      /*size=*/dslx::TypeDim::CreateU32(42)));

  std::vector<const dslx::Type*> param_type_ptrs;
  param_type_ptrs.reserve(param_types.size());
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> values,
                           GenerateInterpValues(rng, param_type_ptrs));
  ASSERT_EQ(values.size(), 1);
  ASSERT_TRUE(values[0].IsUBits());
  EXPECT_THAT(values[0].GetBitCount(), IsOkAndHolds(42));
}

TEST(ValueGeneratorTest, GenerateMixedBitsArguments) {
  std::mt19937_64 rng;
  std::vector<std::unique_ptr<dslx::Type>> param_types;
  param_types.push_back(std::make_unique<dslx::BitsType>(
      /*signed=*/false,
      /*size=*/dslx::TypeDim::CreateU32(123)));
  param_types.push_back(std::make_unique<dslx::BitsType>(
      /*signed=*/true,
      /*size=*/dslx::TypeDim::CreateU32(22)));
  std::vector<const dslx::Type*> param_type_ptrs;
  param_type_ptrs.reserve(param_types.size());
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateInterpValues(rng, param_type_ptrs));
  ASSERT_EQ(arguments.size(), 2);
  ASSERT_TRUE(arguments[0].IsUBits());
  EXPECT_THAT(arguments[0].GetBitCount(), IsOkAndHolds(123));
  ASSERT_TRUE(arguments[1].IsSBits());
  EXPECT_THAT(arguments[1].GetBitCount(), IsOkAndHolds(22));
}

TEST(ValueGeneratorTest, GenerateTupleArgument) {
  std::mt19937_64 rng;
  std::vector<std::unique_ptr<dslx::Type>> param_types;
  std::vector<std::unique_ptr<dslx::Type>> tuple_members;
  tuple_members.push_back(
      std::make_unique<dslx::BitsType>(/*signed=*/false, /*size=*/123));
  tuple_members.push_back(
      std::make_unique<dslx::BitsType>(/*signed=*/true, /*size=*/22));
  param_types.push_back(
      std::make_unique<dslx::TupleType>(std::move(tuple_members)));

  std::vector<const dslx::Type*> param_type_ptrs;
  param_type_ptrs.reserve(param_types.size());
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateInterpValues(rng, param_type_ptrs));
  ASSERT_EQ(arguments.size(), 1);
  EXPECT_TRUE(arguments[0].IsTuple());
  EXPECT_THAT(arguments[0].GetValuesOrDie()[0].GetBitCount(),
              IsOkAndHolds(123));
  EXPECT_THAT(arguments[0].GetValuesOrDie()[1].GetBitCount(), IsOkAndHolds(22));
}

TEST(ValueGeneratorTest, GenerateArrayArgument) {
  std::mt19937_64 rng;
  std::vector<std::unique_ptr<dslx::Type>> param_types;
  param_types.push_back(std::make_unique<dslx::ArrayType>(
      std::make_unique<dslx::BitsType>(
          /*signed=*/true,
          /*size=*/dslx::TypeDim::CreateU32(4)),
      dslx::TypeDim::CreateU32(24)));

  std::vector<const dslx::Type*> param_type_ptrs;
  param_type_ptrs.reserve(param_types.size());
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateInterpValues(rng, param_type_ptrs));
  ASSERT_EQ(arguments.size(), 1);
  ASSERT_TRUE(arguments[0].IsArray());
  EXPECT_THAT(arguments[0].GetLength(), IsOkAndHolds(24));
  EXPECT_TRUE(arguments[0].GetValuesOrDie()[0].IsSBits());
  EXPECT_THAT(arguments[0].GetValuesOrDie()[0].GetBitCount(), IsOkAndHolds(4));
}

TEST(ValueGeneratorTest, GenerateDslxConstantBits) {
  dslx::FileTable file_table;
  dslx::Module module("test", /*fs_path=*/std::nullopt, file_table);
  std::mt19937_64 rng;
  dslx::BuiltinTypeAnnotation* type = module.Make<dslx::BuiltinTypeAnnotation>(
      dslx::FakeSpan(), dslx::BuiltinType::kU32,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           GenerateDslxConstant(rng, &module, type));
  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr->ToString(), HasSubstr("u32:"));
}

TEST(ValueGeneratorTest, GenerateDslxConstantArrayOfBuiltinLessThan64) {
  dslx::FileTable file_table;
  dslx::Module module("test", /*fs_path=*/std::nullopt, file_table);
  std::mt19937_64 rng;
  dslx::BuiltinTypeAnnotation* dim_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kU32,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  dslx::Number* dim = module.Make<dslx::Number>(
      dslx::FakeSpan(), "32", dslx::NumberKind::kOther, dim_type);
  dslx::BuiltinTypeAnnotation* element_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kSN,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kSN));
  dslx::ArrayTypeAnnotation* type = module.Make<dslx::ArrayTypeAnnotation>(
      dslx::FakeSpan(), element_type, dim);
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           GenerateDslxConstant(rng, &module, type));
  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr->ToString(), HasSubstr("sN[u32:32]:"));
}

TEST(ValueGeneratorTest, GenerateDslxConstantArrayOfBuiltinGreaterThan64) {
  dslx::FileTable file_table;
  dslx::Module module("test", /*fs_path=*/std::nullopt, file_table);
  std::mt19937_64 rng;
  dslx::BuiltinTypeAnnotation* dim_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kU32,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  dslx::Number* dim = module.Make<dslx::Number>(
      dslx::FakeSpan(), "65", dslx::NumberKind::kOther, dim_type);
  dslx::BuiltinTypeAnnotation* element_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kSN,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kSN));
  dslx::ArrayTypeAnnotation* type = module.Make<dslx::ArrayTypeAnnotation>(
      dslx::FakeSpan(), element_type, dim);
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           GenerateDslxConstant(rng, &module, type));
  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr->ToString(), HasSubstr("sN[u32:65]:"));
}

TEST(ValueGeneratorTest, GenerateDslxConstantTuple) {
  dslx::FileTable file_table;
  dslx::Module module("test", /*fs_path=*/std::nullopt, file_table);
  std::mt19937_64 rng;
  dslx::BuiltinTypeAnnotation* element0 =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kU32,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  dslx::BuiltinTypeAnnotation* element1 =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kS32,
          module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kS32));
  dslx::TupleTypeAnnotation* type = module.Make<dslx::TupleTypeAnnotation>(
      dslx::FakeSpan(), std::vector<dslx::TypeAnnotation*>{element0, element1});
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           GenerateDslxConstant(rng, &module, type));
  ASSERT_NE(expr, nullptr);
  constexpr const char* kWantPattern = R"(\(u32:[0-9]+, s32:[-0-9]+\))";
  EXPECT_THAT(expr->ToString(), MatchesRegex(kWantPattern));
}

}  // namespace
}  // namespace xls

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

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"

namespace xls {
namespace {

constexpr int kIterations = 1024;

using status_testing::IsOkAndHolds;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

TEST(AstGeneratorTest, RandomBool) {
  const int64_t kSampleCount = 10000;
  ValueGenerator value_gen(std::mt19937{42});
  std::vector<int64_t> bool_values(2);

  for (int64_t i = 0; i < kSampleCount; ++i) {
    bool_values[static_cast<uint64_t>(value_gen.RandomBool())]++;
  }
  // The buckets should not be skewed more the 40/60 with large probability.
  EXPECT_GE(bool_values[true], kSampleCount * 4 / 10);
  EXPECT_GE(bool_values[false], kSampleCount * 4 / 10);
}

TEST(AstGeneratorTest, RandomFloat) {
  const int64_t kSampleCount = 10000;
  ValueGenerator value_gen(std::mt19937{42});

  float min = 1.0f;
  float max = 0.0f;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    float f = value_gen.RandomFloat();
    EXPECT_LT(f, 1.0f);
    min = std::min(min, f);
    max = std::max(max, f);
  }
  EXPECT_LT(min, 0.01);
  EXPECT_GT(max, 0.99);
}

TEST(AstGeneratorTest, RandRangeStartAtZero) {
  const int64_t kSampleCount = 10000;
  ValueGenerator value_gen(std::mt19937{42});

  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = value_gen.RandRange(42);
    EXPECT_GE(x, 0);
    EXPECT_LT(x, 42);
    histogram[x]++;
  }

  // All numbers [0, 42) should have been generated.
  EXPECT_EQ(histogram.size(), 42);
}

TEST(AstGeneratorTest, RandRangeStartAtNonzero) {
  const int64_t kSampleCount = 10000;
  ValueGenerator value_gen(std::mt19937{42});

  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = value_gen.RandRange(10, 20);
    EXPECT_GE(x, 10);
    EXPECT_LT(x, 20);
    histogram[x]++;
  }

  // All numbers [10, 20) should have been generated.
  EXPECT_EQ(histogram.size(), 10);
}

TEST(AstGeneratorTest, RandomIntWithExpectedValue) {
  const int64_t kSampleCount = 10000;
  std::mt19937 rng(42);
  ValueGenerator value_gen(std::mt19937{42});

  int64_t sum = 0;
  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = value_gen.RandomIntWithExpectedValue(10);
    EXPECT_GE(x, 0);
    sum += x;
    histogram[x]++;
  }
  // The expected value rounded to the nearest integer should be 10 with large
  // probability given the large sample size.
  EXPECT_EQ(10, (sum + kSampleCount / 2) / kSampleCount);

  // We should generate at least numbers up to 15 with large probability.
  EXPECT_GT(histogram.size(), 15);
}

TEST(AstGeneratorTest, RandomIntWithExpectedValueWithLowerLimit) {
  const int64_t kSampleCount = 10000;
  ValueGenerator value_gen(std::mt19937{42});

  int64_t sum = 0;
  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = value_gen.RandomIntWithExpectedValue(42, /*lower_limit=*/10);
    EXPECT_GE(x, 10);
    sum += x;
    histogram[x]++;
  }
  // The expected value rounded to the nearest integer should be 42 with large
  // probability given the large sample size.
  EXPECT_EQ(42, (sum + kSampleCount / 2) / kSampleCount);

  // We should generate at least numbers from 10 to 50 with large probability.
  EXPECT_GT(histogram.size(), 40);
}

TEST(ValueGeneratorTest, RngRandRangeBiasedTowardsZero) {
  ValueGenerator value_gen(std::mt19937{});
  constexpr int64_t kLimit = 3;
  std::vector<int64_t> histo(kLimit, 0);
  for (int64_t i = 0; i < kIterations; ++i) {
    histo[value_gen.RandRangeBiasedTowardsZero(kLimit)]++;
  }

  for (int64_t i = 0; i < kLimit; ++i) {
    XLS_LOG(INFO) << i << ": " << histo[i];
    EXPECT_GT(histo[i], 0);
    EXPECT_LT(histo[i], kIterations);
  }

  EXPECT_LT(histo[2], histo[1]);
  EXPECT_LT(histo[1], histo[0]);
}

TEST(SampleGeneratorTest, RngRandRange) {
  ValueGenerator value_gen(std::mt19937{});
  constexpr int64_t kLimit = 3;
  std::vector<int64_t> histo(kLimit, 0);
  for (int64_t i = 0; i < kIterations; ++i) {
    histo[value_gen.RandRange(kLimit)]++;
  }

  for (int64_t i = 0; i < kLimit; ++i) {
    XLS_LOG(INFO) << i << ": " << histo[i];
    EXPECT_GT(histo[i], 0);
    EXPECT_LT(histo[i], kIterations);
  }
}

TEST(SampleGeneratorTest, RngRandomDouble) {
  ValueGenerator value_gen(std::mt19937{});
  for (int64_t i = 0; i < kIterations; ++i) {
    double d = value_gen.RandomDouble();
    EXPECT_GE(d, 0.0);
    EXPECT_LT(d, 1.0);
  }
}

TEST(SampleGeneratorTest, GenerateEmptyValues) {
  ValueGenerator value_gen(std::mt19937{});

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> values,
                           value_gen.GenerateInterpValues(param_type_ptrs));
  ASSERT_TRUE(values.empty());
}

TEST(SampleGeneratorTest, GenerateSingleBitsArgument) {
  ValueGenerator value_gen(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(std::make_unique<dslx::BitsType>(
      /*signed=*/false,
      /*size=*/dslx::ConcreteTypeDim::CreateU32(42)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> values,
                           value_gen.GenerateInterpValues(param_type_ptrs));
  ASSERT_EQ(values.size(), 1);
  ASSERT_TRUE(values[0].IsUBits());
  EXPECT_THAT(values[0].GetBitCount(), IsOkAndHolds(42));
}

TEST(SampleGeneratorTest, GenerateMixedBitsArguments) {
  ValueGenerator value_gen(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(std::make_unique<dslx::BitsType>(
      /*signed=*/false,
      /*size=*/dslx::ConcreteTypeDim::CreateU32(123)));
  param_types.push_back(std::make_unique<dslx::BitsType>(
      /*signed=*/true,
      /*size=*/dslx::ConcreteTypeDim::CreateU32(22)));
  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           value_gen.GenerateInterpValues(param_type_ptrs));
  ASSERT_EQ(arguments.size(), 2);
  ASSERT_TRUE(arguments[0].IsUBits());
  EXPECT_THAT(arguments[0].GetBitCount(), IsOkAndHolds(123));
  ASSERT_TRUE(arguments[1].IsSBits());
  EXPECT_THAT(arguments[1].GetBitCount(), IsOkAndHolds(22));
}

TEST(SampleGeneratorTest, GenerateTupleArgument) {
  ValueGenerator value_gen(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  std::vector<std::unique_ptr<dslx::ConcreteType>> tuple_members;
  tuple_members.push_back(
      std::make_unique<dslx::BitsType>(/*signed=*/false, /*size=*/123));
  tuple_members.push_back(
      std::make_unique<dslx::BitsType>(/*signed=*/true, /*size=*/22));
  param_types.push_back(
      std::make_unique<dslx::TupleType>(std::move(tuple_members)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           value_gen.GenerateInterpValues(param_type_ptrs));
  ASSERT_EQ(arguments.size(), 1);
  EXPECT_TRUE(arguments[0].IsTuple());
  EXPECT_THAT(arguments[0].GetValuesOrDie()[0].GetBitCount(),
              IsOkAndHolds(123));
  EXPECT_THAT(arguments[0].GetValuesOrDie()[1].GetBitCount(), IsOkAndHolds(22));
}

TEST(SampleGeneratorTest, GenerateArrayArgument) {
  ValueGenerator value_gen(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(std::make_unique<dslx::ArrayType>(
      std::make_unique<dslx::BitsType>(
          /*signed=*/true,
          /*size=*/dslx::ConcreteTypeDim::CreateU32(4)),
      dslx::ConcreteTypeDim::CreateU32(24)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           value_gen.GenerateInterpValues(param_type_ptrs));
  ASSERT_EQ(arguments.size(), 1);
  ASSERT_TRUE(arguments[0].IsArray());
  EXPECT_THAT(arguments[0].GetLength(), IsOkAndHolds(24));
  EXPECT_TRUE(arguments[0].GetValuesOrDie()[0].IsSBits());
  EXPECT_THAT(arguments[0].GetValuesOrDie()[0].GetBitCount(), IsOkAndHolds(4));
}

TEST(SampleGeneratorTest, GenerateDslxConstantBits) {
  dslx::Module module("test");
  ValueGenerator value_gen(std::mt19937{});
  dslx::BuiltinTypeAnnotation* type = module.Make<dslx::BuiltinTypeAnnotation>(
      dslx::FakeSpan(), dslx::BuiltinType::kU32,
      module.GetOrCreateBuiltinNameDef("u32"));
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           value_gen.GenerateDslxConstant(&module, type));
  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr->ToString(), HasSubstr("u32:"));
}

TEST(SampleGeneratorTest, GenerateDslxConstantArrayOfBuiltinLessThan64) {
  dslx::Module module("test");
  ValueGenerator value_gen(std::mt19937{});
  dslx::BuiltinTypeAnnotation* dim_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kU32,
          module.GetOrCreateBuiltinNameDef("u32"));
  dslx::Number* dim = module.Make<dslx::Number>(
      dslx::FakeSpan(), "32", dslx::NumberKind::kOther, dim_type);
  dslx::BuiltinTypeAnnotation* element_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kSN,
          module.GetOrCreateBuiltinNameDef("sN"));
  dslx::ArrayTypeAnnotation* type = module.Make<dslx::ArrayTypeAnnotation>(
      dslx::FakeSpan(), element_type, dim);
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           value_gen.GenerateDslxConstant(&module, type));
  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr->ToString(), HasSubstr("sN[u32:32]:"));
}

TEST(SampleGeneratorTest, GenerateDslxConstantArrayOfBuiltinGreaterThan64) {
  dslx::Module module("test");
  ValueGenerator value_gen(std::mt19937{});
  dslx::BuiltinTypeAnnotation* dim_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kU32,
          module.GetOrCreateBuiltinNameDef("u32"));
  dslx::Number* dim = module.Make<dslx::Number>(
      dslx::FakeSpan(), "65", dslx::NumberKind::kOther, dim_type);
  dslx::BuiltinTypeAnnotation* element_type =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kSN,
          module.GetOrCreateBuiltinNameDef("sN"));
  dslx::ArrayTypeAnnotation* type = module.Make<dslx::ArrayTypeAnnotation>(
      dslx::FakeSpan(), element_type, dim);
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           value_gen.GenerateDslxConstant(&module, type));
  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr->ToString(), HasSubstr("sN[u32:65]:"));
}

TEST(SampleGeneratorTest, GenerateDslxConstantTuple) {
  dslx::Module module("test");
  ValueGenerator value_gen(std::mt19937{});
  dslx::BuiltinTypeAnnotation* element0 =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kU32,
          module.GetOrCreateBuiltinNameDef("u32"));
  dslx::BuiltinTypeAnnotation* element1 =
      module.Make<dslx::BuiltinTypeAnnotation>(
          dslx::FakeSpan(), dslx::BuiltinType::kS32,
          module.GetOrCreateBuiltinNameDef("s32"));
  dslx::TupleTypeAnnotation* type = module.Make<dslx::TupleTypeAnnotation>(
      dslx::FakeSpan(), std::vector<dslx::TypeAnnotation*>{element0, element1});
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Expr * expr,
                           value_gen.GenerateDslxConstant(&module, type));
  ASSERT_NE(expr, nullptr);
  constexpr const char* kWantPattern = R"(\(u32:[0-9]+, s32:[-0-9]+\))";
  EXPECT_THAT(expr->ToString(), MatchesRegex(kWantPattern));
}

}  // namespace
}  // namespace xls

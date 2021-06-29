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

#include "xls/fuzzer/sample_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::xls::status_testing::IsOkAndHolds;

constexpr int64_t kIterations = 32 * 1024;

TEST(SampleGeneratorTest, RngRandRangeBiasedTowardsZero) {
  xls::RngState rng(std::mt19937{});
  constexpr int64_t kLimit = 3;
  std::vector<int64_t> histo(kLimit, 0);
  for (int64_t i = 0; i < kIterations; ++i) {
    histo[rng.RandRangeBiasedTowardsZero(kLimit)]++;
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
  xls::RngState rng(std::mt19937{});
  constexpr int64_t kLimit = 3;
  std::vector<int64_t> histo(kLimit, 0);
  for (int64_t i = 0; i < kIterations; ++i) {
    histo[rng.RandRange(kLimit)]++;
  }

  for (int64_t i = 0; i < kLimit; ++i) {
    XLS_LOG(INFO) << i << ": " << histo[i];
    EXPECT_GT(histo[i], 0);
    EXPECT_LT(histo[i], kIterations);
  }
}

TEST(SampleGeneratorTest, RngRandomDouble) {
  xls::RngState rng(std::mt19937{});
  for (int64_t i = 0; i < kIterations; ++i) {
    double d = rng.RandomDouble();
    EXPECT_GE(d, 0.0);
    EXPECT_LT(d, 1.0);
  }
}

TEST(SampleGeneratorTest, GenerateEmptyArguments) {
  xls::RngState rng(std::mt19937{});

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateArguments(param_type_ptrs, &rng));
  ASSERT_TRUE(arguments.empty());
}

TEST(SampleGeneratorTest, GenerateSingleBitsArgument) {
  xls::RngState rng(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(absl::make_unique<dslx::BitsType>(
      /*signed=*/false,
      /*size=*/dslx::ConcreteTypeDim::CreateU32(42)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateArguments(param_type_ptrs, &rng));
  ASSERT_EQ(arguments.size(), 1);
  ASSERT_TRUE(arguments[0].IsUBits());
  EXPECT_THAT(arguments[0].GetBitCount(), IsOkAndHolds(42));
}

TEST(SampleGeneratorTest, GenerateMixedBitsArguments) {
  xls::RngState rng(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(absl::make_unique<dslx::BitsType>(
      /*signed=*/false,
      /*size=*/dslx::ConcreteTypeDim::CreateU32(123)));
  param_types.push_back(absl::make_unique<dslx::BitsType>(
      /*signed=*/true,
      /*size=*/dslx::ConcreteTypeDim::CreateU32(22)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateArguments(param_type_ptrs, &rng));
  ASSERT_EQ(arguments.size(), 2);
  ASSERT_TRUE(arguments[0].IsUBits());
  EXPECT_THAT(arguments[0].GetBitCount(), IsOkAndHolds(123));
  ASSERT_TRUE(arguments[1].IsSBits());
  EXPECT_THAT(arguments[1].GetBitCount(), IsOkAndHolds(22));
}

TEST(SampleGeneratorTest, GenerateTupleArgument) {
  xls::RngState rng(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  std::vector<std::unique_ptr<dslx::ConcreteType>> tuple_members;
  tuple_members.push_back(
      absl::make_unique<dslx::BitsType>(/*signed=*/false, /*size=*/123));
  tuple_members.push_back(
      absl::make_unique<dslx::BitsType>(/*signed=*/true, /*size=*/22));
  param_types.push_back(
      absl::make_unique<dslx::TupleType>(std::move(tuple_members)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateArguments(param_type_ptrs, &rng));
  ASSERT_EQ(arguments.size(), 1);
  EXPECT_TRUE(arguments[0].IsTuple());
  EXPECT_THAT(arguments[0].GetValuesOrDie()[0].GetBitCount(),
              IsOkAndHolds(123));
  EXPECT_THAT(arguments[0].GetValuesOrDie()[1].GetBitCount(), IsOkAndHolds(22));
}

TEST(SampleGeneratorTest, GenerateArrayArgument) {
  xls::RngState rng(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(absl::make_unique<dslx::ArrayType>(
      absl::make_unique<dslx::BitsType>(
          /*signed=*/true,
          /*size=*/dslx::ConcreteTypeDim::CreateU32(4)),
      dslx::ConcreteTypeDim::CreateU32(24)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateArguments(param_type_ptrs, &rng));
  ASSERT_EQ(arguments.size(), 1);
  ASSERT_TRUE(arguments[0].IsArray());
  EXPECT_THAT(arguments[0].GetLength(), IsOkAndHolds(24));
  EXPECT_TRUE(arguments[0].GetValuesOrDie()[0].IsSBits());
  EXPECT_THAT(arguments[0].GetValuesOrDie()[0].GetBitCount(), IsOkAndHolds(4));
}

TEST(SampleGeneratorTest, GenerateBasicSample) {
  xls::RngState rng(std::mt19937{});
  SampleOptions sample_options;
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample,
      GenerateSample(dslx::AstGeneratorOptions{}, /*calls_per_sample=*/3,
                     sample_options, &rng));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_FALSE(sample.options().codegen());
  EXPECT_FALSE(sample.options().simulate());
  EXPECT_EQ(sample.args_batch().size(), 3);
  EXPECT_THAT(sample.input_text(), testing::HasSubstr("fn main"));
}

TEST(SampleGeneratorTest, GenerateCodegenSample) {
  xls::RngState rng(std::mt19937{});
  SampleOptions sample_options;
  sample_options.set_codegen(true);
  sample_options.set_simulate(true);
  constexpr int64_t kCallsPerSample = 0;
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample, GenerateSample(dslx::AstGeneratorOptions{},
                                    kCallsPerSample, sample_options, &rng));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_TRUE(sample.options().codegen());
  EXPECT_TRUE(sample.options().simulate());
  EXPECT_FALSE(sample.options().codegen_args()->empty());
  EXPECT_TRUE(sample.args_batch().empty());
}

}  // namespace
}  // namespace xls

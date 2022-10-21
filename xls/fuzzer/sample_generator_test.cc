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
#include "xls/dslx/concrete_type.h"
#include "xls/fuzzer/value_generator.h"

namespace xls {
namespace {

using ::xls::status_testing::IsOkAndHolds;

TEST(SampleGeneratorTest, GenerateBasicSample) {
  ValueGenerator value_gen(std::mt19937{});
  SampleOptions sample_options;
  sample_options.set_calls_per_sample(3);
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample,
      GenerateSample(dslx::AstGeneratorOptions{}, sample_options, &value_gen));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_FALSE(sample.options().codegen());
  EXPECT_FALSE(sample.options().simulate());
  EXPECT_EQ(sample.args_batch().size(), 3);
  EXPECT_THAT(sample.input_text(), testing::HasSubstr("fn main"));
}

TEST(SampleGeneratorTest, GenerateCodegenSample) {
  ValueGenerator value_gen(std::mt19937{});
  SampleOptions sample_options;
  sample_options.set_codegen(true);
  sample_options.set_simulate(true);
  constexpr int64_t kCallsPerSample = 0;
  sample_options.set_calls_per_sample(kCallsPerSample);
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample,
      GenerateSample(dslx::AstGeneratorOptions{}, sample_options, &value_gen));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_TRUE(sample.options().codegen());
  EXPECT_TRUE(sample.options().simulate());
  EXPECT_FALSE(sample.options().codegen_args()->empty());
  EXPECT_TRUE(sample.args_batch().empty());
}

TEST(SampleGeneratorTest, GenerateChannelArgument) {
  ValueGenerator value_gen(std::mt19937{});
  std::vector<std::unique_ptr<dslx::ConcreteType>> param_types;
  param_types.push_back(
      std::make_unique<dslx::ChannelType>(std::make_unique<dslx::BitsType>(
          /*signed=*/true,
          /*size=*/4)));

  std::vector<const dslx::ConcreteType*> param_type_ptrs;
  for (auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           value_gen.GenerateInterpValues(param_type_ptrs));
  ASSERT_EQ(arguments.size(), 1);
  dslx::InterpValue value = arguments[0];
  ASSERT_TRUE(value.IsSBits());
  EXPECT_THAT(value.GetBitCount(), IsOkAndHolds(4));
}

TEST(SampleGeneratorTest, GenerateBasicProcSample) {
  ValueGenerator value_gen(std::mt19937{});
  SampleOptions sample_options;
  constexpr int64_t kProcTicks = 3;
  sample_options.set_calls_per_sample(0);
  sample_options.set_proc_ticks(kProcTicks);
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample,
      GenerateSample(dslx::AstGeneratorOptions{.generate_proc = true},
                     sample_options, &value_gen));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_FALSE(sample.options().codegen());
  EXPECT_FALSE(sample.options().simulate());
  EXPECT_EQ(sample.args_batch().size(), kProcTicks);
  EXPECT_THAT(sample.input_text(), testing::HasSubstr("proc main"));
}

}  // namespace
}  // namespace xls

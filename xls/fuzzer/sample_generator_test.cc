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

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/value_generator.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;

TEST(SampleGeneratorTest, GenerateBasicFunctionSample) {
  dslx::FileTable file_table;
  std::mt19937_64 rng;
  SampleOptions sample_options;
  constexpr int kCallsPerSample = 3;
  sample_options.set_calls_per_sample(kCallsPerSample);
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample, GenerateSample(dslx::AstGeneratorOptions{}, sample_options,
                                    rng, file_table));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_FALSE(sample.options().codegen());
  EXPECT_FALSE(sample.options().simulate());

  std::vector<std::vector<dslx::InterpValue>> args_batch;
  XLS_EXPECT_OK(sample.GetArgsAndChannels(args_batch));
  EXPECT_EQ(args_batch.size(), kCallsPerSample);
  EXPECT_THAT(sample.input_text(), testing::HasSubstr("fn main"));
}

TEST(SampleGeneratorTest, GenerateCodegenSample) {
  dslx::FileTable file_table;
  std::mt19937_64 rng;
  SampleOptions sample_options;
  sample_options.set_codegen(true);
  sample_options.set_simulate(true);
  constexpr int64_t kCallsPerSample = 0;
  sample_options.set_calls_per_sample(kCallsPerSample);
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample, GenerateSample(dslx::AstGeneratorOptions{}, sample_options,
                                    rng, file_table));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_TRUE(sample.options().codegen());
  EXPECT_TRUE(sample.options().simulate());
  EXPECT_FALSE(sample.options().codegen_args().empty());
  std::vector<std::vector<dslx::InterpValue>> args_batch;
  XLS_EXPECT_OK(sample.GetArgsAndChannels(args_batch));
  EXPECT_EQ(args_batch.size(), kCallsPerSample);
}

TEST(SampleGeneratorTest, GenerateChannelArgument) {
  std::mt19937_64 rng;
  std::vector<std::unique_ptr<dslx::Type>> param_types;
  constexpr int64_t kBitCount = 4;
  param_types.push_back(
      std::make_unique<dslx::ChannelType>(std::make_unique<dslx::BitsType>(
                                              /*signed=*/true,
                                              /*size=*/kBitCount),
                                          dslx::ChannelDirection::kOut));

  std::vector<const dslx::Type*> param_type_ptrs;
  param_type_ptrs.reserve(param_types.size());
  for (const auto& t : param_types) {
    param_type_ptrs.push_back(t.get());
  }
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<dslx::InterpValue> arguments,
                           GenerateInterpValues(rng, param_type_ptrs));
  ASSERT_EQ(arguments.size(), 1);
  ASSERT_EQ(arguments.size(), param_types.size());
  const dslx::InterpValue& value = arguments[0];
  ASSERT_TRUE(value.IsSBits());
  EXPECT_THAT(value.GetBitCount(), IsOkAndHolds(kBitCount));
}

TEST(SampleGeneratorTest, GenerateBasicProcSample) {
  dslx::FileTable file_table;
  std::mt19937_64 rng;
  SampleOptions sample_options;
  constexpr int64_t kProcTicks = 3;
  sample_options.set_sample_type(fuzzer::SampleType::SAMPLE_TYPE_PROC);
  sample_options.set_calls_per_sample(0);
  sample_options.set_proc_ticks(kProcTicks);
  XLS_ASSERT_OK_AND_ASSIGN(
      Sample sample,
      GenerateSample(dslx::AstGeneratorOptions{.generate_proc = true},
                     sample_options, rng, file_table));
  EXPECT_TRUE(sample.options().input_is_dslx());
  EXPECT_TRUE(sample.options().convert_to_ir());
  EXPECT_TRUE(sample.options().optimize_ir());
  EXPECT_FALSE(sample.options().codegen());
  EXPECT_FALSE(sample.options().simulate());

  std::vector<std::vector<dslx::InterpValue>> args_batch;
  std::vector<std::string> ir_channel_names;
  XLS_EXPECT_OK(sample.GetArgsAndChannels(args_batch, &ir_channel_names));
  EXPECT_EQ(args_batch.size(), kProcTicks);

  EXPECT_THAT(sample.input_text(), HasSubstr("proc main"));
}

}  // namespace
}  // namespace xls

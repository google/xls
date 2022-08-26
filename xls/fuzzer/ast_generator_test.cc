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

#include "xls/fuzzer/ast_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using ::testing::MatchesRegex;

// Parses and typechecks the given text to ensure it's valid -- prints errors to
// the screen in a useful way for debugging if they fail parsing / typechecking.
template <typename ModuleMember>
absl::Status ParseAndTypecheck(absl::string_view text,
                               absl::string_view module_name) {
  XLS_LOG_LINES(INFO, text);

  std::string filename = absl::StrCat(module_name, ".x");

  auto get_file_contents =
      [&](absl::string_view path) -> absl::StatusOr<std::string> {
    XLS_CHECK_EQ(path, filename);
    return std::string(text);
  };

  auto import_data = CreateImportDataForTest();
  absl::StatusOr<TypecheckedModule> parsed_or = ParseAndTypecheck(
      text, /*path=*/filename, /*module_name=*/module_name, &import_data);
  TryPrintError(parsed_or.status(), get_file_contents);
  XLS_ASSIGN_OR_RETURN(TypecheckedModule parsed, parsed_or);
  XLS_RETURN_IF_ERROR(
      parsed.module->GetMemberOrError<ModuleMember>("main").status());
  return absl::OkStatus();
}

}  // namespace

// Simply tests that we generate a bunch of valid functions using seed 0 (that
// parse and typecheck).
TEST(AstGeneratorTest, GeneratesValidFunctions) {
  std::mt19937 rng(0);
  AstGeneratorOptions options;
  options.short_samples = true;
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, &rng);
    XLS_LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                             g.Generate("main", module_name));
    std::string text = module->ToString();
    // Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Function>(text, module_name));
  }
}

// Simply tests that we generate a bunch of valid procs using seed 0 (that
// parse and typecheck).
TEST(AstGeneratorTest, GeneratesValidProcs) {
  std::mt19937 rng(0);
  AstGeneratorOptions options;
  options.generate_proc = true;
  options.short_samples = true;
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, &rng);
    XLS_LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                             g.Generate("main", module_name));

    std::string text = module->ToString();
    //  Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Proc>(text, module_name));
  }
}

TEST(AstGeneratorTest, GeneratesParametricBindings) {
  std::mt19937 rng(0);
  AstGenerator g(AstGeneratorOptions(), &rng);
  g.module_ = std::make_unique<Module>("my_mod");
  std::vector<ParametricBinding*> pbs = g.GenerateParametricBindings(2);
  EXPECT_EQ(pbs.size(), 2);
  // TODO(https://github.com/google/googletest/issues/3084): 2021-08-12
  // googletest cannot currently seem to use \d in regexp patterns, which is
  // quite surprising.
  constexpr const char* kWantPattern =
      R"(x[0-9]+: u[0-9]+ = u[0-9]+:0x[0-9a-f_]+)";
  EXPECT_THAT(pbs[0]->ToString(), MatchesRegex(kWantPattern));
  EXPECT_THAT(pbs[1]->ToString(), MatchesRegex(kWantPattern));
}

// Helper function that is used in a TEST_P so we can shard the work.
static void TestRepeatable(int64_t seed) {
  AstGeneratorOptions options;
  options.short_samples = true;
  // Capture first output at a given seed for comparison.
  std::optional<std::string> first;
  // Try 32 generations at a given seed.
  for (int64_t i = 0; i < 32; ++i) {
    std::mt19937 rng(seed);
    AstGenerator g(options, &rng);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                             g.Generate("main", "test"));
    std::string text = module->ToString();
    if (first.has_value()) {
      ASSERT_EQ(text, *first) << "sample " << i << " seed " << seed;
    } else {
      first = text;
      // Parse and typecheck for good measure.
      XLS_ASSERT_OK(ParseAndTypecheck<Function>(text, "test"));
    }
  }
}

class AstGeneratorRepeatableTest : public testing::TestWithParam<int64_t> {};

TEST_P(AstGeneratorRepeatableTest, GenerationRepeatableAtSeed) {
  TestRepeatable(/*seed=*/GetParam());
}

TEST(AstGeneratorTest, RandomBool) {
  const int64_t kSampleCount = 10000;
  std::mt19937 rng(42);
  AstGenerator g(AstGeneratorOptions(), &rng);

  std::vector<int64_t> bool_values(2);
  for (int64_t i = 0; i < kSampleCount; ++i) {
    bool_values[g.RandomBool()]++;
  }
  // The buckets should not be skewed more the 40/60 with large probability.
  EXPECT_GE(bool_values[true], kSampleCount * 4 / 10);
  EXPECT_GE(bool_values[false], kSampleCount * 4 / 10);
}

TEST(AstGeneratorTest, RandomFloat) {
  const int64_t kSampleCount = 10000;
  std::mt19937 rng(42);
  AstGenerator g(AstGeneratorOptions(), &rng);

  float min = 1.0f;
  float max = 0.0f;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    float f = g.RandomFloat();
    EXPECT_GE(f, 0.0f);
    EXPECT_LT(f, 1.0f);
    min = std::min(min, f);
    max = std::max(max, f);
  }
  EXPECT_LT(min, 0.01);
  EXPECT_GT(max, 0.99);
}

TEST(AstGeneratorTest, RandRangeStartAtZero) {
  const int64_t kSampleCount = 10000;
  std::mt19937 rng(42);
  AstGenerator g(AstGeneratorOptions(), &rng);

  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = g.RandRange(42);
    EXPECT_GE(x, 0);
    EXPECT_LT(x, 42);
    histogram[x]++;
  }

  // All numbers [0, 42) should have been generated.
  EXPECT_EQ(histogram.size(), 42);
}

TEST(AstGeneratorTest, RandRangeStartAtNonzero) {
  const int64_t kSampleCount = 10000;
  std::mt19937 rng(42);
  AstGenerator g(AstGeneratorOptions(), &rng);

  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = g.RandRange(10, 20);
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
  AstGenerator g(AstGeneratorOptions(), &rng);

  int64_t sum = 0;
  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = g.RandomIntWithExpectedValue(10);
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
  std::mt19937 rng(42);
  AstGenerator g(AstGeneratorOptions(), &rng);

  int64_t sum = 0;
  absl::flat_hash_map<int64_t, int64_t> histogram;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    int64_t x = g.RandomIntWithExpectedValue(42, /*lower_limit=*/10);
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

INSTANTIATE_TEST_SUITE_P(AstGeneratorRepeatableTestInstance,
                         AstGeneratorRepeatableTest,
                         testing::Range(int64_t{0}, int64_t{1024}));

}  // namespace xls::dslx

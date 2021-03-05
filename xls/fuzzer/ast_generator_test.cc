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
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

// Parses and typechecks the given text to ensure it's valid -- prints errors to
// the screen in a useful way for debugging if they fail parsing / typechecking.
absl::Status ParseAndTypecheck(absl::string_view text,
                               absl::string_view module_name) {
  XLS_LOG_LINES(INFO, text);

  std::string filename = absl::StrCat(module_name, ".x");

  auto get_file_contents =
      [&](absl::string_view path) -> absl::StatusOr<std::string> {
    XLS_CHECK_EQ(path, filename);
    return std::string(text);
  };

  ImportData import_data;
  absl::StatusOr<TypecheckedModule> parsed_or = ParseAndTypecheck(
      text, /*path=*/filename, /*module_name=*/module_name, &import_data,
      /*additional_search_paths=*/{});
  TryPrintError(parsed_or.status(), get_file_contents);
  XLS_ASSIGN_OR_RETURN(TypecheckedModule parsed, parsed_or);
  XLS_RETURN_IF_ERROR(parsed.module->GetFunctionOrError("main").status());
  return absl::OkStatus();
}

// Simply tests that we generate a bunch of valid functions using seed 0 (that
// parse and typecheck).
TEST(AstGeneratorTest, GeneratesValidFunctions) {
  std::mt19937 rng(0);
  AstGeneratorOptions options;
  options.short_samples = true;
  for (int64 i = 0; i < 32; ++i) {
    AstGenerator g(options, &rng);
    XLS_LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(auto generated,
                             g.GenerateFunctionInModule("main", module_name));
    std::string text = generated.second->ToString();
    // Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck(text, module_name));
  }
}

// Helper function that is used in a TEST_P so we can shard the work.
static void TestRepeatable(int64 seed) {
  AstGeneratorOptions options;
  options.short_samples = true;
  // Capture first output at a given seed for comparison.
  absl::optional<std::string> first;
  // Try 32 generations at a given seed.
  for (int64 i = 0; i < 32; ++i) {
    std::mt19937 rng(seed);
    AstGenerator g(options, &rng);
    XLS_ASSERT_OK_AND_ASSIGN(auto generated,
                             g.GenerateFunctionInModule("main", "test"));
    std::string text = generated.second->ToString();
    if (first.has_value()) {
      ASSERT_EQ(text, *first) << "sample " << i << " seed " << seed;
    } else {
      first = text;
      // Parse and typecheck for good measure.
      XLS_ASSERT_OK(ParseAndTypecheck(text, "test"));
    }
  }
}

class AstGeneratorRepeatableTest : public testing::TestWithParam<int64> {};

TEST_P(AstGeneratorRepeatableTest, GenerationRepeatableAtSeed) {
  TestRepeatable(/*seed=*/GetParam());
}

INSTANTIATE_TEST_SUITE_P(AstGeneratorRepeatableTestInstance,
                         AstGeneratorRepeatableTest,
                         testing::Range(int64{0}, int64{1024}));

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);
  return RUN_ALL_TESTS();
}

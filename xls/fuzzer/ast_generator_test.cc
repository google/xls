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
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using ::testing::ContainsRegex;
using ::testing::MatchesRegex;

// Parses and typechecks the given text to ensure it's valid -- prints errors to
// the screen in a useful way for debugging if they fail parsing / typechecking.
template <typename ModuleMember>
absl::Status ParseAndTypecheck(std::string_view text,
                               std::string_view module_name) {
  XLS_LOG_LINES(INFO, text);

  std::string filename = absl::StrCat(module_name, ".x");

  auto get_file_contents =
      [&](std::string_view path) -> absl::StatusOr<std::string> {
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

TEST(AstGeneratorTest, BitsTypeGetMetadata) {
  AstGeneratorOptions options;
  ValueGenerator value_gen(std::mt19937{0});
  AstGenerator g(options, &value_gen);
  g.module_ = std::make_unique<Module>("test_module");

  TypeAnnotation* u7 = g.MakeTypeAnnotation(false, 7);
  XLS_LOG(INFO) << "u7: " << u7->ToString();
  XLS_ASSERT_OK_AND_ASSIGN(int64_t bit_count, g.BitsTypeGetBitCount(u7));
  XLS_ASSERT_OK_AND_ASSIGN(bool is_signed, g.BitsTypeIsSigned(u7));
  EXPECT_EQ(bit_count, 7);
  EXPECT_EQ(is_signed, false);

  TypeAnnotation* s129 = g.MakeTypeAnnotation(true, 129);
  XLS_LOG(INFO) << "s129: " << s129->ToString();
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, g.BitsTypeGetBitCount(s129));
  XLS_ASSERT_OK_AND_ASSIGN(is_signed, g.BitsTypeIsSigned(s129));
  EXPECT_EQ(bit_count, 129);
  EXPECT_EQ(is_signed, true);
}

TEST(AstGeneratorTest, GeneratesParametricBindings) {
  ValueGenerator value_gen(std::mt19937{0});
  AstGenerator g(AstGeneratorOptions(), &value_gen);
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

namespace {
// Simply tests that we generate a bunch of valid functions using seed 0 (that
// parse and typecheck).
TEST(AstGeneratorTest, GeneratesValidFunctions) {
  ValueGenerator value_gen(std::mt19937{0});
  AstGeneratorOptions options;
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, &value_gen);
    XLS_LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                             g.Generate("main", module_name));
    std::string text = module->ToString();
    // Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Function>(text, module_name));
  }
}

// Simply tests that we generate a bunch of valid procs with an empty state type
// using seed 0 (that parse and typecheck).
TEST(AstGeneratorTest, GeneratesValidProcsWithEmptyState) {
  ValueGenerator value_gen(std::mt19937{0});
  AstGeneratorOptions options;
  options.generate_proc = true;
  options.emit_stateless_proc = true;
  // Regex matcher for the next function signature.
  constexpr const char* kWantPattern =
      R"(next\(x[0-9]+: token, x[0-9]+: \(\)\))";
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, &value_gen);
    XLS_LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                             g.Generate("main", module_name));

    std::string text = module->ToString();
    //  Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Proc>(text, module_name));
    EXPECT_THAT(text, ContainsRegex(kWantPattern));
  }
}

// Simply tests that we generate a bunch of valid procs with a random state type
// using seed 0 (that parse and typecheck).
TEST(AstGeneratorTest, GeneratesValidProcsWithRandomState) {
  ValueGenerator value_gen(std::mt19937{0});
  AstGeneratorOptions options;
  options.generate_proc = true;
  // Regex matcher for the next function signature.
  // Although [[:word:]] encapsulates [0-9a-zA-Z_], which would simplify the
  // following regex statement. The following regex statement is more readable.
  constexpr const char* kWantPattern =
      R"(next\(x[0-9]+: token, x[0-9]+: []0-9a-zA-Z_, ()[]+\))";
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, &value_gen);
    XLS_LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                             g.Generate("main", module_name));

    std::string text = module->ToString();
    //  Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Proc>(text, module_name));
    EXPECT_THAT(text, ContainsRegex(kWantPattern));
  }
}

// Helper function that is used in a TEST_P so we can shard the work.
static void TestRepeatable(uint64_t seed) {
  AstGeneratorOptions options;
  // Capture first output at a given seed for comparison.
  std::optional<std::string> first;
  // Try 32 generations at a given seed.
  for (int64_t i = 0; i < 32; ++i) {
    ValueGenerator value_gen(std::mt19937{seed});
    AstGenerator g(options, &value_gen);
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

class AstGeneratorRepeatableTest : public testing::TestWithParam<uint64_t> {};

TEST_P(AstGeneratorRepeatableTest, GenerationRepeatableAtSeed) {
  TestRepeatable(/*seed=*/GetParam());
}

INSTANTIATE_TEST_SUITE_P(AstGeneratorRepeatableTestInstance,
                         AstGeneratorRepeatableTest,
                         testing::Range(uint64_t{0}, uint64_t{1024}));

}  // namespace

}  // namespace xls::dslx

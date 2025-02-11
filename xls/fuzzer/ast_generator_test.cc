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

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

using ::testing::ContainsRegex;

// Parses and typechecks the given text to ensure it's valid -- prints errors to
// the screen in a useful way for debugging if they fail parsing / typechecking.
template <typename ModuleMember>
absl::Status ParseAndTypecheck(std::string_view text,
                               std::string_view module_name) {
  XLS_LOG_LINES(INFO, text);

  std::string filename = absl::StrCat(module_name, ".x");

  auto import_data = CreateImportDataForTest();
  absl::StatusOr<TypecheckedModule> parsed = ParseAndTypecheck(
      text, /*path=*/filename, /*module_name=*/module_name, &import_data);

  UniformContentFilesystem vfs(text, /*expect_path=*/filename);
  TryPrintError(parsed.status(), import_data.file_table(), vfs);
  XLS_RETURN_IF_ERROR(
      parsed->module->GetMemberOrError<ModuleMember>("main").status());
  return absl::OkStatus();
}

}  // namespace

class AstGeneratorTest : public ::testing::Test {
 protected:
  FileTable file_table_;
  AstGeneratorOptions options_;
  std::mt19937_64 rng_{0};
  AstGenerator g_{options_, rng_, file_table_};
};

TEST_F(AstGeneratorTest, BitsTypeGetMetadata) {
  g_.module_ = std::make_unique<Module>("test_module", /*fs_path=*/std::nullopt,
                                        file_table_);

  TypeAnnotation* u7 = g_.MakeTypeAnnotation(/*is_signed=*/false, 7,
                                             /*use_xn=*/false);
  LOG(INFO) << "u7: " << u7->ToString();
  XLS_ASSERT_OK_AND_ASSIGN(int64_t bit_count, g_.BitsTypeGetBitCount(u7));
  XLS_ASSERT_OK_AND_ASSIGN(bool is_signed, g_.BitsTypeIsSigned(u7));
  EXPECT_EQ(bit_count, 7);
  EXPECT_EQ(is_signed, false);

  TypeAnnotation* s129 = g_.MakeTypeAnnotation(/*is_signed=*/true, 129,
                                               /*use_xn=*/false);
  LOG(INFO) << "s129: " << s129->ToString();
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, g_.BitsTypeGetBitCount(s129));
  XLS_ASSERT_OK_AND_ASSIGN(is_signed, g_.BitsTypeIsSigned(s129));
  EXPECT_EQ(bit_count, 129);
  EXPECT_EQ(is_signed, true);

  TypeAnnotation* xn_true_128 =
      g_.MakeTypeAnnotation(/*is_signed=*/true, 128, /*use_xn=*/true);
  EXPECT_EQ(xn_true_128->ToString(), "xN[bool:0x1][128]");
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, g_.BitsTypeGetBitCount(xn_true_128));
  EXPECT_EQ(bit_count, 128);
  XLS_ASSERT_OK_AND_ASSIGN(is_signed, g_.BitsTypeIsSigned(xn_true_128));
  EXPECT_EQ(is_signed, true);
}

TEST_F(AstGeneratorTest, GeneratesParametricBindings) {
  g_.module_ =
      std::make_unique<Module>("my_mod", /*fs_path=*/std::nullopt, file_table_);
  std::vector<ParametricBinding*> pbs = g_.GenerateParametricBindings(4);
  EXPECT_EQ(pbs.size(), 4);
  // Note that the fact we get an unsigned binding is probabilistic, so we
  // generate four examples to try to find one that matches our unsigned
  // pattern.
  //
  // TODO(https://github.com/google/googletest/issues/3084): 2021-08-12
  // googletest cannot currently seem to use \d in regexp patterns, which is
  // quite surprising.
  constexpr const char* kWantPattern =
      R"(x[0-9]+: u[0-9]+ = \{u[0-9]+:0[xb][0-9a-f_]+\})";
  bool found_match = false;
  for (ParametricBinding* pb : pbs) {
    if (RE2::FullMatch(pb->ToString(), kWantPattern)) {
      found_match = true;
      break;
    }
  }
  EXPECT_TRUE(found_match);
}

namespace {
// Simply tests that we generate a bunch of valid functions using seed 0 (that
// parse and typecheck).
TEST(AstGeneratorMultiTest, GeneratesValidFunctions) {
  FileTable file_table;
  std::mt19937_64 rng{0};
  AstGeneratorOptions options;
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, rng, file_table);
    LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(AnnotatedModule module,
                             g.Generate("main", module_name));
    std::string text = module.module->ToString();
    // Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Function>(text, module_name));
  }
}

// Simply tests that we generate a bunch of valid procs with an empty state type
// using seed 0 (that parse and typecheck).
TEST(AstGeneratorMultiTest, GeneratesValidProcsWithEmptyState) {
  FileTable file_table;
  std::mt19937_64 rng{0};
  AstGeneratorOptions options;
  options.generate_proc = true;
  options.emit_stateless_proc = true;
  // Regex matcher for the next function signature.
  constexpr const char* kWantPattern = R"(next\(x[0-9]+: \(\)\))";
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, rng, file_table);
    LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(AnnotatedModule module,
                             g.Generate("main", module_name));

    std::string text = module.module->ToString();
    //  Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Proc>(text, module_name)) << text;
    EXPECT_THAT(text, ContainsRegex(kWantPattern));
  }
}

// Simply tests that we generate a bunch of valid procs with a random state type
// using seed 0 (that parse and typecheck).
TEST(AstGeneratorMultiTest, GeneratesValidProcsWithRandomState) {
  FileTable file_table;
  std::mt19937_64 rng{0};
  AstGeneratorOptions options;
  options.generate_proc = true;
  // Regex matcher for the next function signature.
  // Although [[:word:]] encapsulates [0-9a-zA-Z_], which would simplify the
  // following regex statement. The following regex statement is more readable.
  constexpr const char* kWantPattern =
      R"(next\(x[0-9]+: []0-9a-zA-Z_, ()[]+\))";
  for (int64_t i = 0; i < 32; ++i) {
    AstGenerator g(options, rng, file_table);
    LOG(INFO) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(AnnotatedModule module,
                             g.Generate("main", module_name));

    std::string text = module.module->ToString();
    //  Parses/typechecks as well, which is primarily what we're testing here.
    XLS_ASSERT_OK(ParseAndTypecheck<Proc>(text, module_name));
    EXPECT_THAT(text, ContainsRegex(kWantPattern));
  }
}

// Helper function that is used in a TEST_P so we can shard the work.
static void TestRepeatable(uint64_t seed) {
  FileTable file_table;
  AstGeneratorOptions options;
  // Capture first output at a given seed for comparison.
  std::optional<std::string> first;
  // Try 32 generations at a given seed.
  for (int64_t i = 0; i < 32; ++i) {
    std::mt19937_64 rng{seed};
    AstGenerator g(options, rng, file_table);
    XLS_ASSERT_OK_AND_ASSIGN(AnnotatedModule module,
                             g.Generate("main", "test"));
    std::string text = module.module->ToString();
    if (first.has_value()) {
      ASSERT_EQ(text, *first) << "sample " << i << " seed " << seed;
    } else {
      first = text;
      // Parse and typecheck for good measure.
      XLS_ASSERT_OK(ParseAndTypecheck<Function>(text, "test"));
    }
  }
}

TEST(AstGeneratorMultiTest, GeneratesZeroWidthValues) {
  FileTable file_table;
  std::mt19937_64 rng{0};
  AstGeneratorOptions options;
  options.emit_zero_width_bits_types = true;
  bool saw_zero_width = false;
  // Every couple samples seems to produce a zero-width value somewhere, but set
  // to a high number to catch invalid handling of zero-width values in the
  // generator.
  constexpr int64_t kNumSamples = 5000;
  for (int64_t i = 0; i < kNumSamples; ++i) {
    AstGenerator g(options, rng, file_table);
    VLOG(1) << "Generating sample: " << i;
    std::string module_name = absl::StrFormat("sample_%d", i);
    XLS_ASSERT_OK_AND_ASSIGN(AnnotatedModule module,
                             g.Generate("main", module_name));
    std::string text = module.module->ToString();
    if (absl::StrContains(text, "uN[0]") || absl::StrContains(text, "sN[0]")) {
      VLOG(1) << absl::StrFormat("Saw zero-width type after %d samples", i);
      saw_zero_width = true;
    }
  }
  EXPECT_TRUE(saw_zero_width) << absl::StrFormat(
      "Generated %d samples and did not see a zero-width type", kNumSamples);
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

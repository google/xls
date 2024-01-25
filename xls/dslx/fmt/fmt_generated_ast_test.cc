// Copyright 2024 The XLS Authors
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

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/strings/str_replace.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/fuzzer/ast_generator.h"

namespace xls::dslx {
namespace {

// Autofmt output should be the same as input after whitespace is eliminated
// excepting that we may introduce/remove commas.
std::string RemoveWhitespaceAndCommas(std::string_view input) {
  return absl::StrReplaceAll(input, {
                                        {"\n", ""},
                                        {" ", ""},
                                        {",", ""},
                                    });
}

class FmtGeneratedAstTest : public testing::TestWithParam<int> {};

TEST_P(FmtGeneratedAstTest, RunWithSeed) {
  AstGeneratorOptions options;

  std::mt19937_64 rng(GetParam());

  // TODO(https://github.com/google/xls/issues/1275): When this is increased we
  // observe some failures that seem to be related to very large aggregates such
  // as arrays.
  for (size_t i = 0; i < size_t{1} * 1024; ++i) {
    static const std::string kModuleName = "test";
    AstGenerator gen(options, rng);
    XLS_ASSERT_OK_AND_ASSIGN(
        AnnotatedModule am,
        gen.Generate(/*top_entity_name=*/"main", /*module_name=*/kModuleName));

    std::string stringified = am.module->ToString();

    // We re-parse the module so we can get proper positional annotations -- all
    // positions are trivial in the generated AST structure.
    Scanner scanner("test.x", stringified);
    Parser parser(kModuleName, &scanner);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> parsed,
                             parser.ParseModule());

    Comments comments;
    std::string autoformatted = AutoFmt(*parsed, comments);

    std::string want = RemoveWhitespaceAndCommas(stringified);
    std::string got = RemoveWhitespaceAndCommas(autoformatted);
    EXPECT_EQ(want, got);
    if (want != got) {
      XLS_LOG(ERROR) << "= seed " << GetParam() << " sample " << i << ":";
      XLS_LOG_LINES(ERROR, stringified);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(FmtGeneratedAstTestInstance, FmtGeneratedAstTest,
                         testing::Range(0, 50));

}  // namespace
}  // namespace xls::dslx

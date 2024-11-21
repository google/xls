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
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/random/random.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/fuzzer/ast_generator.h"

namespace xls::dslx {
namespace {

class FmtGeneratedAstTest : public testing::TestWithParam<int> {};

TEST_P(FmtGeneratedAstTest, RunShard) {
  AstGeneratorOptions options;

  absl::BitGen bitgen;
  FileTable file_table;

  for (size_t i = 0; i < 256; ++i) {
    static const std::string kModuleName = "test";
    AstGenerator gen(options, bitgen, file_table);
    XLS_ASSERT_OK_AND_ASSIGN(
        AnnotatedModule am,
        gen.Generate(/*top_entity_name=*/"main", /*module_name=*/kModuleName));

    std::string stringified = am.module->ToString();

    UniformContentFilesystem vfs(stringified);

    // We re-parse the module so we can get proper positional annotations -- all
    // positions are trivial in the generated AST structure.
    Scanner scanner(file_table, Fileno(0), stringified);
    Parser parser(kModuleName, &scanner);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> parsed,
                             parser.ParseModule());

    Comments comments;
    XLS_ASSERT_OK_AND_ASSIGN(std::string autoformatted,
                             AutoFmt(vfs, *parsed, comments));

    // Note that the AST generator currently does not generate any constructs
    // that the "opportunistic postcondition" has difficulty with (such as
    // unnecessary parens and similar), so we expect this to always pass.
    std::optional<AutoFmtPostconditionViolation> maybe_violation =
        ObeysAutoFmtOpportunisticPostcondition(stringified, autoformatted);
    if (maybe_violation.has_value()) {
      FAIL() << "autofmt postcondition violation";
      LOG(ERROR) << "= shard " << GetParam() << " sample " << i << ":";
      XLS_LOG_LINES(ERROR, stringified);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(FmtGeneratedAstTestInstance, FmtGeneratedAstTest,
                         testing::Range(0, 50));

}  // namespace
}  // namespace xls::dslx

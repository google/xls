// Copyright 2023 The XLS Authors
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

#include "xls/dslx/lsp/language_server_adapter.h"

#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

bool operator==(const verible::lsp::Position& lhs,
                const verible::lsp::Position& rhs) {
  return lhs.line == rhs.line && lhs.character == rhs.character;
}

TEST(LanguageServerAdapterTest, TestSingleFunctionModule) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, {"."});
  constexpr std::string_view kUri = "memfile://test.x";
  adapter.Update(kUri, "fn f() { () }");
  std::vector<verible::lsp::Diagnostic> diagnostics =
      adapter.GenerateParseDiagnostics(kUri);
  EXPECT_EQ(diagnostics.size(), 0);
  std::vector<verible::lsp::DocumentSymbol> symbols =
      adapter.GenerateDocumentSymbols(kUri);
  ASSERT_EQ(symbols.size(), 1);
}

TEST(LanguageServerAdapterTest, TestFindDefinitionsFunctionRef) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, {"."});
  constexpr std::string_view kUri = "memfile://test.x";
  adapter.Update(kUri, R"(fn f() { () }
fn main() { f() })");
  // Note: all of the line/column numbers are zero-based in the LSP protocol.
  verible::lsp::Position position{1, 12};
  std::vector<verible::lsp::Location> definition_locations =
      adapter.FindDefinitions(kUri, position);
  ASSERT_EQ(definition_locations.size(), 1);

  verible::lsp::Location definition_location = definition_locations.at(0);
  const auto want_start = verible::lsp::Position{0, 3};
  const auto want_end = verible::lsp::Position{0, 4};
  EXPECT_TRUE(definition_location.range.start == want_start);
  EXPECT_TRUE(definition_location.range.end == want_end);
}

TEST(LanguageServerAdapterTest, TestFindDefinitionsTypeRef) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, {"."});
  constexpr std::string_view kUri = "memfile://test.x";
  adapter.Update(kUri, R"(
type T = ();
fn f() -> T { () }
)");
  // Note: all of the line/column numbers are zero-based in the LSP protocol.
  verible::lsp::Position position{2, 10};
  std::vector<verible::lsp::Location> definition_locations =
      adapter.FindDefinitions(kUri, position);
  ASSERT_EQ(definition_locations.size(), 1);

  verible::lsp::Location definition_location = definition_locations.at(0);
  const auto want_start = verible::lsp::Position{1, 5};
  const auto want_end = verible::lsp::Position{1, 6};
  EXPECT_TRUE(definition_location.range.start == want_start);
  EXPECT_TRUE(definition_location.range.end == want_end);
}

// After we parse an invalid file the language server can still get requests,
// check that works reasonably.
TEST(LanguageServerAdapterTest, TestCallAfterInvalidParse) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, {"."});
  constexpr std::string_view kUri = "memfile://test.x";
  adapter.Update(kUri, "blahblahblah");

  verible::lsp::Position position{1, 12};
  std::vector<verible::lsp::Location> definition_locations =
      adapter.FindDefinitions(kUri, position);
  EXPECT_TRUE(definition_locations.empty());

  std::vector<verible::lsp::Diagnostic> diagnostics =
      adapter.GenerateParseDiagnostics(kUri);
  ASSERT_EQ(diagnostics.size(), 1);
  const verible::lsp::Diagnostic& diag = diagnostics.at(0);
  const auto want_start = verible::lsp::Position{0, 0};
  const auto want_end = verible::lsp::Position{0, 12};

  const verible::lsp::Position& start = diag.range.start;
  const verible::lsp::Position& end = diag.range.end;

  EXPECT_TRUE(start == want_start);
  EXPECT_TRUE(end == want_end);
  EXPECT_EQ(diag.message,
            "Expected start of top-level construct; got: 'blahblahblah'");
}

}  // namespace
}  // namespace xls::dslx

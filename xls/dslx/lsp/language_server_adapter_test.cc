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

TEST(LanguageServerAdapterTest, TestSingleFunctionModule) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, {"."});
  constexpr std::string_view kUri = "unused-for-now";
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
  constexpr std::string_view kUri = "unused-for-now";
  adapter.Update(kUri, R"(fn f() { () }
fn main() { f() })");
  // Note: all of the line/column numbers are zero-based in the LSP protocol.
  verible::lsp::Position position{1, 12};
  std::vector<verible::lsp::Location> definition_locations =
      adapter.FindDefinitions(kUri, position);
  ASSERT_EQ(definition_locations.size(), 1);
  verible::lsp::Location definition_location = definition_locations.at(0);
  EXPECT_EQ(definition_location.range.start.line, 0);
  EXPECT_EQ(definition_location.range.start.character, 3);

  EXPECT_EQ(definition_location.range.end.line, 0);
  EXPECT_EQ(definition_location.range.end.character, 4);
}

TEST(LanguageServerAdapterTest, TestFindDefinitionsTypeRef) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, {"."});
  constexpr std::string_view kUri = "unused-for-now";
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
  EXPECT_EQ(definition_location.range.start.line, 1);
  EXPECT_EQ(definition_location.range.start.character, 5);

  EXPECT_EQ(definition_location.range.end.line, 1);
  EXPECT_EQ(definition_location.range.end.character, 6);
}

}  // namespace
}  // namespace xls::dslx

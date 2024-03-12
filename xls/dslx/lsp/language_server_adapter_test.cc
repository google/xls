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

#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/default_dslx_stdlib_path.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;

std::string DebugString(const verible::lsp::Position& pos) {
  return absl::StrFormat("Position{.line=%d, .character=%d}", pos.line,
                         pos.character);
}

std::string DebugString(const verible::lsp::Range& range) {
  return absl::StrFormat("Range{.start=%s, .end=%s}", DebugString(range.start),
                         DebugString(range.end));
}

bool operator==(const verible::lsp::Position& lhs,
                const verible::lsp::Position& rhs) {
  return lhs.line == rhs.line && lhs.character == rhs.character;
}
bool operator==(const verible::lsp::Range& lhs,
                const verible::lsp::Range& rhs) {
  return lhs.start == rhs.start && lhs.end == rhs.end;
}

TEST(LanguageServerAdapterTest, TestSingleFunctionModule) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, "fn f() { () }"));
  std::vector<verible::lsp::Diagnostic> diagnostics =
      adapter.GenerateParseDiagnostics(kUri);
  EXPECT_EQ(diagnostics.size(), 0);
  std::vector<verible::lsp::DocumentSymbol> symbols =
      adapter.GenerateDocumentSymbols(kUri);
  ASSERT_EQ(symbols.size(), 1);
}

TEST(LanguageServerAdapterTest, LanguageServerRetainsParseResultForAllBuffers) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});

  // Fill language server editor buffers with two files (one valid, one not)
  constexpr std::string_view kUriValidContent = "memfile://valid.x";
  EXPECT_THAT(adapter.Update(kUriValidContent, "fn f() { () }"),
              StatusIs(absl::StatusCode::kOk));

  constexpr std::string_view kUriErrorContent = "memfile://error.x";
  EXPECT_THAT(adapter.Update(kUriErrorContent, "parse-error: not a valid file"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Now, query each buffer individually and get accurate information for each.
  EXPECT_EQ(adapter.GenerateParseDiagnostics(kUriValidContent).size(), 0);
  EXPECT_EQ(adapter.GenerateDocumentSymbols(kUriValidContent).size(), 1);

  EXPECT_EQ(adapter.GenerateParseDiagnostics(kUriErrorContent).size(), 1);
  EXPECT_EQ(adapter.GenerateDocumentSymbols(kUriErrorContent).size(), 0);

  // Query of unknown buffer is handled gracefully.
  EXPECT_EQ(adapter.GenerateParseDiagnostics("non-existent.x").size(), 0);
  EXPECT_EQ(adapter.GenerateDocumentSymbols("non-existent.x").size(), 0);
}

TEST(LanguageServerAdapterTest, TestFindDefinitionsFunctionRef) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(fn f() { () }
fn main() { f() })"));
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
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(
type T = ();
fn f() -> T { () }
)"));
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
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  ASSERT_FALSE(adapter.Update(kUri, "blahblahblah").ok());

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

TEST(LanguageServerAdapterTest, DocumentLinksAreCreatedForImports) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, "import std;"));
  //                             pos: 0123456789A
  std::vector<verible::lsp::DocumentLink> links =
      adapter.ProvideImportLinks(kUri);
  ASSERT_EQ(links.size(), 1);
  EXPECT_EQ(links.front().range.start.line, 0);
  EXPECT_EQ(links.front().range.start.character, 7);  // link over 'std'.
  EXPECT_EQ(links.front().range.end.character, 10);
  EXPECT_TRUE(absl::StrContains(links.front().target, "stdlib/std.x"));
}

TEST(LanguageServerAdapterTest, DocumentLevelFormattingComment) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(// Top of module comment.

fn messy() {
()// retval comment
})"));
  const auto kInputRange =
      verible::lsp::Range{.start = verible::lsp::Position{0, 0},
                          .end = verible::lsp::Position{4, 1}};
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<verible::lsp::TextEdit> edits,
                           adapter.FormatDocument(kUri));

  ASSERT_EQ(edits.size(), 1);

  const verible::lsp::TextEdit& edit = edits.at(0);
  EXPECT_TRUE(edit.range == kInputRange) << DebugString(edit.range);
  EXPECT_EQ(edit.newText, R"(// Top of module comment.

fn messy() {
    ()  // retval comment
}
)");
}

}  // namespace
}  // namespace xls::dslx

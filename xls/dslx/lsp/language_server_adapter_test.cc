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

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "verible/common/lsp/lsp-protocol.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/default_dslx_stdlib_path.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::string DebugString(const verible::lsp::Position& pos) {
  return absl::StrFormat("Position{.line=%d, .character=%d}", pos.line,
                         pos.character);
}

std::string DebugString(const verible::lsp::Range& range) {
  return absl::StrFormat("Range{.start=%s, .end=%s}", DebugString(range.start),
                         DebugString(range.end));
}

std::string DebugString(const verible::lsp::Location& location) {
  return absl::StrFormat("Location{.uri=\"%s\", .range=%s}", location.uri,
                         DebugString(location.range));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<verible::lsp::Location> definition_locations,
      adapter.FindDefinitions(kUri, position));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<verible::lsp::Location> definition_locations,
      adapter.FindDefinitions(kUri, position));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<verible::lsp::Location> definition_locations,
      adapter.FindDefinitions(kUri, position));
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

TEST(LanguageServerAdapterTest, DisableFormatting) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  constexpr std::string_view kInput = R"(// Top of module comment.

// dslx-fmt::off
fn messy(){()// retval comment
}
// dslx-fmt::on)";
  const auto kInputRange =
      verible::lsp::Range{.start = verible::lsp::Position{0, 0},
                          .end = verible::lsp::Position{5, 15}};

  // Notify the adapter of the file contents.
  XLS_ASSERT_OK(adapter.Update(kUri, kInput));

  // Act
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<verible::lsp::TextEdit> edits,
                           adapter.FormatDocument(kUri));

  // Assert
  ASSERT_EQ(edits.size(), 1);
  const verible::lsp::TextEdit& edit = edits.at(0);
  EXPECT_TRUE(edit.range == kInputRange) << DebugString(edit.range);
  // Text should be unchanged, because we disabled formatting.
  EXPECT_EQ(edit.newText, kInput);
}

TEST(LanguageServerAdapterTest, InlayHintForLetStatement) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(fn f(x: u32) -> u32 {
  let y = x;
  let z = y;
  z
})"));

  const auto kInputRange =
      verible::lsp::Range{.start = verible::lsp::Position{1, 0},
                          .end = verible::lsp::Position{2, 0}};
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<verible::lsp::InlayHint> hints,
                           adapter.InlayHint(kUri, kInputRange));

  ASSERT_EQ(hints.size(), 1);

  const verible::lsp::InlayHint& hint = hints.at(0);
  verible::lsp::Position want_position{1, 7};
  EXPECT_TRUE(hint.position == want_position)
      << "got: " << DebugString(hint.position)
      << " want: " << DebugString(want_position);
  EXPECT_EQ(hint.label, ": uN[32]");
}

TEST(LanguageServerAdapterTest, FindDefinitionAcrossFiles) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory tempdir, TempDirectory::Create());
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath,
                                /*dslx_paths=*/{tempdir.path()});

  std::string imported_uri =
      absl::StrFormat("file://%s/imported.x", tempdir.path());
  const std::string_view kImportedContents = "pub fn f() { () }";
  XLS_ASSERT_OK(
      SetFileContents(tempdir.path() / "imported.x", kImportedContents));

  std::string importer_uri =
      absl::StrFormat("file://%s/importer.x", tempdir.path());
  const std::string_view kImporterContents = R"(import imported;

fn main() { imported::f() }
)";
  XLS_ASSERT_OK(adapter.Update(importer_uri, kImporterContents));

  verible::lsp::Position position{2, 13};
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<verible::lsp::Location> definition_locations,
      adapter.FindDefinitions(importer_uri, position));
  ASSERT_EQ(definition_locations.size(), 1);

  verible::lsp::Location definition_location = definition_locations.at(0);
  VLOG(1) << "definition location: " << DebugString(definition_location);
  EXPECT_EQ(definition_location.uri, imported_uri);
  const verible::lsp::Range kWantRange{
      .start = verible::lsp::Position{0, 7},
      .end = verible::lsp::Position{0, 8},
  };
  EXPECT_TRUE(definition_location.range == kWantRange);
}

TEST(LanguageServerAdapterTest, RenameForParameter) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr char kUri[] = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(fn f(x: u32) -> u32 {
  let y = x;
  let z = y;
  z
})"));

  const auto kWantRange =
      verible::lsp::Range{.start = verible::lsp::Position{0, 5},
                          .end = verible::lsp::Position{0, 6}};
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<verible::lsp::Range> to_rename,
                           adapter.PrepareRename(kUri, kWantRange.start));
  ASSERT_TRUE(to_rename.has_value());
  EXPECT_TRUE(kWantRange == to_rename.value());

  // Rename from the use position should also work and resolve to the same
  // definition span.
  verible::lsp::Position use_pos{1, 10};
  XLS_ASSERT_OK_AND_ASSIGN(to_rename, adapter.PrepareRename(kUri, use_pos));
  ASSERT_TRUE(to_rename.has_value());
  EXPECT_TRUE(kWantRange == to_rename.value());

  // See what edits come out.
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<verible::lsp::WorkspaceEdit> edit,
                           adapter.Rename(kUri, kWantRange.start, "foo"));
  ASSERT_TRUE(edit.has_value());

  EXPECT_EQ(edit->changes.at(kUri).size(), 2);
}

TEST(LanguageServerAdapterTest, RenameForModuleScopedConstant) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr char kUri[] = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(const FOO = u32:42;

const BAR: u32 = FOO + FOO;)"));

  const auto kWantRange =
      verible::lsp::Range{.start = verible::lsp::Position{0, 6},
                          .end = verible::lsp::Position{0, 9}};
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<verible::lsp::Range> to_rename,
                           adapter.PrepareRename(kUri, kWantRange.start));
  ASSERT_TRUE(to_rename.has_value());
  EXPECT_TRUE(kWantRange == to_rename.value());

  // Rename from the use position should also work and resolve to the same
  // definition span.
  verible::lsp::Position use_pos{2, 17};
  XLS_ASSERT_OK_AND_ASSIGN(to_rename, adapter.PrepareRename(kUri, use_pos));
  ASSERT_TRUE(to_rename.has_value());
  EXPECT_TRUE(kWantRange == to_rename.value());

  // See what edits come out.
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<verible::lsp::WorkspaceEdit> edit,
                           adapter.Rename(kUri, kWantRange.start, "FT"));
  ASSERT_TRUE(edit.has_value());

  EXPECT_EQ(edit->changes.at(kUri).size(), 3);
}

// Currently we cannot rename across files so we refuse to rename `pub`
// visibility members.
TEST(LanguageServerAdapterTest, RenameForPublicModuleScopedConstant) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(pub const FOO = u32:42;

const BAR: u32 = FOO + FOO;)"));

  const auto kWantRange =
      verible::lsp::Range{.start = verible::lsp::Position{0, 10},
                          .end = verible::lsp::Position{0, 13}};
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<verible::lsp::Range> to_rename,
                           adapter.PrepareRename(kUri, kWantRange.start));
  ASSERT_TRUE(to_rename.has_value());
  EXPECT_TRUE(kWantRange == to_rename.value());

  // See what edits come out.
  absl::StatusOr<std::optional<verible::lsp::WorkspaceEdit>> edit =
      adapter.Rename(kUri, kWantRange.start, "FT");
  XLS_EXPECT_OK(edit.status());
  EXPECT_EQ(edit.value(), std::nullopt);
}

TEST(LanguageServerAdapterTest, DocumentHighlight) {
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath, /*dslx_paths=*/{"."});
  constexpr std::string_view kUri = "memfile://test.x";
  XLS_ASSERT_OK(adapter.Update(kUri, R"(pub const FOO = u32:42;

const BAR: u32 = FOO + FOO;

fn f() -> u32 { FOO })"));
  const auto kTargetPos = verible::lsp::Position{4, 16};
  absl::StatusOr<std::vector<verible::lsp::DocumentHighlight>> highlights_or =
      adapter.DocumentHighlight(kUri, kTargetPos);
  XLS_EXPECT_OK(highlights_or.status());
  const auto& highlights = highlights_or.value();

  // There are four instances in the document including the definition.
  EXPECT_EQ(highlights.size(), 4);

  // Definition comes first.
  EXPECT_EQ(highlights[0].range.start.line, 0);

  // Then uses in the const.
  EXPECT_EQ(highlights[1].range.start.line, 2);
  EXPECT_EQ(highlights[2].range.start.line, 2);

  // Then use in the function definition.
  EXPECT_EQ(highlights[3].range.start.line, 4);
}

// This models a scenario where we observe a problem in `outer.x`, but that
// problem actually stems from the import of `inner.x`.
//
// Even though `outer.x` does not successfully import `inner.x` we test that
// the module DAG information contains "outer tried to import inner" -- we use
// this DAG information to walk upwards and check whether `outer.x` is ok once
// `inner.x` is fixed.
TEST(LanguageServerAdapterTest, DagShowsUnsuccessfulImports) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory tempdir, TempDirectory::Create());
  LanguageServerAdapter adapter(kDefaultDslxStdlibPath,
                                /*dslx_paths=*/{tempdir.path()});

  std::string inner_uri = absl::StrFormat("file://%s/inner.x", tempdir.path());
  const std::string bad_inner_contents = R"(const FOO = u32:42;)";
  const std::string good_inner_contents = R"(pub const FOO = u32:42;)";
  XLS_ASSERT_OK(
      SetFileContents(tempdir.path() / "inner.x", bad_inner_contents));
  XLS_ASSERT_OK(adapter.Update(inner_uri, bad_inner_contents));

  std::string outer_uri = absl::StrFormat("file://%s/outer.x", tempdir.path());
  std::string outer_contents = R"(import inner;

const OUTER_FOO = inner::FOO;  // this is not public, at first
)";
  EXPECT_THAT(
      adapter.Update(outer_uri, outer_contents),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attempted to refer to module member const FOO")));
  std::vector<verible::lsp::Diagnostic> diags =
      adapter.GenerateParseDiagnostics(outer_uri);
  ASSERT_EQ(diags.size(), 1);

  // Now we make our correction to inner in our text buffer.
  //
  // This should cause outer to be re-evaluated, and we should no longer see
  // error diagnostics for it.
  XLS_ASSERT_OK(adapter.Update(inner_uri, good_inner_contents));
  auto sensitive_set =
      adapter.import_sensitivity().GatherAllSensitiveToChangeIn(inner_uri);
  ASSERT_EQ(sensitive_set.size(), 2);
  EXPECT_THAT(sensitive_set,
              testing::UnorderedElementsAre(inner_uri, outer_uri));

  for (const std::string& sensitive : sensitive_set) {
    if (sensitive == inner_uri) {
      continue;
    }

    XLS_ASSERT_OK(adapter.Update(sensitive, std::nullopt))
        << "due to update of: " << sensitive;
  }

  diags = adapter.GenerateParseDiagnostics(outer_uri);
  ASSERT_TRUE(diags.empty());
}

}  // namespace
}  // namespace xls::dslx

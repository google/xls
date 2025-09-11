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

#include "xls/dslx/fmt/format_disabler.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr std::string_view kFmtOff = "// dslx-fmt::off\n";
constexpr std::string_view kFmtOn = "// dslx-fmt::on\n";

TEST(FormatDisablerTest, NotDisabled) {
  // Arrange.
  const std::string kProgram = "import bar;\n";

  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  // Act.
  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto actual, disabler(import_node, m.get(), {}));

  // Assert that it is nullopt, which indicates "node not modified".
  EXPECT_EQ(actual, std::nullopt);
}

TEST(FormatDisablerTest, NotDisabled_WithComments) {
  // There are comments but not enable/disable comments.
  const std::string kProgram = R"(
      // comment
      import bar;
      // another comment
  )";

  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto actual, disabler(import_node, m.get(), {}));

  EXPECT_EQ(actual, std::nullopt);
}

TEST(FormatDisablerTest, DisabledAroundImport) {
  const std::string kImport = "  import\n  bar;\n";
  const std::string kProgram = absl::StrCat(kFmtOff, kImport, kFmtOn);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto actual, disabler(import_node, m.get(), {}));
  ASSERT_TRUE(actual.has_value());
  auto it_map = actual->find(import_node);
  ASSERT_NE(it_map, actual->end());
  VerbatimNode* actual_node = down_cast<VerbatimNode*>(it_map->second);
  ASSERT_NE(actual_node, nullptr);

  EXPECT_EQ(actual_node->text(), absl::StrCat(kImport, kFmtOn));
}

TEST(FormatDisablerTest, EnabledOnSameLine) {
  // Note trailing space, which we want to be part of the unformatted text.
  const std::string kImport = "  import  bar; ";
  const std::string kProgram = absl::StrCat(kFmtOff, kImport, kFmtOn);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto actual, disabler(import_node, m.get(), {}));
  ASSERT_TRUE(actual.has_value());
  auto it_map = actual->find(import_node);
  ASSERT_NE(it_map, actual->end());
  VerbatimNode* actual_node = down_cast<VerbatimNode*>(it_map->second);
  ASSERT_NE(actual_node, nullptr);

  EXPECT_EQ(actual_node->text(), absl::StrCat(kImport, kFmtOn));
}

TEST(FormatDisablerTest, EnabledOnSameLineWithNewlineBetween) {
  // Note trailing space, which we want to be part of the unformatted text.
  const std::string kImport = "  import\n bar; ";
  const std::string kProgram = absl::StrCat(kFmtOff, kImport, kFmtOn);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto actual, disabler(import_node, m.get(), {}));
  ASSERT_TRUE(actual.has_value());
  auto it_map = actual->find(import_node);
  ASSERT_NE(it_map, actual->end());
  std::optional<AstNode*> actual_node_opt = it_map->second;

  ASSERT_TRUE(actual_node_opt.has_value());
  VerbatimNode* actual_node = down_cast<VerbatimNode*>(*actual_node_opt);
  ASSERT_NE(actual_node, nullptr);

  EXPECT_EQ(actual_node->text(), absl::StrCat(kImport, kFmtOn));
}

TEST(FormatDisablerTest, MultipleDisabledStatements) {
  const std::string kTwoImports = "  import\n  foo;\n  import  bar;\n";
  const std::string kProgram = absl::StrCat(kFmtOff, kTwoImports, kFmtOn);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* first_import_node = std::get<Import*>(m->top().at(0));
  Import* second_import_node = std::get<Import*>(m->top().at(1));
  ASSERT_NE(first_import_node, nullptr);
  ASSERT_NE(second_import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto first_actual,
                           disabler(first_import_node, m.get(), {}));
  ASSERT_TRUE(first_actual.has_value());
  auto it_first = first_actual->find(first_import_node);
  ASSERT_NE(it_first, first_actual->end());
  std::optional<AstNode*> first_node_opt = it_first->second;

  ASSERT_TRUE(first_node_opt.has_value());
  VerbatimNode* first_verbatim_node = down_cast<VerbatimNode*>(*first_node_opt);
  ASSERT_NE(first_verbatim_node, nullptr);

  // Text should be the two imports concatenated.
  EXPECT_EQ(first_verbatim_node->text(), absl::StrCat(kTwoImports, kFmtOn));

  // The second node should be replaced with an empty verbatim node since it's
  // within the "disable" range.
  XLS_ASSERT_OK_AND_ASSIGN(auto second_actual,
                           disabler(second_import_node, m.get(), {}));
  ASSERT_TRUE(second_actual.has_value());
  auto it_second = second_actual->find(second_import_node);
  ASSERT_NE(it_second, second_actual->end());
  std::optional<AstNode*> second_node_opt = it_second->second;
  ASSERT_TRUE(second_node_opt.has_value());
  VerbatimNode* second_verbatim_node =
      down_cast<VerbatimNode*>(*second_node_opt);
  ASSERT_NE(second_verbatim_node, nullptr);

  EXPECT_TRUE(second_verbatim_node->IsEmpty());
}

TEST(FormatDisablerTest, OneDisabledOneEnabledStatement) {
  const std::string kUnformattedImport = "  import\n  foo;\n";
  const std::string kProgram =
      absl::StrCat(kFmtOff, kUnformattedImport, kFmtOn, "import bar;\n");
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* first_import_node = std::get<Import*>(m->top().at(0));
  Import* second_import_node = std::get<Import*>(m->top().at(1));
  ASSERT_NE(first_import_node, nullptr);
  ASSERT_NE(second_import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto first_actual,
                           disabler(first_import_node, m.get(), {}));
  std::optional<AstNode*> first_node_opt = std::nullopt;
  if (first_actual.has_value()) {
    auto it = first_actual->find(first_import_node);
    if (it != first_actual->end()) first_node_opt = it->second;
  }

  ASSERT_TRUE(first_node_opt.has_value());
  VerbatimNode* actual_node = down_cast<VerbatimNode*>(*first_node_opt);
  ASSERT_NE(actual_node, nullptr);

  // Text should be just the first import.
  EXPECT_EQ(actual_node->text(), absl::StrCat(kUnformattedImport, kFmtOn));

  // The second import should be left as-is since it's outside the "disable"
  // range.
  XLS_ASSERT_OK_AND_ASSIGN(auto second_actual,
                           disabler(second_import_node, m.get(), {}));
  std::optional<AstNode*> second_node_opt = std::nullopt;
  if (second_actual.has_value()) {
    auto it = second_actual->find(second_import_node);
    if (it != second_actual->end()) second_node_opt = it->second;
  }
  EXPECT_EQ(second_node_opt, std::nullopt);
}

TEST(FormatDisablerTest, MultipleEnabledStatements) {
  const std::string kTwoImports = "  import\n  foo;\n  import  bar;\n";
  const std::string kProgram =
      absl::StrCat("// comment 1\n", kTwoImports, "// comment 2\n");
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* first_import_node = std::get<Import*>(m->top().at(0));
  Import* second_import_node = std::get<Import*>(m->top().at(1));
  ASSERT_NE(first_import_node, nullptr);
  ASSERT_NE(second_import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  // First node should be returned as-is, since there's no "start disable"
  // before it.
  XLS_ASSERT_OK_AND_ASSIGN(auto first_actual,
                           disabler(first_import_node, m.get(), {}));
  std::optional<AstNode*> first_node_opt = std::nullopt;
  if (first_actual.has_value()) {
    auto it = first_actual->find(first_import_node);
    if (it != first_actual->end()) first_node_opt = it->second;
  }
  EXPECT_EQ(first_node_opt, std::nullopt);

  // The second node should be returned as-is too.
  XLS_ASSERT_OK_AND_ASSIGN(auto second_actual,
                           disabler(second_import_node, m.get(), {}));
  std::optional<AstNode*> second_node_opt = std::nullopt;
  if (second_actual.has_value()) {
    auto it = second_actual->find(second_import_node);
    if (it != second_actual->end()) second_node_opt = it->second;
  }
  EXPECT_EQ(second_node_opt, std::nullopt);
}

TEST(FormatDisablerTest, EnabledOnly) {
  const std::string kImport = "  import\n  bar;\n";
  const std::string kProgram = absl::StrCat(kImport, kFmtOn);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto maybe_map, disabler(import_node, m.get(), {}));
  std::optional<AstNode*> actual = std::nullopt;
  if (maybe_map.has_value()) {
    auto it = maybe_map->find(import_node);
    if (it != maybe_map->end()) actual = it->second;
  }

  // No change.
  EXPECT_EQ(actual, std::nullopt);
}

TEST(FormatDisablerTest, NeverEnabled) {
  const std::string kImport = "  import\n  bar;\n";
  const std::string kProgram = absl::StrCat(kFmtOff, kImport);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto maybe_map, disabler(import_node, m.get(), {}));
  ASSERT_TRUE(maybe_map.has_value());
  auto it_map = maybe_map->find(import_node);
  ASSERT_NE(it_map, maybe_map->end());
  std::optional<AstNode*> actual = it_map->second;

  ASSERT_TRUE(actual.has_value());
  VerbatimNode* actual_node = down_cast<VerbatimNode*>(*actual);
  ASSERT_NE(actual_node, nullptr);

  EXPECT_EQ(actual_node->text(), kImport);
}

TEST(FormatDisablerTest, NoSpan) {
  // First we have to get it into "unformatted" mode.
  const std::string kProgram = "// dslx-fmt::off\nimport m;\n";
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);
  UniformContentFilesystem vfs(kProgram, "fake.x");
  Import* import_node = std::get<Import*>(m->top().at(0));
  FormatDisabler disabler(vfs, comments, kProgram);
  XLS_ASSERT_OK_AND_ASSIGN(auto first_map, disabler(import_node, m.get(), {}));
  std::optional<AstNode*> actual = std::nullopt;
  if (first_map.has_value()) {
    auto it = first_map->find(import_node);
    if (it != first_map->end()) actual = it->second;
  }

  // Now we can give it a node with no span.
  BuiltinNameDef built_in(nullptr, "identifier");
  XLS_ASSERT_OK_AND_ASSIGN(auto no_span_map, disabler(&built_in, m.get(), {}));
  actual = std::nullopt;
  if (no_span_map.has_value()) {
    auto it = no_span_map->find(&built_in);
    if (it != no_span_map->end()) actual = it->second;
  }

  ASSERT_FALSE(actual.has_value());
}

TEST(FormatDisablerTest, TwoDisables) {
  constexpr std::string_view kProgram = R"(
// dslx-fmt::off
// dslx-fmt::off
import m;
)";

  std::vector<CommentData> comments_vec;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_vec));
  Comments comments = Comments::Create(comments_vec);
  UniformContentFilesystem vfs(kProgram, "fake.x");

  Import* import_node = std::get<Import*>(m->top().at(0));
  FormatDisabler disabler(vfs, comments, kProgram);
  EXPECT_THAT(disabler(import_node, m.get(), {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("between module start and")));
}

TEST(FormatDisablerTest, TwoDisablesAfterImport) {
  constexpr std::string_view kProgram = R"(
import m;

// dslx-fmt::off
// dslx-fmt::off
    import m2;
)";

  std::vector<CommentData> comments_vec;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_vec));
  Comments comments = Comments::Create(comments_vec);
  UniformContentFilesystem vfs(kProgram, "fake.x");

  FormatDisabler disabler(vfs, comments, kProgram);
  Import* first_import_node = std::get<Import*>(m->top().at(0));
  XLS_ASSERT_OK(disabler(first_import_node, m.get(), {}).status());

  Import* second_import_node = std::get<Import*>(m->top().at(1));
  EXPECT_THAT(disabler(second_import_node, m.get(), {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("between import m; and import m2")));
}

TEST(FormatDisablerTest, InternalCommentIncludedInVerbatimNode) {
  const std::string kImport = "  import\n  bar;\n";
  const std::string kInternalComment = "  // internal comment\n";
  const std::string kExternalComment = "  // external comment\n";
  const std::string kProgram = absl::StrCat(kFmtOff, kImport, kInternalComment,
                                            kFmtOn, kExternalComment);
  std::vector<CommentData> comments_list;
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kProgram, "fake.x", "fake", file_table, &comments_list));
  Comments comments = Comments::Create(comments_list);

  Import* import_node = std::get<Import*>(m->top().at(0));
  ASSERT_NE(import_node, nullptr);

  UniformContentFilesystem vfs(kProgram, "fake.x");
  FormatDisabler disabler(vfs, comments, kProgram);
  EXPECT_EQ(comments.GetComments(m->GetSpan().value()).size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(auto maybe_map, disabler(import_node, m.get(), {}));
  ASSERT_TRUE(maybe_map.has_value());
  auto it_map = maybe_map->find(import_node);
  ASSERT_NE(it_map, maybe_map->end());
  std::optional<AstNode*> actual = it_map->second;

  ASSERT_TRUE(actual.has_value());
  VerbatimNode* actual_node = down_cast<VerbatimNode*>(*actual);
  ASSERT_NE(actual_node, nullptr);

  // There should be zero comments in the verbatim node's span, and two fewer in
  // the module span (the enable formatting comment is removed too.)
  EXPECT_EQ(comments.GetComments(actual_node->GetSpan().value()).size(), 0);
  ASSERT_EQ(comments.GetComments(m->GetSpan().value()).size(), 2);
  EXPECT_THAT(kFmtOff,
              HasSubstr(comments.GetComments(m->GetSpan().value())[0]->text));
  EXPECT_THAT(kExternalComment,
              HasSubstr(comments.GetComments(m->GetSpan().value())[1]->text));

  // The text should include the internal comment.
  EXPECT_EQ(actual_node->text(),
            absl::StrCat(kImport, kInternalComment, kFmtOn));
}

}  // namespace
}  // namespace xls::dslx

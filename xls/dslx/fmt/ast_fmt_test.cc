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

#include "xls/dslx/fmt/ast_fmt.h"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(AstFmtTest, FormatLet) {
  Scanner s{"fake.x", "{ let x: u32 = u32:42; }"};
  Parser p("fake", &s);
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p.ParseBlockExpression(bindings));
  Statement* stmt = block->statements().at(0);

  Comments comments = Comments::Create(s.comments());

  DocArena arena;
  DocRef doc = Fmt(*stmt, comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "let x: u32 = u32:42");
}

TEST(AstFmtTest, FormatLetWithInlineComment) {
  Scanner s{"fake.x", R"({
  let x: u32 = u32:42; // short
})"};
  Parser p("fake", &s);
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p.ParseBlockExpression(bindings));

  Statement* stmt = block->statements().at(0);

  Comments comments = Comments::Create(s.comments());

  DocArena arena;
  DocRef doc = Fmt(*stmt, comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100),
            "let x: u32 = u32:42 // short");
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/20), R"(// short
let x: u32 = u32:42)");
}

class FunctionFmtTest : public testing::Test {
 public:
  absl::StatusOr<std::string> DoFmt(std::string_view original) {
    XLS_CHECK(!scanner_.has_value());
    scanner_.emplace("fake.x", std::string{original});
    parser_.emplace("fake", &scanner_.value());
    XLS_ASSIGN_OR_RETURN(
        f_, parser_->ParseFunction(/*is_public=*/false, bindings_));
    Comments comments = Comments::Create(scanner_->comments());

    DocRef doc = Fmt(*f_, comments, arena_);
    return PrettyPrint(arena_, doc, /*text_width=*/100);
  }

 private:
  DocArena arena_;
  std::optional<Scanner> scanner_;
  std::optional<Parser> parser_;
  Bindings bindings_;
  Function* f_ = nullptr;
};

TEST_F(FunctionFmtTest, FormatIdentityFn) {
  const std::string_view original = "fn f(x:u32)->u32{x}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, "fn f(x: u32) -> u32 { x }");
}

TEST_F(FunctionFmtTest, FormatEmptyFn) {
  const std::string_view original = "fn f(){}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, "fn f() {}");
}

TEST_F(FunctionFmtTest, FormatLetYEqualsXFn) {
  const std::string_view original = "fn f(x:u32)->u32{let y=x;y}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u32 {
    let y = x;
    y
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, FormatTupleDestructure) {
  const std::string_view original = "fn f(t:(u32,u64))->u32{let(x,y)=t;x}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(t: (u32, u64)) -> u32 {
    let (x, y) = t;
    x
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, FormatMultiParameter) {
  const std::string_view original = "fn f(x:u32,y:u64)->(u32,u64){(x,y)}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32, y: u64) -> (u32, u64) { (x, y) })";
  EXPECT_EQ(got, want);
}

TEST(ModuleFmtTest, TwoSimpleFunctions) {
  const std::string_view kProgram =
      "fn double(x:u32)->u32{u32:2*x}fn triple(x: u32)->u32{u32:3*x}";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, R"(fn double(x: u32) -> u32 { u32:2 * x }

fn triple(x: u32) -> u32 { u32:3 * x }
)");
}

}  // namespace
}  // namespace xls::dslx

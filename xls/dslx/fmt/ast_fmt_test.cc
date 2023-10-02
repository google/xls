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

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"

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
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/0), R"(let
x:
u32
=
u32:
42)");
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
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/0), R"(// short
let
x:
u32
=
u32:
42)");
}

}  // namespace
}  // namespace xls::dslx

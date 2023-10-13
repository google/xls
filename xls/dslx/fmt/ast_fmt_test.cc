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
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_test_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(BuiltAstFmtTest, FormatCastThatNeedsParens) {
  auto [module, lt] = MakeCastWithinLtComparison();
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*lt, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x as t) < x");
}

TEST(BuiltAstFmtTest, FormatIndexThatNeedsParens) {
  auto [module, index] = MakeCastWithinIndexExpression();
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*index, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x as u32[42])[i]");
}

TEST(BuiltAstFmtTest, FormatTupleIndexThatNeedsParens) {
  auto [module, tuple_index] = MakeIndexWithinTupleIndexExpression();
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*tuple_index, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x[i]).2");
}

TEST(BuiltAstFmtTest, FormatUnopThatNeedsParensOnOperand) {
  auto [module, unop] = MakeCastWithinNegateExpression();
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*unop, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "-(x as u32)");
}

TEST(BuiltAstFmtTest, FormatAttrThatNeedsParensOnOperand) {
  auto [module, attr] = MakeArithWithinAttrExpression();
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*attr, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x * y).my_attr");
}

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

// Fixture for test that format entire (single) functions -- expected usage:
//
//  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt("fn ..."));
//  EXPECT_EQ(got, ...);
//
// Use `ModuleFmtTest` for formatting of entire modules.
class FunctionFmtTest : public testing::Test {
 public:
  // Args:
  //  original: The text to parse-then-auto-format.
  //  builtin_name_defs: Names to initialize in the bindings as built-in.
  absl::StatusOr<std::string> DoFmt(
      std::string_view original,
      const absl::flat_hash_set<std::string>& builtin_name_defs = {}) {
    XLS_CHECK(!scanner_.has_value());
    scanner_.emplace("fake.x", std::string{original});
    parser_.emplace("fake", &scanner_.value());

    for (const std::string& name : builtin_name_defs) {
      bindings_.Add(name, parser_->module().GetOrCreateBuiltinNameDef(name));
    }

    XLS_ASSIGN_OR_RETURN(
        f_, parser_->ParseFunction(/*is_public=*/false, bindings_));
    Comments comments = Comments::Create(scanner_->comments());

    DocRef doc = Fmt(*f_, comments, arena_);
    return PrettyPrint(arena_, doc, /*text_width=*/100);
  }

  Bindings& bindings() { return bindings_; }

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

TEST_F(FunctionFmtTest, SimpleUnopNegate) {
  const std::string_view original = "fn f(x:u32)->u32{-x}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u32 { -x })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SimpleUnopInvert) {
  const std::string_view original = "fn f(x:u32)->u32{!x}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u32 { !x })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SimpleCast) {
  const std::string_view original = "fn f(x:u32)->u64{x as u64}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u64 { x as u64 })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, DoubleParensLhsOfCast) {
  const std::string_view original = "fn f(x:u32)->u64{((x++x)) as u64}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u64 { (x ++ x) as u64 })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, DoubleParensLhsOfIndex) {
  const std::string_view original = "fn f(x:u32[2],i:u32)->u32{((x++x))[i]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32[2], i: u32) -> u32 { (x ++ x)[i] })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, NestedCast) {
  const std::string_view original = "fn f(x:u32)->u64{x as u48 as u64}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u64 { x as u48 as u64 })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, CastOfConcatAndSlice) {
  const std::string_view original = "fn f(x:u32)->u64{x++(x[:16]) as u64}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u64 { x ++ (x[:16]) as u64 })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SliceStartAndLimitValues) {
  const std::string_view original = "fn f(x:u32)->u16{x[16:32]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u16 { x[16:32] })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, WidthSlice) {
  const std::string_view original = "fn f(x:u32)->u16{x[16+:u16]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u16 { x[16+:u16] })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, TupleIndex) {
  const std::string_view original = "fn f(t:(u1,u2,u3))->u3{t.2}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(t: (u1, u2, u3)) -> u3 { t.2 })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConstAssert) {
  const std::string_view original =
      "fn f(){const_assert!(u32:2+u32:3 == u32:5);}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f() { const_assert!(u32:2 + u32:3 == u32:5); })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConditionalInTernaryStyle) {
  const std::string_view original =
      "fn f(x:bool,y:u32,z:u32)->u32{if x{y}else{z}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: bool, y: u32, z: u32) -> u32 { if x { y } else { z } })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConditionalMultiStatementCausesHardBreaks) {
  const std::string_view original =
      "fn f(x:bool,y:u32,z:u32)->u32{if x{y;z}else{z;y}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: bool, y: u32, z: u32) -> u32 {
    if x {
        y;
        z
    } else {
        z;
        y
    }
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConditionalWithElseIf) {
  const std::string_view original =
      "fn f(a:bool[2],x:u32[3])->u32{if a[0]{x[0]}else if "
      "a[1]{x[1]}else{x[2]}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(a: bool[2], x: u32[3]) -> u32 {
    if a[0] {
        x[0]
    } else if a[1] {
        x[1]
    } else {
        x[2]
    }
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SimpleForOneStatementNoTypeAnnotation) {
  const std::string_view original =
      "fn f(x:u32)->u32{for(i,accum)in u32:0..u32:4{accum+i}(x)}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u32 {
    for (i, accum) in u32:0..u32:4 {
        accum + i
    }(x)
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SimpleForOneStatementTypeAnnotated) {
  const std::string_view original =
      "fn f(x:u32)->u32{for(i,accum):(u32,u32)in u32:0..u32:4{accum+i}(x)}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u32 {
    for (i, accum): (u32, u32) in u32:0..u32:4 {
        accum + i
    }(x)
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SimpleForLetBinding) {
  const std::string_view original =
      "fn f(x:u32)->u32{let y=for(i,accum):(u32,u32)in "
      "u32:0..u32:4{accum+i}(x);y}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u32 {
    let y = for (i, accum): (u32, u32) in u32:0..u32:4 {
        accum + i
    }(x);
    y
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, SimpleMatchOnBool) {
  const std::string_view original =
      "fn f(b:bool)->u32{match b{true=>u32:42,_=>u32:64}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(b: bool) -> u32 {
    match b {
        true => u32:42,
        _ => u32:64,
    }
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, MatchMultiPattern) {
  const std::string_view original =
      "fn f(b:bool)->u32{match b{true|false=>u32:42,_=>u32:64}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(b: bool) -> u32 {
    match b {
        true | false => u32:42,
        _ => u32:64,
    }
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ZeroMacro) {
  const std::string_view original = "fn f()->u32{zero!<u32>()}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"zero!"}));
  const std::string_view want = R"(fn f() -> u32 { zero!<u32>() })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, TraceFormatMacro) {
  const std::string_view original =
      R"(fn f(x:u32,y:u32){trace_fmt!("x is {} y is {}",x,y)})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"trace_fmt!"}));
  const std::string_view want =
      R"(fn f(x: u32, y: u32) { trace_fmt!("x is {} y is {}", x, y) })";
  EXPECT_EQ(got, want);
}

// -- ModuleFmtTest cases, formatting entire modules

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

TEST(ModuleFmtTest, OverLongImport) {
  const std::string_view kProgram =
      "import very_long.name_here.made_of.dotted_components";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments), 14);
  EXPECT_EQ(got,
            "import very_long.\n"
            "       name_here.\n"
            "       made_of.\n"
            "       dotted_components\n");
}

TEST(ModuleFmtTest, ConstantDef) {
  const std::string_view kProgram = "pub const MOL = u32:42;\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ConstantDefArray) {
  const std::string_view kProgram = "pub const VALS = u32[2]:[32, 64];\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ConstantDefArrayMultiline) {
  const std::string_view kProgram = R"(pub const VALS = u64[5]:[
    0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0a447a02823ad868,
    0x1df74948b3fbea7e, 0x1bc8b594bcf01a39, 0x07b767ca9520e99a, 0x05e28b4320bfd20e,
    0x0105906a24823f57, 0x1a1e7d14a6d24384, 0x2a7326df322e084d, 0x120bc9cc3fac4ec7,
    0x2c8f193a1b46a9c5, 0x2b9c95743bbe3f90, 0x0dcfc5b1d0398b46, 0x006ba47b3448bea3,
    0x3fe4fbf9a522891b, 0x23e1a50ad6aebca3, 0x1b263d39ea62be44, 0x13581d282e643b0e,
];
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ConstantDefArrayMultilineWithEllipsis) {
  const std::string_view kProgram = R"(pub const VALS = u64[8]:[
    0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0a447a02823ad868,
    0x1df74948b3fbea7e, 0x1bc8b594bcf01a39, 0x07b767ca9520e99a, ...
];
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ConstantDefArrayEllipsis) {
  const std::string_view kProgram = "pub const VALS = u32[2]:[32, ...];\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, EnumDefTwoValues) {
  const std::string_view kInputProgram = "pub enum MyEnum:u32{A=1,B=2}\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kInputProgram, "fake.x", "fake", &comments));

  const std::string_view kWant = R"(pub enum MyEnum : u32 {
    A = 1,
    B = 2,
}
)";
  std::string got_multiline = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got_multiline, kWant);
}

TEST(ModuleFmtTest, StructDefTwoFields) {
  const std::string_view kProgram =
      "pub struct Point<N: u32> { x: bits[N], y: u64 }\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  // At normal 100 char width it can be in single line form.
  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }

  const std::string_view kWantMultiline = R"(pub struct Point<N: u32> {
    x: bits[N],
    y: u64,
}
)";
  std::string got_multiline =
      AutoFmt(*m, Comments::Create(comments), /*text_width=*/32);
  EXPECT_EQ(got_multiline, kWantMultiline);
}

TEST(ModuleFmtTest, StructDefTwoParametrics) {
  const std::string_view kProgram =
      "pub struct Point<M: u32, N: u32> { x: bits[M], y: bits[N] }\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  // At normal 100 char width it can be in single line form.
  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }

  const std::string_view kWantMultiline = R"(pub struct Point<M: u32, N: u32> {
    x: bits[M],
    y: bits[N],
}
)";
  std::string got_multiline =
      AutoFmt(*m, Comments::Create(comments), /*text_width=*/35);
  EXPECT_EQ(got_multiline, kWantMultiline);
}

TEST(ModuleFmtTest, SimpleTestFunction) {
  const std::string_view kProgram =
      R"(fn id(x: u32) -> u32 { x }

#[test]
fn my_test() {
    assert_eq(id(u32:64), u32:64);
    assert_eq(id(u32:128), u32:128);
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, SimpleParametricInvocation) {
  const std::string_view kProgram =
      R"(fn p<N: u32>(x: bits[N]) -> bits[N] { x }

fn f() -> u8 { p<8>(u8:42) }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, TypeRefTypeAnnotation) {
  const std::string_view kProgram =
      R"(type MyU32 = u32;

fn f() -> MyU32 { MyU32:42 }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, ColonRefWithImportSubject) {
  const std::string_view kProgram =
      R"(import foo

fn f() -> u32 { foo::bar }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, NestedColonRefWithImportSubject) {
  const std::string_view kProgram =
      R"(import foo

fn f() -> u32 { foo::bar::baz::bat }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, ModuleLevelConstAssert) {
  const std::string_view kProgram =
      R"(import foo

const_assert!(foo::bar == u32:42);
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, ConstStructInstance) {
  const std::string_view kProgram =
      R"(struct Point { x: u32, y: u32 }

const P = Point { x: u32:42, y: u32:64 };
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, ConstStructInstanceEmpty) {
  const std::string_view kProgram =
      R"(struct Nothing {}

const NOTHING = Nothing {};
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, StructAttr) {
  const std::string_view kProgram =
      R"(struct Point { x: u32, y: u32 }

fn get_x(p: Point) -> u32 { p.x }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, ConstStructInstanceWithSplatVariantOneUpdate) {
  const std::string_view kProgram =
      R"(struct Point { x: u32, y: u32 }

const P = Point { x: u32:42, y: u32:64 };

const Q = Point { x: u32:32, ..P };
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, ConstStructInstanceWithSplatVariantNoUpdate) {
  const std::string_view kProgram =
      R"(struct Point { x: u32, y: u32 }

const P = Point { x: u32:42, y: u32:64 };

const Q = Point { ..P };
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));

  {
    std::string got = AutoFmt(*m, Comments::Create(comments));
    EXPECT_EQ(got, kProgram);
  }
}

TEST(ModuleFmtTest, SimpleQuickCheck) {
  const std::string_view kProgram =
      R"(#[quickcheck]
fn f() -> bool { true }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimplePublicFunction) {
  const std::string_view kProgram =
      R"(pub fn id(x: u32) -> u32 { x }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, OneModuleLevelCommentNoReflow) {
  const std::string_view kProgram =
      R"(// This is a module level comment at the top of the file.
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, TwoModuleLevelCommentsNoReflow) {
  const std::string_view kProgram =
      R"(// This is a module level comment at the top of the file.

// This is another one slightly farther down in the file.
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, OneMultiLineCommentNoReflow) {
  const std::string_view kProgram =
      R"(// This is a module level comment at the top of the file.
// It spans multiple lines in a single block of comment text.
// Three, to be precise. And then the file ends.
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, OneMultiLineCommentWithAnEmptyLineNoReflow) {
  const std::string_view kProgram =
      R"(// This is a module level comment at the top of the file.
//
// There's a blank on the second line. And then the file ends after the third.
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

// TODO(leary): 2023-10-12 This is not working because of a quirk in the
// PrefixedReflow command in the pretty printer -- fix and re-enable.
TEST(ModuleFmtTest, DISABLED_OneOverlongCommentLineWithOneToken) {
  const std::string_view kProgram =
      R"(// abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ModuleAndFunctionLevelComments) {
  const std::string_view kProgram =
      R"(// This is a module level comment at the top of the file.

// This is a function level comment.
fn f(x: u32) -> u32 { x }

// This is another function level comment.
fn g(x: u32) -> u32 {
    let y = x + u32:1;
    y
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, TwoModuleLevelCommentBlocksBeforeFunction) {
  const std::string_view kProgram =
      R"(// Module comment one.

// Module comment two.

fn uncommented_fn(x: u32) -> u32 { x }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

}  // namespace
}  // namespace xls::dslx

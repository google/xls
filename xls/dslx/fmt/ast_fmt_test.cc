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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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

TEST(BuildAstFmtTest, FormatSingleElementTuple) {
  auto [module, tuple] =
      MakeNElementTupleExpression(1, /*has_trailing_comma=*/true);
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*tuple, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x0,)");
}

TEST(BuildAstFmtTest, FormatShortTupleWithoutTrailingComma) {
  auto [module, tuple] =
      MakeNElementTupleExpression(2, /*has_trailing_comma=*/false);
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*tuple, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x0, x1)");
}

TEST(BuildAstFmtTest, FormatShortTupleWithTrailingComma) {
  auto [module, tuple] =
      MakeNElementTupleExpression(2, /*has_trailing_comma=*/true);
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*tuple, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x0, x1)");
}

TEST(BuildAstFmtTest, FormatLongTupleShouldTrailingComma) {
  auto [module, tuple] =
      MakeNElementTupleExpression(40, /*has_trailing_comma=*/true);
  const Comments empty_comments = Comments::Create({});

  DocArena arena;
  DocRef doc = Fmt(*tuple, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100),
            R"((
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39,
))");
  auto [_, tuple_without_trailing_comma] =
      MakeNElementTupleExpression(40, /*has_trailing_comma=*/false);

  doc = Fmt(*tuple_without_trailing_comma, empty_comments, arena);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100),
            R"((
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39,
))");
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
  DocRef doc = FmtStatement(*stmt, comments, arena, /*trailing_semi=*/false);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "let x: u32 = u32:42");
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
      const absl::flat_hash_set<std::string>& builtin_name_defs = {},
      int64_t text_width = 100) {
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
    return PrettyPrint(arena_, doc, text_width);
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

TEST_F(FunctionFmtTest, FormatLetYEqualsXWithTypeFn) {
  const std::string_view original = "fn f(x:u32)->u32{let y:u32=x;y}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u32 {
    let y: u32 = x;
    y
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, FormatLetWithLongRhs) {
  const std::string_view original =
      R"(fn f(x: u32, partial_narrowed_result: u32) -> bool {
    let overflow_narrowed_result_upper_sum = false;
    let overflow_detected = or_reduce(x) || or_reduce(x) || or_reduce(x) || or_reduce(x) ||
                            or_reduce(partial_narrowed_result[x as s32:]) ||
                            overflow_narrowed_result_upper_sum;
    overflow_detected
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           DoFmt(original, {"or_reduce"}, 100));
  EXPECT_EQ(got, original);
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

TEST_F(FunctionFmtTest, ConditionalWithUnnecessaryParens) {
  const std::string_view original =
      "fn f(a:u32,b:u32)->u32{if(a<b){a}else if(b<a){b}else{a}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(a: u32, b: u32) -> u32 {
    if a < b {
        a
    } else if b < a {
        b
    } else {
        a
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

TEST_F(FunctionFmtTest, MatchWithCommentsOnArms) {
  const std::string_view original = R"(fn f(b:bool)->u32{match b{
  // comment on first arm
  true|false=>u32:42,
  // comment on second arm
  _=>u32:64
  }
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(b: bool) -> u32 {
    match b {
        // comment on first arm
        true | false => u32:42,
        // comment on second arm
        _ => u32:64,
    }
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, MatchWithInlineCommentsOnArms) {
  const std::string_view original = R"(fn f(b:bool)->u32{match b{
  true|false=>u32:42,// comment on first arm
  _=>u32:64,// comment on second arm
  }
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(b: bool) -> u32 {
    match b {
        true | false => u32:42,  // comment on first arm
        _ => u32:64,  // comment on second arm
    }
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, MatchWithCommentOnNonBlockExpression) {
  const std::string_view original = R"(fn f(x: u32) -> u32 {
    match x {
        _ =>  // some comment on conditional
            if true { u32:42 } else { u32:64 },
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

// As above but the comment starts on the next line -- we canonicalize it to be
// next to the arrow as above.
TEST_F(FunctionFmtTest, MatchWithCommentOnNonBlockExpressionSecondaryLine) {
  const std::string_view original = R"(fn f(x: u32) -> u32 {
    match x {
        _ =>
            // some comment on conditional
            if true { u32:42 } else { u32:64 },
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want = R"(fn f(x: u32) -> u32 {
    match x {
        _ =>  // some comment on conditional
            if true { u32:42 } else { u32:64 },
    }
})";
  EXPECT_EQ(got, want);
}

// Note that the parametrics for the invocation don't fit on the same line with
// the identifier.
TEST_F(FunctionFmtTest, MatchWithOverLongRhs) {
  const std::string_view original = R"(fn f(yyyyyyyyyyyyyy: u32) {
    match yyyyyyyyyyyyyy {
        _ => fffffffffffffffffffffffffffffffff<
            u32:7, AAAAAAAAAAAAAAAAAAAAA, BBBBBBBBBBB, CCCCCCCCCC>(
            yyyyyyyyyyyyyy),
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string got,
      DoFmt(original, {"fffffffffffffffffffffffffffffffff",
                       "AAAAAAAAAAAAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCC"}));
  EXPECT_EQ(got, original);
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

TEST_F(FunctionFmtTest, FunctionWithCommentsOnLastLines) {
  const std::string_view original =
      R"(fn f() {
    let x = u32:42;
    let y = x + x;

    // I like to put comments
    // down here at the end...
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, ParametricFunction) {
  const std::string_view original =
      R"(fn concat3<X: u32, Y: u32, Z: u32, R: u32 = {X + Y + Z}>
    (x: bits[X], y: bits[Y], z: bits[Z]) -> bits[R] {
    x ++ y ++ z
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, LetWithCommentsInBlock) {
  const std::string_view original =
      R"(fn f() -> u32 {
    let x = {
        // A comment on the top statement
        let y = u32:42 + u32:64;
        // And a comment on the second statement.
        y
        // And a comment on the last line!
    };
    x
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

// Shows that when we separate statement chunks with a newline (i.e. in the
// style of paragraphs) it is retained.
TEST_F(FunctionFmtTest, FunctionWithParagraphStyleCodeChunks) {
  const std::string_view original =
      R"(fn f() -> u32 {
    // A comment on the top statement
    let y = u32:42 + u32:64;

    // And a comment on the second statement.
    y
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest,
       FunctionWithParagraphStyleCodeChunksIntermediateCommentParagraph) {
  const std::string_view original =
      R"(fn f() -> u32 {
    // A comment on the top statement
    let y = u32:42 + u32:64;

    // A trailing paragraph-style comment on this first chunk.
    // This could be multiple lines.

    // And a comment associated with the second statement.
    y
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, FunctionWithParagraphStyleTrailingComment) {
  const std::string_view original =
      R"(fn f() -> u32 {
    // A comment on the top statement
    let y = u32:42 + u32:64;

    // A trailing comment in its own "paragraph".
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, FunctionWithParagraphStyleChunksOfStatements) {
  const std::string_view original =
      R"(fn f() -> u32 {
    let y = u32:42 + u32:64;
    let z = y * y;

    let a = z - y;
    let b = a * a;
    b
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, FunctionWithStmtThatNeedsReflow) {
  const std::string_view original =
      R"(fn f() {
    assert_eq(
        aaaaaaaaaaa,
        ffffffffffffffff(xxxxxxxxxxx, yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy, zzzzzzz));
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string got,
      DoFmt(original,
            {"assert_eq", "aaaaaaaaaaa", "ffffffffffffffff", "xxxxxxxxxxx",
             "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", "zzzzzzz"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, FunctionWithSmallInlineArrayLiteral) {
  const std::string_view original =
      R"(fn f() { let arr = map([u2:1, u2:2], self_append); })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           DoFmt(original, {"map", "self_append"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, CharLiteral) {
  const std::string_view original = R"(fn f() -> u8 { u8:'!' })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, CharLiteralTab) {
  const std::string_view original = R"(fn f() -> u8 { u8:'\t' })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, CharLiteralNul) {
  const std::string_view original = R"(fn f() -> u8 { u8:'\0' })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, LetWithInlineCommentThatFits) {
  const std::string_view original =
      R"(fn f() {
    // some comment immediately above
    // that is multi-line
    let (a_wide, b_wide) = (a as uN[WIDTH + u32:1], b as uN[WIDTH + u32:1]);  // room for carry
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           DoFmt(original, {"a", "b", "WIDTH"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, LetWithInlineCommentThatDoesNotFit) {
  const std::string_view original =
      R"(fn f() {
    let (a_wide, b_wide) = (a as uN[WIDTH + u32:1], b as uN[WIDTH + u32:1]);  // blah blah blah blah blah
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           DoFmt(original, {"a", "b", "WIDTH"}));
  const std::string_view want =
      R"(fn f() {
    let (a_wide, b_wide) = (a as uN[WIDTH + u32:1], b as uN[WIDTH + u32:1]);  // blah blah blah blah
                                                                              // blah
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, LetWithInlineCommentAndStatementOnSubsequentLine) {
  const std::string_view original =
      R"(fn f() -> u32 {
    let a = u32:42;  // May be the meaning of life.

    let b = a + a;
    b
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, MultiLineTernary) {
  const std::string_view original =
      R"(fn f() {
    let dddddddddddddddddddddddddddddddddddddddddddddddddddd = if ccccccccccccccccccccccc {
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    } else {
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    };
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string got,
      DoFmt(original, {"ccccccccccccccccccccccc",
                       "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                       "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, CommentParagraphThenStatement) {
  const std::string_view original =
      R"(fn f() {
    // I am an explainer at the top of the function.

    let x = u32:42;
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, LetRhsIsOverlongFor) {
  const std::string_view original =
      R"(fn f() {
    let (_, _, _, div_result) = for
        (idx, (shifted_y, shifted_index_bit, running_product, running_result)) in
        range(u32:0, REALLY_LONG_NAME_HERE) {
        idx
    }((
        (y as uN[DN]) << (init_shift_amount as uN[DN]), uN[N]:1 << init_shift_amount, uN[DN]:0,
        uN[N]:0,
    ));
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string got, DoFmt(original, {"range", "REALLY_LONG_NAME_HERE", "y",
                                        "DN", "init_shift_amount", "N"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, ForWithArrayInitValue) {
  const std::string_view original =
      R"(fn f() -> bool[4] {
    for (idx, accum) in range(u32:0, REALLY_LONG_NAME_HERE) {
        accum
    }(bool[4]:[true, false, true, false]);
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           DoFmt(original, {"range", "REALLY_LONG_NAME_HERE"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, ParametricInvocationWithExpression) {
  const std::string_view original = R"(fn f() { p<A, u32:42, {A + B}>() })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"p", "A", "B"}));
  EXPECT_EQ(got, original);
}

// The semicolon at the end of the let RHS pushes this over 100 chars.
TEST_F(FunctionFmtTest, LetWith100Chars) {
  const std::string_view original = R"(fn f(integer_part: s9) {
    let integer_part =
        if input_fraction_in_upper_half { integer_part + s9:1 } else { integer_part };
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           DoFmt(original, {"input_fraction_in_upper_half"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, SliceExprOverlong) {
  const std::string_view original = R"(fn f() {
    if a {
        if b {
            let middle_bits = upper_bits ++
                              x[smin(from_inclusive as s32 - fixed_shift as s32, N as s32)
                                :smin(to_exclusive as s32 - fixed_shift as s32, N as s32)];
        } else {
            ()
        }
    } else {
        ()
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string got,
      DoFmt(original, {"a", "b", "x", "smin", "upper_bits", "from_inclusive",
                       "to_exclusive", "fixed_shift", "N"}));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, InlineBlockExpression) {
  const std::string_view original = R"(fn f() -> u32 {
    let x = { u32:42 };
    x
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
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

TEST(ModuleFmtTest, ImportAs) {
  const std::string_view kProgram = "import foo as bar\n";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ImportGroups) {
  const std::string_view kProgram = R"(import thing1
import thing2

import other
import stuff
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ImportSuperLongName) {
  const std::string_view kProgram = R"(// Module-level comment
import blahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
    as blah
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, TypeAliasGroups) {
  const std::string_view kProgram = R"(import thing1
import float32

type F32 = float32::F32;
type FloatTag = float32::FloatTag;

type TaggedF32 = float32::TaggedF32;
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ConstantDefGroups) {
  const std::string_view kProgram = R"(const A = u32:42;
const B = u32:64;

const C = u32:128;
const D = u32:256;
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
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

// We want these arrays to not have e.g. extra newlines introduced between them,
// since they are abutted.
TEST(ModuleFmtTest, ConstantDefMultipleArray) {
  const std::string_view kProgram = R"(// Module level comment.
const W_A0 = u32:32;
const W_A1 = u32:32;
const W_A2 = u32:32;
const NUM_PIECES = u32:16;

pub const A2 = sN[W_A2][NUM_PIECES]:[
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 111, 111, 111, 111, 111,
];
pub const A1 = sN[W_A1][NUM_PIECES]:[
    1111, 1111, 1111, 1111, 1111, 1111, 1111, 1111, 1111, 1111, 1111, 1111, 1111, 11111, 11111,
    11111,
];
pub const A0 = sN[W_A0][NUM_PIECES]:[
    111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111,
    111111, 111111, 111111, 111111,
];
)";
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

TEST(ModuleFmtTest, EnumDefCommentOnEachMember) {
  const std::string_view kInputProgram = R"(pub enum MyEnum:u32{
// This is the first member comment.
FIRST = 0,
// This is the second member comment.
SECOND = 1,
// This is a trailing comment.
})";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> m,
      ParseModule(kInputProgram, "fake.x", "fake", &comments));

  const std::string_view kWant = R"(pub enum MyEnum : u32 {
    // This is the first member comment.
    FIRST = 0,
    // This is the second member comment.
    SECOND = 1,
    // This is a trailing comment.
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

TEST(ModuleFmtTest, SimpleTestFunctionWithLeadingComment) {
  const std::string_view kProgram =
      R"(fn id(x: u32) -> u32 { x }

// This is a test function. Now you know.
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

TEST(ModuleFmtTest, SimpleParametricStructInstantiation) {
  const std::string_view kProgram =
      R"(import mol

struct Point<N: u32> { x: bits[N], y: bits[N] }

fn f() -> Point<mol::MOL> { Point<mol::MOL> { x: u8:42, y: u8:64 } }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, TypeRefTypeAnnotationModuleLevel) {
  constexpr std::string_view kProgram =
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

TEST(ModuleFmtTest, TypeRefTypeAnnotationInBody) {
  constexpr std::string_view kProgram =
      R"(fn f() -> u32 {
    type MyU32 = u32;
    MyU32:42
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

TEST(ModuleFmtTest, TypeRefChannelTypeAnnotation) {
  constexpr std::string_view kProgram =
      R"(type MyChan = chan<u32> out;
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

TEST(ModuleFmtTest, StructInstanceWithNamesViaBindings) {
  const std::string_view kProgram =
      R"(struct Point { x: u32, y: u16 }

fn f() {
    let x = u32:42;
    let y = u16:64;
    Point { x, y }
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

TEST(ModuleFmtTest, StructInstanceWithNamesViaBindingsBackwards) {
  const std::string_view kProgram =
      R"(struct Point { x: u32, y: u16 }

fn f() {
    let x = u32:42;
    let y = u16:64;
    Point { y, x }
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

TEST(ModuleFmtTest, OneOverlongCommentLineWithOneToken) {
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

TEST(ModuleFmtTest, SimpleProc) {
  const std::string_view kProgram =
      R"(pub proc p {
    config() { () }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimpleProcWithMembers) {
  const std::string_view kProgram =
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config(cin: chan<u32> in, cout: chan<u32> out) { (cin, cout) }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimpleParametricProc) {
  const std::string_view kProgram =
      R"(pub proc p<N: u32> {
    config() { () }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimpleProcWithChannelDecl) {
  const std::string_view kProgram =
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config() {
        let (cin, cout) = chan<u32>;
        (cin, cout)
    }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimpleProcWithChannelDeclWithFifoDepth) {
  const std::string_view kProgram =
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config() {
        let (cin, cout) = chan<u32, u32:4>;
        (cin, cout)
    }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimpleProcWithSpawn) {
  const std::string_view kProgram =
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config(cin: chan<u32> in, cout: chan<u32> out) { (cin, cout) }

    init { () }

    next(tok: token, state: ()) { () }
}

pub proc q {
    config() {
        let (cin, cout) = chan<u32, u32:4>;
        spawn p(cin, cout);
        ()
    }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

// Based on report in https://github.com/google/xls/issues/1216
TEST(ModuleFmtTest, ProcSpawnImported) {
  const std::string_view kProgram =
      R"(import some_import

proc p {
    config() {
        spawn some_import::some_proc();
        ()
    }

    init { () }

    next(tok: token, state: ()) { () }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, SimpleTestProc) {
  constexpr std::string_view kProgram =
      R"(#[test_proc]
proc p_test {
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) { (terminator,) }

    init { () }

    next(tok: token, state: ()) { send(tok, terminator, true); }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, MatchLongWildcardArmExpression) {
  const std::string_view kProgram =
      R"(import float32

fn f(input_float: float32::F32) -> float32::F32 {
    match f.bexp {
        _ => float32::F32 {
            sign: input_float.sign, bexp: input_float.bexp - u8:1, fraction: input_float.fraction
        },
    }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ProcCallFarRhs) {
  const std::string_view kProgram =
      R"(struct DelayState {}

const DELAY = u32:42;
const DATA_WIDTH = u8:42;

fn eq() {}

proc p {
    config() { () }

    init { () }

    next(tok: token, state: DelayState) {
        let data_in = ();
        let (recv_tok, input_data, data_in_valid) =
            recv_if_non_blocking(tok, data_in, !eq(state.occupancy, DELAY), uN[DATA_WIDTH]:0);
    }
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ParametricFnWithManyArgs) {
  const std::string_view kProgram =
      R"(fn umax() {}

pub fn uadd_with_overflow
    <V: u32, N: u32, M: u32, MAX_N_M: u32 = {umax(N, M)}, MAX_N_M_V: u32 = {umax(MAX_N_M, V)}>
    (x: uN[N], y: uN[M]) -> (bool, uN[V]) {
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, TypeAliasToColonRefInstantiated) {
  const std::string_view kProgram =
      R"(import float32

type F32 = float32::F32;

pub fn f() -> F32 { F32 { blah: u32:42 } }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, AttrEquality) {
  const std::string_view kProgram =
      R"(import m

const SOME_BOOL = true;

fn f(x: m::MyStruct, y: m::MyStruct) -> bool { (x.foo == y.foo) || SOME_BOOL }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ArrowReturnTypePackedOnOneLine) {
  const std::string_view kProgram =
      R"(import apfloat

fn n_path<EXP_SZ: u32, FRACTION_SZ: u32>
    (a: apfloat::APFloat<EXP_SZ, FRACTION_SZ>, b: apfloat::APFloat<EXP_SZ, FRACTION_SZ>)
    -> (apfloat::APFloat<EXP_SZ, FRACTION_SZ>, bool) {
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, LongParametricList) {
  const std::string_view kProgram = R"(// Signed max routine.
pub fn smax<N: u32>(x: sN[N], y: sN[N]) -> sN[N] { if x > y { x } else { y } }

pub fn extract_bits
    <from_inclusive: u32, to_exclusive: u32, fixed_shift: u32, N: u32,
     extract_width: u32 = {smax(s32:0, to_exclusive as s32 - from_inclusive as s32) as u32}>
    (x: uN[N]) -> uN[extract_width] {
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, NestedBinopLogicalOr) {
  const std::string_view kProgram =
      R"(// Define some arbitrary constants at various identifier widths.
const AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA = true;
const BBBBBBBBBBBBBBBBBBBBBBB = true;
const CCCCCCCCCCCCCCCCCCCCCCC = true;

fn f() -> bool {
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ||
    BBBBBBBBBBBBBBBBBBBBBBB || CCCCCCCCCCCCCCCCCCCCCCC
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ModuleConstantsWithInlineComments) {
  const std::string_view kProgram =
      R"(pub const MOL = u32:42;  // may be important

const TWO_TO_FIFTH = u32:32;  // 2^5
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

// If the array constant is placed on a single line it is overly long -- we
// check that we smear it across multiple lines to stay within 100 chars.
TEST(ModuleFmtTest, OverLongArrayConstant) {
  const std::string_view kProgram = R"(// Top of module comment.
const W_A0 = u32:32;
const NUM_PIECES = u32:8;
pub const A0 = sN[W_A0][NUM_PIECES]:[
    111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111,
];
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, InvocationWithOneStructArg) {
  const std::string_view kProgram = R"(struct APFloat {}

fn unbiased_exponent() {}

fn f() {
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0 });
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

// See github issue https://github.com/google/xls/issues/1193
TEST(ModuleFmtTest, LongLetLeader) {
  const std::string_view kProgram = R"(import std

fn foo(some_value_that_is_pretty_long: u32, some_other_value_that_is_also_not_too_short: u32) {
    type SomeTypeNameThatIsNotTooShort = s64;
    let very_somewhat_long_variable_name: SomeTypeNameThatIsNotTooShort = std::to_signed(
        some_value_that_is_pretty_long ++ some_other_value_that_is_also_not_too_short);
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, LongLetRhs) {
  const std::string_view kProgram = R"(import std

fn foo(some_value_that_is_pretty_long: u32, some_other_value_that_is_also_not_too_short: u32) {
    type SomeTypeNameThatIsNotTooShort = sN[u32:96];
    let very_somewhat_long_variable_name: SomeTypeNameThatIsNotTooShort = std::to_signed(
        some_value_that_is_pretty_long ++ some_other_value_that_is_also_not_too_short ++
        some_value_that_is_pretty_long);
}
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, QuickcheckWithCount) {
  const std::string_view kProgram = R"(// Comment on quickcheck.
#[quickcheck(test_count=100000)]
fn prop_eq(x: u32, y: u32) -> bool { x == y }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

TEST(ModuleFmtTest, ModuleLevelAnnotation) {
  const std::string_view kProgram = R"(#![allow(nonstandard_constant_naming)]

fn id(x: u32) { x }
)";
  std::vector<CommentData> comments;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m,
                           ParseModule(kProgram, "fake.x", "fake", &comments));
  std::string got = AutoFmt(*m, Comments::Create(comments));
  EXPECT_EQ(got, kProgram);
}

}  // namespace
}  // namespace xls::dslx

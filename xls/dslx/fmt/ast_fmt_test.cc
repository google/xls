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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_test_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(BuiltAstFmtTest, FormatCastThatNeedsParens) {
  auto [file_table, module, lt] = MakeCastWithinLtComparison();
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*lt);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x as t) < x");
}

TEST(BuiltAstFmtTest, FormatIndexThatNeedsParens) {
  auto [file_table, module, index] = MakeCastWithinIndexExpression();
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*index);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x as u32[42])[i]");
}

TEST(BuiltAstFmtTest, FormatTupleIndexThatNeedsParens) {
  auto [file_table, module, tuple_index] =
      MakeIndexWithinTupleIndexExpression();
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*tuple_index);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x[i]).2");
}

TEST(BuiltAstFmtTest, FormatSingleElementTuple) {
  auto [file_table, module, tuple] =
      MakeNElementTupleExpression(1, /*has_trailing_comma=*/true);
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*tuple);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x0,)");
}

TEST(BuiltAstFmtTest, FormatShortTupleWithoutTrailingComma) {
  auto [file_table, module, tuple] =
      MakeNElementTupleExpression(2, /*has_trailing_comma=*/false);
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*tuple);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x0, x1)");
}

TEST(BuiltAstFmtTest, FormatShortTupleWithTrailingComma) {
  auto [file_table, module, tuple] =
      MakeNElementTupleExpression(2, /*has_trailing_comma=*/true);
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*tuple);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x0, x1)");
}

TEST(BuiltAstFmtTest, FormatLongTupleShouldAddTrailingComma) {
  Comments empty_comments = Comments::Create({});
  {
    auto [file_table, module, tuple] =
        MakeNElementTupleExpression(40, /*has_trailing_comma=*/true);

    DocArena arena(file_table);
    DocRef doc = Formatter(empty_comments, arena).Format(*tuple);
    EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100),
              R"((
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39,
))");
  }

  {
    auto [file_table, module, tuple_without_trailing_comma] =
        MakeNElementTupleExpression(40, /*has_trailing_comma=*/false);

    DocArena arena(file_table);
    DocRef doc =
        Formatter(empty_comments, arena).Format(*tuple_without_trailing_comma);
    EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100),
              R"((
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39,
))");
  }
}

TEST(BuiltAstFmtTest, FormatUnopThatNeedsParensOnOperand) {
  auto [file_table, module, unop] = MakeCastWithinNegateExpression();
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*unop);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "-(x as u32)");
}

TEST(BuiltAstFmtTest, FormatAttrThatNeedsParensOnOperand) {
  auto [file_table, module, attr] = MakeArithWithinAttrExpression();
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena).Format(*attr);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "(x * y).my_attr");
}

TEST(AstFmtTest, FormatLet) {
  FileTable file_table;
  Scanner s{file_table, Fileno(0), "{ let x: u32 = u32:42; }"};
  Parser p("fake", &s);
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(StatementBlock * block,
                           p.ParseBlockExpression(bindings));
  Statement* stmt = block->statements().at(0);

  Comments comments = Comments::Create(s.comments());

  DocArena arena(file_table);
  Formatter fmt(comments, arena);
  DocRef doc = fmt.Format(*stmt, /*trailing_semi=*/false);
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/100), "let x: u32 = u32:42");
}

TEST(AstFmtTest, FormatVerbatimNodeTop) {
  FileTable file_table;
  Module m("test", /*fs_path=*/std::nullopt, file_table);

  const std::string verbatim_text = "anything // goes\n  even here";
  VerbatimNode verbatim(&m, Span(), verbatim_text);
  XLS_ASSERT_OK(m.AddTop(&verbatim, /*make_collision_error=*/nullptr));
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  XLS_ASSERT_OK_AND_ASSIGN(DocRef doc,
                           Formatter(empty_comments, arena).Format(m));

  // Intentionally small text width, should still be formatted verbatim.
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/10), verbatim_text);
}

TEST(AstFmtTest, FormatVerbatimNodeStatement) {
  FileTable file_table;
  Module m("test", /*fs_path=*/std::nullopt, file_table);

  const std::string verbatim_text = "anything // goes\n  even here";
  VerbatimNode verbatim(&m, Span(), verbatim_text);
  Statement statement(&m, &verbatim);
  Comments empty_comments = Comments::Create({});

  DocArena arena(file_table);
  DocRef doc = Formatter(empty_comments, arena)
                   .Format(statement, /*trailing_semi=*/false);

  // Intentionally small text width, should still be formatted verbatim.
  EXPECT_EQ(PrettyPrint(arena, doc, /*text_width=*/10), verbatim_text);
}

// Fixture for test that format entire (single) functions -- expected usage:
//
//  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt("fn ..."));
//  EXPECT_EQ(got, ...);
//
// Use `ModuleFmtTest` for formatting of entire modules.
class FunctionFmtTest : public testing::Test {
 public:
  FunctionFmtTest() : arena_(file_table_) {}

  // Args:
  //  original: The text to parse-then-auto-format.
  //  builtin_name_defs: Names to initialize in the bindings as built-in.
  absl::StatusOr<std::string> DoFmt(
      std::string_view original,
      const absl::flat_hash_set<std::string>& builtin_name_defs = {},
      bool opportunistic_postcondition = true) {
    CHECK(!scanner_.has_value());
    scanner_.emplace(file_table_, Fileno(0), std::string{original});
    parser_.emplace("fake", &scanner_.value());

    for (const std::string& name : builtin_name_defs) {
      bindings_.Add(name, parser_->module().GetOrCreateBuiltinNameDef(name));
    }

    XLS_ASSIGN_OR_RETURN(
        f_, parser_->ParseFunction(Pos(), /*is_public=*/false,
                                   /*is_test_utility=*/false, bindings_));
    Comments comments = Comments::Create(scanner_->comments());

    DocRef doc = Formatter(comments, arena_).Format(*f_);
    std::string formatted = PrettyPrint(arena_, doc, kDslxDefaultTextWidth);

    std::optional<AutoFmtPostconditionViolation> maybe_violation =
        ObeysAutoFmtOpportunisticPostcondition(original, formatted);
    if (maybe_violation.has_value() && opportunistic_postcondition) {
      LOG(ERROR) << "= original";
      XLS_LOG_LINES(ERROR, original);
      LOG(ERROR) << "= autofmt";
      XLS_LOG_LINES(ERROR, formatted);
      LOG(ERROR) << "= original (transformed)";
      XLS_LOG_LINES(ERROR, maybe_violation->original_transformed);
      LOG(ERROR) << "= autofmt (transformed)";
      XLS_LOG_LINES(ERROR, maybe_violation->autofmt_transformed);
      return absl::InternalError(
          "Sample did not obey auto-formatting postcondition");
    }

    return formatted;
  }

  absl::StatusOr<std::string> DoFmtNoPostcondition(std::string_view original) {
    return DoFmt(original, {}, /*opportunistic_postcondition=*/false);
  }

  Bindings& bindings() { return bindings_; }

 private:
  FileTable file_table_;
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

TEST_F(FunctionFmtTest, LogicalOrLhsForArrayIndex) {
  const std::string_view original = "fn f(a: u32, b: u32){(a||b)[3]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, "fn f(a: u32, b: u32) { (a || b)[3] }");
}

TEST_F(FunctionFmtTest, LogicalOrLhsForAttrIndex) {
  const std::string_view original = "fn f(a: u32, b: u32){(a||b).attr[0]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, "fn f(a: u32, b: u32) { (a || b).attr[0] }");
}

TEST_F(FunctionFmtTest, LogicalOrLhsForAttr) {
  const std::string_view original = "fn f(a: u32, b: u32){(a||b).attr}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, "fn f(a: u32, b: u32) { (a || b).attr }");
}

TEST_F(FunctionFmtTest, AttrIndexChain) {
  const std::string_view original =
      "fn f(a: u32, b: u32, c: u32){a[0][1].b[2][3].c[4]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmtNoPostcondition(original));
  EXPECT_EQ(got, "fn f(a: u32, b: u32, c: u32) { a[0][1].b[2][3].c[4] }");
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
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmtNoPostcondition(original));
  const std::string_view want = R"(fn f(x: u32) -> u64 { (x ++ x) as u64 })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, DoubleParensLhsOfIndex) {
  const std::string_view original = "fn f(x:u32[2],i:u32)->u32{((x++x))[i]}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmtNoPostcondition(original));
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

TEST_F(FunctionFmtTest, ConditionalWithElseIfAfterLet) {
  const std::string_view original =
      R"(fn f(a: bool[2], x: u32[3]) -> u32 {
    let result = if a[0] {
        x[0]
    } else if a[1] {
        x[1]
    } else {
        x[2]
    };
    result
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, ConditionalWithoutElseShort) {
  const std::string_view original =
      R"(fn f(x:bool){if x {trace_fmt!("Hello DSLX");};})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"trace_fmt!"}));
  const std::string_view want =
      R"(fn f(x: bool) { if x { trace_fmt!("Hello DSLX"); }; })";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConditionalWithoutElseMultiStatementCausesHardBreaks) {
  const std::string_view original =
      R"(fn f(x:bool){if x{trace_fmt!("Hello");trace_fmt!("DSLX");};})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"trace_fmt!"}));
  const std::string_view want =
      R"(fn f(x: bool) {
    if x {
        trace_fmt!("Hello");
        trace_fmt!("DSLX");
    };
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConditionalWithElseIfButWithoutElse) {
  const std::string_view original =
      R"(fn f(a:bool[2]){if a[0]{trace_fmt!("A0");}else if a[1]{trace_fmt!("A1");};})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"trace_fmt!"}));
  const std::string_view want =
      R"(fn f(a: bool[2]) {
    if a[0] {
        trace_fmt!("A0");
    } else if a[1] {
        trace_fmt!("A1");
    };
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, ConditionalWithUnnecessaryParens) {
  const std::string_view original =
      "fn f(a:u32,b:u32)->u32{if(a<b){a}else if(b<a){b}else{a}}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmtNoPostcondition(original));
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

TEST_F(FunctionFmtTest, SimpleUnrollForOneStatementNoTypeAnnotation) {
  const std::string_view original =
      "fn f(x:u32)->u32{unroll_for!(i,accum)in u32:0..u32:4{accum+i}(x)}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u32 {
    unroll_for! (i, accum) in u32:0..u32:4 {
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

TEST_F(FunctionFmtTest, SimpleUnrollForOneStatementTypeAnnotated) {
  const std::string_view original =
      "fn f(x:u32)->u32{unroll_for!(i,accum):(u32,u32)in "
      "u32:0..u32:4{accum+i}(x)}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u32 {
    unroll_for! (i, accum): (u32, u32) in u32:0..u32:4 {
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

TEST_F(FunctionFmtTest, SimpleUnrollForLetBinding) {
  const std::string_view original =
      "fn f(x:u32)->u32{let y=unroll_for!(i,accum):(u32,u32)in "
      "u32:0..u32:4{accum+i}(x);y}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(x: u32) -> u32 {
    let y = unroll_for! (i, accum): (u32, u32) in u32:0..u32:4 {
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

TEST_F(FunctionFmtTest, SimpleLetEqualsMatchOnBool) {
  const std::string_view original =
      "fn f(b:bool)->u32{let x=match b{true=>u32:42,_=>u32:64};x}";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  const std::string_view want =
      R"(fn f(b: bool) -> u32 {
    let x = match b {
        true => u32:42,
        _ => u32:64,
    };
    x
})";
  EXPECT_EQ(got, want);
}

TEST_F(FunctionFmtTest, LongConstLetStyleBinding) {
  const std::string_view original = R"(fn f() -> u64[8] {
    const X = u64[8]:[
        0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0a447a02823ad868,
        0x1df74948b3fbea7e, 0x1bc8b594bcf01a39, 0x07b767ca9520e99a, 0x05e28b4320bfd20e,
    ];
    X
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
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

TEST_F(FunctionFmtTest, SingleStatementWithInlineComment) {
  const std::string_view original = R"(fn f() {
    ()  // inline comment here
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, SingleStatementWithMultilineInlineComment) {
  const std::string_view original = R"(fn f() {
    ()  // inline comment here
        // second half
})";
  EXPECT_THAT(DoFmt(original), IsOkAndHolds(original));
}

TEST_F(FunctionFmtTest, SingleStatementWithUnalignedMultilineInlineComment) {
  const std::string_view original = R"(fn f() {
    ()  // inline comment here
          // second half unindents
})";
  const std::string_view want = R"(fn f() {
    ()  // inline comment here
    // second half unindents
})";
  EXPECT_THAT(DoFmt(original), IsOkAndHolds(want));
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

TEST_F(FunctionFmtTest, MatchWithMultilineInlineCommentsOnArms) {
  const std::string_view original = R"(fn f(b:bool)->u32{match b{
  true|false=>u32:42,// comment on first arm
                     // continued comment on first arm.
  _=>u32:64,// comment on second arm
            // continued here.
  }
})";
  const std::string_view want =
      R"(fn f(b: bool) -> u32 {
    match b {
        true | false => u32:42,  // comment on first arm
                                 // continued comment on first arm.
        _ => u32:64,  // comment on second arm
                      // continued here.
    }
})";
  EXPECT_THAT(DoFmt(original), IsOkAndHolds(want));
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

TEST_F(FunctionFmtTest, VTraceFormatMacro) {
  const std::string_view original =
      R"(fn f(x:u32,y:u32){vtrace_fmt!(4, "x is {} y is {}",x,y)})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original, {"vtrace_fmt!"}));
  const std::string_view want =
      R"(fn f(x: u32, y: u32) { vtrace_fmt!(4, "x is {} y is {}", x, y) })";
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

TEST_F(FunctionFmtTest,
       FunctionWithLetBindingOfSmallInlineArrayLiteralWithType) {
  const std::string_view original = R"(fn f() { let arr = u2[2]:[1, 2]; })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
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

TEST_F(FunctionFmtTest, CharLiteralSingleQuote) {
  const std::string_view original = R"(fn f() -> u8 { u8:'\'' })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, CharLiteralDoubleQuote) {
  const std::string_view original = R"(fn f() -> u8 { u8:'"' })";
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

TEST_F(FunctionFmtTest,
       LetWithMultilineInlineCommentAndStatementOnSubsequentLine) {
  const std::string_view original =
      R"(fn f() -> u32 {
    let a = u32:42;  // May be the meaning of life.
                     // but probably not.

    let b = a + a;
    b
})";
  EXPECT_THAT(DoFmt(original), IsOkAndHolds(original));
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

TEST_F(FunctionFmtTest, LetRhsIsOverLongFor) {
  const std::string_view original =
      R"(fn f() {
    let (_, _, _, div_result) =
        for (idx, (shifted_y, shifted_index_bit, running_product, running_result)) in
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

// Regression test for https://github.com/google/xls/issues/1195
TEST_F(FunctionFmtTest, TupleWithTrailingComment) {
  const std::string_view original = R"(fn foo() {
    let a = (
        u32:1,
        u32:2,
        u32:3, // after third
        // After last
    );
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, TupleWithInternalComments) {
  const std::string_view original = R"(fn foo() {
    let a = (
        u32:1,  // after first
        u32:2,  // after second
      // another after second
        u32:3,  // after third
    );
})";
  const std::string_view expected = R"(fn foo() {
    let a = (
        u32:1, // after first
        u32:2, // after second
        // another after second
        u32:3, // after third
    );
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string actual, DoFmt(original));
  EXPECT_EQ(actual, expected);
}

TEST_F(FunctionFmtTest, TupleWithComments) {
  const std::string_view original = R"(fn foo() {
    let a = (
        // Before first
        u32:1, // after first
        u32:2, // after second
        u32:3, // after third
        // After last
    );
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, original);
}

TEST_F(FunctionFmtTest, SingletonTupleWithTrailingComment) {
  const std::string_view original = R"(fn foo() {
    let a = ( u32  :  1,
    // after first
    );
})";
  const std::string_view expected = R"(fn foo() {
    let a = (
        u32:1, // after first
    );
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, expected);
}

TEST_F(FunctionFmtTest, SingletonTupleWithLeadingComment) {
  const std::string_view original = R"(fn foo() {
    let a = (// before first
    u32  :  1,      );
})";
  const std::string_view expected = R"(fn foo() {
    let a = (
        // before first
        u32:1,
    );
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string got, DoFmt(original));
  EXPECT_EQ(got, expected);
}

// -- ModuleFmtTest cases, formatting entire modules

class ModuleFmtTest : public testing::Test {
 public:
  void DoFmt(std::string input,
             std::optional<std::string_view> want = std::nullopt,
             int64_t text_width = kDslxDefaultTextWidth,
             bool opportunistic_postcondition = true) {
    std::vector<CommentData> comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Module> m,
        ParseModule(input, "fake.x", "fake", file_table_, &comments_vec));
    Comments comments = Comments::Create(comments_vec);
    AllErrorsFilesystem vfs;
    XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                             AutoFmt(vfs, *m, comments, input, text_width));

    ASSERT_EQ(got, want.value_or(input));

    if (opportunistic_postcondition) {
      std::optional<AutoFmtPostconditionViolation> maybe_violation =
          ObeysAutoFmtOpportunisticPostcondition(input, got);
      if (maybe_violation.has_value()) {
        LOG(ERROR) << "= original (transformed)";
        XLS_LOG_LINES(ERROR, maybe_violation->original_transformed);
        LOG(ERROR) << "= autofmt (transformed)";
        XLS_LOG_LINES(ERROR, maybe_violation->autofmt_transformed);
        FAIL() << "auto-formatter postcondition was violated";
      }
    }
  }

  void DoFmtNoPostcondition(std::string input,
                            std::optional<std::string_view> want) {
    DoFmt(input, want, kDslxDefaultTextWidth,
          /*opportunistic_postcondition=*/false);
  }

  void AutoFmtExpectsError(
      std::string input, std::string_view error_substr,
      absl::StatusCode code = absl::StatusCode::kInternal) {
    std::vector<CommentData> comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Module> m,
        ParseModule(input, "fake.x", "fake", file_table_, &comments_vec));
    Comments comments = Comments::Create(comments_vec);
    AllErrorsFilesystem vfs;
    EXPECT_THAT(AutoFmt(vfs, *m, comments, input, kDslxDefaultTextWidth),
                StatusIs(code, HasSubstr(error_substr)));
  }

 private:
  FileTable file_table_;
};

// See https://github.com/google/xls/issues/1617
TEST_F(ModuleFmtTest, OverlongTernary) {
  DoFmt(R"(type TypeTTTTTTTTTTTTTTTTT = u32;

const test000000000000000000000000000000 = u32:42;
const consequent000000000000000000000000000000000 = u32:64;
const alternate0000000000000000000000000 = u32:128;

fn f() {
    let output0000000000000000000000000000 =
        if test000000000000000000000000000000 == TypeTTTTTTTTTTTTTTTTT:0 {
            consequent000000000000000000000000000000000
        } else {
            alternate0000000000000000000000000
        };
}
)");
}

TEST_F(ModuleFmtTest, TwoSimpleFunctions) {
  DoFmt("fn double(x:u32)->u32{u32:2*x}fn triple(x: u32)->u32{u32:3*x}",
        R"(fn double(x: u32) -> u32 { u32:2 * x }

fn triple(x: u32) -> u32 { u32:3 * x }
)");
}

TEST_F(ModuleFmtTest, OverLongImport) {
  DoFmt("import very_long.name_here.made_of.dotted_components;",
        "import very_long.\n"
        "       name_here.\n"
        "       made_of.\n"
        "       dotted_components;\n",
        14);
}

TEST_F(ModuleFmtTest, ImportAs) { DoFmt("import foo as bar;\n"); }

TEST_F(ModuleFmtTest, ImportGroups) {
  DoFmt(R"(import thing1;
import thing2;

import other;
import stuff;
)");
}

TEST_F(ModuleFmtTest, ImportSuperLongName) {
  DoFmt(R"(// Module-level comment
import blahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
    as blah;
)");
}

TEST_F(ModuleFmtTest, TypeAliasGroups) {
  DoFmt(R"(import thing1;
import float32;

type F32 = float32::F32;
type FloatTag = float32::FloatTag;

type TaggedF32 = float32::TaggedF32;
)");
}

TEST_F(ModuleFmtTest, ConstantDefGroups) {
  DoFmt(R"(const A = u32:42;
const B = u32:64;

const C = u32:128;
const D = u32:256;
)");
}

TEST_F(ModuleFmtTest, ConstantDef) { DoFmt("pub const MOL = u32:42;\n"); }

TEST_F(ModuleFmtTest, ConstantDefLongName) {
  DoFmt(R"(pub
const
this_is_a_very_long_name_this_is_a_very_long_name_this_is_a_very_long_name_this_is_a_very_long_name
= bool:0;
)");
}

TEST_F(ModuleFmtTest, ConstantDefWithType) {
  DoFmt("pub const MOL: u32 = u32:42;\n");
}

TEST_F(ModuleFmtTest, ConstantDefWithTypeAliasTypeAnnotation) {
  DoFmt(R"(type MyU32 = u32;

pub const MOL: MyU32 = MyU32:42;
)");
}

TEST_F(ModuleFmtTest, TypeAliasSvType) {
  DoFmt(R"(#[sv_type("foo")]type MyU32 = u32;)",
        R"(#[sv_type("foo")]
type MyU32 = u32;
)");
}

TEST_F(ModuleFmtTest, ConstantDefArray) {
  DoFmt("pub const VALS = u32[2]:[32, 64];\n");
}

TEST_F(ModuleFmtTest, ConstantDefArrayMultiline) {
  DoFmt(R"(pub const VALS = u64[5]:[
    0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0a447a02823ad868,
    0x1df74948b3fbea7e, 0x1bc8b594bcf01a39, 0x07b767ca9520e99a, 0x05e28b4320bfd20e,
    0x0105906a24823f57, 0x1a1e7d14a6d24384, 0x2a7326df322e084d, 0x120bc9cc3fac4ec7,
    0x2c8f193a1b46a9c5, 0x2b9c95743bbe3f90, 0x0dcfc5b1d0398b46, 0x006ba47b3448bea3,
    0x3fe4fbf9a522891b, 0x23e1a50ad6aebca3, 0x1b263d39ea62be44, 0x13581d282e643b0e,
];
)");
}

TEST_F(ModuleFmtTest, ConstantDefArrayMultilineWithEllipsis) {
  DoFmt(R"(pub const VALS = u64[8]:[
    0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0a447a02823ad868,
    0x1df74948b3fbea7e, 0x1bc8b594bcf01a39, 0x07b767ca9520e99a, ...
];
)");
}

TEST_F(ModuleFmtTest, ConstantDefArrayEllipsis) {
  DoFmt("pub const VALS = u32[2]:[32, ...];\n");
}

// We want these arrays to not have e.g. extra newlines introduced between them,
// since they are abutted.
TEST_F(ModuleFmtTest, ConstantDefMultipleArray) {
  DoFmt(R"(// Module level comment.
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
)");
}

TEST_F(ModuleFmtTest, EnumDefTwoValues) {
  DoFmt("pub enum MyEnum:u32{A=1,B=2}\n",
        R"(pub enum MyEnum : u32 {
    A = 1,
    B = 2,
}
)");
}

TEST_F(ModuleFmtTest, EnumDefTwoValuesSvType) {
  DoFmt(R"(#[sv_type("foo")] pub enum MyEnum:u32{A=1,B=2}
)",
        R"(#[sv_type("foo")]
pub enum MyEnum : u32 {
    A = 1,
    B = 2,
}
)");
}

TEST_F(ModuleFmtTest, EnumDefCommentOnEachMember) {
  const std::string_view kWant = R"(pub enum MyEnum : u32 {
    // This is the first member comment.
    FIRST = 0,
    // This is the second member comment.
    SECOND = 1,
    // This is a trailing comment.
}
)";
  DoFmt(R"(pub enum MyEnum:u32{
// This is the first member comment.
FIRST = 0,
// This is the second member comment.
SECOND = 1,
// This is a trailing comment.
})",
        kWant);
}

TEST_F(ModuleFmtTest, FunctionRefWithExplicitParametrics) {
  DoFmt(
      R"(fn f<X: u32>() -> u32 { X }

fn g() -> u32[3] { map([u32:1, u32:2, u32:3], f<u32:4>) }
)");
}

TEST_F(ModuleFmtTest, StructDefTwoFields) {
  const std::string kInput =
      "pub struct Point<N: u32> { x: bits[N], y: u64 }\n";

  // At normal 100 char width it can be in single line form.
  DoFmt(kInput);

  DoFmt(kInput, R"(pub struct Point<N: u32> {
    x: bits[N],
    y: u64,
}
)",
        32);
}

TEST_F(ModuleFmtTest, StructDefTwoFieldsSvType) {
  const std::string kInput =
      R"(#[sv_type("cool")] pub struct Point { x: u32, y: u64 })";

  DoFmt(kInput, R"(#[sv_type("cool")]
pub struct Point { x: u32, y: u64 }
)");
}

TEST_F(ModuleFmtTest, StructDefEmptyWithComment) {
  DoFmt(
      R"(pub struct Point {
    // Very empty.
}
)");
}

TEST_F(ModuleFmtTest, StructDefWithInlineCommentsOnFields) {
  DoFmt(
      R"(pub struct Point {
    x: u32,  // Comment on the first field
    y: u64,  // Comment on the second field
}
)");
}

TEST_F(ModuleFmtTest, StructDefWithAbuttedCommentsOnFields) {
  DoFmt(R"(pub struct Point {
    // Above the first member
    x: u32,
    // Above the second member
    y: u64,
    // After the second member
}
)");
}

TEST_F(ModuleFmtTest, StructDefWithMixedCommentAnnotations) {
  DoFmt(
      R"(pub struct Point {
    x: u32,  // short inline comment
    // This has a long, long discussion for some reason.
    // It spreads over multiple lines and refers to what is below.
    y: u64,
}
)");
}

TEST_F(ModuleFmtTest, StructDefWithMultilineInlineComment) {
  DoFmt(
      R"(pub struct Point {
    x: u32,  // this is a longer comment
             // it wants to be multi-line for some reason
    y: u64,
}
)");
}

TEST_F(ModuleFmtTest, StructDefGithub1260) {
  DoFmt(
      R"(// Foos do what they do.
struct Foo {
    // the foo top of body comment
    bar: u1,  // the bar
    hop: u2,  // the hop
}
)");
}

TEST_F(ModuleFmtTest, UnaryWithCommentGithub1372) {
  DoFmt(
      R"(fn main(x: bool) -> bool {
    !x  // Gotta negate it!
}
)");
}

TEST_F(ModuleFmtTest, StructDefTwoParametrics) {
  const std::string kProgram =
      "pub struct Point<M: u32, N: u32> { x: bits[M], y: bits[N] }\n";
  DoFmt(kProgram);

  const std::string_view kWantMultiline = R"(pub struct Point<M: u32, N: u32> {
    x: bits[M],
    y: bits[N],
}
)";
  DoFmt(kProgram, kWantMultiline, 35);
}

TEST_F(ModuleFmtTest, ImplSimple) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {}
)");
}

TEST_F(ModuleFmtTest, ImplWithConstants) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    const SOME_CONST = u32:3;
    // Always include this value.
    const ANOTHER = "another";
}
)");
}

TEST_F(ModuleFmtTest, ImplWithConstantsAndFunctions) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    const SOME_CONST = u32:3;
    // Always include this value.
    const ANOTHER = "another";

    // Some function.
    fn my_function() -> u32 {
        // Do something important.
        SOME_CONST
    }

    fn second_func() -> u32 { u32:13; }

    const RESULT = my_function();
}
)");
}

TEST_F(ModuleFmtTest, ImplFunctionsAndComments) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    // First function.
    fn my_function() -> u32 {
        // Do something important.
        u32:5
    }

    // Another function.
    fn second_func() -> u32 { u32:13; }
}
)");
}

TEST_F(ModuleFmtTest, ImplWithDisabledFmtFn) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    // dslx-fmt::off
fn my_function() -> u32 {
// Do something important.
u32:3
}
// dslx-fmt::on

}
)");
}

TEST_F(ModuleFmtTest, ImplWithDisabledFmtExpr) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    fn my_function() -> u32 {
        // dslx-fmt::off
// Do something important.
u32:3
// dslx-fmt::on

    }
}
)");
}

TEST_F(ModuleFmtTest, DISABLED_ImplWithDisabledFmtImpl) {
  // This test is disabled because the impl becomes a VerbatimNode. Then the
  // cloner tries to set it into the struct as its impl, but structs cannot
  // take VerbatimNodes as their impl yet.
  DoFmt(
      R"(struct MyStruct {}

// dslx-fmt::off
impl MyStruct { fn my_function() -> u32 {
// Do something important.
u32:3
}}
// dslx-fmt::on
)");
}

TEST_F(ModuleFmtTest, ImplMethodWithVars) {
  DoFmt(R"(struct MyStruct { x: u32 }

impl MyStruct {
    fn my_function(self, a: u32, b: u32) -> u32 {
        // Do something important.
        a * b * self.x
    }
}
)");
}

TEST_F(ModuleFmtTest, ImplMethodExplicitType) {
  DoFmt(R"(struct MyStruct { x: u32 }

impl MyStruct {
    fn my_function(self: Self, a: u32, b: u32) -> u32 {
        // Do something important.
        a * b * self.x
    }
}
)");
}

TEST_F(ModuleFmtTest, ImplFunctionWithVars) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    fn my_function(a: u32, b: u32) -> u32 {
        // Do something important.
        a * b
    }
}
)");
}

TEST_F(ModuleFmtTest, ImplWithOneFunction) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    fn my_function() -> u32 {
        // Do something important.
        u32:5
    }
}
)");
}

TEST_F(ModuleFmtTest, ImplEndsWithFunction) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    const SOME_CONST = u32:3;
    // Always include this value.
    const ANOTHER = "another";

    fn my_function() -> u32 {
        // Do something important.
        SOME_CONST
    }

    fn second_func() -> u32 { u32:13; }
}
)");
}

TEST_F(ModuleFmtTest, ImplWithInlineComment) {
  DoFmt(
      R"(struct MyStruct {}

impl MyStruct {
    const SOME_CONST = u32:3;  // three
    // Something else.
    const ANOTHER = "another";
}
)");
}

TEST_F(ModuleFmtTest, SimpleTestFunction) {
  DoFmt(
      R"(fn id(x: u32) -> u32 { x }

#[test]
fn my_test() {
    assert_eq(id(u32:64), u32:64);
    assert_eq(id(u32:128), u32:128);
}
)");
}

TEST_F(ModuleFmtTest, SimpleTestFunctionWithLeadingComment) {
  DoFmt(
      R"(fn id(x: u32) -> u32 { x }

// This is a test function. Now you know.
#[test]
fn my_test() {
    assert_eq(id(u32:64), u32:64);
    assert_eq(id(u32:128), u32:128);
}
)");
}

TEST_F(ModuleFmtTest, SimpleTestUtilityFunction) {
  DoFmt(
      R"(#[cfg(test)]
fn assert_value_is_0<N: u32>(a: uN[N]) {
    assert_eq(0, a);
}
)");
}

TEST_F(ModuleFmtTest, SimpleParametricInvocation) {
  DoFmt(
      R"(fn p<N: u32>(x: bits[N]) -> bits[N] { x }

fn f() -> u8 { p<8>(u8:42) }
)");
}

TEST_F(ModuleFmtTest, StructInstantiationWithRedundantAttributeSpecifier) {
  DoFmtNoPostcondition(
      R"(struct MyStruct { x: u32 }

fn f() -> MyStruct {
    let x = u32:42;
    MyStruct { x: x }
}
)",
      R"(struct MyStruct { x: u32 }

fn f() -> MyStruct {
    let x = u32:42;
    MyStruct { x }
}
)");
}

TEST_F(ModuleFmtTest, StructInstantiationWithExactSizedCondExpr) {
  DoFmt(
      R"(struct MyStruct { field: u32 }

const TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT = true;

fn f(b: bool) -> MyStruct {
    MyStruct {
        field: if TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT {
            u32:42
        } else {
            u32:64
        },
    }
}
)");
}

TEST_F(ModuleFmtTest, StructInstantiationWithExactOneCharOverlyLargeCondExpr) {
  DoFmt(
      R"(struct MyStruct { field: u32 }

const TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT = true;

fn f(b: bool) -> MyStruct {
    MyStruct {
        field: if TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT {
            u32:42
        } else {
            u32:64
        },
    }
}
)",
      /*want=*/R"(struct MyStruct { field: u32 }

const TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT = true;

fn f(b: bool) -> MyStruct {
    MyStruct {
        field:
            if TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT {
                u32:42
            } else {
                u32:64
            },
    }
}
)");
}

TEST_F(ModuleFmtTest, SimpleParametricStructInstantiation) {
  DoFmt(
      R"(import mol;

struct Point<N: u32> { x: bits[N], y: bits[N] }

fn f() -> Point<mol::MOL> { Point<mol::MOL> { x: u8:42, y: u8:64 } }
)");
}

TEST_F(ModuleFmtTest, TypeRefTypeAnnotationModuleLevel) {
  DoFmt(
      R"(type MyU32 = u32;

fn f() -> MyU32 { MyU32:42 }
)");
}

TEST_F(ModuleFmtTest, TypeRefTypeAnnotationInBody) {
  DoFmt(
      R"(fn f() -> u32 {
    type MyU32 = u32;
    MyU32:42
}
)");
}

TEST_F(ModuleFmtTest, TypeRefChannelTypeAnnotation) {
  DoFmt("type MyChan = chan<u32> out;\n");
}

TEST_F(ModuleFmtTest, TypeRefChannelArrayTypeAnnotation) {
  DoFmt("type MyChan = chan<u32>[2] out;\n");
}

TEST_F(ModuleFmtTest, ColonRefWithImportSubject) {
  DoFmt(
      R"(import foo;

fn f() -> u32 { foo::bar }
)");
}

TEST_F(ModuleFmtTest, NestedColonRefWithImportSubject) {
  DoFmt(
      R"(import foo;

fn f() -> u32 { foo::bar::baz::bat }
)");
}

TEST_F(ModuleFmtTest, ModuleLevelConstAssert) {
  DoFmt(
      R"(import foo;

const_assert!(foo::bar == u32:42);
)");
}

TEST_F(ModuleFmtTest, ConstStructInstance) {
  DoFmt(R"(struct Point { x: u32, y: u32 }

const P = Point { x: u32:42, y: u32:64 };
)");
}

TEST_F(ModuleFmtTest, ConstStructInstanceEmpty) {
  DoFmt(
      R"(struct Nothing {}

const NOTHING = Nothing {};
)");
}

TEST_F(ModuleFmtTest, LetEqualStructInstance) {
  DoFmt(
      R"(struct Point0000000000000000000000000000000000000000000000000000000000000000000000 {
    x: u32,
    y: u32,
}

fn f() {
    let q = Point0000000000000000000000000000000000000000000000000000000000000000000000 {
        x: u32:42,
        y: u32:64,
    };
}
)");
}

TEST_F(ModuleFmtTest, LetEqualSplatStructInstance) {
  DoFmt(
      R"(struct Point0000000000000000000000000000000000000000000000000000000000000000000000 {
    x: u32,
    y: u32,
}

fn f(p: Point0000000000000000000000000000000000000000000000000000000000000000000000) {
    let q = Point0000000000000000000000000000000000000000000000000000000000000000000000 {
        x: u32:42,
        ..p
    };
}
)");
}

TEST_F(ModuleFmtTest, StructAttr) {
  DoFmt(
      R"(struct Point { x: u32, y: u32 }

fn get_x(p: Point) -> u32 { p.x }
)");
}

TEST_F(ModuleFmtTest, ConstStructInstanceWithSplatVariantOneUpdate) {
  DoFmt(
      R"(struct Point { x: u32, y: u32 }

const P = Point { x: u32:42, y: u32:64 };

const Q = Point { x: u32:32, ..P };
)");
}

TEST_F(ModuleFmtTest, ConstStructInstanceWithSplatVariantNoUpdate) {
  DoFmt(
      R"(struct Point { x: u32, y: u32 }

const P = Point { x: u32:42, y: u32:64 };

const Q = Point { ..P };
)");
}

TEST_F(ModuleFmtTest, StructInstanceWithNamesViaBindings) {
  DoFmt(R"(struct Point { x: u32, y: u16 }

fn f() {
    let x = u32:42;
    let y = u16:64;
    Point { x, y }
}
)");
}

TEST_F(ModuleFmtTest, StructInstanceWithNamesViaBindingsBackwards) {
  DoFmt(
      R"(struct Point { x: u32, y: u16 }

fn f() {
    let x = u32:42;
    let y = u16:64;
    Point { y, x }
}
)");
}

TEST_F(ModuleFmtTest, StructInstanceLetBinding) {
  DoFmt(
      R"(struct Point { x: u32, y: u16 }

fn f() -> Point {
    let x = u32:42;
    let y = u16:64;
    let p = Point { y, x };
    p
}
)");
}

TEST_F(ModuleFmtTest, SimpleQuickCheck) {
  DoFmt(
      R"(#[quickcheck]
fn f() -> bool { true }
)");
}

TEST_F(ModuleFmtTest, QuickCheckExhaustive) {
  DoFmt(
      R"(#[quickcheck(exhaustive)]
fn f(x: u1) -> bool { true }
)");
}

TEST_F(ModuleFmtTest, SimplePublicFunction) {
  DoFmt(
      R"(pub fn id(x: u32) -> u32 { x }
)");
}

TEST_F(ModuleFmtTest, OneModuleLevelCommentNoReflow) {
  DoFmt(
      R"(// This is a module level comment at the top of the file.
)");
}

TEST_F(ModuleFmtTest, TwoModuleLevelCommentsNoReflow) {
  DoFmt(
      R"(// This is a module level comment at the top of the file.

// This is another one slightly farther down in the file.
)");
}

TEST_F(ModuleFmtTest, OneMultiLineCommentNoReflow) {
  DoFmt(
      R"(// This is a module level comment at the top of the file.
// It spans multiple lines in a single block of comment text.
// Three, to be precise. And then the file ends.
)");
}

TEST_F(ModuleFmtTest, OneMultiLineCommentWithAnEmptyLineNoReflow) {
  DoFmt(
      R"(// This is a module level comment at the top of the file.
//
// There's a blank on the second line. And then the file ends after the third.
)");
}

TEST_F(ModuleFmtTest, OneOverlongCommentLineWithOneToken) {
  DoFmt(
      R"(// abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz
)");
}

TEST_F(ModuleFmtTest, ModuleAndFunctionLevelComments) {
  DoFmt(
      R"(// This is a module level comment at the top of the file.

// This is a function level comment.
fn f(x: u32) -> u32 { x }

// This is another function level comment.
fn g(x: u32) -> u32 {
    let y = x + u32:1;
    y
}
)");
}

TEST_F(ModuleFmtTest, TwoModuleLevelCommentBlocksBeforeFunction) {
  DoFmt(
      R"(// Module comment one.

// Module comment two.

fn uncommented_fn(x: u32) -> u32 { x }
)");
}

TEST_F(ModuleFmtTest, SimpleProc) {
  DoFmt(
      R"(pub proc p {
    config() { () }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithConstant) {
  DoFmt(
      R"(pub proc p {
    // My constant.
    const MY_CONST = u32:8;
    // My second constant.
    const ANOTHER_CONST = "second";

    config() { () }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithComments) {
  DoFmt(
      R"(// Proc-level comment.
pub proc p {
    // Member-level comment on cin.
    cin: chan<u32> in;
    // Member-level comment on cout.
    cout: chan<u32> out;
    // Type alias comment.
    type MyType = u32;

    // Config-level comment.
    config() { () }

    // Init-level comment.
    init { () }

    // Next-level comment.
    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcEmptyConfigBlock) {
  DoFmt(
      R"(pub proc p {
    config() {  }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithMembers) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config(cin: chan<u32> in, cout: chan<u32> out) { (cin, cout) }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithMembersImplStyle) {
  DoFmt(
      R"(pub proc P { foo: u32, bar: s16[2] }

impl P {}
)");
}

TEST_F(ModuleFmtTest, SimpleParametricProc) {
  DoFmt(
      R"(pub proc p<N: u32> {
    config() { () }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithChannelDecl) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config() {
        let (cin, cout) = chan<u32>("c");
        (cin, cout)
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithChannelArrayDecl) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32>[2] in;
    cout: chan<u32>[2] out;

    config() {
        let (cin, cout) = chan<u32>[2]("c");
        (cin, cout)
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithChannelDeclWithChannelConfig) {
  DoFmt(
      R"(#![feature(channel_attributes)]

pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config() {
        let (cin, cout) = #[channel(depth=0)]
                          chan<u32>("c");
        let _ = #[channel(depth=1, bypass=true, register_push_outputs=true, register_pop_outputs=true, input_flop_kind=skid, output_flop_kind=zero_latency)]
                chan<u32>("unused");
        (cin, cout)
    }

    init { () }

    next(state: ()) { () }
}
)");
}
TEST_F(ModuleFmtTest, SimpleProcWithChannelDeclWithFifoDepth) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config() {
        let (cin, cout) = chan<u32, u32:4>("c");
        (cin, cout)
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithSpawn) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config(cin: chan<u32> in, cout: chan<u32> out) { (cin, cout) }

    init { () }

    next(state: ()) { () }
}

pub proc q {
    config() {
        let (cin, cout) = chan<u32, u32:4>("c");
        spawn p(cin, cout);
        ()
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithSpawnNoTrailingTuple) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32> in;
    cout: chan<u32> out;

    config(cin: chan<u32> in, cout: chan<u32> out) { (cin, cout) }

    init { () }

    next(state: ()) { () }
}

pub proc q {
    config() {
        let (cin, cout) = chan<u32, u32:4>("c");
        spawn p(cin, cout);
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithLotsOfChannels) {
  DoFmt(
      R"(pub proc p {
    cin: chan<u32> in;
    ca: chan<u32> out;
    cb: chan<u32> out;
    cc: chan<u32> out;
    cd: chan<u32> out;
    ce: chan<u32> out;
    cf: chan<u32> out;
    cg: chan<u32> out;

    config(cin: chan<u32> in, ca: chan<u32> out, cb: chan<u32> out, cc: chan<u32> out,
           cd: chan<u32> out, ce: chan<u32> out, cf: chan<u32> out, cg: chan<u32> out) {
        (cin, ca, cb, cc, cd, ce, cf, cg)
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleProcWithTypeAlias) {
  DoFmt(
      R"(pub proc p {
    type MyU32OutChan = chan<u32> out;
    cg: MyU32OutChan;

    config(c: MyU32OutChan) { (c,) }

    init { () }

    next(state: ()) { () }
}
)");
}

// Based on report in https://github.com/google/xls/issues/1216
TEST_F(ModuleFmtTest, ProcSpawnImported) {
  DoFmt(
      R"(import some_import;

proc p {
    config() {
        spawn some_import::some_proc();
        ()
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, SimpleTestProc) {
  DoFmt(
      R"(#[test_proc]
proc p_test {
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) { (terminator,) }

    init { () }

    next(state: ()) { send(join(), terminator, true); }
}
)");
}

TEST_F(ModuleFmtTest, SimpleTestProcWithExpectedFailLabelAttribute) {
  DoFmt(
      R"(#[test_proc(expected_fail_label="my_fail")]
proc tester {
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) { (terminator,) }

    init {  }

    next(_: ()) {
        assert!(false, "my_fail");
        send(join(), terminator, true);
    }
}
)");
}

TEST_F(ModuleFmtTest, SimpleTestUtilityProc) {
  DoFmt(
      R"(#[cfg(test)]
proc TestUtilityProc {
    req_r: chan<()> in;
    resp_s: chan<()> out;

    config(req_r: chan<()> in, resp_s: chan<()> out) { (req_r, resp_s) }

    init {  }

    next(state: ()) {
        let (tok, _) = recv(join(), req_r);
        trace_fmt!("Message from a TestUtilityProc");
        send(tok, resp_s, ());
    }
}
)");
}

TEST_F(ModuleFmtTest, MatchLongWildcardArmExpression) {
  DoFmt(
      R"(import float32;

fn f(input_float: float32::F32) -> float32::F32 {
    match f.bexp {
        _ => float32::F32 {
            sign: input_float.sign,
            bexp: input_float.bexp - u8:1,
            fraction: input_float.fraction,
        },
    }
}
)");
}

TEST_F(ModuleFmtTest, ProcCallFarRhs) {
  DoFmt(
      R"(struct DelayState {}

const DELAY = u32:42;
const DATA_WIDTH = u8:42;

fn eq() {}

proc p {
    config() { () }

    init { () }

    next(state: DelayState) {
        let data_in = ();
        let (recv_tok, input_data, data_in_valid) =
            recv_if_non_blocking(join(), data_in, !eq(state.occupancy, DELAY), uN[DATA_WIDTH]:0);
    }
}
)");
}

TEST_F(ModuleFmtTest, ParametricFnWithManyArgs) {
  DoFmt(
      R"(fn umax() {}

pub fn uadd_with_overflow
    <V: u32, N: u32, M: u32, MAX_N_M: u32 = {umax(N, M)}, MAX_N_M_V: u32 = {umax(MAX_N_M, V)}>
    (x: uN[N], y: uN[M]) -> (bool, uN[V]) {
}
)");
}

TEST_F(ModuleFmtTest, TypeAliasToColonRefInstantiated) {
  DoFmt(
      R"(import float32;

type F32 = float32::F32;

pub fn f() -> F32 { F32 { blah: u32:42 } }
)");
}

TEST_F(ModuleFmtTest, AttrEquality) {
  DoFmt(
      R"(import m;

const SOME_BOOL = true;

fn f(x: m::MyStruct, y: m::MyStruct) -> bool { (x.foo == y.foo) || SOME_BOOL }
)");
}

TEST_F(ModuleFmtTest, ArrowReturnTypePackedOnOneLine) {
  DoFmt(
      R"(import apfloat;

fn n_path<EXP_SZ: u32, FRACTION_SZ: u32>
    (a: apfloat::APFloat<EXP_SZ, FRACTION_SZ>, b: apfloat::APFloat<EXP_SZ, FRACTION_SZ>)
    -> (apfloat::APFloat<EXP_SZ, FRACTION_SZ>, bool) {
}
)");
}

TEST_F(ModuleFmtTest, LongParametricList) {
  DoFmt(R"(// Signed max routine.
pub fn smax<N: u32>(x: sN[N], y: sN[N]) -> sN[N] { if x > y { x } else { y } }

pub fn extract_bits
    <from_inclusive: u32, to_exclusive: u32, fixed_shift: u32, N: u32,
     extract_width: u32 = {smax(s32:0, to_exclusive as s32 - from_inclusive as s32) as u32}>
    (x: uN[N]) -> uN[extract_width] {
}
)");
}

TEST_F(ModuleFmtTest, NestedBinopLogicalOr) {
  DoFmt(
      R"(// Define some arbitrary constants at various identifier widths.
const AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA = true;
const BBBBBBBBBBBBBBBBBBBBBBB = true;
const CCCCCCCCCCCCCCCCCCCCCCC = true;

fn f() -> bool {
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ||
    BBBBBBBBBBBBBBBBBBBBBBB || CCCCCCCCCCCCCCCCCCCCCCC
}
)");
}

TEST_F(ModuleFmtTest, ModuleConstantsWithInlineComments) {
  DoFmt(
      R"(pub const MOL = u32:42;  // may be important

const TWO_TO_FIFTH = u32:32;  // 2^5
)");
}

TEST_F(ModuleFmtTest, ModuleConstantsWithMultilineInlineComments) {
  DoFmt(
      R"(pub const MOL = u32:42;  // may be important

const TWO_TO_FIFTH = u32:32;  // 2^5
                              // My favorite const.
)");
}

// If the array constant is placed on a single line it is overly long -- we
// check that we smear it across multiple lines to stay within 100 chars.
TEST_F(ModuleFmtTest, OverLongArrayConstant) {
  DoFmt(R"(// Top of module comment.
const W_A0 = u32:32;
const NUM_PIECES = u32:8;
pub const A0 = sN[W_A0][NUM_PIECES]:[
    111111, 111111, 111111, 111111, 111111, 111111, 111111, 111111,
];
)");
}

TEST_F(ModuleFmtTest, InvocationWithOneStructArg) {
  DoFmt(R"(struct APFloat {}

fn unbiased_exponent() {}

fn f() {
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0 });
}
)");
}

// See github issue https://github.com/google/xls/issues/1193
TEST_F(ModuleFmtTest, LongLetLeader) {
  DoFmt(R"(import std;

fn foo(some_value_that_is_pretty_long: u32, some_other_value_that_is_also_not_too_short: u32) {
    type SomeTypeNameThatIsNotTooShort = s64;
    let very_somewhat_long_variable_name: SomeTypeNameThatIsNotTooShort = std::to_signed(
        some_value_that_is_pretty_long ++ some_other_value_that_is_also_not_too_short);
}
)");
}

TEST_F(ModuleFmtTest, LongLetRhs) {
  DoFmt(R"(import std;

fn foo(some_value_that_is_pretty_long: u32, some_other_value_that_is_also_not_too_short: u32) {
    type SomeTypeNameThatIsNotTooShort = sN[u32:96];
    let very_somewhat_long_variable_name: SomeTypeNameThatIsNotTooShort = std::to_signed(
        some_value_that_is_pretty_long ++ some_other_value_that_is_also_not_too_short ++
        some_value_that_is_pretty_long);
}
)");
}

TEST_F(ModuleFmtTest, QuickcheckWithCount) {
  DoFmt(R"(// Comment on quickcheck.
#[quickcheck(test_count=100000)]
fn prop_eq(x: u32, y: u32) -> bool { x == y }
)");
}

TEST_F(ModuleFmtTest, ModuleLevelAnnotation) {
  DoFmt(R"(#![allow(nonstandard_constant_naming)]

fn id(x: u32) { x }
)");
}

TEST_F(ModuleFmtTest, TypeInferenceVersionAnnotation) {
  DoFmt(R"(#![feature(type_inference_v2)]

fn id(x: u32) { x }
)");
}

TEST_F(ModuleFmtTest, TypeInferenceVersionAnnotationWithCommentAbove) {
  DoFmt(R"(// Foobar

#![feature(type_inference_v2)]

fn id(x: u32) { x }
)");
}

TEST_F(ModuleFmtTest, GithubIssue1229) {
  // Note: we just need it to parse, no need for it to typecheck.
  DoFmt(R"(struct ReadReq<X: u32> {}
struct ReadResp<X: u32> {}
struct WriteReq<X: u32, Y: u32> {}
struct WriteResp {}
struct AllCSR<X: u32, Y: u32> {}

proc CSR<X: u32, Y: u32, Z: u32> {
    config() { () }

    init { () }

    next(state: ()) { () }
}

proc csr_8_32_14 {
    config(read_req: chan<ReadReq<u32:8>> in, read_resp: chan<ReadResp<u32:32>> out,
           write_req: chan<WriteReq<u32:8, u32:32>> in, write_resp: chan<WriteResp> out,
           all_csr: chan<AllCSR<u32:32, u32:14>> out) {
        spawn CSR<u32:8, u32:32, u32:14>(read_req, read_resp, write_req, write_resp, all_csr);
        ()
    }

    init { () }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, GithubIssue1329) {
  DoFmt(R"(struct StructA { data: u64[8] }
struct StructB { a: StructA }
struct StructC { b: StructB }

#[test]
fn struct_c_test() {
    assert_eq(
        zero!<StructA>(),
        StructC {
            b: StructB {
                a: StructA {
                    data: u64[8]:[
                        0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0, ...
                    ],
                },
            },
        });
}
)");
}

TEST_F(ModuleFmtTest, GithubIssue1354) {
  DoFmt(R"(pub struct B { value: u32 }

pub struct A { value: B[5] }

pub fn f(a: A) -> A { if a.B[0].value == u32:0 { zero!<A>() } else { a } }
)");
}

TEST_F(ModuleFmtTest, LongStructInstanceFieldExpr) {
  DoFmt(R"(struct X { xxxxxxxxxxxxxxxxxxx: bool, yyyyyyyyyyyyyyyyyyy: u32 }

const aaaaaaaaaaaaaaaaaaaaaaaaa = u32:0;
const bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = u32:0;
const ccccccccccccccccccccccccc = bool:false;

fn f() -> X {
    X {
        xxxxxxxxxxxxxxxxxxx:
            aaaaaaaaaaaaaaaaaaaaaaaaa == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
        yyyyyyyyyyyyyyyyyyy: u32:0,
    }
}
)");
}

TEST_F(ModuleFmtTest, TraceTest) {
  DoFmt(R"(fn trace_test() {
    let x0 = clz(u3:0b011);
    let x1 = (x0 as u8) * u8:3;
    trace!(x1);
}
)");
}

TEST_F(ModuleFmtTest, IfElseIf) {
  DoFmt(R"(fn elseif_sample(s: bool, x: u32, y: u32) -> u32 {
    if s == true {
        x
    } else if x == u32:7 {
        y
    } else {
        u32:42
    }
}
)");
}

TEST_F(ModuleFmtTest, IfWithoutElse) {
  DoFmt(
      R"(fn elseif_without_else(s: bool, x: u32) { if s == true { trace_fmt!("{}", x); }; }
)");
}

TEST_F(ModuleFmtTest, IfElseIfWithoutElse) {
  DoFmt(R"(fn elseif_without_else(s: bool, x: u32, y: u32) {
    if s == true {
        trace_fmt!("{}", x);
    } else if x == u32:7 {
        trace_fmt!("{}", y);
    };
}
)");
}

TEST_F(ModuleFmtTest, ImportGroupsWithComments) {
  DoFmt(R"(import thing1;
import thing2;

// Starting comment
import other;
import stuff;
// Ending comment
)");
}

TEST_F(ModuleFmtTest, ImportGroupsWithDisableComment) {
  DoFmt(R"(import thing1;
import thing2;

// dslx-fmt::off
import other;
import stuff;
// dslx-fmt::on

const foo = u32:26;
)");
}

TEST_F(ModuleFmtTest, MultipleImportGroupsWithDisableComment) {
  DoFmt(R"(import thing1;
import thing2;

// dslx-fmt::off
  import other;
import stuff;
    import morestuff
    ;
 // dslx-fmt::on

import other2;
// The above import needs an extra newline before it because the formatter
// can't recognize the unformatted imports as being part of the same group as
// the last one, so it inserts a newline before the next group.
)");
}

TEST_F(ModuleFmtTest, ImportWithFmtDisabled) {
  DoFmt(R"(import thing1;
import thing2;

// dslx-fmt::off
        import
 other;
  // dslx-fmt::on

const foo = u32:26;
)");
}

TEST_F(ModuleFmtTest, ConstWithFmtDisabled) {
  DoFmt(R"(import thing1;

// Note intentional trailing space
// dslx-fmt::off
        const foo=u32:    26; const
      bar=u16:    26;
// dslx-fmt::on
)");
}

TEST_F(ModuleFmtTest, ImportGroupsWithFmtDisabled) {
  DoFmt(R"(import thing1;
import thing2;

// dslx-fmt::off
        import
 other;
  import   stuff;
// dslx-fmt::on
)");
}

TEST_F(ModuleFmtTest, ImportGroupsWithFmtDisabledNeverEnabled) {
  DoFmt(R"(import thing1;
import thing2;

// dslx-fmt::off
        import
 other;
  import   stuff;

)");
}

TEST_F(ModuleFmtTest, FnWithFmtDisabled) {
  DoFmt(R"(import thing1;
import thing2;

// dslx-fmt::off
  fn f() -> (u32, u16) {
let x = u32:42;
let y = u16:64;   (x,   y)
}
  // dslx-fmt::on
)");
}

TEST_F(ModuleFmtTest, FnStatementsWithFmtDisabled) {
  DoFmt(R"(import thing1;
import thing2;

fn f() -> (u32, u16) {
    let y = u16:64;

    // dslx-fmt::off
let y = u16:64;
let x =
  u32:41;
         (x ,   y)

          // dslx-fmt::on

}
)");
}

TEST_F(ModuleFmtTest, StatementsWithFmtDisabledThenComment) {
  DoFmt(R"(const result = {
    let y = u16:63;

    // dslx-fmt::off
let y = u16:64;
let x =
  u32:41;
         (x ,   y)

          // dslx-fmt::on

    // The previous line should *not* be indented because it's
    // *within* the "disable formatting" range.
};
)");
}

TEST_F(ModuleFmtTest, StatementsWithFmtDisabled) {
  DoFmt(R"(const result = {
    let y = u16:63;

    // dslx-fmt::off
let y = u16:64;
let x =
  u32:41;
         (x ,   y)

// dslx-fmt::on
};
)");
}

TEST_F(ModuleFmtTest, TooLongLines) {
  DoFmt(R"(const result = {
    // dslx-fmt::off
                   let         y =         u16:64;         let x =         u32:41        ;     let y = u16:63;    (x ,   y)

// dslx-fmt::on
};
)");
}

TEST_F(ModuleFmtTest, ExternVerilog) {
  // Note alternate string literal delimiter * so it can use )" on the
  // last line of the annotation.
  DoFmt(R"*(#[extern_verilog("external_divmod #(
     .divisor_width({B_WIDTH})
    ) {fn} (
     .dividend({a}),
     .by_zero({return.1})
    )")]
fn divmod() {
}
)*");
}

TEST_F(ModuleFmtTest, ConstantDefWithComment) {
  // It intentionally moves the comment so we can't test with the
  // opportunistic_postcondition.
  DoFmt(R"(pub // 1 between pub and const
const // 2 between const and name
foo // 3 between name and colon
: // 4 between colon and type
u32 // 5 between type and =
= // 6 between = and value
u32:42 // 7 between value and semi
;)",
        R"(// 1 between pub and const
// 2 between const and name
pub const foo // 3 between name and colon
// 4 between colon and type
    : u32 // 5 between type and =
    // 6 between = and value
    = u32:42 // 7 between value and semi
    ;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

// NOTE: the numbers in the following tests are referencing the 7 possible
// places a comment can be in a constant definition:
// pub 1 const 2 name 3 : 4 type 5 = 6 value 7 ;

TEST_F(ModuleFmtTest, ConstantDefWithComment1) {
  // It intentionally moves the comment so we can't test with the
  // opportunistic_postcondition.
  DoFmt(R"(pub // between pub and const
const foo = u32:42;
)",
        R"(// between pub and const
pub const foo = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithCommentPub2) {
  DoFmt(R"(pub const // between const and name
    foo = u32:42;
)",
        R"(// between const and name
pub const foo = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithCommentPub1And2) {
  DoFmt(R"(pub // between pub and const
const // between const and name
    foo = u32:42;
)",
        R"(// between pub and const
// between const and name
pub const foo = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithComment2) {
  DoFmt(R"(const // between const and name
    foo = u32:42;
)",
        R"(// between const and name
const foo = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithComment3) {
  DoFmt(R"(const foo // between name and colon
    : u32 = u32:42;
)");
}

TEST_F(ModuleFmtTest, ConstantDefWithComment4) {
  DoFmt(R"(const foo: // between colon and type
    u32 = u32:42;
)",
        R"(const foo // between colon and type
    : u32 = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithComment5) {
  DoFmt(R"(const foo: u32 // between type and =
    = u32:42;
)",
        R"(const foo: u32 // between type and =
    = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithComment5NoType) {
  DoFmt(R"(const foo // between name and =
    = u32:42;
)");
}

TEST_F(ModuleFmtTest, ConstantDefWithComment6) {
  DoFmt(R"(const foo: u32 = // after equals
    u32:42;
)",
        R"(const foo: u32 // after equals
    = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithComment6NoType) {
  DoFmt(R"(const foo = // after equals
    u32:42;
)",
        R"(const foo // after equals
    = u32:42;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, ConstantDefWithComment7) {
  DoFmt(R"(const foo: u32 = u32:42 // after value
    ;
)");
}

TEST_F(ModuleFmtTest, DisableFormatSingleExpr) {
  // This test flushes out a bug when disabling formatting around a single
  // `Expr`ession.

  // Unfortunately after fixing https://github.com/google/xls/issues/1697,
  // comments in the middle of `const` statements are rearranged, so we have to
  // contrive this example to flush out the bug.
  DoFmt(R"(const x = // dslx-fmt::off
 u32 :1
    // dslx-fmt::on
;
)",
        R"(const x // dslx-fmt::off
    =  u32 :1
    // dslx-fmt::on
;
)",
        kDslxDefaultTextWidth, /*opportunistic_postcondition=*/false);
}

TEST_F(ModuleFmtTest, SimplestProc) {
  // There's an extra space in the blocks due to bug
  // https://github.com/google/xls/issues/1703
  DoFmt(R"(proc p {
    config() {  }

    init {  }

    next(state: ()) { () }
}
)");
}

TEST_F(ModuleFmtTest, DisableFmtSimpleProc) {
  DoFmt(R"(// dslx-fmt::off
proc p { config() { } init { } next(state: ()) { () } }
// dslx-fmt::on
)");
}

TEST_F(ModuleFmtTest, DisableFmtSimpleProcNext) {
  DoFmt(R"(proc p {
    config() {  }

    init {  }

    // dslx-fmt::off
    next(state: ()) { () }
    // dslx-fmt::on
}
)");
}

TEST_F(ModuleFmtTest, DisableFmtInProc_GH1735) {
  // This reproduces the issue from https://github.com/google/xls/issues/1735,
  // and is still broken, but at least doesn't delete the whole proc!
  AutoFmtExpectsError(R"(struct FooType { a: u32, b: u32 }
struct BarType { c: u32, d: u32 }
const NUM_ELEMS = u32:8;
const NUM_BLOCKS = u32:2;
proc A {
    config() {  }
    init {  }
    next(_: ()) {
        // some comment
        let _some_import_code_here = true;
        let (_foo, _bar, _baz) =
            for (i, (foo, bar, baz)): (
                // dslx-fmt::off
                u32, (FooType[NUM_ELEMS][NUM_BLOCKS], BarType[NUM_ELEMS][NUM_BLOCKS],
                      bool[NUM_ELEMS][NUM_BLOCKS])
                ) in range(u32:0, 8) {
                // dslx-fmt::on

                // this is another cool comment
                (foo, bar, baz)
            }((zero!<FooType[8][2]>(), zero!<BarType[8][2]>(), all_ones!<bool[8][2]>()));

        // the end
        let _my_grand_finale_here = true;
    }
})",
                      "would be deleted");
}

TEST_F(ModuleFmtTest, BinopWithCommentBeforeOpNoNewline) {
  DoFmt(R"(const val = u32:1 // before op
    + u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithCommentBeforeOp) {
  DoFmt(R"(const val = u32:1
    // before op
    + u32:2;
)",
        R"(const val = u32:1 // before op
    + u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithMultipleCommentsBeforeOp) {
  DoFmt(R"(const val = u32:1 // before op
    // second line before op
    + u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithCommentAfterOpNoNewline) {
  DoFmt(R"(const val = u32:1 + // after op
    u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithCommentAfterOp) {
  DoFmt(R"(const val = u32:1 +
    // after op
    u32:2;
)",
        R"(const val = u32:1 + // after op
    u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithMultipleCommentsAfterOp) {
  DoFmt(R"(const val = u32:1 + // after op
    // second line after op
    // third line after op
    u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithCommentsBeforeAndAfterOp) {
  DoFmt(R"(const val = u32:1 // before op
    + // after op
    u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithCommentsBeforeAndAfterOpWithNewlines) {
  DoFmt(R"(const val = u32:1
    // before op
    +
    // after op
    u32:2;
)",
        R"(const val = u32:1 // before op
    + // after op
    u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithMultipleCommentsBeforeAndAfterOp) {
  DoFmt(R"(const val = u32:1
    // before op
    // before op second line
    // before op third line
    +
    // after op
    // after op second line
    // after op third line
    u32:2;
)",
        R"(const val = u32:1 // before op
    // before op second line
    // before op third line
    + // after op
    // after op second line
    // after op third line
    u32:2;
)");
}

TEST_F(ModuleFmtTest, BinopWithMultipleCommentsBeforeAndAfterOpWithNewlines) {
  DoFmt(R"(const val = u32:1 // before op
    // before op second line
    // before op third line
    + // after op
    // after op second line
    // after op third line
    u32:2;
)");
}

// TODO: https://github.com/google/xls/issues/1756 - we don't support disabling
// formatting in the middle of a struct definition yet, but at least it doesn't
// crash.
TEST_F(ModuleFmtTest, DisableFmtInStructDefDisallowed_GH1756) {
  AutoFmtExpectsError(R"(
  struct Point {
    // dslx-fmt::off
x  :   u32,
    // dslx-fmt::on
    y: u32,
    }
    )",
                      "Cannot disable", absl::StatusCode::kInvalidArgument);
}

// TODO: https://github.com/google/xls/issues/1760 - we don't support disabling
// formatting in this situation yet, but at least it doesn't crash.
TEST_F(ModuleFmtTest, DisableFmtAroundTestFnDisallowed_GH1760) {
  AutoFmtExpectsError(R"(#[test]
// dslx-fmt::off
fn test() {
}
// dslx-fmt::on
)",
                      "Cannot disable", absl::StatusCode::kInvalidArgument);
}

TEST_F(ModuleFmtTest, DisableFmtAroundStatementBlockDisallowed_GH1760) {
  // Cannot disable just the statement block of a function.
  AutoFmtExpectsError(R"(fn test()
// dslx-fmt::off
{}
// dslx-fmt::on
)",
                      "Cannot disable", absl::StatusCode::kInvalidArgument);
}

// You *can* completely disable formatting around a test function.
TEST_F(ModuleFmtTest, DisableFmtAroundTestFn) {
  DoFmt(R"(// dslx-fmt::off
#[test] fn test() { }
// dslx-fmt::on
)");
}

TEST_F(ModuleFmtTest, DisableFmtViaFnAttribute) {
  DoFmt(R"(#[dslx_format_disable]
fn
test() -> u32
    {
// first line
        u32:1
}
)");
}

TEST_F(ModuleFmtTest, DisableFmtViaFnAttributeOnlyAppliesToCurrentFn) {
  DoFmt(R"(#[dslx_format_disable]
fn test() -> u32
    {
// first line
        u32:1
}

// This one will be formatted:
fn test2() ->
u32 {
    u32:1
}
)",
        R"(#[dslx_format_disable]
fn test() -> u32
    {
// first line
        u32:1
}

// This one will be formatted:
fn test2() -> u32 { u32:1 }
)");
}

TEST_F(ModuleFmtTest, CommentBeforeFirstItemStructLiteral) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        // Comment before
        a: u1:0,
        b: u1:0,
    };
}
)");
}

TEST_F(ModuleFmtTest, CommentBetweenItemsInStructLiteral) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        a: u1:0,
        // and between
        b: u1:0,
    };
}
)");
}

TEST_F(ModuleFmtTest, CommentAfterLastItemInStructLiteral) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        a: u1:0,
        b: u1:0,
        // and after.
    };
}
)");
}

TEST_F(ModuleFmtTest, CommentThroughoutInStructLiteral) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        // Comment before
        a: u1:0,
        // and between
        b: u1:0,
        // and after.
    };
}
)");
}

TEST_F(ModuleFmtTest, CommentSameLineAsItemInStructLiteral) {
  // TODO: https://github.com/google/xls/issues/1719 - if there is a comment
  // on the same line as this item, don't insert a hard-line.
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        a: u1:0, // and between
        b: u1:0, // and after.
    };
}
)",
        R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        a: u1:0,
        // and between
        b: u1:0,
        // and after.
    };
}
)");
}

TEST_F(ModuleFmtTest, CommentAfterStructLiteralSpan) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo { a: u1:0, b: u1:0 };  // after only
}
)");
}

TEST_F(ModuleFmtTest, StructLiteral) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() { let foo = Foo { a: u1:0, b: u1:0 }; }
)");
}

TEST_F(ModuleFmtTest, StructLiteralShorthand) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let a = u1:0;
    let foo = Foo {
        a,
        b: u1:0 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1,
    };
}
)");
}

TEST_F(ModuleFmtTest, StructLiteralBreak) {
  DoFmt(R"(struct Foo { a: u1, b: u1 }

fn test() {
    let foo = Foo {
        a: u1:0 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1,
        b: u1:0 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1 + u1:1,
    };
}
)");
}

TEST_F(ModuleFmtTest, CommentsDoNotReflow100Chars) {
  // This is 100 characters, and should not reflow.
  DoFmt(
      R"(// Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labo
struct Foo { a: u1, b: u1 }
)");
}

TEST_F(ModuleFmtTest, CommentsReflow101Chars) {
  // This is 101 characters, and wasn't reflowing before.
  DoFmt(
      R"(// Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labor
struct Foo { a: u1, b: u1 }
)",
      R"(// Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
// labor
struct Foo { a: u1, b: u1 }
)");
}

TEST_F(ModuleFmtTest, BetterBreaksOnTupleTypes_GH1685) {
  DoFmt(R"(fn foo() {
    const NUM_ELEMENTS = u32:82;
    let (points_per_game, foo_bar, foo_bar_valid) =
        for (game_idx, (points_per_game, foo_bar, foo_bar_valid)): (u32, (bits[NUM_ELEMENTS], bits[NUM_ELEMENTS
        ][NUM_ELEMENTS
        ], bits[NUM_ELEMENTS])) in range(u32:0, NUM_ELEMENTS) {


        }(zero!<(bits[NUM_ELEMENTS], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS])>());
})",
        R"(fn foo() {
    const NUM_ELEMENTS = u32:82;
    let (points_per_game, foo_bar, foo_bar_valid) =
        for (game_idx, (points_per_game, foo_bar, foo_bar_valid)): (
            u32, (bits[NUM_ELEMENTS], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS])
        ) in range(u32:0, NUM_ELEMENTS) {


        }(zero!<(bits[NUM_ELEMENTS], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS])>());
}
)");
}

TEST_F(ModuleFmtTest, BetterBreaksOnTupleTypesNested_GH1685) {
  DoFmt(R"(fn foo() {
    const NUM_ELEMENTS = u32:82;
    let (points_per_game, foo_bar, foo_bar_valid) =
        for (game_idx, (points_per_game, foo_bar, foo_bar_valid, foo_bar_invalid)): (u32, (bits[NUM_ELEMENTS
        ], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS], bits[NUM_ELEMENTS])) in
            range(u32:0, NUM_ELEMENTS) {


        }(zero!<(bits[NUM_ELEMENTS], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS], bits[NUM_ELEMENTS
        ])>());
})",
        R"(fn foo() {
    const NUM_ELEMENTS = u32:82;
    let (points_per_game, foo_bar, foo_bar_valid) =
        for (game_idx, (points_per_game, foo_bar, foo_bar_valid, foo_bar_invalid)): (
            u32,
            (
                bits[NUM_ELEMENTS], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS],
                bits[NUM_ELEMENTS]
            )
        ) in range(u32:0, NUM_ELEMENTS) {


        }(zero!<(
            bits[NUM_ELEMENTS], bits[NUM_ELEMENTS][NUM_ELEMENTS], bits[NUM_ELEMENTS],
            bits[NUM_ELEMENTS]
        )>());
}
)");
}

TEST_F(ModuleFmtTest, TupleWithComment_GH_1678) {
  DoFmt(R"(fn foo(bar: u32) {
    let some_data_to_make_single_update_per_line = u32:0xabcdef;
    (
        bit_slice_update(
            some_data_to_make_single_update_per_line, 1,
            if bar > u32:0xdeadbeef { u1:1 } else { u1:0 }), // this is yet another comment
    )
}
)");
}

TEST_F(ModuleFmtTest, TupleWithMultipleComments_GH_1678) {
  DoFmt(R"(fn foo(bar: u32) {
    let some_data_to_make_single_update_per_line = u32:0xabcdef;
    (
        // TODO: davidplass - if the previous comment is not on the same line as
        // the previous element, insert a hard line before the comment.
        bit_slice_update(some_data_to_make_single_update_per_line, 0, u1:1),
        // This is an important comment
        bit_slice_update(
            some_data_to_make_single_update_per_line, 1,
            if bar > u32:0xdeadbeef { u1:1 } else { u1:0 }), // this is yet another comment
        bit_slice_update(some_data_to_make_single_update_per_line, 2, u1:1)
    )
}
)",
        R"(fn foo(bar: u32) {
    let some_data_to_make_single_update_per_line = u32:0xabcdef;
    (
        // TODO: davidplass - if the previous comment is not on the same line as
        // the previous element, insert a hard line before the comment.
        bit_slice_update(some_data_to_make_single_update_per_line, 0, u1:1), // This is an important
        // comment
        bit_slice_update(
            some_data_to_make_single_update_per_line, 1,
            if bar > u32:0xdeadbeef { u1:1 } else { u1:0 }), // this is yet another comment
        bit_slice_update(some_data_to_make_single_update_per_line, 2, u1:1)
    )
}
)");
}

TEST_F(ModuleFmtTest, LongConstAssert_GH_1959) {
  DoFmt(R"(import foo_bar_baz_cat_dog;

const_assert!(
    foo_bar_baz_cat_dog::FOO_BAR_ONE_TWO_THREE == foo_bar_baz_cat_dog::FOO_BAZ_ONE_TWO_THREE);
)");
}

TEST_F(ModuleFmtTest, LongConstAssertViaTernaryInside) {
  DoFmt(R"(import m;

const_assert!(if m::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {
    m::BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
} else {
    m::CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
});
)");
}

// One extra char on the test expression vs the above, which causes the
// expression to break/nest.
TEST_F(ModuleFmtTest, LongConstAssertViaTernaryInsideOverlongTestNests) {
  DoFmt(R"(import m;

const_assert!(
    if m::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {
        m::BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    } else {
        m::CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    });
)");
}

TEST_F(ModuleFmtTest, LongConstAssertMultilineBinopExpression) {
  DoFmt(R"(import m;

const_assert!(m::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ==
              m::BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB);
)");
}

}  // namespace
}  // namespace xls::dslx

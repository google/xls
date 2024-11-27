// Copyright 2020 The XLS Authors
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

#include "xls/dslx/frontend/parser.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/error_test_utils.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class ParserTest : public ::testing::Test {
 public:
  absl::StatusOr<std::unique_ptr<Module>> Parse(std::string_view program) {
    scanner_.emplace(file_table_, Fileno(0), std::string(program));
    parser_.emplace("test", &*scanner_);
    return parser_->ParseModule();
  }

  std::unique_ptr<Module> RoundTrip(
      std::string_view program,
      std::optional<std::string_view> target = std::nullopt) {
    absl::StatusOr<std::unique_ptr<Module>> module = Parse(program);
    if (!module.ok()) {
      UniformContentFilesystem vfs(program);
      TryPrintError(module.status(), file_table_, vfs);
      XLS_EXPECT_OK(module) << module.status();
      return nullptr;
    }
    std::unique_ptr<Module> module_ptr = std::move(*module);
    if (target.has_value()) {
      EXPECT_EQ(module_ptr->ToString(), *target);
    } else {
      EXPECT_EQ(module_ptr->ToString(), program);
    }
    return module_ptr;
  }

  std::unique_ptr<Module> ExpectParsesSuccessfully(std::string_view program) {
    absl::StatusOr<std::unique_ptr<Module>> module = Parse(program);
    if (!module.ok()) {
      UniformContentFilesystem vfs(program);
      TryPrintError(module.status(), file_table_, vfs);
      XLS_EXPECT_OK(module) << module.status();
      return nullptr;
    }
    return std::move(*module);
  }

  // Note: given expression text should have no free variables other than those
  // in "predefine": those are defined a builtin name definitions (like the DSLX
  // builtins are).
  absl::StatusOr<Expr*> ParseExpr(std::string_view expr_text,
                                  absl::Span<const std::string> predefine = {},
                                  bool populate_dslx_builtins = false) {
    scanner_.emplace(file_table_, Fileno(0), std::string{expr_text});
    parser_.emplace("test", &*scanner_);
    Bindings b;

    Module& mod = *parser_->module_;
    if (populate_dslx_builtins) {
      for (auto const& it : GetParametricBuiltins()) {
        std::string name(it.first);
        b.Add(name, mod.GetOrCreateBuiltinNameDef(name));
      }
    }

    for (const std::string& s : predefine) {
      b.Add(s, mod.GetOrCreateBuiltinNameDef(s));
    }
    auto expr_or = parser_->ParseExpression(/*bindings=*/b);
    if (!expr_or.ok()) {
      UniformContentFilesystem vfs(expr_text);
      TryPrintError(expr_or.status(), file_table_, vfs);
    }
    return expr_or;
  }

  Expr* RoundTripExpr(std::string_view expr_text,
                      absl::Span<const std::string> predefine = {},
                      bool populate_dslx_builtins = false,
                      std::optional<std::string> target = std::nullopt) {
    absl::StatusOr<Expr*> e_or =
        ParseExpr(expr_text, predefine, populate_dslx_builtins);
    if (!e_or.ok()) {
      XLS_EXPECT_OK(e_or.status());
      return nullptr;
    }
    Expr* e = e_or.value();
    if (target.has_value()) {
      EXPECT_EQ(e->ToString(), *target);
    } else {
      EXPECT_EQ(e->ToString(), expr_text);
    }

    return e;
  }

  // Allows the private ParseTypeAnnotation method to be used by subtypes (test
  // instances).
  absl::StatusOr<TypeAnnotation*> ParseTypeAnnotation(Parser& p,
                                                      Bindings& bindings) {
    return p.ParseTypeAnnotation(bindings);
  }

  FileTable file_table_;
  std::optional<Scanner> scanner_;
  std::optional<Parser> parser_;
};

TEST(BindingsTest, BindingsStack) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  Bindings top;
  Bindings leaf0(&top);
  Bindings leaf1(&top);

  auto* a = module.GetOrCreateBuiltinNameDef("a");
  auto* b = module.GetOrCreateBuiltinNameDef("b");
  auto* c = module.GetOrCreateBuiltinNameDef("c");

  top.Add("a", a);
  leaf0.Add("b", b);
  leaf1.Add("c", c);

  Pos pos(Fileno(0), 0, 0);
  Span span(pos, pos);

  // Everybody can resolve the binding in "top".
  EXPECT_THAT(leaf0.ResolveNodeOrError("a", span, file_table), IsOkAndHolds(a));
  EXPECT_THAT(leaf1.ResolveNodeOrError("a", span, file_table), IsOkAndHolds(a));
  EXPECT_THAT(top.ResolveNodeOrError("a", span, file_table), IsOkAndHolds(a));

  EXPECT_THAT(top.ResolveNodeOrError("b", span, file_table),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"b\"")));
  EXPECT_THAT(leaf1.ResolveNodeOrError("b", span, file_table),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"b\"")));
  EXPECT_THAT(leaf0.ResolveNodeOrError("c", span, file_table),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"c\"")));

  EXPECT_THAT(leaf0.ResolveNodeOrError("b", span, file_table), IsOkAndHolds(b));
  EXPECT_THAT(leaf1.ResolveNodeOrError("c", span, file_table), IsOkAndHolds(c));
}

TEST_F(ParserTest, TestRoundTripFailsOnSyntaxError) {
  EXPECT_NONFATAL_FAILURE(RoundTrip("invalid-program"), "ParseError:");
}

TEST_F(ParserTest, TestIdentityFunction) {
  RoundTrip(R"(fn f(x: u32) -> u32 {
    x
})");
}

TEST_F(ParserTest, DuplicateFn) {
  EXPECT_NONFATAL_FAILURE(
      RoundTrip(R"(fn f(){} fn f(){})"),
      "Function 'f' is defined in this module multiple times");
}

TEST_F(ParserTest, DuplicateMember) {
  EXPECT_NONFATAL_FAILURE(RoundTrip(R"(fn f(){} struct f {})"),
                          "already contains a member named `f`");
}

TEST_F(ParserTest, TestIdentityFunctionWithLet) {
  std::unique_ptr<Module> module = RoundTrip(R"(fn f(x: u32) -> u32 {
    let y = x;
    y
})");
  std::optional<Function*> maybe_f = module->GetFunction("f");
  ASSERT_TRUE(maybe_f.has_value());
  Function* f = maybe_f.value();
  ASSERT_NE(f, nullptr);
  StatementBlock* f_body = f->body();
  ASSERT_EQ(f_body->statements().size(), 2);
}

TEST_F(ParserTest, TestBlockOfUnitNoSemi) {
  RoundTripExpr(R"({
    ()
})");
}

TEST_F(ParserTest, TestBlockOfUnitWithSemi) {
  RoundTripExpr(R"({
    ();
})");
}

TEST_F(ParserTest, TestBlockOfTwoUnits) {
  Expr* e = RoundTripExpr(R"({
    ();
    ()
})");
  ASSERT_NE(e, nullptr);
  auto* block = dynamic_cast<StatementBlock*>(e);
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements().size(), 2);
  EXPECT_TRUE(
      std::holds_alternative<Expr*>(block->statements().at(0)->wrapped()));
  EXPECT_TRUE(
      std::holds_alternative<Expr*>(block->statements().at(1)->wrapped()));
}

TEST_F(ParserTest, TestTokenIdentity) {
  RoundTrip(R"(fn f(t: token) -> token {
    t
})");
}

TEST_F(ParserTest, ImplDefRoundTrip) {
  RoundTrip(R"(pub struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
})");
}

TEST_F(ParserTest, ImplWithFunctionDefRoundTrip) {
  RoundTrip(R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func() -> u9 {
        u9:0
    }
})");
}

TEST_F(ParserTest, ImplWithMethodDefRoundTrip) {
  std::unique_ptr<Module> module = RoundTrip(R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func(self) -> u9 {
        self.a
    }
}
fn main() -> u9 {
    let st = foo { a: u9:5, b: u16:0 };
    st.my_func()
})");
  Impl* impl = module->GetImpls().at(0);
  EXPECT_TRUE(impl->GetFunction("my_func").has_value());
}

TEST_F(ParserTest, ImplWithExplicitSelfType) {
  RoundTrip(R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func(self: Self) -> u9 {
        self.a
    }
}
fn main() -> u9 {
    let st = foo { a: u9:5, b: u16:0 };
    st.my_func()
})");
}

TEST_F(ParserTest, ImplWithMethodMultipleParams) {
  RoundTrip(R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func(self, mult: u9) -> u9 {
        self.a * mult
    }
})");
}

TEST(ParserErrorTest, ImplWithMisplacedSelf) {
  constexpr std::string_view kProgram = R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func(b: u32, self, mult: u9) -> u9 {
        self.a * mult
    }
}
fn main() -> u9 {
    self.a
})";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or.status(),
              IsPosError("ParseError",
                         HasSubstr("`self` must be the first parameter")));
}

TEST(ParserErrorTest, SelfOutsideImpl) {
  constexpr std::string_view kProgram = R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func(self, mult: u9) -> u9 {
        self.a * mult
    }
}
fn main() -> u9 {
    self.a
})";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError",
                 HasSubstr("Cannot find a definition for name: \"self\"")));
}

TEST(ParserErrorTest, ImplUsingSelfInFunction) {
  constexpr std::string_view kProgram = R"(struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    fn my_func(self, mult: u9) -> u9 {
        self.a * mult
    }

    fn static_func() -> u16 {
        self.b
    }
})";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError",
                 HasSubstr("Cannot find a definition for name: \"self\"")));
}

TEST_F(ParserTest, ImplSpanSetsCorrectSpan) {
  std::unique_ptr<Module> module = RoundTrip(R"(struct foo {
}
impl foo {
})");
  Impl* impl = module->GetImpls().at(0);
  EXPECT_EQ(impl->span().start().lineno(), 2);
  EXPECT_EQ(impl->span().start().colno(), 0);
  EXPECT_EQ(impl->span().limit().lineno(), 3);
  EXPECT_EQ(impl->span().limit().colno(), 1);
}

TEST_F(ParserTest, PubImplSetsCorrectSpan) {
  std::unique_ptr<Module> module = RoundTrip(R"(struct foo {
}
pub impl foo {
})");
  Impl* impl = module->GetImpls().at(0);
  EXPECT_EQ(impl->span().start().lineno(), 2);
  EXPECT_EQ(impl->span().start().colno(), 0);
  EXPECT_EQ(impl->span().limit().lineno(), 3);
  EXPECT_EQ(impl->span().limit().colno(), 1);
}

TEST_F(ParserTest, PubStructSetsCorrectSpan) {
  std::unique_ptr<Module> module = RoundTrip(R"(pub struct foo {
})");
  StructDef* struct_def = module->GetStructDefs().at(0);
  EXPECT_EQ(struct_def->span().start().lineno(), 0);
  EXPECT_EQ(struct_def->span().start().colno(), 0);
  EXPECT_EQ(struct_def->span().limit().lineno(), 1);
  EXPECT_EQ(struct_def->span().limit().colno(), 1);
}

TEST_F(ParserTest, ProcWithImplRoundTrip) {
  RoundTrip(R"(pub proc Foo {
    a: bits[9],
    b: bits[16],
}
impl Foo {
})");
}

TEST_F(ParserTest, ProcWithImplAndSemicolonSeparator) {
  constexpr std::string_view kProgram = R"(pub proc Foo {
    a: bits[9];
    b: bits[16];
}
impl Foo {
})";

  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Impl-style procs must use commas to separate members.")));
}

TEST_F(ParserTest, ProcWithImplEmptyInstantiation) {
  RoundTrip(R"(proc Foo {
}
impl Foo {
}
fn foo() -> Foo {
    Foo {  }
})");
}

TEST_F(ParserTest, ProcWithImplInstantiation) {
  RoundTrip(R"(proc Foo<N: u32> {
    foo: u32,
    bar: bits[N],
}
impl Foo {
}
fn foo() -> Foo<u32:8> {
    Foo<u32:8> { foo: u32:5, bar: u8:6 }
})");
}

TEST_F(ParserTest, ProcWithNoImplAndCommaSeparator) {
  constexpr std::string_view kProgram = R"(pub proc Foo {
    x: u32,
    y: u32,
    config() { () }
    init { (u32:0, u32:0) }
    next(addend: u32) { x + y + addend }
})";

  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Non-impl-style procs must use semicolons to "
                                 "separate members.")));
}

TEST_F(ParserTest, ImplWithConstRoundTrip) {
  RoundTrip(R"(pub struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    const FOO_VAL = u32:5;
    const FOO_STRING = "foo";
})");
}

TEST_F(ParserTest, PublicImplWithConstRoundTrip) {
  RoundTrip(R"(pub struct foo {
    a: bits[9],
    b: bits[16],
}
pub impl foo {
    const FOO_VAL = u32:5;
    const FOO_STRING = "foo";
})");
}

TEST_F(ParserTest, ImplWithParametricsRoundTrip) {
  RoundTrip(R"(pub struct foo<N: u32> {
    a: bits[9],
    b: bits[16],
}
impl foo<N> {
    const FOO_VAL = N;
    const FOO_STRING = "foo";
})");
}

TEST_F(ParserTest, ImplUseConstantDefRoundTrip) {
  RoundTrip(R"(pub struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    const FOO_VAL = u32:5;
    const FOO_STRING = "foo";
}
fn f(my_foo: foo) -> u32 {
    my_foo::FOO_VAL
})");
}

TEST_F(ParserTest, ImplWithFnDefinitionRoundTrip) {
  RoundTrip(R"(pub struct Foo {
    a: bits[9],
    b: bits[16],
}
impl Foo {
    fn get_foo() -> u32 {
        u32:5
    }
}
fn f(my_foo: Foo) -> u32 {
    my_foo.get_foo()
})");
}

TEST_F(ParserTest, ImplWithFnAndConstsDefinitionRoundTrip) {
  RoundTrip(R"(pub struct Foo {
    a: bits[9],
    b: bits[16],
}
impl Foo {
    const FOO_VAL = u32:5;
    fn get_foo() -> u32 {
        FOO_VAL
    }
    const SECOND_CONST = u32:1;
    fn nada() {}
}
fn f(my_foo: Foo) -> u32 {
    my_foo.get_foo()
})");
}

TEST_F(ParserTest, ParseErrorForDuplicateImpl) {
  constexpr std::string_view kProgram = R"(pub struct foo {
    a: bits[9],
    b: bits[16],
}
impl foo {
    const FOO_VAL = u32:5;
}
impl foo {
    const FOO_STRING = "foo";
})";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or.status(),
              IsPosError("ParseError", HasSubstr("can only be defined once")));
}

TEST(ParserErrorTest, ParseErrorForUseOutsideStruct) {
  constexpr std::string_view kProgram = R"(pub struct foo {
    a: bits[9],
    b: bits[16],
}

impl foo {
    const FOO_VAL = u32:5;
}

const MY_FOO = FOO_VAL;
)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError", HasSubstr("Cannot find a definition for name")));
}

TEST(ParserErrorTest, ParseErrorForImplOnBuiltin) {
  constexpr std::string_view kProgram = R"(impl u32 {
    const ONE = u32:1;
})";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError",
                 HasSubstr("'impl' can only be defined for a 'struct'")));
}

TEST(ParserErrorTest, ParseErrorForImplOnNonStruct) {
  constexpr std::string_view kProgram = R"(type foo = u32;

impl foo {
    const ONE = u32:1;
})";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError",
                 HasSubstr("'impl' can only be defined for a 'struct'")));
}

TEST(ParserErrorTest, ParseErrorForImplWithoutStructDef) {
  constexpr std::string_view kProgram = (R"(impl foo {
    const FOO_VAL = u32:5;
    const FOO_STRING = "foo";
})");
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError", HasSubstr("Cannot find a definition for name")));
}

TEST_F(ParserTest, StructDefRoundTrip) {
  RoundTrip(R"(pub struct foo<A: u32, B: bits[16]> {
    a: bits[A],
    b: bits[16][B],
})");
}

TEST_F(ParserTest, ParseErrorSpan) {
  Scanner scanner(file_table_, Fileno(0), "+");
  Parser parser("test_module", &scanner);
  Bindings b;
  absl::StatusOr<Expr*> expr_or = parser.ParseExpression(b);
  ASSERT_THAT(expr_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ParseError: <no-file>:1:1-1:2 Expected start of "
                       "an expression; got: +"));
}

TEST_F(ParserTest, EmptyTupleWithComma) {
  Scanner scanner(file_table_, Fileno(0), "(,)");
  Parser parser("test_module", &scanner);
  Bindings b;
  absl::StatusOr<Expr*> expr_or = parser.ParseExpression(b);
  ASSERT_THAT(
      expr_or,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "<no-file>:1:2-1:3 Expected start of an expression; got: ,")));
}

TEST_F(ParserTest, ParseLet) {
  const char* text = R"({
    let x: u32 = 2;
    x
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings b;
  XLS_ASSERT_OK_AND_ASSIGN(StatementBlock * block,
                           p.ParseBlockExpression(/*bindings=*/b));

  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 2);

  Let* let = std::get<Let*>(stmts.at(0)->wrapped());
  NameDef* name_def = std::get<NameDef*>(let->name_def_tree()->leaf());
  EXPECT_EQ(name_def->identifier(), "x");
  EXPECT_EQ(let->type_annotation()->ToString(), "u32");
  EXPECT_EQ(let->rhs()->ToString(), "2");

  Expr* e = std::get<Expr*>(stmts.at(1)->wrapped());
  auto* name_ref = dynamic_cast<NameRef*>(e);
  EXPECT_EQ(name_ref->ToString(), "x");
}

TEST_F(ParserTest, ParseLetWildcardBinding) {
  const char* text = R"({
  let _ = 2;
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings b;
  XLS_ASSERT_OK_AND_ASSIGN(StatementBlock * block,
                           p.ParseBlockExpression(/*bindings=*/b));

  ASSERT_TRUE(block->trailing_semi());
  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 1);

  Let* let = std::get<Let*>(stmts.at(0)->wrapped());
  EXPECT_EQ(
      AstNodeKindToString(ToAstNode(let->name_def_tree()->leaf())->kind()),
      "wildcard pattern");
  WildcardPattern* wildcard =
      std::get<WildcardPattern*>(let->name_def_tree()->leaf());
  ASSERT_NE(wildcard, nullptr);
  ASSERT_TRUE(let->name_def_tree()->IsWildcardLeaf());
}

TEST_F(ParserTest, ParseLetWildcardBindingTwice) {
  const char* text = R"({
  let (y, y) = (u32:0, u32:0);
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings b;
  auto block_or = p.ParseBlockExpression(/*bindings=*/b);
  ASSERT_FALSE(block_or.ok());
  EXPECT_THAT(block_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Name 'y' is defined twice in this pattern")));
}

TEST_F(ParserTest, ParseLetExpressionWithShadowing) {
  const char* text = R"({
    let x: u32 = 2;
    let x: u32 = 4;
    x
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings b;
  XLS_ASSERT_OK_AND_ASSIGN(StatementBlock * block,
                           p.ParseBlockExpression(/*bindings=*/b));

  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 3);

  Let* second_let = std::get<Let*>(stmts.at(1)->wrapped());

  Expr* e = std::get<Expr*>(stmts.at(2)->wrapped());
  auto* name_ref = dynamic_cast<NameRef*>(e);
  EXPECT_EQ(name_ref->ToString(), "x");
  EXPECT_EQ(std::get<const NameDef*>(name_ref->name_def()),
            std::get<NameDef*>(second_let->name_def_tree()->leaf()));
}

TEST_F(ParserTest, ParseBlockMultiLet) {
  const char* kProgram = R"({
    let x = f();
    let y = g(x);
    x + y
})";
  Scanner s{file_table_, Fileno(0), std::string{kProgram}};
  Parser p{"test", &s};
  Bindings bindings;
  Module& mod = p.module();
  bindings.Add("f", mod.GetOrCreateBuiltinNameDef("f"));
  bindings.Add("g", mod.GetOrCreateBuiltinNameDef("g"));
  XLS_ASSERT_OK_AND_ASSIGN(StatementBlock * block,
                           p.ParseBlockExpression(/*bindings=*/bindings));

  EXPECT_EQ(3, block->statements().size());
  Expr* add_expr = std::get<Expr*>(block->statements().back()->wrapped());
  auto* add = dynamic_cast<Binop*>(add_expr);
  NameRef* lhs = dynamic_cast<NameRef*>(add->lhs());
  const NameDef* lhs_def = std::get<const NameDef*>(lhs->name_def());
  EXPECT_NE(lhs_def->definer(), nullptr);
  EXPECT_EQ(lhs_def->definer()->ToString(), "f()");

  NameRef* rhs = dynamic_cast<NameRef*>(add->rhs());
  const NameDef* rhs_def = std::get<const NameDef*>(rhs->name_def());
  EXPECT_NE(rhs_def->definer(), nullptr);
  EXPECT_EQ(rhs_def->definer()->ToString(), "g(x)");
}

TEST_F(ParserTest, ParseIdentityFunction) {
  const char* text = "fn ident(x: bits) { x }";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));

  StatementBlock* block = f->body();
  ASSERT_TRUE(block != nullptr);
  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 1);
  Statement* stmt = stmts.at(0);

  NameRef* body = dynamic_cast<NameRef*>(std::get<Expr*>(stmt->wrapped()));
  ASSERT_TRUE(body != nullptr);
  EXPECT_EQ(body->identifier(), "x");
}

TEST_F(ParserTest, ParseSimpleProc) {
  const char* text = R"(proc simple {
    x: u32;
    config() {
        ()
    }
    init {
        u32:0
    }
    next(addend: u32) {
        x + addend
    }
})";

  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  auto proc =
      parser.ParseProc(Pos(), /*is_public=*/false, /*bindings=*/bindings);
  if (!proc.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(proc.status(), file_table_, vfs);
    XLS_ASSERT_OK(proc.status());
  }
  const Proc* p = std::get<Proc*>(*proc);
  EXPECT_EQ(p->ToString(), text);
}

TEST_F(ParserTest, TraceFmtNoArgs) {
  RoundTrip(R"(fn main() {
    trace_fmt!("Hello world!");
})");
}

TEST_F(ParserTest, TraceFmtWithArgs) {
  RoundTrip(R"(fn main() {
    let foo = u32:2;
    trace_fmt!("Hello {} {:x}!", "world", foo + u32:1);
})");
}

TEST_F(ParserTest, VTraceFmtNoArgs) {
  RoundTrip(R"(fn main() {
    vtrace_fmt!(3, "Hello world!");
})");
}

TEST_F(ParserTest, VTraceFmtWithArgs) {
  RoundTrip(R"(fn main() {
    const VERBOSITY = u32:4;
    let foo = u32:3;
    vtrace_fmt!(VERBOSITY + u32:1, "Hello {} {:x}!", "world", foo + u32:1);
})");
}

TEST_F(ParserTest, ParseProcWithConst) {
  RoundTrip(R"(proc simple {
    const MAX_X = u32:10;
    x: u32;
    config() {
        ()
    }
    init {
        u32:0
    }
    next(addend: u32) {
        if x > MAX_X { x } else { x + addend }
    }
})");
}

TEST_F(ParserTest, ParseParametricProcWithConstant) {
  RoundTrip(R"(proc MyProc<X: u32> {
    const INC = u32:20;
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(state: ()) {
        state + INC
    }
})");
}

TEST_F(ParserTest, ParseParametricProcWithConstantRefParam) {
  RoundTrip(R"(proc MyProc<X: u32> {
    const DOUBLE_X = u32:2 * X;
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(state: ()) {
        state + DOUBLE_X
    }
})");
}

TEST_F(ParserTest, ParseProcWithConstantRefFunction) {
  RoundTrip(R"(fn double(x: u32) -> u32 {
    x * u32:2
}
proc MyProc<X: u32> {
    const DOUBLE_X = double(X);
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(state: ()) {
        state + DOUBLE_X
    }
})");
}

TEST_F(ParserTest, ParseProcWithConstantRefGlobalConstant) {
  RoundTrip(R"(const MY_VAL = u32:15;
proc MyProc {
    const DOUBLE_VAL = u32:2 * MY_VAL;
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(state: ()) {
        state + DOUBLE_VAL
    }
})");
}

TEST_F(ParserTest, ParseProcWithConstantInType) {
  RoundTrip(R"(const MY_VAL = u32:15;
proc MyProc {
    const DOUBLE_VAL = u32:2 * MY_VAL;
    c: chan<uN[DOUBLE_VAL]> in;
    config(c: chan<uN[DOUBLE_VAL]> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(state: ()) {
        state + DOUBLE_VAL
    }
})");
}

TEST_F(ParserTest, ParseProcWithConstantRefConstant) {
  RoundTrip(R"(proc MyProc {
    const MY_VAL = u32:15;
    const DOUBLE_VAL = u32:2 * MY_VAL;
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(state: ()) {
        state + DOUBLE_VAL
    }
})");
}

TEST_F(ParserTest, ParseNextTooManyArgs) {
  const char* text = R"(proc confused {
    config() { () }
    init { () }
    next(state: (), more: u32, even_more: u64) { () }
})";

  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  auto proc =
      parser.ParseProc(Pos(), /*is_public=*/false, /*bindings=*/bindings);
  if (!proc.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(proc.status(), file_table_, vfs);
  }
  EXPECT_THAT(proc,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "<no-file>:4:47-4:47 A Proc next function takes one "
                           "argument (a recurrent state element); got 3 "
                           "parameters: [state, more, even_more]")));
}

TEST_F(ParserTest, ParseSimpleProcWithAlias) {
  const char* text = R"(proc simple {
    type MyU32 = u32;
    x: MyU32;
    config() {
        ()
    }
    init {
        MyU32:0
    }
    next(addend: MyU32) {
        x + addend
    }
})";

  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  auto proc =
      parser.ParseProc(Pos(), /*is_public=*/false, /*bindings=*/bindings);
  if (!proc.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(proc.status(), file_table_, vfs);
    XLS_ASSERT_OK(proc.status());
  }
  const Proc* p = std::get<Proc*>(*proc);
  EXPECT_EQ(p->ToString(), text)
      << "Proc ToString() did not match original text.";
}

TEST_F(ParserTest, ParseSimpleProcWithDepdenentTypeAlias) {
  const char* text = R"(proc simple {
    type MyU32 = u32;
    type MyOtherU32 = MyU32;
    x: MyOtherU32;
    config() {
        ()
    }
    init {
        MyOtherU32:0
    }
    next(addend: MyOtherU32) {
        x + addend
    }
})";

  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  auto proc =
      parser.ParseProc(Pos(), /*is_public=*/false, /*bindings=*/bindings);
  if (!proc.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(proc.status(), file_table_, vfs);
    XLS_ASSERT_OK(proc.status());
  }
  const Proc* p = std::get<Proc*>(*proc);
  EXPECT_EQ(p->ToString(), text)
      << "Proc ToString() did not match original text.";
}

// Parses the "iota" example.
TEST_F(ParserTest, ParseProcNetwork) {
  std::string_view kModule = R"(proc producer {
    c_: chan<u32> out;
    limit_: u32;
    config(limit: u32, c: chan<u32> out) {
        (c, limit)
    }
    init {
        u32:0
    }
    next(i: u32) {
        let tok = send(join(), c_, i);
        i + 1
    }
}
proc consumer<N: u32> {
    c_: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(i: u32) {
        let (tok1, e) = recv(join(), c_);
        i + 1
    }
}
proc main {
    config() {
        let (p, c) = chan<u32>("my_chan");
        spawn producer(u32:10, p);
        spawn consumer(range(10), c);
        ()
    }
    init {
        ()
    }
    next(state: ()) {
        ()
    }
})";

  RoundTrip(std::string(kModule));
}

// Parses the "iota" example with fifo_depth set on the internal channel.
TEST_F(ParserTest, ParseProcNetworkWithFifoDepthOnInternalChannel) {
  std::string_view kModule = R"(proc producer {
    c_: chan<u32> out;
    limit_: u32;
    config(limit: u32, c: chan<u32> out) {
        (c, limit)
    }
    init {
        u32:0
    }
    next(i: u32) {
        let tok = send(join(), c_, i);
        i + 1
    }
}
proc consumer<N: u32> {
    c_: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        u32:0
    }
    next(i: u32) {
        let (tok1, e) = recv(join(), c_);
        i + 1
    }
}
proc main {
    config() {
        let (p, c) = chan<u32, 2>("my_chan");
        spawn producer(u32:10, p);
        spawn consumer(range(10), c);
        ()
    }
    init {
        ()
    }
    next(state: ()) {
        ()
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ChannelsNotAsNextArgs) {
  const char* text = R"(proc producer {
    c: chan<u32> out;
    config(c: chan<u32> out) {
        (c,)
    }
    next(i: (chan<u32> out, u32)) {
        let tok = send(join(), c, i);
        (c, i + i)
    }
})";

  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  auto status_or_module = parser.ParseModule();
  EXPECT_THAT(status_or_module,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Channels cannot be Proc next params.")));
}

TEST_F(ParserTest, ChannelArraysOneD) {
  constexpr std::string_view kModule = R"(proc consumer {
    c: chan<u32> out;
    config(c: chan<u32> out) {
        (c,)
    }
    init {
        u32:100
    }
    next(i: u32) {
        recv(join(), c);
        i + i
    }
}
proc producer {
    channels: chan<u32>[32] out;
    config() {
        let (ps, cs) = chan<u32>[32]("my_chan");
        spawn consumer(cs[0]);
        (ps,)
    }
    init {
        ()
    }
    next(state: ()) {
        send(join(), channels[0], u32:0);
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ChannelArraysThreeD) {
  constexpr std::string_view kModule = R"(proc consumer {
    c: chan<u32> out;
    config(c: chan<u32> out) {
        (c,)
    }
    init {
        u32:0
    }
    next(i: u32) {
        let tok = recv(join(), c);
        i + i
    }
}
proc producer {
    channels: chan<u32>[32][64][128] out;
    config() {
        let (ps, cs) = chan<u32>[32][64][128]("my_chan");
        spawn consumer(cs[0]);
        (ps,)
    }
    init {
        ()
    }
    next(state: ()) {
        send(join(), channels[0][1][2], u32:0);
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ParseSendIfAndRecvIf) {
  constexpr std::string_view kModule = R"(proc producer {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        false
    }
    next(do_send: bool) {
        send_if(join(), c, do_send, do_send as u32);
        (!do_send,)
    }
}
proc consumer {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        false
    }
    next(do_recv: bool) {
        let (_, foo) = recv_if(join(), c, do_recv, u32:0);
        let _ = assert_eq(foo, true);
        (!do_recv,)
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ParseSendIfAndRecvNb) {
  constexpr std::string_view kModule = R"(proc producer {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        false
    }
    next(do_send: bool) {
        let _ = send_if(join(), c, do_send, do_send as u32);
        (!do_send,)
    }
}
proc consumer {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        ()
    }
    next(state: ()) {
        let (_, foo, valid) = recv_non_blocking(join(), c, u32:0);
        assert_eq(!(foo ^ valid), true);
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ParseRecvIfNb) {
  constexpr std::string_view kModule = R"(proc foo {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        (c,)
    }
    init {
        ()
    }
    next(state: ()) {
        recv_if_non_blocking(join(), c, true, u32:0);
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ParseJoin) {
  constexpr std::string_view kModule = R"(proc foo {
    c0: chan<u32> out;
    c1: chan<u32> out;
    c2: chan<u32> out;
    c3: chan<u32> in;
    config(c0: chan<u32> out, c1: chan<u32> out, c2: chan<u32> out, c3: chan<u32> in) {
        (c0, c1, c2, c3)
    }
    init {
        u32:0
    }
    next(state: u32) {
        let tok = join();
        let tok0 = send(tok, c0, state as u32);
        let tok1 = send(tok, c1, state as u32);
        let tok2 = send(tok, c2, state as u32);
        let tok3 = send(tok0, c0, state as u32);
        let tok = join(tok0, tok1, tok2, send(tok0, c0, state as u32));
        let tok = recv(tok, c3);
        state + u32:1
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ParseTestProc) {
  constexpr std::string_view kModule = R"(proc testee {
    input: chan<u32> in;
    output: chan<u32> out;
    config(input: chan<u32> in, output: chan<u32> out) {
        (input, output)
    }
    init {
        u32:0
    }
    next(x: u32) {
        let (tok, y) = recv(join(), input);
        let tok = send(join(), output, x + y);
        (x + y,)
    }
}
#[test_proc]
proc tester {
    p: chan<u32> out;
    c: chan<u32> in;
    terminator: chan<u32> out;
    config(terminator: chan<u32> out) {
        let (input_p, input_c) = chan<u32>("input");
        let (output_p, output_c) = chan<u32>("output");
        spawn testee(input_c, output_p);
        (input_p, output_c, terminator)
    }
    init {
        u32:0
    }
    next(iter: u32) {
        let tok = join();
        let tok = send(tok, p, u32:0);
        let tok = send(tok, p, u32:1);
        let tok = send(tok, p, u32:2);
        let tok = send(tok, p, u32:3);
        let (tok, exp) = recv(tok, c);
        assert_eq(exp, u32:0);
        let (tok, exp) = recv(tok, c);
        assert_eq(exp, u32:1);
        let (tok, exp) = recv(tok, c);
        assert_eq(exp, u32:3);
        let (tok, exp) = recv(tok, c);
        assert_eq(exp, u32:6);
        let tok = send_if(tok, terminator, iter == u32:4, true);
        (iter + u32:1,)
    }
})";

  RoundTrip(std::string(kModule));
}

TEST_F(ParserTest, ParseStructSplat) {
  const char* text = R"(struct Point {
    x: u32,
    y: u32,
}
fn f(p: Point) -> Point {
    Point { x: u32:42, ..p }
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m, parser.ParseModule());
  XLS_ASSERT_OK_AND_ASSIGN(TypeDefinition c, m->GetTypeDefinition("Point"));
  ASSERT_TRUE(std::holds_alternative<StructDef*>(c));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, m->GetMemberOrError<Function>("f"));

  StatementBlock* block = f->body();
  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 1);
  Statement* stmt = stmts.at(0);

  SplatStructInstance* ssi =
      dynamic_cast<SplatStructInstance*>(std::get<Expr*>(stmt->wrapped()));
  ASSERT_TRUE(ssi != nullptr) << f->body()->ToString();
  NameRef* splatted = dynamic_cast<NameRef*>(ssi->splatted());
  ASSERT_TRUE(splatted != nullptr) << ssi->splatted()->ToString();
  EXPECT_EQ(splatted->identifier(), "p");
}

TEST_F(ParserTest, ConcatFunction) {
  // TODO(leary): 2021-01-24 Notably just "bits" is not a valid type here,
  // should make a test that doesn't make it through typechecking if it's not a
  // parse-time error.
  const char* text = "fn concat(x: bits, y: bits) { x ++ y }";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));
  EXPECT_EQ(f->params().size(), 2);

  StatementBlock* block = f->body();
  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 1);
  Statement* stmt = stmts.at(0);

  Binop* body = dynamic_cast<Binop*>(std::get<Expr*>(stmt->wrapped()));
  ASSERT_TRUE(body != nullptr);
  EXPECT_EQ(body->binop_kind(), BinopKind::kConcat);
  NameRef* lhs = dynamic_cast<NameRef*>(body->lhs());
  ASSERT_TRUE(lhs != nullptr);
  EXPECT_EQ(lhs->identifier(), "x");
  NameRef* rhs = dynamic_cast<NameRef*>(body->rhs());
  EXPECT_EQ(rhs->identifier(), "y");
}

// Verifies that the parameter sequence to a function can have a trailing comma
// and it's not a parse error.
TEST_F(ParserTest, TrailingParameterComma) {
  const char* text = R"(
fn concat(
  x: bits,
  y: bits,
) {
  x ++ y
}
)";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));
  EXPECT_EQ(f->params().size(), 2);
}

TEST_F(ParserTest, BitSlice) {
  const char* text = R"(
fn f(x: u32) -> u8 {
  x[0:8]
}
)";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));

  StatementBlock* block = f->body();
  absl::Span<Statement* const> stmts = block->statements();
  ASSERT_EQ(stmts.size(), 1);
  Statement* stmt = stmts.at(0);

  auto* index = dynamic_cast<Index*>(std::get<Expr*>(stmt->wrapped()));
  ASSERT_NE(index, nullptr);
  IndexRhs rhs = index->rhs();
  ASSERT_TRUE(std::holds_alternative<Slice*>(rhs));
  auto* slice = std::get<Slice*>(rhs);
  EXPECT_EQ(slice->start()->ToString(), "0");
  EXPECT_EQ(slice->limit()->ToString(), "8");
}

TEST_F(ParserTest, LocalConstBinding) {
  const char* text = R"(fn f() -> u8 {
    const FOO = u8:42;
    FOO
})";
  RoundTrip(text);
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));
  StatementBlock* body = f->body();
  absl::Span<Statement* const> stmts = body->statements();
  ASSERT_EQ(stmts.size(), 2);

  Let* const_let = std::get<Let*>(stmts.at(0)->wrapped());
  ASSERT_NE(const_let, nullptr);
  ASSERT_TRUE(const_let->is_const());
  EXPECT_EQ("u8:42", const_let->rhs()->ToString());

  auto* const_ref =
      dynamic_cast<ConstRef*>(std::get<Expr*>(stmts.at(1)->wrapped()));
  ASSERT_NE(const_ref, nullptr);
  const NameDef* name_def = const_ref->name_def();
  EXPECT_EQ(name_def->ToString(), "FOO");
  AstNode* definer = name_def->definer();
  EXPECT_EQ(definer, const_let);
}

TEST_F(ParserTest, ParenthesizedUnop) {
  Expr* e = RoundTripExpr("(!x)", {"x"});
  EXPECT_EQ(e->span().ToString(file_table_), "<no-file>:1:2-1:4");
}

TEST_F(ParserTest, BitSliceOfCall) { RoundTripExpr("id(x)[0:8]", {"id", "x"}); }

TEST_F(ParserTest, BitSliceOfBitSlice) { RoundTripExpr("x[0:8][4:]", {"x"}); }

TEST_F(ParserTest, BitSliceWithWidth) { RoundTripExpr("x[1+:u8]", {"x"}); }

TEST_F(ParserTest, CmpChainParensOnLhs) {
  RoundTripExpr("(x == y) == z", {"x", "y", "z"});
}

TEST_F(ParserTest, CmpChainParensOnRhs) {
  RoundTripExpr("x == (y == z)", {"x", "y", "z"});
}

TEST_F(ParserTest, CmpChainParensOnLhsAndRhs) {
  RoundTripExpr("(x == y) == (y == z)", {"x", "y", "z"});
}

TEST_F(ParserTest, ZeroMacroSimple) {
  RoundTripExpr("zero!<u32>()", {}, /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, ZeroMacroSimpleStruct) {
  RoundTripExpr("zero!<MyType>()", {"MyType"}, /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, ZeroMacroSimpleArray) {
  RoundTripExpr("zero!<u32[10]>()", {}, /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, ZeroMacroSimpleBitsArray) {
  RoundTripExpr("zero!<bits[32][10]>()", {}, /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, ZeroMacroArrayOfImportedType) {
  RoundTrip(R"(import bar;
fn foo() -> bar::T[2] {
    zero!<bar::T[2]>()
})");
}

TEST_F(ParserTest, ZeroMacroImportedTypeWithParametric) {
  RoundTrip(R"(import bar;
fn foo() -> bar::T<2> {
    zero!<bar::T<2>>()
})");
}

TEST_F(ParserTest, ZeroMacroArrayOfImportedTypeWithParametric) {
  RoundTrip(R"(import bar;
fn foo() -> bar::T<2>[3] {
    zero!<bar::T<2>[3]>()
})");
}

TEST_F(ParserTest, ZeroMacroSimpleStructArray) {
  const char* text = R"(zero!<MyType[10]>())";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};

  Bindings b;

  Module& mod = p.module();
  for (auto const& it : GetParametricBuiltins()) {
    std::string name(it.first);
    b.Add(name, mod.GetOrCreateBuiltinNameDef(name));
  }

  NameDef* name_def =
      mod.Make<NameDef>(Span::Fake(), std::string("MyType"), nullptr);

  auto* struct_def = mod.Make<StructDef>(Span::Fake(), name_def,
                                         std::vector<ParametricBinding*>(),
                                         std::vector<StructMember>{}, false);
  b.Add(name_def->identifier(), struct_def);

  auto expr_or = p.ParseExpression(/*bindings=*/b);
  if (!expr_or.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(expr_or.status(), file_table_, vfs);
  }
  ASSERT_TRUE(expr_or.ok());
}

TEST_F(ParserTest, ZeroMacroParametricStruct) {
  const char* text = R"(zero!<MyType<MyParm0, MyParm1>>())";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};

  Bindings b;

  Module& mod = p.module();
  for (auto const& it : GetParametricBuiltins()) {
    std::string name(it.first);
    b.Add(name, mod.GetOrCreateBuiltinNameDef(name));
  }

  std::vector<ParametricBinding*> params;

  NameDef* name_def =
      mod.Make<NameDef>(Span::Fake(), std::string("MyParm0"), nullptr);
  b.Add(name_def->identifier(), name_def);
  BuiltinType builtin_type = BuiltinTypeFromString("u32").value();
  TypeAnnotation* elem_type = mod.Make<BuiltinTypeAnnotation>(
      Span::Fake(), builtin_type, mod.GetOrCreateBuiltinNameDef(builtin_type));
  params.push_back(mod.Make<ParametricBinding>(name_def, elem_type, nullptr));

  name_def = mod.Make<NameDef>(Span::Fake(), std::string("MyParm1"), nullptr);
  b.Add(name_def->identifier(), name_def);
  builtin_type = BuiltinTypeFromString("u32").value();
  elem_type = mod.Make<BuiltinTypeAnnotation>(
      Span::Fake(), builtin_type, mod.GetOrCreateBuiltinNameDef(builtin_type));
  params.push_back(mod.Make<ParametricBinding>(name_def, elem_type, nullptr));

  name_def = mod.Make<NameDef>(Span::Fake(), std::string("MyType"), nullptr);

  auto* struct_def = mod.Make<StructDef>(Span::Fake(), name_def, params,
                                         std::vector<StructMember>{}, false);
  b.Add(name_def->identifier(), struct_def);

  auto expr_or = p.ParseExpression(/*bindings=*/b);
  if (!expr_or.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(expr_or.status(), file_table_, vfs);
  }
  ASSERT_TRUE(expr_or.ok());
}

TEST_F(ParserTest, ZeroMacroParametricStructArray) {
  const char* text = R"(zero!<MyType<MyParm0, MyParm1>[10]>())";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};

  Bindings b;

  Module& mod = p.module();
  for (auto const& it : GetParametricBuiltins()) {
    std::string name(it.first);
    b.Add(name, mod.GetOrCreateBuiltinNameDef(name));
  }

  std::vector<ParametricBinding*> params;

  NameDef* name_def =
      mod.Make<NameDef>(Span::Fake(), std::string("MyParm0"), nullptr);
  b.Add(name_def->identifier(), name_def);
  BuiltinType builtin_type = BuiltinTypeFromString("u32").value();
  TypeAnnotation* elem_type = mod.Make<BuiltinTypeAnnotation>(
      Span::Fake(), builtin_type, mod.GetOrCreateBuiltinNameDef(builtin_type));
  params.push_back(mod.Make<ParametricBinding>(name_def, elem_type, nullptr));

  name_def = mod.Make<NameDef>(Span::Fake(), std::string("MyParm1"), nullptr);
  b.Add(name_def->identifier(), name_def);
  builtin_type = BuiltinTypeFromString("u32").value();
  elem_type = mod.Make<BuiltinTypeAnnotation>(
      Span::Fake(), builtin_type, mod.GetOrCreateBuiltinNameDef(builtin_type));
  params.push_back(mod.Make<ParametricBinding>(name_def, elem_type, nullptr));

  name_def = mod.Make<NameDef>(Span::Fake(), std::string("MyType"), nullptr);

  auto* struct_def = mod.Make<StructDef>(Span::Fake(), name_def, params,
                                         std::vector<StructMember>{}, false);
  b.Add(name_def->identifier(), struct_def);

  auto expr_or = p.ParseExpression(/*bindings=*/b);
  if (!expr_or.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(expr_or.status(), file_table_, vfs);
  }
  ASSERT_TRUE(expr_or.ok());
}

TEST_F(ParserTest, AllonesMacroSimple) {
  RoundTripExpr("all_ones!<u32>()", {}, /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, AllOnesMacroSimpleStruct) {
  RoundTripExpr("all_ones!<MyType>()", {"MyType"},
                /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, AllOnesMacroSimpleArray) {
  RoundTripExpr("all_ones!<u32[10]>()", {}, /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, AllOnesMacroSimpleBitsArray) {
  RoundTripExpr("all_ones!<bits[32][10]>()", {},
                /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, LocalParametricStruct) {
  ExpectParsesSuccessfully(R"(struct my_struct<N: u32> {
    my_field: uN[N],
}
const local_struct = my_struct<5> { my_field: u5:10 };)");
}

TEST_F(ParserTest, LocalParametricStructArray) {
  ExpectParsesSuccessfully(R"(struct my_struct<N: u32> {
    my_field: uN[N],
}
const local_struct = my_struct<5>[1]:[
  my_struct { my_field: u5:10 },
  my_struct { my_field: u5:11 }
];)");
}

TEST_F(ParserTest, ImportedStruct) {
  ExpectParsesSuccessfully(R"(import lib;
const lib_struct = lib::my_lib_struct { my_field: u5:10 };)");
}

// Exercises https://github.com/google/xls/issues/1030
TEST_F(ParserTest, ImportedParametricStruct) {
  ExpectParsesSuccessfully(R"(import lib;
const lib_struct = lib::my_lib_struct<5> { my_field: u5:10 };)");
}

TEST_F(ParserTest, SyntaxErrorParametricStruct) {
  constexpr std::string_view kProgram = R"(import lib;
const lib_struct = lib::my_lib_struct<{ my_field: u5:10 };)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected start of an expression; got: {")));
}

TEST_F(ParserTest, SyntaxErrorNoParametricForParametricStruct) {
  constexpr std::string_view kProgram = R"(import lib;
const lib_struct = lib::my_lib_struct<,>{ my_field: u5:10 };)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected start of an expression; got: ,")));
}

TEST_F(ParserTest, ImportedEnum) {
  ExpectParsesSuccessfully(R"(import lib;
const lib_enum_value = lib::my_lib_enum.value;
)");
}

TEST_F(ParserTest, ImportedParametricFn) {
  ExpectParsesSuccessfully(R"(import lib;
fn main() -> () {
    lib::my_proc<5>();
})");
}

TEST_F(ParserTest, ImportedParametricStructArray) {
  ExpectParsesSuccessfully(R"(import lib;
const imported_structs = lib::my_struct<5>[2]:[
  lib::my_struct { my_field: u5:10 },
  lib::my_struct { my_field: u5:11 }
];)");
}

TEST_F(ParserTest, ImportedParametricStructArrayField) {
  ExpectParsesSuccessfully(R"(import lib;
const imported_structs = lib::my_struct<5>[1]:[lib::my_struct { my_field: u5:10 }];
const imported_field = imported_structs[0].my_field;
)");
}

TEST_F(ParserTest, ImportedParametricLt) {
  ExpectParsesSuccessfully(R"(import lib;

fn main() -> bool {
  lib::OFFSET < u32:32
}
)");
}

TEST_F(ParserTest, ParseBlockWithTwoStatements) {
  RoundTripExpr(R"({
    type MyU32 = u32;
    MyU32:42
})");
}

TEST_F(ParserTest, ModuleConstWithEnumInside) {
  std::unique_ptr<Module> module = RoundTrip(R"(enum MyEnum : u2 {
    FOO = 0,
    BAR = 1,
}
const MY_TUPLE = (MyEnum::FOO, MyEnum::BAR) as (MyEnum, MyEnum);)");
  ASSERT_NE(module, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * my_enum,
                           module->GetMemberOrError<EnumDef>("MyEnum"));
  EXPECT_EQ(my_enum->span().ToString(file_table_), "<no-file>:1:1-4:2");
}

TEST_F(ParserTest, Struct) {
  const char* text = R"(struct Point {
    x: u32,
    y: u32,
})";
  RoundTrip(text);
}

TEST_F(ParserTest, StructAnnotation) {
  const char* text = R"(#[sv_type("cool")]
struct Point {
    x: u32,
    y: u32,
})";
  RoundTrip(text);
}

TEST_F(ParserTest, StructWithAccessFn) {
  const char* text = R"(struct Point {
    x: u32,
    y: u32,
}
fn f(p: Point) -> u32 {
    p.x
}
fn g(xy: u32) -> Point {
    Point { x: xy, y: xy }
})";
  RoundTrip(text);
}

TEST_F(ParserTest, ParametricWithEnumColonRefInvocation) {
  const char* text = R"(enum OneValue : u3 {
    ONE = 4,
}
fn p<X: OneValue>() -> OneValue {
    X
}
fn main() {
    p<OneValue::ONE>()
})";
  RoundTrip(text);
}

TEST_F(ParserTest, LetDestructureFlat) {
  RoundTripExpr(R"({
    let (x, y, z): (u32, u32, u32) = (1, 2, 3);
    y
})");
}

TEST_F(ParserTest, LetDestructureNested) {
  RoundTripExpr(
      R"({
    let (w, (x, (y)), z): (u32, (u32, (u32,)), u32) = (1, (2, (3,)), 4);
    y
})");
}

TEST_F(ParserTest, LetDestructureWildcard) {
  RoundTripExpr(R"({
    let (x, y, _): (u32, u32, u32) = (1, 2, 3);
    y
})");
}

TEST_F(ParserTest, LetDestructureRestEnd) {
  RoundTripExpr(R"({
    let (x, ..): (u32, u32, u32) = (1, 2, 3);
    x
})");
}

TEST_F(ParserTest, LetDestructureRestBeginning) {
  RoundTripExpr(R"({
    let (.., x): (u32, u32, u32) = (1, 2, 3);
    x
})");
}

TEST_F(ParserTest, LetDestructureRestMiddle) {
  RoundTripExpr(R"({
    let (x, .., y): (u32, u32, u32) = (1, 2, 3);
    x
})");
}

TEST_F(ParserTest, LetDestructureZeroMatches) {
  RoundTripExpr(R"({
    let (x, .., y): (u32, u32) = (1, 2);
    x
})");
}

TEST_F(ParserTest, For) {
  RoundTripExpr(R"({
    let accum: u32 = 0;
    let accum: u32 = for (i, accum): (u32, u32) in range(u32:0, u32:4) {
        let new_accum: u32 = accum + i;
        new_accum
    }(accum);
    accum
})",
                {"range"});
}

TEST_F(ParserTest, ForSansTypeAnnotation) {
  RoundTripExpr(
      R"({
    let init = ();
    for (i, accum) in range(u32:0, u32:4) {
        accum
    }(init)
})",
      {"range"});
}

TEST_F(ParserTest, MatchWithConstPattern) {
  RoundTrip(R"(const FOO = u32:64;
fn f(x: u32) {
    match x {
        FOO => u32:64,
        _ => u32:42,
    }
})");
}

TEST_F(ParserTest, MatchWithMultiplePatterns) {
  RoundTrip(R"(fn f(x: u32) {
    match x {
        u32:42 | u32:64 => u32:64,
        _ => u32:42,
    }
})");
}

TEST_F(ParserTest, MatchWithNumberRangePattern) {
  RoundTrip(R"(fn f(x: u32) {
    match x {
        u32:42..u32:64 => u32:64,
        _ => u32:42,
    }
})");
}

TEST_F(ParserTest, MatchWithRestOfTuple) {
  RoundTrip(R"(const FOO = u32:64;
fn f(x: u32) {
    match x {
        (FOO, ..) => u32:61,
        (.., FOO) => u32:62,
        (FOO, .., FOO) => u32:63,
        _ => u32:42,
    }
})");
}

TEST_F(ParserTest, MatchWithDuplicateRestOfTuple) {
  const char* kProgram = R"(const FOO = u32:64;
fn f(x: u32) {
    match x {
        (FOO, .., ..) => u32:61,
        _ => u32:42,
    }
})";
  EXPECT_NONFATAL_FAILURE(
      RoundTrip(kProgram),
      "Rest-of-tuple (`..`) can only be used once per tuple pattern");
}

TEST_F(ParserTest, MatchWithRestOfTupleOnly) {
  const char* kProgram = R"(const FOO = u32:64;
fn f(x: u32) {
    match x {
        .. => u32:41,
        _ => u32:42,
    }
})";
  EXPECT_NONFATAL_FAILURE(RoundTrip(kProgram),
                          "Cannot use both wildcard (`_`) and rest-of-tuple "
                          "(`..`) as match arms");
}

TEST_F(ParserTest, ArrayTypeAnnotation) {
  std::string s = "u8[2]";
  scanner_.emplace(file_table_, Fileno(0), s);
  parser_.emplace("test", &*scanner_);
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(TypeAnnotation * ta,
                           ParseTypeAnnotation(parser_.value(), bindings));

  auto* array_type = dynamic_cast<ArrayTypeAnnotation*>(ta);
  EXPECT_EQ(array_type->span(),
            Span(Pos(Fileno(0), 0, 0), Pos(Fileno(0), 0, 5)));
  EXPECT_EQ(array_type->ToString(), "u8[2]");
  EXPECT_EQ(array_type->element_type()->span(),
            Span(Pos(Fileno(0), 0, 0), Pos(Fileno(0), 0, 2)));
  EXPECT_EQ(array_type->element_type()->ToString(), "u8");
}

TEST_F(ParserTest, TupleArrayAndInt) {
  Expr* e = RoundTripExpr("(u8[4]:[1, 2, 3, 4], 7)", {}, false, std::nullopt);
  auto* tuple = dynamic_cast<XlsTuple*>(e);
  EXPECT_EQ(2, tuple->members().size());
  auto* array = tuple->members()[0];
  EXPECT_NE(dynamic_cast<ConstantArray*>(array), nullptr);
}

TEST_F(ParserTest, Cast) { RoundTripExpr("foo() as u32", {"foo"}); }

TEST_F(ParserTest, CastOfCast) { RoundTripExpr("x as s32 as u32", {"x"}); }

TEST_F(ParserTest, CheckedCast) {
  RoundTripExpr("checked_cast<u32>(foo())", {"foo"},
                /*populate_dslx_builtins=*/true, "checked_cast<u32>(foo())");
}

TEST_F(ParserTest, WideningCast) {
  RoundTripExpr("widening_cast<u32>(foo())", {"foo"},
                /*populate_dslx_builtins=*/true, "widening_cast<u32>(foo())");
}

TEST_F(ParserTest, WideningCastOfCheckedCastOfCast) {
  RoundTripExpr("widening_cast<u32>(checked_cast<u16>(x as u24))", {"x"},
                /*populate_dslx_builtins=*/true);
}

TEST_F(ParserTest, CastOfCastEnum) {
  RoundTrip(R"(enum MyEnum : u3 {
    SOME_VALUE = 0,
}
fn f(x: u8) -> MyEnum {
    x as u3 as MyEnum
})");
}

TEST_F(ParserTest, CastToTypeAlias) {
  RoundTrip(R"(type u128 = bits[128];
fn f(x: u32) -> u128 {
    x as u128
})");
}

TEST_F(ParserTest, xNTypeConstructor) {
  RoundTrip(R"(type u128 = xN[false][128];)");
  RoundTrip(R"(type s128 = xN[true][128];)");
}

// Note: parens around the left hand side are required or we attempt to parse
// the `u128<` as the start of a parameterized type for purposes of the cast.
TEST_F(ParserTest, CastToTypeAliasLtIdentifier) {
  RoundTrip(R"(type u128 = bits[128];
fn f(x: u32, y: u128) -> u128 {
    (x as u128) < y
})");
}

TEST_F(ParserTest, Enum) {
  RoundTrip(R"(enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
})");
}

TEST_F(ParserTest, EnumSvType) {
  RoundTrip(R"(#[sv_type("cool")]
enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
})");
}

TEST_F(ParserTest, ModuleWithSemis) {
  RoundTrip(R"(fn f() -> s32 {
    let x: s32 = 42;
    x
})");
}

TEST_F(ParserTest, ModuleWithParametric) {
  RoundTrip(R"(fn parametric<X: u32, Y: u32 = {X + X}>() -> (u32, u32) {
    (X, Y)
})");
}

TEST_F(ParserTest, ParametricInvocation) { RoundTripExpr("f<u32:2>()", {"f"}); }

TEST_F(ParserTest, ParametricInvocationWithCastToTypeAlias) {
  RoundTrip(R"(type foo_t = bits[32];
fn f<N: u32>(x: u32) -> u32 {
    N + x
}
fn main(x: u32) -> u32 {
    f<foo_t:1>(x)
})");
}

TEST_F(ParserTest, ParametricInvocationWithCastToColonRef) {
  RoundTrip(R"(import foo;
fn f<N: u32>(x: u32) -> u32 {
    N + x
}
fn main(x: u32) -> u32 {
    f<foo::bar_t:1>(x)
})");
}

TEST_F(ParserTest, ColonRef) {
  Expr* e = RoundTripExpr("BuiltinEnum::VALUE", /*predefine=*/{"BuiltinEnum"});
  EXPECT_EQ(e->span(), Span(Pos(Fileno(0), 0, 0), Pos(Fileno(0), 0, 18)));
}

TEST_F(ParserTest, ParametricColonRefInvocation) {
  RoundTripExpr("f<BuiltinEnum::VALUE>()", {"f", "BuiltinEnum"});
}

TEST_F(ParserTest, ModuleWithTypeAlias) { RoundTrip("type MyType = u32;"); }
TEST_F(ParserTest, ModuleWithTypeAliasSvType) {
  RoundTrip(R"(#[sv_type("cool")]
type MyType = u32;)");
}

TEST_F(ParserTest, ModuleWithImport) { RoundTrip("import thing;"); }

TEST_F(ParserTest, ModuleWithImportDots) {
  RoundTrip("import thing.subthing;");
}

TEST_F(ParserTest, ModuleWithImportAs) { RoundTrip("import thing as other;"); }

TEST_F(ParserTest, ConstArrayOfEnumRefs) {
  RoundTrip(R"(enum MyEnum : u3 {
    FOO = 1,
    BAR = 2,
}
const A = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR];)");
}

TEST_F(ParserTest, ImplicitWidthEnum) {
  RoundTrip(R"(const A = u32:42;
const B = u32:64;
enum ImplicitWidthEnum {
    FOO = A,
    BAR = B,
})");
}

TEST_F(ParserTest, ConstWithTypeAnnotation) {
  RoundTrip(R"(const MOL: u32 = u32:42;)");
}

TEST_F(ParserTest, ConstArrayOfConstRefs) {
  RoundTrip(R"(const MOL = u32:42;
const ZERO = u32:0;
const ARR = u32[2]:[MOL, ZERO];)");
}

// As above, but uses a trailing ellipsis in the array definition.
TEST_F(ParserTest, ConstArrayOfConstRefsEllipsis) {
  RoundTrip(R"(const MOL = u32:42;
const ZERO = u32:0;
const ARR = u32[2]:[MOL, ZERO, ...];)");
}

TEST_F(ParserTest, EllipsisNotInTrailingMiddle) {
  const char* text = R"({
  const ARR = u32[2]:[u32:0, ..., u32:0];
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings b;
  auto block_or = p.ParseBlockExpression(/*bindings=*/b);
  ASSERT_FALSE(block_or.ok());
  EXPECT_THAT(block_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Ellipsis may only be in trailing position.")));
}

TEST_F(ParserTest, EllipsisNotInTrailingLeading) {
  const char* text = R"({
  const ARR = u32[2]:[..., u32:0];
})";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings b;
  auto block_or = p.ParseBlockExpression(/*bindings=*/b);
  ASSERT_FALSE(block_or.ok());
  EXPECT_THAT(block_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Ellipsis may only be in trailing position.")));
}

TEST_F(ParserTest, QuickCheckDirective) {
  RoundTrip(R"(#[quickcheck]
fn foo(x: u5) -> bool {
    true
})");
}

TEST_F(ParserTest, QuickCheckDirectiveWithTestCount) {
  RoundTrip(R"(#[quickcheck(test_count=1024)]
fn foo(x: u5) -> bool {
    true
})");
}

TEST_F(ParserTest, ModuleWithTypeAliasArrayTuple) {
  RoundTrip(R"(type MyType = u32;
type MyTupleType = (MyType[2],);)");
}

TEST_F(ParserTest, ModuleWithEmptyTestFunction) {
  std::unique_ptr<Module> mod = RoundTrip(R"(#[test]
fn example() {
    ()
})");
  ASSERT_EQ(mod->top().size(), 1);
  auto* tf = std::get<TestFunction*>(mod->top()[0]);
  EXPECT_EQ(tf->span().ToString(file_table_), "<no-file>:1:1-4:2");
}

TEST_F(ParserTest, ModuleWithEmptyExternVerilogFunction) {
  RoundTrip(R"(#[extern_verilog("unit")]
fn example() {
    ()
})");
}

TEST_F(ParserTest, ModuleWithTestFunction) {
  RoundTrip(R"(fn id(x: u32) -> u32 {
    x
}
#[test]
fn id_4() {
    assert_eq(u32:4, id(u32:4))
})");
}

TEST_F(ParserTest, TypeAliasForTupleWithConstSizedArray) {
  RoundTrip(R"(const HOW_MANY_THINGS = u32:42;
type MyTupleType = (u32[HOW_MANY_THINGS],);
fn get_things(x: MyTupleType) -> u32[HOW_MANY_THINGS] {
    x[0]
})");
}

TEST_F(ParserTest, ArrayOfNameRefs) {
  RoundTripExpr("[a, b, c, d]", {"a", "b", "c", "d"});
}

TEST_F(ParserTest, EmptyTuple) {
  Expr* e = RoundTripExpr("()", {}, false, std::nullopt);
  auto* tuple = dynamic_cast<XlsTuple*>(e);
  ASSERT_NE(tuple, nullptr);
  EXPECT_TRUE(tuple->empty());
}

TEST_F(ParserTest, SingleElementTuple) {
  Expr* e = RoundTripExpr("(a,)", {"a"}, false, std::nullopt);
  auto* tuple = dynamic_cast<XlsTuple*>(e);
  ASSERT_NE(tuple, nullptr);
  EXPECT_EQ(tuple->members().size(), 1);
  EXPECT_TRUE(tuple->has_trailing_comma());
}

TEST_F(ParserTest, Match) {
  RoundTripExpr(R"(match x {
    u32:42 => u32:64,
    _ => u32:42,
})",
                /*predefine=*/{"x"});
}

TEST_F(ParserTest, MatchFreevars) {
  Expr* e = RoundTripExpr(R"(match x {
    y => z,
})",
                          {"x", "y", "z"});
  FreeVariables fv = GetFreeVariablesByPos(e, &e->span().start());
  EXPECT_THAT(fv.Keys(), testing::ContainerEq(
                             absl::flat_hash_set<std::string>{"x", "y", "z"}));
}

TEST_F(ParserTest, ForFreevars) {
  Expr* e = RoundTripExpr(R"(for (i, accum): (u32, u32) in range(u32:4) {
    let new_accum: u32 = accum + i + j;
    new_accum
}(u32:0))",
                          {"range", "j"});
  FreeVariables fv = GetFreeVariablesByPos(e, &e->span().start());
  EXPECT_THAT(fv.Keys(), testing::ContainerEq(
                             absl::flat_hash_set<std::string>{"j", "range"}));
}

TEST_F(ParserTest, EmptyTernary) { RoundTripExpr("if true {} else {}"); }

TEST_F(ParserTest, TernaryConditional) {
  Expr* e = RoundTripExpr("if true { u32:42 } else { u32:24 }", {});

  EXPECT_FALSE(down_cast<Conditional*>(e)->HasElseIf());
  EXPECT_FALSE(down_cast<Conditional*>(e)->HasMultiStatementBlocks());

  RoundTripExpr(R"(if really_long_identifier_so_that_this_is_too_many_chars {
    u32:42
} else {
    u32:24
})",
                {"really_long_identifier_so_that_this_is_too_many_chars"});
}

TEST_F(ParserTest, LadderedConditional) {
  Expr* e = RoundTripExpr(
      "if true { u32:42 } else if false { u32:33 } else { u32:24 }");

  EXPECT_TRUE(down_cast<Conditional*>(e)->HasElseIf());
  EXPECT_FALSE(down_cast<Conditional*>(e)->HasMultiStatementBlocks());

  RoundTripExpr(
      R"(if really_long_identifier_so_that_this_is_too_many_chars {
    u32:42
} else if another_really_long_identifier_so_that_this_is_too_many_chars {
    u32:22
} else {
    u32:24
})",
      {"really_long_identifier_so_that_this_is_too_many_chars",
       "another_really_long_identifier_so_that_this_is_too_many_chars"});
}

TEST_F(ParserTest, TernaryWithComparisonTest) {
  RoundTripExpr("if a <= b { u32:42 } else { u32:24 }", {"a", "b"});
}

TEST_F(ParserTest, TernaryWithComparisonToColonRefTest) {
  RoundTripExpr("if a <= m::b { u32:42 } else { u32:24 }", {"a", "m"});
}

TEST_F(ParserTest, ForInWithColonRefAsRangeLimit) {
  RoundTripExpr(R"(for (x, s) in u32:0..m::SOME_CONST {
    x
}(i))",
                {"m", "i"});
}

TEST_F(ParserTest, TernaryWithOrExpressionTest) {
  RoundTripExpr("if a || b { u32:42 } else { u32:24 }", {"a", "b"});
}

TEST_F(ParserTest, TernaryWithComparisonStructInstanceTest) {
  RoundTrip(R"(struct MyStruct {
    x: u32
}
fn f(a: MyStruct) -> u32 {
    if a.x <= MyStruct { x: u32:42 }.x { u32:42 } else { u32:24 }
})",
            /*target=*/R"(struct MyStruct {
    x: u32,
}
fn f(a: MyStruct) -> u32 {
    if a.x <= MyStruct { x: u32:42 }.x { u32:42 } else { u32:24 }
})");
}

TEST_F(ParserTest, ConstantArray) {
  Expr* e = RoundTripExpr("u32[2]:[0, 1]", {}, false, std::nullopt);
  ASSERT_TRUE(dynamic_cast<ConstantArray*>(e) != nullptr);
}

TEST_F(ParserTest, MixedArrayIsNotConstantArray) {
  std::unique_ptr<Module> module = RoundTrip(R"(const a = u32:0;
const b = u32[2]:[a, 1];)");
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant_def,
                           module->GetConstantDef("b"));
  auto array = constant_def->value();
  ASSERT_TRUE(dynamic_cast<Array*>(array) != nullptr);
}

TEST_F(ParserTest, NonConstantArrayIsNotConstantArray) {
  std::unique_ptr<Module> module = RoundTrip(R"(const a = u32:0;
const b = u32[2]:[a, a];)");
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant_def,
                           module->GetConstantDef("b"));
  auto array = constant_def->value();
  ASSERT_TRUE(dynamic_cast<Array*>(array) != nullptr);
}

TEST_F(ParserTest, EmptyArrayIsConstant) {
  Expr* e = RoundTripExpr("u32[2]:[]", {}, false, std::nullopt);
  ASSERT_TRUE(dynamic_cast<ConstantArray*>(e) != nullptr);
}

TEST_F(ParserTest, DoubleNegation) { RoundTripExpr("!!x", {"x"}, false); }

TEST_F(ParserTest, ArithmeticOperatorPrecedence) {
  Expr* e = RoundTripExpr("-a + b % c", {"a", "b", "c"});
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->binop_kind(), BinopKind::kAdd);

  auto* unop = dynamic_cast<Unop*>(binop->lhs());
  EXPECT_EQ(unop->unop_kind(), UnopKind::kNegate);

  auto* binop_rhs = dynamic_cast<Binop*>(binop->rhs());
  EXPECT_EQ(binop_rhs->binop_kind(), BinopKind::kMod);
}

TEST_F(ParserTest, LogicalOperatorPrecedence) {
  Expr* e = RoundTripExpr("!a || !b && c", {"a", "b", "c"});
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->binop_kind(), BinopKind::kLogicalOr);
  auto* binop_rhs = dynamic_cast<Binop*>(binop->rhs());
  EXPECT_EQ(binop_rhs->binop_kind(), BinopKind::kLogicalAnd);
  auto* unop = dynamic_cast<Unop*>(binop_rhs->lhs());
  EXPECT_EQ(unop->unop_kind(), UnopKind::kInvert);
}

TEST_F(ParserTest, LogicalEqualityPrecedence) {
  Expr* e = RoundTripExpr("a ^ !b == f()", {"a", "b", "f"});
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->binop_kind(), BinopKind::kEq);
  auto* binop_lhs = dynamic_cast<Binop*>(binop->lhs());
  EXPECT_EQ(binop_lhs->binop_kind(), BinopKind::kXor);
  auto* unop = dynamic_cast<Unop*>(binop_lhs->rhs());
  EXPECT_EQ(unop->unop_kind(), UnopKind::kInvert);
}

TEST_F(ParserTest, CastVsComparatorPrecedence) {
  Expr* e = RoundTripExpr("x >= y as u32", /*predefine=*/{"x", "y"});
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->binop_kind(), BinopKind::kGe);
  auto* cast = dynamic_cast<Cast*>(binop->rhs());
  ASSERT_NE(cast, nullptr);
  auto* casted_name_ref = dynamic_cast<NameRef*>(cast->expr());
  ASSERT_NE(casted_name_ref, nullptr);
  EXPECT_EQ(casted_name_ref->identifier(), "y");
}

TEST_F(ParserTest, CastVsUnaryPrecedence) {
  Expr* e = RoundTripExpr("-x as s32", /*predefine=*/{"x", "y"});
  auto* cast = dynamic_cast<Cast*>(e);
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->type_annotation()->ToString(), "s32");
}

TEST_F(ParserTest, NameDefTree) {
  RoundTripExpr(R"({
    let (a, (b, (c, d), e), f) = x;
    a
})",
                {"x"});
}

TEST_F(ParserTest, Strings) {
  RoundTripExpr(R"({
    let x = "dummy --> \" <-- string";
    x
})");
  RoundTripExpr(R"({
    let x = "dummy --> \"";
    x
})");
}

TEST_F(ParserTest, TupleCast) {
  const char* text = R"(
fn f() -> u8 {
    let a = (u8, u16):(u8:8, u16:64);
    a.0
}
)";
  Scanner s(file_table_, Fileno(0), std::string(text));
  Parser p("test", &s);
  Bindings bindings;
  XLS_ASSERT_OK(
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));
}

TEST_F(ParserTest, TupleArrayCast) {
  const char* text = R"(
fn f() -> u8 {
    let a = (u8, u16)[1]:[(u8:8, u16:64)];
    a[0].0
}
)";
  Scanner s(file_table_, Fileno(0), std::string(text));
  Parser p("test", &s);
  Bindings bindings;
  XLS_ASSERT_OK(
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));
}

TEST_F(ParserTest, InvalidParenthetical) {
  const char* text = R"(
fn f() -> u8 {
    let a = (u8, u16);
    a.0
}
)";
  Scanner s(file_table_, Fileno(0), std::string(text));
  Parser p("test", &s);
  Bindings bindings;
  auto function_or =
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings);
  EXPECT_THAT(function_or.status(),
              IsPosError("ParseError",
                         HasSubstr("Expected ':', got ',': Expect colon after "
                                   "type annotation in cast")));
}

TEST_F(ParserTest, NestedTupleCast) {
  const char* text = R"(
fn f() -> u8 {
    type MyTuple = (u8, u16);

    let a = 2*(MyTuple:(u8:5, u16: 42) + (u8, u16):(u8:8, u16:64)) + MyTuple:(u8:7, u16:42);
    a.0
}
)";
  Scanner s(file_table_, Fileno(0), std::string(text));
  Parser p("test", &s);
  Bindings bindings;
  XLS_ASSERT_OK(
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));
}

TEST_F(ParserTest, TupleIndex) {
  const char* text = R"(
fn f(x: u32) -> u8 {
    (u32:6, u32:7).1
}
)";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(Pos(), /*is_public=*/false, /*bindings=*/bindings));

  StatementBlock* body = f->body();
  absl::Span<Statement* const> stmts = body->statements();
  ASSERT_EQ(stmts.size(), 1);

  auto* tuple_index =
      dynamic_cast<TupleIndex*>(std::get<Expr*>(stmts.at(0)->wrapped()));
  ASSERT_NE(tuple_index, nullptr);

  Expr* lhs = tuple_index->lhs();
  EXPECT_EQ(lhs->ToString(), "(u32:6, u32:7)");
  Number* index = tuple_index->index();
  EXPECT_EQ(index->ToString(), "1");

  RoundTripExpr(R"({
    let foo = tuple.0;
    foo
})",
                {"tuple"});
  RoundTripExpr(R"({
    let foo = (u32:6, u32:7).1;
    foo
})",
                {"tuple"});
}

TEST_F(ParserTest, BlockWithinBlock) {
  const char* kInput = R"({
    let a = u32:0;
    let b = {
        let c = u32:1;
        c
    };
    let d = u32:2;
})";
  RoundTripExpr(kInput);
}

TEST_F(ParserTest, UnrollFor) {
  RoundTripExpr(
      R"({
    let bar = u32:0;
    let res = unroll_for! (i, acc) in range(u32:0, u32:4) {
        let foo = i + 1;
    }(u32:0);
    let baz = u32:0;
    res
})",
      /*predefine=*/{"range"});
}

TEST_F(ParserTest, Range) {
  RoundTripExpr(R"({
    let foo = u32:8..u32:16;
    foo
})");
  RoundTripExpr(R"({
    let foo = a..b;
    foo
})",
                {"a", "b"});
}

TEST_F(ParserTest, BuiltinFailWithLabels) {
  constexpr std::string_view kProgram = R"(fn main(x: u32) -> u32 {
    let _ = if x == u32:7 { fail!("x_is_7", u32:0) } else { u32:0 };
    let _ = {
        if x == u32:8 { fail!("x_is_8", u32:0) } else { u32:0 }
    };
    x
})";
  RoundTrip(std::string(kProgram));
}

TEST_F(ParserTest, ProcWithInit) {
  constexpr std::string_view kProgram = R"(proc foo {
    member: u32;
    config() {
        (u32:1,)
    }
    init {
        u32:0
    }
    next(state: u32) {
        state
    }
})";

  RoundTrip(std::string(kProgram));
}

// -- Parse-time errors

TEST_F(ParserTest, BadEnumRef) {
  const char* text = R"(
enum MyEnum : u1 {
    FOO = 0
}

fn my_fun() -> MyEnum {
    FOO  // Should be qualified as MyEnum::FOO!
}
)";
  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  auto module_status = parser.ParseModule();
  ASSERT_THAT(module_status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ParseError: <no-file>:7:5-7:8 Cannot find "
                       "a definition for name: \"FOO\""));
}

TEST_F(ParserTest, ProcConfigCannotSeeMembers) {
  constexpr std::string_view kProgram = R"(
proc main {
    x12: chan<u8> in;
    config(x27: chan<u8> in) {
        (x12,)
    }
    next(s: ()) {
        ()
    }
})";
  Scanner s{file_table_, Fileno(0), std::string{kProgram}};
  Parser parser{"test", &s};
  auto module_status = parser.ParseModule();
  ASSERT_THAT(
      module_status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               "ParseError: <no-file>:5:10-5:13 "
               "Cannot find a definition for name: \"x12\"; "
               "\"x12\" is a proc member, but those cannot be referenced "
               "from within a proc config function."));
}

TEST_F(ParserTest, ProcConfigCanSeeTypeAlias) {
  constexpr std::string_view kProgram = R"(
proc main {
    type MyU8RecvChan = chan<u8> in;
    x12: MyU8RecvChan;
    config(x27: MyU8RecvChan) {
        (x27,)
    }
    init { () }
    next(s: ()) { () }
})";
  Scanner s{file_table_, Fileno(0), std::string{kProgram}};
  Parser parser{"test", &s};
  XLS_ASSERT_OK(parser.ParseModule());
}

TEST_F(ParserTest, ProcDuplicateConfig) {
  constexpr std::string_view kProgram = R"(
proc main {
    x12: chan<u8> in;
    config(x27: chan<u8> in) { (x27,) }
    config(x27: chan<u8> in) { (x27,) }
    next(s: ()) { () }
})";
  Scanner s{file_table_, Fileno(0), std::string{kProgram}};
  Parser parser{"test", &s};
  auto module_status = parser.ParseModule();
  ASSERT_THAT(module_status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("proc `main` config function was already "
                                 "specified @ <no-file>:5:5-5:11")));
}

TEST_F(ParserTest, ProcDuplicateNext) {
  constexpr std::string_view kProgram = R"(
proc main {
    x12: chan<u8> in;
    config(x27: chan<u8> in) { (x27,) }
    next(s: ()) { () }
    next(s: ()) { () }
})";
  Scanner s{file_table_, Fileno(0), std::string{kProgram}};
  Parser parser{"test", &s};
  auto module_status = parser.ParseModule();
  ASSERT_THAT(module_status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("proc `main` next function was already "
                                 "specified @ <no-file>:6:5-6:9")));
}

TEST_F(ParserTest, NumberSpan) {
  XLS_ASSERT_OK_AND_ASSIGN(Expr * e, ParseExpr("u32:42"));
  auto* number = dynamic_cast<Number*>(e);
  ASSERT_NE(number, nullptr);
  EXPECT_EQ(number->span(), Span(Pos(Fileno(0), 0, 0), Pos(Fileno(0), 0, 6)));
}

TEST_F(ParserTest, DetectsDuplicateFailLabels) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u32 {
    let _ = if x == u32:7 { fail!("x_is_7", u32:0) } else { u32:0 };
    let _ = if x == u32:7 { fail!("x_is_7", u32:0) } else { u32:0 };
    x
}
)";

  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  absl::StatusOr<std::unique_ptr<Module>> module_or = parser.ParseModule();
  ASSERT_FALSE(module_or.ok()) << module_or.status();
  LOG(INFO) << "status: " << module_or.status();
  EXPECT_THAT(module_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("A fail label must be unique")));
}

TEST_F(ParserTest, ParseAllowNonstandardConstantNamingAnnotation) {
  constexpr std::string_view kProgram = R"(
#![allow(nonstandard_constant_naming)]
)";

  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  absl::StatusOr<std::unique_ptr<Module>> module_or = parser.ParseModule();
  ASSERT_TRUE(module_or.ok()) << module_or.status();
  EXPECT_THAT(
      module_or.value()->annotations(),
      testing::ElementsAre(ModuleAnnotation::kAllowNonstandardConstantNaming));
}

TEST_F(ParserTest, ParseTypeInferenceVersionAttributeWithValue1) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
#![type_inference_version = 1]
)"));
  EXPECT_THAT(module->annotations(), testing::IsEmpty());
}

TEST_F(ParserTest, ParseTypeInferenceVersionAttributeWithValue2) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
#![type_inference_version = 2]
)"));
  EXPECT_THAT(module->annotations(),
              testing::ElementsAre(ModuleAnnotation::kTypeInferenceVersion2));
}

TEST_F(ParserTest, ParseTypeInferenceVersionAttributeWithValue3Fails) {
  absl::StatusOr<std::unique_ptr<Module>> module = Parse(R"(
#![type_inference_version = 3]
)");
  EXPECT_THAT(module,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type inference version must be 1 or 2.")));
}

TEST_F(ParserTest, ParseTypeInferenceVersionAttributeWithNonIntegerFails) {
  absl::StatusOr<std::unique_ptr<Module>> module = Parse(R"(
#![type_inference_version = "2"]
)");
  EXPECT_THAT(
      module,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Type inference version must be an unquoted integer.")));
}

TEST_F(ParserTest, ParseTypeInferenceVersionAttributeWithParenFormatFails) {
  absl::StatusOr<std::unique_ptr<Module>> module = Parse(R"(
#![type_inference_version(2)]
)");
  EXPECT_THAT(module, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Expected '=', got '('")));
}

// Verifies that we can walk backwards through a tree. In this case, from the
// terminal node to the defining expr.
TEST(ParserBackrefTest, CanFindDefiner) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
    let foo = u32:0 + u32:1;
    let bar = u32:3 + foo;
    let baz = bar + foo;
    foo
}
)";

  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                           parser.ParseModule());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));

  StatementBlock* body = f->body();
  absl::Span<Statement* const> stmts = body->statements();
  ASSERT_EQ(stmts.size(), 4);

  // Get the terminal expr.
  Expr* current_expr = std::get<Expr*>(stmts.back()->wrapped());
  NameRef* nameref = dynamic_cast<NameRef*>(current_expr);
  ASSERT_NE(nameref, nullptr);

  Let* foo_parent = std::get<Let*>(stmts.at(0)->wrapped());
  ASSERT_NE(foo_parent, nullptr);
  // The easiest way to verify what we've got the right node is just to do a
  // string comparison, even if it's not pretty.
  EXPECT_EQ(foo_parent->rhs()->ToString(), "u32:0 + u32:1");
}

TEST_F(ParserTest, ChainedEqualsComparisonError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32, y: u32, z: bool) -> bool {
    x == y == z
}
)";

  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(parser.ParseModule(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ParseError: <no-file>:3:12-3:14 comparison "
                                 "operators cannot be chained")));
}

TEST_F(ParserTest, ChainedLtComparisonError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32, y: u32, z: bool) -> bool {
    x < y < z
}
)";

  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(parser.ParseModule(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ParseError: <no-file>:3:11-3:12 comparison "
                                 "operators cannot be chained")));
}

TEST_F(ParserTest, ChainedLtGtComparisonError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32, y: u32, z: bool) -> bool {
    x < y > z
}
)";

  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(parser.ParseModule(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected '}', got 'identifier'")));
}

TEST_F(ParserTest, ChannelDeclWithFifoDepthExpression) {
  constexpr std::string_view kProgram = R"(proc foo<N: u32, M: u32> {
    c_in: chan<u32> in;
    c_out: chan<u32> out;
    config() {
        let (c_p, c_c) = chan<u32, {N + M}>("my_chan");
        (c_p, c_c)
    }
    init {}
    next(state: ()) {
      ()
    }
}
)";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  XLS_EXPECT_OK(parser.ParseModule());
}

TEST_F(ParserTest, UnterminatedString) {
  constexpr std::string_view kProgram = R"(const A=")";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(
      parser.ParseModule(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("ScanError: <no-file>:1:10-1:10 Reached end of file "
                         "without finding a closing double quote")));
}

TEST_F(ParserTest, UnterminatedEscapedChar) {
  constexpr std::string_view kProgram = "'\\d";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(parser.ParseModule(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ScanError: <no-file>:1:2-1:3 Unrecognized "
                                 "escape sequence: `\\d`")));
}

TEST_F(ParserTest, UnterminatedEscapedUnicodeChar) {
  constexpr std::string_view kProgram = R"(const A="\u)";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unexpected EOF in escaped unicode character")));
}

TEST_F(ParserTest, UnterminatedEscapedHexChar) {
  constexpr std::string_view kProgram = "const A='\\x";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unexpected EOF in escaped hexadecimal character")));
}

TEST_F(ParserTest, ConstShadowsImport) {
  constexpr std::string_view kProgram = R"(import x;
const x)";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(
      parser.ParseModule(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "ParseError: <no-file>:2:7-2:8 Constant definition is "
               "shadowing an existing definition from <no-file>:1:1-1:10"));
}

TEST_F(ParserTest, ZeroLengthStringAtEof) {
  constexpr std::string_view kProgram = "const A=\"\"";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Zero-length strings are not supported.")));
}

TEST_F(ParserTest, RepetitiveImport) {
  constexpr std::string_view kProgram = R"(import repetitively;
import repetitively;)";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Import of `repetitively` is shadowing an existing "
                         "definition at <no-file>:1:1-1:21")));
}

TEST_F(ParserTest, UnreasonablyDeepExpr) {
  // Note: this is the minimum number of parentheses required to trigger the
  // error.
  constexpr std::string_view kProgram = R"(const E=
((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((()";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expression is too deeply nested")));
}

TEST_F(ParserTest, NonTypeDefinitionBeforeArrayLiteralColon) {
  constexpr std::string_view kProgram = "const A=4[5]:[";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Type before ':' for presumed array literal was not a "
                         "type definition; got `4` (kind: number)")));
}

TEST(ParserErrorTest, WildcardPatternExpressionStatement) {
  constexpr std::string_view kProgram = R"(
const MOL = u32:42;
#[test]
fn test_f() {
    let _ = assert_eq(u32:42, MOL);
    _
}
)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError(
          "ParseError",
          HasSubstr("Wildcard pattern `_` cannot be used as a reference")));
}

TEST(ParserErrorTest, BadTestTarget) {
  constexpr std::string_view kProgram = R"(#[test] let x=42;)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or.status(),
              IsPosError("ParseError", HasSubstr("Invalid test type: let")));
}

TEST(ParserErrorTest, BadDirectiveTokenType) {
  constexpr std::string_view kProgram = R"(#[3])";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(
      module_or.status(),
      IsPosError("ParseError", HasSubstr("Expected attribute identifier")));
}

TEST(ParserErrorTest, BadDirective) {
  constexpr std::string_view kProgram = R"(#[foo])";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();
  EXPECT_THAT(module_or.status(),
              IsPosError("ParseError", HasSubstr("Unknown directive: 'foo'")));
}

TEST(ParserErrorTest, FailLabelVerilogIdentifierConstraintReported) {
  constexpr std::string_view kProgram = R"(
fn will_fail(x:u32) -> u32 {
   fail!("not a proper label", x)
}
)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();

  // Check expected message
  EXPECT_THAT(
      module_or.status(),
      IsPosError(
          "ParseError",
          HasSubstr("The label parameter to fail!() must be a valid Verilog")));

  // Highlight span of the label parameter
  EXPECT_THAT(module_or.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                           HasSubstr("<no-file>:3:10-3:30")));
}

TEST(ParserErrorTest, FailLabelAtTopLevel) {
  constexpr std::string_view kProgram = R"(
const ZERO_FAIL = fail!("always_fails", u32:0);
)";
  FileTable file_table;
  Scanner s{file_table, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  auto module_or = parser.ParseModule();

  // Check expected message
  EXPECT_THAT(
      module_or.status(),
      IsPosError(
          "ParseError",
          HasSubstr("Cannot use the `fail!` builtin at module top-level.")));
}

TEST_F(ParserTest, ParseParametricProcWithConstAssert) {
  const char* text = R"(proc MyProc<N: u32> {
    const_assert!(N == u32:42);
    config() {
        ()
    }
    init {
        ()
    }
    next(state: ()) {
        ()
    }
})";

  Scanner s{file_table_, Fileno(0), std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  auto proc =
      parser.ParseProc(Pos(), /*is_public=*/false, /*bindings=*/bindings);
  if (!proc.ok()) {
    UniformContentFilesystem vfs(text);
    TryPrintError(proc.status(), file_table_, vfs);
    XLS_ASSERT_OK(proc.status());
  }
  const Proc* p = std::get<Proc*>(*proc);
  EXPECT_EQ(p->ToString(), text);
}

TEST_F(ParserTest, ParseParametricInMapBuiltin) {
  constexpr std::string_view kProgram = R"(
fn truncate<OUT: u32, IN: u32>(x: bits[IN]) -> bits[OUT] {
    x[0:OUT]
}

fn main(x: u32[64]) -> u16[64] {
    map(x, truncate<16>)
}
)";

  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  XLS_EXPECT_OK(parser.ParseModule());
}

TEST_F(ParserTest, ParseWithUncompletedStateTransaction) {
  constexpr std::string_view kProgram =
      "import st0;fn n(x:u6){let()=st0::w<\xb0";
  Scanner s{file_table_, Fileno(0), std::string(kProgram)};
  Parser parser{"test", &s};
  EXPECT_THAT(parser.ParseModule(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unrecognized character: '\xb0' (0xb0)")));
}

}  // namespace xls::dslx

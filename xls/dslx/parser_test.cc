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

#include "xls/dslx/parser.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::HasSubstr;

class ParserTest : public ::testing::Test {
 public:
  static constexpr absl::string_view kFilename = "test.x";

  void RoundTrip(std::string program,
                 absl::optional<absl::string_view> target = absl::nullopt) {
    scanner_.emplace(std::string(kFilename), program);
    parser_.emplace("test", &*scanner_);
    XLS_ASSERT_OK_AND_ASSIGN(auto module, parser_->ParseModule());
    if (target.has_value()) {
      EXPECT_EQ(module->ToString(), *target);
    } else {
      EXPECT_EQ(module->ToString(), program);
    }
  }

  // Note: given expression text should have no free variables other than those
  // in "predefine": those are defined a builtin name definitions (like the DSLX
  // builtins are).
  absl::StatusOr<Expr*> ParseExpr(
      std::string expr_text, absl::Span<const std::string> predefine = {}) {
    scanner_.emplace(std::string(kFilename), expr_text);
    parser_.emplace("test", &*scanner_);
    Bindings b;
    for (const std::string& s : predefine) {
      b.Add(s, parser_->module_->Make<BuiltinNameDef>(s));
    }
    return parser_->ParseExpression(/*bindings=*/&b);
  }

  void RoundTripExpr(std::string expr_text,
                     absl::Span<const std::string> predefine = {},
                     absl::optional<std::string> target = absl::nullopt,
                     Expr** parsed = nullptr) {
    XLS_ASSERT_OK_AND_ASSIGN(Expr * e, ParseExpr(expr_text, predefine));
    if (target.has_value()) {
      EXPECT_EQ(e->ToString(), *target);
    } else {
      EXPECT_EQ(e->ToString(), expr_text);
    }

    if (parsed != nullptr) {
      *parsed = e;
    }
  }

  absl::optional<Scanner> scanner_;
  absl::optional<Parser> parser_;
};

TEST(BindingsTest, BindingsStack) {
  Module module("test");
  Bindings top;
  Bindings leaf0(&top);
  Bindings leaf1(&top);

  auto* a = module.Make<BuiltinNameDef>("a");
  auto* b = module.Make<BuiltinNameDef>("b");
  auto* c = module.Make<BuiltinNameDef>("c");

  top.Add("a", a);
  leaf0.Add("b", b);
  leaf1.Add("c", c);

  const char* kFakeFilename = "fake.x";
  Pos pos(kFakeFilename, 0, 0);
  Span span(pos, pos);

  // Everybody can resolve the binding in "top".
  EXPECT_THAT(leaf0.ResolveNodeOrError("a", span), IsOkAndHolds(a));
  EXPECT_THAT(leaf1.ResolveNodeOrError("a", span), IsOkAndHolds(a));
  EXPECT_THAT(top.ResolveNodeOrError("a", span), IsOkAndHolds(a));

  EXPECT_THAT(top.ResolveNodeOrError("b", span),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: 'b'")));
  EXPECT_THAT(leaf1.ResolveNodeOrError("b", span),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: 'b'")));
  EXPECT_THAT(leaf0.ResolveNodeOrError("c", span),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: 'c'")));

  EXPECT_THAT(leaf0.ResolveNodeOrError("b", span), IsOkAndHolds(b));
  EXPECT_THAT(leaf1.ResolveNodeOrError("c", span), IsOkAndHolds(c));
}

TEST_F(ParserTest, TestIdentityFunction) {
  RoundTrip(R"(fn f(x: u32) -> u32 {
  x
})");
}

TEST_F(ParserTest, TestIdentityFunctionWithLet) {
  RoundTrip(R"(fn f(x: u32) -> u32 {
  let y = x;
  y
})");
}

TEST_F(ParserTest, TestTokenIdentity) {
  RoundTrip(R"(fn f(t: token) -> token {
  t
})");
}

TEST_F(ParserTest, StructDefRoundTrip) {
  RoundTrip(R"(pub struct foo<A: u32, B: bits[16]> {
  a: bits[A],
  b: bits[16][B],
})");
}

TEST_F(ParserTest, ParseErrorSpan) {
  const char* kFakeFilename = "fake.x";
  Scanner scanner(kFakeFilename, "+");
  Parser parser("test_module", &scanner);
  absl::StatusOr<Expr*> expr_or = parser.ParseExpression(/*bindings=*/nullptr);
  ASSERT_THAT(expr_or, StatusIs(absl::StatusCode::kInvalidArgument,
                                "ParseError: fake.x:1:1-1:2 Expected start of "
                                "an expression; got: +"));
}

TEST_F(ParserTest, ParseLetExpression) {
  const char* text = "let x: u32 = 2; x";
  Scanner s{"test.x", std::string{text}};
  Parser p{"test", &s};
  XLS_ASSERT_OK_AND_ASSIGN(Expr * e, p.ParseExpression(/*bindings=*/nullptr));
  Let* let = dynamic_cast<Let*>(e);
  ASSERT_TRUE(let != nullptr) << e->ToString();
  NameDef* name_def = absl::get<NameDef*>(let->name_def_tree()->leaf());
  EXPECT_EQ(name_def->identifier(), "x");
  EXPECT_EQ(let->type_annotation()->ToString(), "u32");
  EXPECT_EQ(let->rhs()->ToString(), "2");
  EXPECT_EQ(let->body()->ToString(), "x");
}

TEST_F(ParserTest, ParseIdentityFunction) {
  const char* text = "fn ident(x: bits) { x }";
  Scanner s{"test.x", std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(/*is_public=*/false, /*bindings=*/&bindings));
  NameRef* body = dynamic_cast<NameRef*>(f->body());
  ASSERT_TRUE(body != nullptr);
  EXPECT_EQ(body->identifier(), "x");
}

TEST_F(ParserTest, ParseSimpleProc) {
  const char* text = R"(proc simple(addend: u32) {
  next(x: u32) {
    next((x) + (addend))
  }
})";

  Scanner s{"test.x", std::string{text}};
  Parser parser{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * p,
      parser.ParseProc(/*is_public=*/false, /*outer_bindings=*/&bindings));
  EXPECT_EQ(p->ToString(), text);
}

TEST_F(ParserTest, ParseStructSplat) {
  const char* text = R"(struct Point {
  x: u32,
  y: u32,
}
fn f(p: Point) -> Point {
  Point { x: u32:42, ..p }
})";
  Scanner s{"test.x", std::string{text}};
  Parser parser{"test", &s};
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> m, parser.ParseModule());
  XLS_ASSERT_OK_AND_ASSIGN(TypeDefinition c, m->GetTypeDefinition("Point"));
  ASSERT_TRUE(absl::holds_alternative<StructDef*>(c));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, m->GetFunctionOrError("f"));
  SplatStructInstance* ssi = dynamic_cast<SplatStructInstance*>(f->body());
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
  Scanner s{"test.x", std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(/*is_public=*/false, /*bindings=*/&bindings));
  EXPECT_EQ(f->params().size(), 2);
  Binop* body = dynamic_cast<Binop*>(f->body());
  ASSERT_TRUE(body != nullptr);
  EXPECT_EQ(body->kind(), BinopKind::kConcat);
  NameRef* lhs = dynamic_cast<NameRef*>(body->lhs());
  ASSERT_TRUE(lhs != nullptr);
  EXPECT_EQ(lhs->identifier(), "x");
  NameRef* rhs = dynamic_cast<NameRef*>(body->rhs());
  EXPECT_EQ(rhs->identifier(), "y");
}

TEST_F(ParserTest, TrailingParameterComma) {
  const char* text = R"(
fn concat(
  x: bits,
  y: bits,
) {
  x ++ y
}
)";
  Scanner s{"test.x", std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(/*is_public=*/false, /*bindings=*/&bindings));
  EXPECT_EQ(f->params().size(), 2);
}

TEST_F(ParserTest, BitSlice) {
  const char* text = R"(
fn f(x: u32) -> u8 {
  x[0:8]
}
)";
  Scanner s{"test.x", std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(/*is_public=*/false, /*bindings=*/&bindings));
  auto* index = dynamic_cast<Index*>(f->body());
  ASSERT_NE(index, nullptr);
  IndexRhs rhs = index->rhs();
  ASSERT_TRUE(absl::holds_alternative<Slice*>(rhs));
  auto* slice = absl::get<Slice*>(rhs);
  EXPECT_EQ(slice->start()->ToString(), "0");
  EXPECT_EQ(slice->limit()->ToString(), "8");
}

TEST_F(ParserTest, LocalConstBinding) {
  const char* text = R"(fn f() -> u8 {
  const FOO = u8:42;
  FOO
})";
  RoundTrip(text);
  Scanner s{"test.x", std::string{text}};
  Parser p{"test", &s};
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      p.ParseFunction(/*is_public=*/false, /*bindings=*/&bindings));
  auto* const_let = dynamic_cast<Let*>(f->body());
  ASSERT_NE(const_let, nullptr);
  auto* constant_def = const_let->constant_def();
  ASSERT_NE(constant_def, nullptr);
  EXPECT_EQ("u8:42", constant_def->value()->ToString());

  auto* const_ref = dynamic_cast<ConstRef*>(const_let->body());
  ASSERT_NE(const_ref, nullptr);
  NameDef* name_def = const_ref->name_def();
  EXPECT_EQ(name_def->ToString(), "FOO");
  AstNode* definer = name_def->definer();
  EXPECT_EQ(definer, constant_def);

  std::vector<AstNode*> const_let_children =
      const_let->GetChildren(/*want_types=*/false);
  EXPECT_THAT(const_let_children, testing::Contains(constant_def));
}

TEST_F(ParserTest, BitSliceOfCall) {
  // TODO(leary): 2021-01-25 Eliminate unnecessary parens with a precedence
  // query.
  RoundTripExpr("id(x)[0:8]", {"id", "x"}, "(id(x))[0:8]");
}

TEST_F(ParserTest, BitSliceOfBitSlice) {
  // TODO(leary): 2021-01-25 Eliminate unnecessary parens with a precedence
  // query.
  RoundTripExpr("x[0:8][4:]", {"x"}, "((x)[0:8])[4:]");
}

TEST_F(ParserTest, BitSliceWithWidth) {
  // TODO(leary): 2021-01-25 Eliminate unnecessary parens with a precedence
  // query.
  RoundTripExpr("x[1+:u8]", {"x"}, "(x)[1+:u8]");
}

TEST_F(ParserTest, ModuleConstWithEnumInside) {
  // TODO(leary): 2021-01-26 This doesn't round trip properly, note the type
  // annotation on the tuple constant is dropped.
  absl::string_view expect = R"(enum MyEnum : u2 {
  FOO = 0,
  BAR = 1,
}
const MY_TUPLE = (MyEnum::FOO, MyEnum::BAR);)";
  RoundTrip(R"(enum MyEnum : u2 {
  FOO = 0,
  BAR = 1,
}
const MY_TUPLE = (MyEnum, MyEnum):(MyEnum::FOO, MyEnum::BAR);)",
            expect);
}

TEST_F(ParserTest, Struct) {
  const char* text = R"(struct Point {
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

TEST_F(ParserTest, LetDestructureFlat) {
  RoundTripExpr(R"(let (x, y, z): (u32, u32, u32) = (1, 2, 3);
y)");
}

TEST_F(ParserTest, LetDestructureNested) {
  RoundTripExpr(
      R"(let (w, (x, (y)), z): (u32, (u32, (u32,)), u32) = (1, (2, (3,)), 4);
y)");
}

TEST_F(ParserTest, LetDestructureWildcard) {
  RoundTripExpr(R"(let (x, y, _): (u32, u32, u32) = (1, 2, 3);
y)");
}

TEST_F(ParserTest, For) {
  RoundTripExpr(R"(let accum: u32 = 0;
let accum: u32 = for (i, accum): (u32, u32) in range(u32:0, u32:4) {
  let new_accum: u32 = (accum) + (i);
  new_accum
}(accum);
accum)",
                {"range"});
}

TEST_F(ParserTest, ForSansTypeAnnotation) {
  RoundTripExpr(
      R"(let init = ();
for (i, accum) in range(u32:0, u32:4) {
  accum
}(init))",
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

TEST_F(ParserTest, TupleArrayAndInt) {
  Expr* e;
  RoundTripExpr("(u8[4]:[1, 2, 3, 4], 7)", {}, absl::nullopt, &e);
  auto* tuple = dynamic_cast<XlsTuple*>(e);
  EXPECT_EQ(2, tuple->members().size());
  auto* array = tuple->members()[0];
  EXPECT_NE(dynamic_cast<ConstantArray*>(array), nullptr);
}

TEST_F(ParserTest, Cast) {
  // TODO(leary): 2021-01-24 We'll want the formatter to be precedence-aware in
  // its insertion of parens to avoid the round trip target value being special
  // here.
  RoundTripExpr("foo() as u32", {"foo"}, "((foo()) as u32)");
}

TEST_F(ParserTest, CastOfCast) {
  // TODO(leary): 2021-01-24 We'll want the formatter to be precedence-aware in
  // its insertion of parens to avoid the round trip target value being special
  // here.
  RoundTripExpr("x as s32 as u32", {"x"}, "((((x) as s32)) as u32)");
}

TEST_F(ParserTest, CastOfCastEnum) {
  // TODO(leary): 2021-01-24 We'll want the formatter to be precedence-aware in
  // its insertion of parens to avoid the round trip target value being special
  // here.
  RoundTrip(R"(enum MyEnum : u3 {
  SOME_VALUE = 0,
}
fn f(x: u8) -> MyEnum {
  ((((x) as u3)) as MyEnum)
})");
}

TEST_F(ParserTest, CastToTypeAlias) {
  // TODO(leary): 2021-01-24 We'll want the formatter to be precedence-aware in
  // its insertion of parens to avoid the round trip target value being special
  // here.
  RoundTrip(R"(type u128 = bits[128];
fn f(x: u32) -> u128 {
  ((x) as u128)
})");
}

TEST_F(ParserTest, Enum) {
  // TODO(leary): 2021-01-24 We'll want the formatter to be precedence-aware in
  // its insertion of parens to avoid the round trip target value being special
  // here.
  RoundTrip(R"(enum MyEnum : u2 {
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
  RoundTrip(R"(fn parametric<X: u32, Y: u32 = (X) + (X)>() -> (u32, u32) {
  (X, Y)
})");
}

TEST_F(ParserTest, ModuleWithTypeAlias) { RoundTrip("type MyType = u32;"); }

TEST_F(ParserTest, ModuleWithImport) { RoundTrip("import thing"); }

TEST_F(ParserTest, ModuleWithImportDots) { RoundTrip("import thing.subthing"); }

TEST_F(ParserTest, ModuleWithImportAs) { RoundTrip("import thing as other"); }

TEST_F(ParserTest, ConstArrayOfEnumRefs) {
  RoundTrip(R"(enum MyEnum : u3 {
  FOO = 1,
  BAR = 2,
}
const A = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR];)");
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

TEST_F(ParserTest, QuickCheckDirective) {
  RoundTrip(R"(#![quickcheck]
fn foo(x: u5) -> bool {
  true
})");
}

TEST_F(ParserTest, QuickCheckDirectiveWithTestCount) {
  RoundTrip(R"(#![quickcheck(test_count=1024)]
fn foo(x: u5) -> bool {
  true
})");
}

TEST_F(ParserTest, ModuleWithTypeAliasArrayTuple) {
  RoundTrip(R"(type MyType = u32;
type MyTupleType = (MyType[2],);)");
}

TEST_F(ParserTest, ModuleWithEmptyTestFunction) {
  RoundTrip(R"(#![test]
fn example() {
  ()
})");
}

TEST_F(ParserTest, ModuleWithTestFunction) {
  RoundTrip(R"(fn id(x: u32) -> u32 {
  x
}
#![test]
fn id_4() {
  assert_eq(u32:4, id(u32:4))
})");
}

TEST_F(ParserTest, TypeAliasForTupleWithConstSizedArray) {
  RoundTrip(R"(const HOW_MANY_THINGS = u32:42;
type MyTupleType = (u32[HOW_MANY_THINGS],);
fn get_things(x: MyTupleType) -> u32[HOW_MANY_THINGS] {
  (x)[0]
})");
}

TEST_F(ParserTest, ArrayOfNameRefs) {
  RoundTripExpr("[a, b, c, d]", {"a", "b", "c", "d"});
}

TEST_F(ParserTest, EmptyTuple) {
  Expr* e;
  RoundTripExpr("()", {}, absl::nullopt, &e);
  auto* tuple = dynamic_cast<XlsTuple*>(e);
  ASSERT_NE(tuple, nullptr);
  EXPECT_TRUE(tuple->empty());
}

TEST_F(ParserTest, Match) {
  RoundTripExpr(R"(match x {
  u32:42 => u32:64,
  _ => u32:42,
})",
                {"x"});
}

TEST_F(ParserTest, MatchFreevars) {
  Expr* e;
  RoundTripExpr(R"(match x {
  y => z,
})",
                {"x", "y", "z"}, absl::nullopt, &e);
  FreeVariables fv = e->GetFreeVariables(&e->span().start());
  EXPECT_THAT(fv.Keys(), testing::ContainerEq(
                             absl::flat_hash_set<std::string>{"x", "y", "z"}));
}

TEST_F(ParserTest, ForFreevars) {
  Expr* e;
  RoundTripExpr(R"(for (i, accum): (u32, u32) in range(u32:4) {
  let new_accum: u32 = ((accum) + (i)) + (j);
  new_accum
}(u32:0))",
                {"range", "j"}, absl::nullopt, &e);
  FreeVariables fv = e->GetFreeVariables(&e->span().start());
  EXPECT_THAT(fv.Keys(), testing::ContainerEq(
                             absl::flat_hash_set<std::string>{"j", "range"}));
}

TEST_F(ParserTest, Ternary) {
  RoundTripExpr("u32:42 if true else u32:24", {},
                "(u32:42) if (true) else (u32:24)");
}

TEST_F(ParserTest, ConstantArray) {
  Expr* e;
  RoundTripExpr("u32[2]:[0, 1]", {}, absl::nullopt, &e);
  ASSERT_TRUE(dynamic_cast<ConstantArray*>(e) != nullptr);
}

TEST_F(ParserTest, DoubleNegation) { RoundTripExpr("!!x", {"x"}, "!(!(x))"); }

TEST_F(ParserTest, LogicalOperatorPrecedence) {
  Expr* e;
  RoundTripExpr("!a || !b && c", {"a", "b", "c"}, "(!(a)) || ((!(b)) && (c))",
                &e);
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->kind(), BinopKind::kLogicalOr);
  auto* binop_rhs = dynamic_cast<Binop*>(binop->rhs());
  EXPECT_EQ(binop_rhs->kind(), BinopKind::kLogicalAnd);
  auto* unop = dynamic_cast<Unop*>(binop_rhs->lhs());
  EXPECT_EQ(unop->kind(), UnopKind::kInvert);
}

TEST_F(ParserTest, LogicalEqualityPrecedence) {
  Expr* e;
  RoundTripExpr("a ^ !b == f()", {"a", "b", "f"}, "((a) ^ (!(b))) == (f())",
                &e);
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->kind(), BinopKind::kEq);
  auto* binop_lhs = dynamic_cast<Binop*>(binop->lhs());
  EXPECT_EQ(binop_lhs->kind(), BinopKind::kXor);
  auto* unop = dynamic_cast<Unop*>(binop_lhs->rhs());
  EXPECT_EQ(unop->kind(), UnopKind::kInvert);
}

TEST_F(ParserTest, CastVsComparatorPrecedence) {
  Expr* e;
  RoundTripExpr("x >= y as u32", {"x", "y"}, "(x) >= (((y) as u32))", &e);
  auto* binop = dynamic_cast<Binop*>(e);
  EXPECT_EQ(binop->kind(), BinopKind::kGe);
  auto* cast = dynamic_cast<Cast*>(binop->rhs());
  ASSERT_NE(cast, nullptr);
  auto* casted_name_ref = dynamic_cast<NameRef*>(cast->expr());
  ASSERT_NE(casted_name_ref, nullptr);
  EXPECT_EQ(casted_name_ref->identifier(), "y");
}

TEST_F(ParserTest, CastVsUnaryPrecedence) {
  Expr* e;
  RoundTripExpr("-x as s32", {"x", "y"}, "((-(x)) as s32)", &e);
  auto* cast = dynamic_cast<Cast*>(e);
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->type_annotation()->ToString(), "s32");
}

TEST_F(ParserTest, NameDefTree) {
  RoundTripExpr(R"(let (a, (b, (c, d), e), f) = x;
a)",
                {"x"});
}

TEST_F(ParserTest, Strings) {
  RoundTripExpr(R"(let x = "dummy --> \" <-- string";
x)",
                {"x"});
  RoundTripExpr(R"(let x = "dummy --> \"";
x)",
                {"x"});
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
  Scanner s{"test.x", std::string{text}};
  Parser parser{"test", &s};
  auto module_status = parser.ParseModule();
  ASSERT_THAT(module_status, StatusIs(absl::StatusCode::kInvalidArgument,
                                      "ParseError: test.x:7:3-7:6 Cannot find "
                                      "a definition for name: 'FOO'"));
}

TEST_F(ParserTest, NumberSpan) {
  XLS_ASSERT_OK_AND_ASSIGN(Expr * e, ParseExpr("u32:42"));
  auto* number = dynamic_cast<Number*>(e);
  ASSERT_NE(number, nullptr);
  // TODO(https://github.com/google/xls/issues/438): 2021-05-24 Fix the
  // parsing/reporting of number spans so that the span starts at 0,0.
  EXPECT_EQ(number->span(), Span(Pos(std::string(kFilename), 0, 4),
                                 Pos(std::string(kFilename), 0, 6)));
}

}  // namespace xls::dslx

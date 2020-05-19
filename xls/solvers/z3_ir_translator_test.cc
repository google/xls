// Copyright 2020 Google LLC
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

#include "xls/solvers/z3_ir_translator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"

namespace xls {
namespace {

using solvers::z3::IrTranslator;
using solvers::z3::Predicate;
using solvers::z3::TryProve;

TEST(Z3IrTranslatorTest, ZeroIsZero) {
  Package p("test");
  FunctionBuilder b("f", &p);
  auto x = b.Literal(UBits(0, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven,
      TryProve(f, x.node(), Predicate::EqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ZeroTwoBitsIsZero) {
  Package p("test");
  FunctionBuilder b("f", &p);
  auto x = b.Literal(UBits(0, /*bit_count=*/2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven,
      TryProve(f, x.node(), Predicate::EqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, OneIsNotEqualToZero) {
  Package p("test");
  FunctionBuilder b("f", &p);
  auto x = b.Literal(UBits(1, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven,
      TryProve(f, x.node(), Predicate::EqualToZero(), absl::Seconds(1)));
  EXPECT_FALSE(proven);
}

TEST(Z3IrTranslatorTest, OneIsNotEqualToZeroPredicate) {
  Package p("test");
  FunctionBuilder b("f", &p);
  auto x = b.Literal(UBits(1, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven,
      TryProve(f, x.node(), Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ParamMinusSelfIsZero) {
  Package p("test");
  Type* u32 = p.GetBitsType(32);
  FunctionBuilder b("f", &p);
  auto x = b.Param("x", u32);
  auto res = b.Subtract(x, x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven,
      TryProve(f, res.node(), Predicate::EqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, XPlusYMinusYIsX) {
  const std::string program = R"(
fn f(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  ret sub.2: bits[32] = sub(add.1, y)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(),
                            Predicate::EqualTo(f->GetParamByName("x").value()),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, TupleIndexMinusSelf) {
  const std::string program = R"(
fn f(p: (bits[1], bits[32])) -> bits[32] {
  x: bits[32] = tuple_index(p, index=1)
  ret z: bits[32] = sub(x, x)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ConcatThenSliceIsSelf) {
  const std::string program = R"(
fn f(x: bits[4], y: bits[4], z: bits[4]) -> bits[1] {
  a: bits[12] = concat(x, y, z)
  b: bits[4] = bit_slice(a, start=4, width=4)
  ret c: bits[1] = eq(y, b)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ValueUgtSelf) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  ret result: bits[1] = ugt(p, p)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ValueUltSelf) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  ret result: bits[1] = ult(p, p)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ZeroExtBitAlwaysZero) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  x: bits[5] = zero_ext(p, new_bit_count=5)
  ret msb: bits[1] = bit_slice(x, start=4, width=1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, ZeroMinusParamHighBit) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  one: bits[4] = literal(value=1)
  zero_b4: bits[4] = literal(value=0)
  pz: bits[1] = eq(p, zero_b4)
  p2: bits[4] = sel(pz, cases=[p, one])
  zero: bits[5] = literal(value=0)
  x: bits[5] = zero_ext(p2, new_bit_count=5)
  result: bits[5] = sub(zero, x)
  ret msb: bits[1] = bit_slice(result, start=4, width=1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

// Since the value can wrap around, we should not be able to prove that adding
// one to a value is unsigned-greater-than itself.
TEST(Z3IrTranslatorTest, BumpByOneUgtSelf) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  one: bits[4] = literal(value=1)
  x: bits[4] = add(p, one)
  ret result: bits[1] = ugt(x, p)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_FALSE(proven_ez);

  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_FALSE(proven_nez);
}

TEST(Z3IrTranslatorTest, MaskAndReverse) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[2] = literal(value=1)
  x: bits[2] = and(p, one)
  rev: bits[2] = reverse(x)
  ret result: bits[1] = bit_slice(rev, start=0, width=1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, ReverseSlicesEq) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  p0: bits[1] = bit_slice(p, start=0, width=1)
  rp: bits[2] = reverse(p)
  rp1: bits[1] = bit_slice(rp, start=1, width=1)
  ret result: bits[1] = eq(p0, rp1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, ShiftRightLogicalFillsZero) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[2] = literal(value=1)
  x: bits[2] = shrl(p, one)
  ret result: bits[1] = bit_slice(x, start=1, width=1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, ShiftLeftLogicalFillsZero) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[2] = literal(value=1)
  x: bits[2] = shll(p, one)
  ret result: bits[1] = bit_slice(x, start=0, width=1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, ShiftLeftLogicalDifferentSize) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[1] = literal(value=1)
  x: bits[2] = shll(p, one)
  ret result: bits[1] = bit_slice(x, start=0, width=1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, XAndNotXIsZero) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  np: bits[1] = not(p)
  ret result: bits[1] = and(p, np)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, XNandNotXIsZero) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  np: bits[1] = not(p)
  ret result: bits[1] = nand(p, np)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, XOrNotXIsNotZero) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  np: bits[1] = not(p)
  ret result: bits[1] = or(p, np)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, SignExtendBitsAreEqual) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  p2: bits[2] = sign_ext(p, new_bit_count=2)
  b0: bits[1] = bit_slice(p2, start=0, width=1)
  b1: bits[1] = bit_slice(p2, start=1, width=1)
  ret eq: bits[1] = eq(b0, b1)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, XPlusNegX) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[4] {
  np: bits[4] = neg(p)
  ret result: bits[4] = add(p, np)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, XNeX) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  ret result: bits[1] = ne(p, p)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, OneHot) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[2] {
  ret result: bits[2] = one_hot(p, lsb_prio=true)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, EncodeZeroIsZero) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  z: bits[2] = xor(x, x)
  ret result: bits[1] = encode(z)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

TEST(Z3IrTranslatorTest, EncodeWithIndex1SetIsNotZero) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  literal.1: bits[2] = literal(value=0b10)
  or.2: bits[2] = or(x, literal.1)
  ret result: bits[1] = encode(or.2)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, SelWithDefault) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  literal.1: bits[1] = literal(value=0b1)
  literal.2: bits[1] = literal(value=0b0)
  ret sel.3: bits[1] = sel(x, cases=[literal.1], default=literal.2)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_FALSE(proven_nez);
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_FALSE(proven_ez);
}

TEST(Z3IrTranslatorTest, SgeVsSlt) {
  const std::string program = R"(
fn f(x: bits[2], y: bits[2]) -> bits[1] {
  sge: bits[1] = sge(x, y)
  slt: bits[1] = slt(x, y)
  ret and: bits[1] = and(sge, slt)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_ez, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                               absl::Seconds(1)));
  EXPECT_TRUE(proven_ez);
}

// TODO(b/153195241): Re-enable these.
#ifdef NDEBUG
TEST(Z3IrTranslatorTest, AddToMostNegativeSge) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  most_negative: bits[2] = literal(value=0b10)
  add: bits[2] = add(most_negative, x)
  ret result: bits[1] = sge(add, most_negative)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

TEST(Z3IrTranslatorTest, SltVsMaxPositive) {
  const std::string program = R"(
fn f(x: bits[3]) -> bits[1] {
  most_positive: bits[3] = literal(value=0b011)
  most_negative: bits[3] = literal(value=0b100)
  eq_mp: bits[1] = eq(x, most_positive)
  sel: bits[3] = sel(eq_mp, cases=[x, most_negative])
  ret result: bits[1] = slt(sel, most_positive)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}
#endif

TEST(Z3IrTranslatorTest, TupleAndAccess) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  t: (bits[2], bits[2]) = tuple(x, x)
  u: ((bits[2], bits[2]), bits[2]) = tuple(t, x)
  lhs: (bits[2], bits[2]) = tuple_index(u, index=0)
  y: bits[2] = tuple_index(lhs, index=0)
  z: bits[2] = tuple_index(t, index=1)
  ret eq: bits[1] = eq(y, z)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_nez, TryProve(f, f->return_value(),
                                Predicate::NotEqualToZero(), absl::Seconds(1)));
  EXPECT_TRUE(proven_nez);
}

// This test verifies that selects with tuple values can be translated.
TEST(Z3IrTranslatorTest, TupleSelect) {
  const std::string program = R"(
package p

fn f() -> bits[1] {
  lit_true: bits[1] = literal(value=1)
  lit_false: bits[1] = literal(value=0)
  truple: (bits[1], bits[1]) = tuple(lit_true, lit_true)
  falseple: (bits[1], bits[1]) = tuple(lit_false, lit_false)
  mix1: (bits[1], bits[1]) = tuple(lit_false, lit_true)
  mix2: (bits[1], bits[1]) = tuple(lit_true, lit_false)
  selector: bits[2] = literal(value=2)
  choople: (bits[1], bits[1]) = sel(selector, cases=[falseple,mix1,truple,mix2])
  elem0: bits[1] = tuple_index(choople, index=0)
  elem1: bits[1] = tuple_index(choople, index=1)
  ret result: bits[1] = and(elem0, elem1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq, TryProve(f, f->return_value(),
                               Predicate::NotEqualToZero(), absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

TEST(Z3IrTranslatorTest, TupleSelectsMore) {
  const std::string program = R"(
package p

fn f() -> bits[4] {
 literal.1: bits[4] = literal(value=1)
 literal.2: bits[4] = literal(value=2)
 literal.3: bits[4] = literal(value=3)
 literal.4: bits[4] = literal(value=4)
 literal.5: bits[4] = literal(value=5)
 tuple.6: (bits[4], bits[4], bits[4], bits[4], bits[4]) = tuple(literal.1, literal.2, literal.3, literal.4, literal.5)
 tuple.7: (bits[4], bits[4], bits[4], bits[4], bits[4]) = tuple(literal.2, literal.3, literal.4, literal.5, literal.1)
 tuple.8: (bits[4], bits[4], bits[4], bits[4], bits[4]) = tuple(literal.3, literal.4, literal.5, literal.1, literal.2)
 tuple.9: (bits[4], bits[4], bits[4], bits[4], bits[4]) = tuple(literal.4, literal.5, literal.1, literal.2, literal.3)
 tuple.10: (bits[4], bits[4], bits[4], bits[4], bits[4]) = tuple(literal.5, literal.1, literal.2, literal.3, literal.4)
 literal.11: bits[4] = literal(value=1)
 sel.12: (bits[4], bits[4], bits[4], bits[4], bits[4]) = sel(literal.11, cases=[tuple.6, tuple.7, tuple.8, tuple.9, tuple.10], default=tuple.6)
 ret tuple_index.13: bits[4] = tuple_index(sel.12, index=1)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  Node* to_compare = nullptr;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("literal.3") != std::string::npos) {
      to_compare = node;
      break;
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq,
      TryProve(f, f->return_value(), Predicate::EqualTo(to_compare),
               absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

// Array test 1: Can we properly handle arrays of bits!
TEST(Z3IrTranslatorTest, ArrayOfBits) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=4)
  literal.4: bits[32] = literal(value=8)
  literal.5: bits[32] = literal(value=16)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  ret array_index.7: bits[32] = array_index(array.6, literal.3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  Node* eq_node;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("literal.5") != std::string::npos) {
      eq_node = node;
      break;
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq, TryProve(f, f->return_value(),
                               Predicate::EqualTo(eq_node), absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

// Array test 2: Can we properly handle arrays...OF ARRAYS?
TEST(Z3IrTranslatorTest, ArrayOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  literal.3: bits[32] = literal(value=2)
  literal.4: bits[32] = literal(value=3)
  literal.5: bits[32] = literal(value=4)
  subarray.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  subarray.7: bits[32][5] = array(literal.2, literal.3, literal.4, literal.5, literal.1)
  subarray.8: bits[32][5] = array(literal.3, literal.4, literal.5, literal.1, literal.2)
  subarray.9: bits[32][5] = array(literal.4, literal.5, literal.1, literal.2, literal.3)
  subarray.10: bits[32][5] = array(literal.5, literal.1, literal.2, literal.3, literal.4)
  big_array.11: bits[32][5][5] = array(subarray.6, subarray.7, subarray.8, subarray.9, subarray.10)
  big_array_index.12: bits[32][5] = array_index(big_array.11, literal.3)
  ret sub_array_index.13: bits[32] = array_index(big_array_index.12, literal.2)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  Node* eq_node;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("literal.4") != std::string::npos) {
      eq_node = node;
      break;
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq, TryProve(f, f->return_value(),
                               Predicate::EqualTo(eq_node), absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

// Array test 3! Arrays...OF TUPLES
TEST(Z3IrTranslatorTest, ArrayOfTuples) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  tuple.6: (bits[32], bits[32], bits[32]) = tuple(literal.1, literal.2, literal.3)
  tuple.7: (bits[32], bits[32], bits[32]) = tuple(literal.2, literal.3, literal.4)
  tuple.8: (bits[32], bits[32], bits[32]) = tuple(literal.3, literal.4, literal.5)
  tuple.9: (bits[32], bits[32], bits[32]) = tuple(literal.4, literal.5, literal.1)
  tuple.10: (bits[32], bits[32], bits[32]) = tuple(literal.5, literal.1, literal.2)
  array.11: (bits[32], bits[32], bits[32])[5] = array(tuple.6, tuple.7, tuple.8, tuple.9, tuple.10)
  array_index.12: (bits[32], bits[32], bits[32]) = array_index(array.11, literal.4)
  ret tuple_index.13: bits[32] = tuple_index(array_index.12, index=0)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  Node* eq_node;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("literal.5") != std::string::npos) {
      eq_node = node;
      break;
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq, TryProve(f, f->return_value(),
                               Predicate::EqualTo(eq_node), absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

TEST(Z3IrTranslatorTest, ArrayOfTuplesOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  array.7: bits[32][5] = array(literal.2, literal.3, literal.4, literal.5, literal.1)
  array.8: bits[32][5] = array(literal.3, literal.4, literal.5, literal.1, literal.2)
  array.9: bits[32][5] = array(literal.4, literal.5, literal.1, literal.2, literal.3)
  array.10: bits[32][5] = array(literal.5, literal.1, literal.2, literal.3, literal.4)
  tuple.11: (bits[32][5], bits[32][5], bits[32][5]) = tuple(array.6, array.7, array.8)
  tuple.12: (bits[32][5], bits[32][5], bits[32][5]) = tuple(array.7, array.8, array.9)
  tuple.13: (bits[32][5], bits[32][5], bits[32][5]) = tuple(array.8, array.9, array.10)
  tuple.14: (bits[32][5], bits[32][5], bits[32][5]) = tuple(array.9, array.10, array.6)
  tuple.15: (bits[32][5], bits[32][5], bits[32][5]) = tuple(array.10, array.6, array.7)
  array.16: (bits[32][5], bits[32][5], bits[32][5])[5] = array(tuple.11, tuple.12, tuple.13, tuple.14, tuple.15)
  array_index.17: (bits[32][5], bits[32][5], bits[32][5]) = array_index(array.16, literal.2)
  tuple_index.18: bits[32][5] = tuple_index(array_index.17, index=1)
  ret array_index.19: bits[32] = array_index(tuple_index.18, literal.3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  Node* eq_node;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("literal.2") != std::string::npos) {
      eq_node = node;
      break;
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq, TryProve(f, f->return_value(),
                               Predicate::EqualTo(eq_node), absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

TEST(Z3IrTranslatorTest, OverflowingArrayIndex) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  ret array_index.7: bits[32] = array_index(array.6, literal.5)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  Node* eq_node;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("literal.5") != std::string::npos) {
      eq_node = node;
      break;
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven_eq, TryProve(f, f->return_value(),
                               Predicate::EqualTo(eq_node), absl::Seconds(10)));
  EXPECT_TRUE(proven_eq);
}

TEST(Z3IrTranslatorTest, ParamReuse) {
  // Have the two programs do slightly different things, just to avoid paranoia
  // over potential evaluation short-circuits.
  const std::string program_1 = R"(
package p1

fn f(x: bits[32], y: bits[16], z: bits[8]) -> bits[16] {
  tuple.1: (bits[32], bits[16], bits[8]) = tuple(x, y, z)
  ret tuple_index.2: bits[16] = tuple_index(tuple.1, index=1)
}
)";

  const std::string program_2 = R"(
package p2

fn f(x: bits[32], y: bits[16], z: bits[8]) -> bits[16] {
  ret y
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p1,
                           Parser::ParsePackage(program_1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, p1->GetFunction("f"));
  XLS_ASSERT_OK_AND_ASSIGN(auto translator_1,
                           IrTranslator::CreateAndTranslate(f1));
  std::vector<Z3_ast> imported_params;
  for (auto* param : f1->params()) {
    imported_params.push_back(translator_1->GetTranslation(param));
  }

  Z3_context ctx = translator_1->ctx();

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p2,
                           Parser::ParsePackage(program_2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, p2->GetFunction("f"));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto translator_2,
      IrTranslator::CreateAndTranslate(translator_1->ctx(), f2,
                                       absl::MakeSpan(imported_params)));

  Z3_ast return_1 = translator_1->GetReturnNode();
  Z3_ast return_2 = translator_2->GetReturnNode();

  // Solvers and params need explicit references to be taken, or they'll be very
  // eagerly destroyed.
  Z3_solver solver = Z3_mk_solver(ctx);

  // Remember: we try to prove the condition by searching for a model that
  // produces the opposite result. Thus, we want to find a model where the
  // results are _not_ equal.
  Z3_ast objective = Z3_mk_not(ctx, Z3_mk_eq(ctx, return_1, return_2));
  Z3_solver_assert(ctx, solver, objective);

  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  EXPECT_EQ(satisfiable, Z3_L_FALSE);
}

}  // namespace
}  // namespace xls

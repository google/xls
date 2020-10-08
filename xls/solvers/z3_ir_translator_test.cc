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

#include "xls/solvers/z3_ir_translator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3.h"
#include "../z3/src/api/z3_api.h"

namespace xls {
namespace {

using solvers::z3::IrTranslator;
using solvers::z3::Predicate;
using solvers::z3::TryProve;
using status_testing::IsOkAndHolds;

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

TEST(Z3IrTranslatorTest, InBoundsDynamicSlice) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=1)
  dynamic_slice: bits[3] = dynamic_bit_slice(p, start, width=3)
  slice: bits[3] = bit_slice(p, start=1, width=3)
  ret result: bits[1] = eq(slice, dynamic_slice)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, PartialOutOfBoundsDynamicSlice) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=2)
  slice: bits[3] = dynamic_bit_slice(p, start, width=3)
  out_of_bounds: bits[1] = bit_slice(slice, start=2, width=1)
  zero: bits[1] = literal(value=0)
  ret result: bits[1] = eq(out_of_bounds, zero)
}
)";
  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

TEST(Z3IrTranslatorTest, CompletelyOutOfBoundsDynamicSlice) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=7)
  slice: bits[3] = dynamic_bit_slice(p, start, width=3)
  zero: bits[3] = literal(value=0)
  ret result: bits[1] = eq(slice, zero)
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

TEST(Z3IrTranslatorTest, BasicAfterAllTokenTest) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  after_all.10: token = after_all()
  literal.2: bits[32] = literal(value=2)
  after_all.11: token = after_all()
  literal.3: bits[32] = literal(value=4)
  after_all.12: token = after_all()
  literal.4: bits[32] = literal(value=8)
  after_all.13: token = after_all(after_all.10, after_all.11, after_all.12)
  literal.5: bits[32] = literal(value=16)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  ret array_index.7: bits[32] = array_index(array.6, literal.3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  // Check that non-token logic is not affected.
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

  std::vector<Node*> token_nodes;
  for (Node* node : f->nodes()) {
    if (node->GetType()->IsToken()) {
      token_nodes.push_back(node);
    }
  }
  assert(token_nodes.size() == 4);

  for (int l_idx = 0; l_idx < token_nodes.size(); ++l_idx) {
    for (int r_idx = l_idx + 1; r_idx < token_nodes.size(); ++r_idx) {
      // All tokens are equal to each other.
      XLS_ASSERT_OK_AND_ASSIGN(
          bool proven_eq, TryProve(f, token_nodes.at(l_idx),
                                   Predicate::EqualTo(token_nodes.at(r_idx)),
                                   absl::Seconds(10)));
      EXPECT_TRUE(proven_eq);
    }
    // Can't prove a token is 0 or non-zero because it is a non-bit type.
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::EqualToZero(),
                          absl::Seconds(10))
                     .status()
                     .ok());
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::NotEqualToZero(),
                          absl::Seconds(10))
                     .status()
                     .ok());
  }
}

TEST(Z3IrTranslatorTest, TokensNotEqualToEmptyTuples) {
  const std::string program = R"(
package p

fn f(empty_tuple: ()) -> bits[32] {
  after_all.10: token = after_all()
  ret literal.1: bits[32] = literal(value=1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  Node* token_node;
  Node* tuple_node;
  for (Node* node : f->nodes()) {
    if (node->ToString().find("after_all") != std::string::npos) {
      token_node = node;
    }
    if (node->ToString().find("empty_tuple") != std::string::npos) {
      tuple_node = node;
    }
  }
  // Even though we represent tokens as empty tuples as a convenient hack, we
  // should not evaluate tokens == empty tuples.  Evaluation should fail becaue
  // an empty tuple is not a bit type.
  EXPECT_FALSE(
      TryProve(f, token_node, Predicate::EqualTo(tuple_node), absl::Seconds(10))
          .status()
          .ok());
  EXPECT_FALSE(
      TryProve(f, tuple_node, Predicate::EqualTo(token_node), absl::Seconds(10))
          .status()
          .ok());
}

TEST(Z3IrTranslatorTest, TokenArgsAndReturn) {
  const std::string program = R"(
package p

fn f(arr1: token, arr2: token, arr3: token) -> token {
  ret after_all.1: token = after_all(arr1, arr2, arr3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  std::vector<Node*> token_nodes;
  for (Node* node : f->nodes()) {
    if (node->GetType()->IsToken()) {
      token_nodes.push_back(node);
    }
  }
  ASSERT_EQ(token_nodes.size(), 4);

  for (int l_idx = 0; l_idx < token_nodes.size(); ++l_idx) {
    for (int r_idx = l_idx + 1; r_idx < token_nodes.size(); ++r_idx) {
      // All tokens are equal to each other.
      ASSERT_THAT(TryProve(f, token_nodes.at(l_idx),
                           Predicate::EqualTo(token_nodes.at(r_idx)),
                           absl::Seconds(10)),
                  IsOkAndHolds(true));
    }
    // Can't prove a token is 0 or non-zero because it is a non-bit type.
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::EqualToZero(),
                          absl::Seconds(10))
                     .status()
                     .ok());
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::NotEqualToZero(),
                          absl::Seconds(10))
                     .status()
                     .ok());
  }
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
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  array.7: bits[32][5] = array(literal.2, literal.3, literal.4, literal.5, literal.1)
  array.8: bits[32][5] = array(literal.3, literal.4, literal.5, literal.1, literal.2)
  array.9: bits[32][5] = array(literal.4, literal.5, literal.1, literal.2, literal.3)
  array.10: bits[32][5] = array(literal.5, literal.1, literal.2, literal.3, literal.4)
  array.11: bits[32][5][5] = array(array.6, array.7, array.8, array.9, array.10)
  array_index.12: bits[32][5] = array_index(array.11, literal.3)
  ret array_index.13: bits[32] = array_index(array_index.12, literal.2)
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

// UpdateArray test 1: Array of bits
TEST(Z3IrTranslatorTest, UpdateArrayOfBits) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  array.6: bits[32][2] = array(literal.1, literal.1)
  array_update.8: bits[32][2] = array_update(array.6, literal.1, literal.2)
  array_index.9: bits[32] = array_index(array_update.8, literal.1)
  array_index.10: bits[32] = array_index(array_update.8, literal.2)
  ret literal.2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  auto assign_nodes = [f](std::vector<std::string>& m_string,
                          std::vector<Node*>& m_nodes) {
    m_nodes.reserve(m_string.size());
    for (int idx = 0; idx < m_string.size(); ++idx) {
      m_nodes.push_back(nullptr);
    }
    for (Node* func_node : f->nodes()) {
      for (int idx = 0; idx < m_nodes.size(); ++idx) {
        if (m_nodes[idx] == nullptr &&
            func_node->ToString().find(m_string[idx]) != std::string::npos) {
          m_nodes[idx] = func_node;
        }
      }
    }
    for (auto node : m_nodes) {
      EXPECT_NE(node, nullptr);
    }
  };
  std::vector<Node*> expect_node;
  std::vector<std::string> expect_str = {"literal.2", "literal.1"};
  std::vector<Node*> observe_node;
  std::vector<std::string> observe_str = {"array_index.9", "array_index.10"};
  assign_nodes(expect_str, expect_node);
  assign_nodes(observe_str, observe_node);

  for (int idx = 0; idx < expect_node.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        bool proven_eq,
        TryProve(f, expect_node[idx], Predicate::EqualTo(observe_node[idx]),
                 absl::Seconds(10)));
    EXPECT_TRUE(proven_eq);
  }
}

// UpdateArray test 2: Array of Arrays
TEST(Z3IrTranslatorTest, UpdateArrayOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  array.3: bits[32][2] = array(literal.1, literal.1)
  array.4: bits[32][2] = array(literal.2, literal.2)
  array.6: bits[32][2][2] = array(array.3, array.3)
  array_update.8: bits[32][2][2] = array_update(array.6, literal.2, array.4)
  array_index.9: bits[32][2] = array_index(array_update.8, literal.1)
  array_index.10: bits[32] = array_index(array_index.9, literal.1)
  array_index.11: bits[32] = array_index(array_index.9, literal.2)
  array_index.12: bits[32][2] = array_index(array_update.8, literal.2)
  array_index.13: bits[32] = array_index(array_index.12, literal.1)
  array_index.14: bits[32] = array_index(array_index.12, literal.2)
  ret literal.2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  auto assign_nodes = [f](std::vector<std::string>& m_string,
                          std::vector<Node*>& m_nodes) {
    m_nodes.reserve(m_string.size());
    for (int idx = 0; idx < m_string.size(); ++idx) {
      m_nodes.push_back(nullptr);
    }
    for (Node* func_node : f->nodes()) {
      for (int idx = 0; idx < m_nodes.size(); ++idx) {
        if (m_nodes[idx] == nullptr &&
            func_node->ToString().find(m_string[idx]) != std::string::npos) {
          m_nodes[idx] = func_node;
        }
      }
    }
    for (auto node : m_nodes) {
      EXPECT_NE(node, nullptr);
    }
  };
  std::vector<Node*> expect_node;
  std::vector<std::string> expect_str = {"literal.1", "literal.1", "literal.2",
                                         "literal.2"};
  std::vector<Node*> observe_node;
  std::vector<std::string> observe_str = {"array_index.10", "array_index.11",
                                          "array_index.13", "array_index.14"};
  assign_nodes(expect_str, expect_node);
  assign_nodes(observe_str, observe_node);

  for (int idx = 0; idx < expect_node.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        bool proven_eq,
        TryProve(f, expect_node[idx], Predicate::EqualTo(observe_node[idx]),
                 absl::Seconds(10)));
    EXPECT_TRUE(proven_eq);
  }
}

// UpdateArray test 3: Array of Tuples
TEST(Z3IrTranslatorTest, UpdateArrayOfTuples) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  tuple.3: (bits[32], bits[32]) = tuple(literal.1, literal.2)
  tuple.4: (bits[32], bits[32]) = tuple(literal.2, literal.1)
  array.6: (bits[32], bits[32])[2] = array(tuple.3, tuple.3)
  array_update.8:(bits[32], bits[32])[2] = array_update(array.6, literal.2, tuple.4)
  array_index.9: (bits[32], bits[32]) = array_index(array_update.8, literal.1)
  tuple_index.10: bits[32] = tuple_index(array_index.9, index=0)
  tuple_index.11: bits[32] = tuple_index(array_index.9, index=1)
  array_index.12: (bits[32], bits[32]) = array_index(array_update.8, literal.2)
  tuple_index.13: bits[32] = tuple_index(array_index.12, index=0)
  tuple_index.14: bits[32] = tuple_index(array_index.12, index=1)
  ret literal.2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  auto assign_nodes = [f](std::vector<std::string>& m_string,
                          std::vector<Node*>& m_nodes) {
    m_nodes.reserve(m_string.size());
    for (int idx = 0; idx < m_string.size(); ++idx) {
      m_nodes.push_back(nullptr);
    }
    for (Node* func_node : f->nodes()) {
      for (int idx = 0; idx < m_nodes.size(); ++idx) {
        if (m_nodes[idx] == nullptr &&
            func_node->ToString().find(m_string[idx]) != std::string::npos) {
          m_nodes[idx] = func_node;
        }
      }
    }
    for (auto node : m_nodes) {
      EXPECT_NE(node, nullptr);
    }
  };
  std::vector<Node*> expect_node;
  std::vector<std::string> expect_str = {"literal.1", "literal.2", "literal.2",
                                         "literal.1"};
  std::vector<Node*> observe_node;
  std::vector<std::string> observe_str = {"tuple_index.10", "tuple_index.11",
                                          "tuple_index.13", "tuple_index.14"};
  assign_nodes(expect_str, expect_node);
  assign_nodes(observe_str, observe_node);

  for (int idx = 0; idx < expect_node.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        bool proven_eq,
        TryProve(f, expect_node[idx], Predicate::EqualTo(observe_node[idx]),
                 absl::Seconds(10)));
    EXPECT_TRUE(proven_eq);
  }
}

// UpdateArray test 4: Array of Tuples of Arrays
TEST(Z3IrTranslatorTest, UpdateArrayOfTuplesOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  array.3: bits[32][2] = array(literal.1, literal.2)
  array.4: bits[32][2] = array(literal.2, literal.1)
  tuple.5: (bits[32][2], bits[32][2]) = tuple(array.3, array.4)
  tuple.6: (bits[32][2], bits[32][2]) = tuple(array.4, array.3)
  array.7: (bits[32][2], bits[32][2])[2] = array(tuple.5, tuple.5)
  array_update.8: (bits[32][2], bits[32][2])[2] = array_update(array.7, literal.2, tuple.6)
  array_index.9: (bits[32][2], bits[32][2]) = array_index(array_update.8, literal.1)
  tuple_index.10: bits[32][2] = tuple_index(array_index.9, index=0)
  tuple_index.11: bits[32][2] = tuple_index(array_index.9, index=1)
  array_index.12: bits[32] = array_index(tuple_index.10, literal.1)
  array_index.13: bits[32] = array_index(tuple_index.10, literal.2)
  array_index.14: bits[32] = array_index(tuple_index.11, literal.1)
  array_index.15: bits[32] = array_index(tuple_index.11, literal.2)
  array_index.16: (bits[32][2], bits[32][2]) = array_index(array_update.8, literal.2)
  tuple_index.17: bits[32][2] = tuple_index(array_index.16, index=0)
  tuple_index.18: bits[32][2] = tuple_index(array_index.16, index=1)
  array_index.19: bits[32] = array_index(tuple_index.17, literal.1)
  array_index.20: bits[32] = array_index(tuple_index.17, literal.2)
  array_index.21: bits[32] = array_index(tuple_index.18, literal.1)
  array_index.22: bits[32] = array_index(tuple_index.18, literal.2)
  ret literal.2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  auto assign_nodes = [f](std::vector<std::string>& m_string,
                          std::vector<Node*>& m_nodes) {
    m_nodes.reserve(m_string.size());
    for (int idx = 0; idx < m_string.size(); ++idx) {
      m_nodes.push_back(nullptr);
    }
    for (Node* func_node : f->nodes()) {
      for (int idx = 0; idx < m_nodes.size(); ++idx) {
        if (m_nodes[idx] == nullptr &&
            func_node->ToString().find(m_string[idx]) != std::string::npos) {
          m_nodes[idx] = func_node;
        }
      }
    }
    for (auto node : m_nodes) {
      EXPECT_NE(node, nullptr);
    }
  };
  std::vector<Node*> expect_node;
  std::vector<std::string> expect_str = {"literal.1", "literal.2", "literal.2",
                                         "literal.1", "literal.2", "literal.1",
                                         "literal.1", "literal.2"};
  std::vector<Node*> observe_node;
  std::vector<std::string> observe_str = {
      "array_index.12", "array_index.13", "array_index.14", "array_index.15",
      "array_index.19", "array_index.20", "array_index.21", "array_index.22"};
  assign_nodes(expect_str, expect_node);
  assign_nodes(observe_str, observe_node);

  for (int idx = 0; idx < expect_node.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        bool proven_eq,
        TryProve(f, expect_node[idx], Predicate::EqualTo(observe_node[idx]),
                 absl::Seconds(10)));
    EXPECT_TRUE(proven_eq);
  }
}

// UpdateArray test 4: Out of bounds index
TEST(Z3IrTranslatorTest, UpdateArrayOutOfBoundsIndex) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  literal.3: bits[32] = literal(value=99)
  array.6: bits[32][2] = array(literal.1, literal.1)
  array_update.8: bits[32][2] = array_update(array.6, literal.3, literal.2)
  array_index.9: bits[32] = array_index(array_update.8, literal.1)
  array_index.10: bits[32] = array_index(array_update.8, literal.2)
  ret literal.2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  auto assign_nodes = [f](std::vector<std::string>& m_string,
                          std::vector<Node*>& m_nodes) {
    m_nodes.reserve(m_string.size());
    for (int idx = 0; idx < m_string.size(); ++idx) {
      m_nodes.push_back(nullptr);
    }
    for (Node* func_node : f->nodes()) {
      for (int idx = 0; idx < m_nodes.size(); ++idx) {
        if (m_nodes[idx] == nullptr &&
            func_node->ToString().find(m_string[idx]) != std::string::npos) {
          m_nodes[idx] = func_node;
        }
      }
    }
    for (auto node : m_nodes) {
      EXPECT_NE(node, nullptr);
    }
  };
  std::vector<Node*> expect_node;
  std::vector<std::string> expect_str = {"literal.1", "literal.1"};
  std::vector<Node*> observe_node;
  std::vector<std::string> observe_str = {"array_index.9", "array_index.10"};
  assign_nodes(expect_str, expect_node);
  assign_nodes(observe_str, observe_node);

  for (int idx = 0; idx < expect_node.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        bool proven_eq,
        TryProve(f, expect_node[idx], Predicate::EqualTo(observe_node[idx]),
                 absl::Seconds(10)));
    EXPECT_TRUE(proven_eq);
  }
}

// UpdateArray test 5: Unknown index
TEST(Z3IrTranslatorTest, UpdateArrayUnknownIndex) {
  const std::string program = R"(
package p

fn f(index: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  literal.2: bits[32] = literal(value=1)
  literal.3: bits[32] = literal(value=99)
  array.6: bits[32][2] = array(literal.1, literal.1)
  array_update.8: bits[32][2] = array_update(array.6, index, literal.2)
  array_index.9: bits[32] = array_index(array_update.8, literal.1)
  array_index.10: bits[32] = array_index(array_update.8, literal.2)
  ret literal.2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));

  auto assign_nodes = [f](std::vector<std::string>& m_string,
                          std::vector<Node*>& m_nodes) {
    m_nodes.reserve(m_string.size());
    for (int idx = 0; idx < m_string.size(); ++idx) {
      m_nodes.push_back(nullptr);
    }
    for (Node* func_node : f->nodes()) {
      for (int idx = 0; idx < m_nodes.size(); ++idx) {
        if (m_nodes[idx] == nullptr &&
            func_node->ToString().find(m_string[idx]) != std::string::npos) {
          m_nodes[idx] = func_node;
        }
      }
    }
    for (auto node : m_nodes) {
      EXPECT_NE(node, nullptr);
    }
  };
  std::vector<Node*> in_node;
  std::vector<std::string> in_str = {"literal.1", "literal.2", "literal.1",
                                     "literal.2"};
  std::vector<Node*> out_node;
  std::vector<std::string> out_str = {"array_index.9", "array_index.9",
                                      "array_index.10", "array_index.10"};
  assign_nodes(in_str, in_node);
  assign_nodes(out_str, out_node);

  // If we don't know the update index, we don't know if the final
  // value at an index is 0 or 1.
  for (int idx = 0; idx < in_node.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        bool proven_eq,
        TryProve(f, in_node[idx], Predicate::EqualTo(out_node[idx]),
                 absl::Seconds(10)));
    EXPECT_FALSE(proven_eq);
  }
}

// Array Concat #0a - Test bits after concat are traced back to input (part a)
TEST(Z3IrTranslatorTest, ConcatZero) {
  const std::string program = R"(
fn f(x: bits[4][1], y: bits[4][1]) -> bits[4] {
  array_concat.3: bits[4][4] = array_concat(x, x, y, y)

  literal.4: bits[32] = literal(value=0)
  literal.5: bits[32] = literal(value=1)
  literal.6: bits[32] = literal(value=2)
  literal.7: bits[32] = literal(value=3)

  array_index.8: bits[4] = array_index(array_concat.3, literal.4)
  array_index.9: bits[4] = array_index(array_concat.3, literal.5)
  array_index.10: bits[4] = array_index(array_concat.3, literal.6)
  array_index.11: bits[4] = array_index(array_concat.3, literal.7)

  xor.12: bits[4] = xor(array_index.8, array_index.11)
  xor.13: bits[4] = xor(xor.12, array_index.9)
  ret result: bits[4] = xor(xor.13, array_index.10)
}
)";

  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::EqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
}

// Array Concat #0b - Test bits after concat are traced back to input (part b)
TEST(Z3IrTranslatorTest, ConcatNotZero) {
  const std::string program = R"(
fn f(x: bits[4][1], y: bits[4][1]) -> bits[1] {
  array_concat.3: bits[4][4] = array_concat(x, x, y, y)

  literal.4: bits[32] = literal(value=0)
  literal.5: bits[32] = literal(value=1)
  literal.6: bits[32] = literal(value=2)
  literal.7: bits[32] = literal(value=3)

  array_index.8: bits[4] = array_index(array_concat.3, literal.4)
  array_index.9: bits[4] = array_index(array_concat.3, literal.5)
  array_index.10: bits[4] = array_index(array_concat.3, literal.6)
  array_index.11: bits[4] = array_index(array_concat.3, literal.7)

  xor.12: bits[4] = xor(array_index.8, array_index.11)
  xor.13: bits[4] = xor(xor.12, array_index.9)

  array_index.14: bits[4] = array_index(x, literal.4)
  array_index.15: bits[4] = array_index(y, literal.4)

  ret result: bits[1] = eq(xor.13, array_index.15)
}
)";

  Package p("test");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(
      bool proven, TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
                            absl::Seconds(1)));
  EXPECT_TRUE(proven);
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

  Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);

  // Remember: we try to prove the condition by searching for a model that
  // produces the opposite result. Thus, we want to find a model where the
  // results are _not_ equal.
  Z3_ast objective = Z3_mk_not(ctx, Z3_mk_eq(ctx, return_1, return_2));
  Z3_solver_assert(ctx, solver, objective);

  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  EXPECT_EQ(satisfiable, Z3_L_FALSE);
  Z3_solver_dec_ref(ctx, solver);
}

TEST(Z3IrTranslatorTest, HandlesZeroOneHotSelector) {
  const std::string program = R"(
package p

fn f(selector: bits[2]) -> bits[4] {
  literal.1: bits[4] = literal(value=0xf)
  literal.2: bits[4] = literal(value=0x5)
  ret one_hot_sel.3: bits[4] = one_hot_sel(selector, cases=[literal.1, literal.2])
})";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
  XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                           IrTranslator::CreateAndTranslate(f));
  Z3_context ctx = translator->ctx();
  Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
  // We want to prove that the result can be 0x0 - without the fix for this case
  // (selector_can_be_zero=false -> true), that can not be the case.
  Z3_ast z3_zero = Z3_mk_int(ctx, 0, Z3_mk_bv_sort(ctx, 4));
  Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), z3_zero);
  Z3_solver_assert(ctx, solver, objective);
  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  EXPECT_EQ(satisfiable, Z3_L_TRUE);
  Z3_solver_dec_ref(ctx, solver);
}

TEST(Z3IrTranslatorTest, HandlesUMul) {
  const std::string tmpl = R"(
package p

fn f() -> bits[6] {
  literal.1: bits[4] = literal(value=$0)
  literal.2: bits[8] = literal(value=$1)
  ret umul.3: bits[6] = umul(literal.1, literal.2)
}
)";

  std::vector<std::pair<int, int>> test_cases({
      {0x0, 0x5},
      {0x1, 0x5},
      {0xf, 0x4},
      {0x3, 0x7f},
      {0xf, 0xff},
  });

  for (std::pair<int, int> test_case : test_cases) {
    std::string program =
        absl::Substitute(tmpl, test_case.first, test_case.second);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    uint32 mask = 127;
    Z3_ast expected =
        Z3_mk_int(ctx, (test_case.first * test_case.second) & mask,
                  Z3_mk_bv_sort(ctx, 6));
    Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
    Z3_solver_assert(ctx, solver, objective);
    Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
    EXPECT_EQ(satisfiable, Z3_L_TRUE);
    Z3_solver_dec_ref(ctx, solver);
  }
}

TEST(Z3IrTranslatorTest, HandlesSMul) {
  const std::string tmpl = R"(
package p

fn f() -> bits[6] {
  literal.1: bits[4] = literal(value=$0)
  literal.2: bits[8] = literal(value=$1)
  ret smul.3: bits[6] = smul(literal.1, literal.2)
}
)";

  std::vector<std::pair<int, int>> test_cases({
      {0, 5},
      {1, 5},
      {-1, 5},
      {1, -5},
      {-1, -5},
      {6, -5},
      {-5, 7},
  });

  for (std::pair<int, int> test_case : test_cases) {
    std::string program =
        absl::Substitute(tmpl, test_case.first, test_case.second);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    // To avoid boom in the last case (-35 requires 7 bits to represent), put
    // everything in a 7-bit Bits and truncate by one.
    Bits expected_bits = SBits(test_case.first * test_case.second, 7);
    expected_bits = expected_bits.Slice(0, 6);
    Z3_ast expected =
        Z3_mk_int(ctx, expected_bits.ToInt64().value(), Z3_mk_bv_sort(ctx, 6));
    Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
    Z3_solver_assert(ctx, solver, objective);
    Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
    EXPECT_EQ(satisfiable, Z3_L_TRUE);
    Z3_solver_dec_ref(ctx, solver);
  }
}

}  // namespace
}  // namespace xls

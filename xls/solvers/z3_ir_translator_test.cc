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

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/solvers/z3_ir_translator_matchers.h"
#include "xls/solvers/z3_utils.h"
#include "external/z3/src/api/z3.h"  // IWYU pragma: keep
#include "external/z3/src/api/z3_api.h"

namespace xls {
namespace {

using solvers::z3::IrTranslator;
using solvers::z3::Predicate;
using solvers::z3::PredicateOfNode;
using solvers::z3::ProverResult;
using solvers::z3::TryProve;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;

using solvers::z3::IsProvenFalse;
using solvers::z3::IsProvenTrue;
using testing::HasSubstr;

class Z3IrTranslatorTest : public IrTestBase {};

// This parameterized fixture allows a single value-parameterized Z3-based test
// to be instantiated across a range of different bitwidths. Each individual
// test may decide how to use this width parameter, e.g., to set the width of
// a function input.
class Z3ParameterizedWidthBitVectorIrTranslatorTest
    : public Z3IrTranslatorTest,
      public testing::WithParamInterface<int> {
 protected:
  int BitWidth() const { return GetParam(); }
};

// The complexity of the SMT formula underlying a width-parameterized test grows
// rapidly with width, so this suite picks a sampling of small bitwidths.
INSTANTIATE_TEST_SUITE_P(Z3BitVectorTestWidthSweep,
                         Z3ParameterizedWidthBitVectorIrTranslatorTest,
                         testing::Values(1, 2, 3, 8));

TEST_F(Z3IrTranslatorTest, ZeroIsZero) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Literal(UBits(0, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProverResult proven,
                           TryProve(f, x.node(), Predicate::EqualToZero(),
                                    absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ZeroIsZeroAndOneIsOne) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Literal(UBits(0, /*bit_count=*/1));
  auto y = b.Literal(UBits(1, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  std::vector<PredicateOfNode> terms = {
      PredicateOfNode{x.node(), Predicate::EqualToZero()},
      PredicateOfNode{y.node(), Predicate::NotEqualToZero()},
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProveConjunction(f, terms, absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ParamsEqualToSelfButUnequalToEachOther) {
  std::unique_ptr<Package> package = CreatePackage();
  Type* u32 = package->GetBitsType(32);
  FunctionBuilder b("f", package.get());
  auto x = b.Param("x", u32);
  auto y = b.Param("y", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  std::vector<PredicateOfNode> terms = {
      PredicateOfNode{x.node(), Predicate::IsEqualTo(x.node())},
      PredicateOfNode{y.node(), Predicate::IsEqualTo(y.node())},
      // This term should not prove.
      PredicateOfNode{x.node(), Predicate::IsEqualTo(y.node())},
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProveConjunction(f, terms, absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenFalse());
}

TEST_F(Z3IrTranslatorTest, ParamAddOneIsGeParam) {
  std::unique_ptr<Package> package = CreatePackage();
  Type* u32 = package->GetBitsType(32);
  FunctionBuilder b("f", package.get());
  auto x = b.Param("x", u32);
  auto one = b.Literal(UBits(1, /*bit_count=*/32));
  auto xp1 = b.Add(x, one);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  std::vector<PredicateOfNode> terms = {
      PredicateOfNode{xp1.node(), Predicate::UnsignedGreaterOrEqual(
                                      UBits(1, /*bit_count=*/32))},
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProveConjunction(f, terms, absl::InfiniteDuration()));
  // The all-ones value will cause rollover such that the assertion `xp1 >= 1`
  // is false.
  EXPECT_THAT(proven, IsProvenFalse());
}

TEST_F(Z3IrTranslatorTest, ZeroTwoBitsIsZero) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Literal(UBits(0, /*bit_count=*/2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProverResult proven,
                           TryProve(f, x.node(), Predicate::EqualToZero(),
                                    absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, OneIsNotEqualToZero) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Literal(UBits(1, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProverResult proven,
                           TryProve(f, x.node(), Predicate::EqualToZero(),
                                    absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenFalse());
}

TEST_F(Z3IrTranslatorTest, OneIsNotEqualToZeroPredicate) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Literal(UBits(1, /*bit_count=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProverResult proven,
                           TryProve(f, x.node(), Predicate::NotEqualToZero(),
                                    absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ParamMinusSelfIsZero) {
  std::unique_ptr<Package> package = CreatePackage();
  Type* u32 = package->GetBitsType(32);
  FunctionBuilder b("f", package.get());
  auto x = b.Param("x", u32);
  auto res = b.Subtract(x, x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProverResult proven,
                           TryProve(f, res.node(), Predicate::EqualToZero(),
                                    absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XPlusYMinusYIsX) {
  const std::string program = R"(
fn f(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  ret sub.2: bits[32] = sub(add.1, y)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(),
               Predicate::IsEqualTo(f->GetParamByName("x").value()),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, TupleIndexMinusSelf) {
  const std::string program = R"(
fn f(p: (bits[1], bits[32])) -> bits[32] {
  x: bits[32] = tuple_index(p, index=1)
  ret z: bits[32] = sub(x, x)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ConcatThenSliceIsSelf) {
  const std::string program = R"(
fn f(x: bits[4], y: bits[4], z: bits[4]) -> bits[1] {
  a: bits[12] = concat(x, y, z)
  b: bits[4] = bit_slice(a, start=4, width=4)
  ret c: bits[1] = eq(y, b)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, InBoundsDynamicSlice) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=1)
  dynamic_slice: bits[3] = dynamic_bit_slice(p, start, width=3)
  slice: bits[3] = bit_slice(p, start=1, width=3)
  ret result: bits[1] = eq(slice, dynamic_slice)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, PartialOutOfBoundsDynamicSlice) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=2)
  slice: bits[3] = dynamic_bit_slice(p, start, width=3)
  out_of_bounds: bits[1] = bit_slice(slice, start=2, width=1)
  zero: bits[1] = literal(value=0)
  ret result: bits[1] = eq(out_of_bounds, zero)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, CompletelyOutOfBoundsDynamicSlice) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=7)
  slice: bits[3] = dynamic_bit_slice(p, start, width=3)
  zero: bits[3] = literal(value=0)
  ret result: bits[1] = eq(slice, zero)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, BitSliceUpdate) {
  const std::string program = R"(
fn f(x: bits[8], v: bits[4]) -> bits[1] {
  start: bits[4] = literal(value=2)
  update: bits[8] = bit_slice_update(x, start, v)
  x_lsb: bits[2] = bit_slice(x, start=0, width=2)
  x_msb: bits[2] = bit_slice(x, start=6, width=2)
  expected: bits[8] = concat(x_msb, v, x_lsb)
  ret result: bits[1] = eq(update, expected)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, BitSliceUpdateOutOfBounds) {
  const std::string program = R"(
fn f(x: bits[8], v: bits[4]) -> bits[1] {
  start: bits[32] = literal(value=200)
  update: bits[8] = bit_slice_update(x, start, v)
  ret result: bits[1] = eq(update, x)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, BitSliceUpdateZeroStart) {
  const std::string program = R"(
fn f(x: bits[8], v: bits[16]) -> bits[1] {
  start: bits[32] = literal(value=0)
  update: bits[8] = bit_slice_update(x, start, v)
  expected: bits[8] = bit_slice(v, start=0, width=8)
  ret result: bits[1] = eq(update, expected)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ValueUgtSelf) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  ret result: bits[1] = ugt(p, p)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ValueUltSelf) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  ret result: bits[1] = ult(p, p)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ZeroExtBitAlwaysZero) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  x: bits[5] = zero_ext(p, new_bit_count=5)
  ret msb: bits[1] = bit_slice(x, start=4, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ZeroMinusParamHighBit) {
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
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

// Since the value can wrap around, we should not be able to prove that adding
// one to a value is unsigned-greater-than itself.
TEST_F(Z3IrTranslatorTest, BumpByOneUgtSelf) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  one: bits[4] = literal(value=1)
  x: bits[4] = add(p, one)
  ret result: bits[1] = ugt(x, p)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenFalse());

  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenFalse());
}

TEST_F(Z3IrTranslatorTest, MaskAndReverse) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[2] = literal(value=1)
  x: bits[2] = and(p, one)
  rev: bits[2] = reverse(x)
  ret result: bits[1] = bit_slice(rev, start=0, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ReverseSlicesEq) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  p0: bits[1] = bit_slice(p, start=0, width=1)
  rp: bits[2] = reverse(p)
  rp1: bits[1] = bit_slice(rp, start=1, width=1)
  ret result: bits[1] = eq(p0, rp1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ShiftRightLogicalFillsZero) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[2] = literal(value=1)
  x: bits[2] = shrl(p, one)
  ret result: bits[1] = bit_slice(x, start=1, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ShiftLeftLogicalFillsZero) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[2] = literal(value=1)
  x: bits[2] = shll(p, one)
  ret result: bits[1] = bit_slice(x, start=0, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, ShiftLeftLogicalDifferentSize) {
  const std::string program = R"(
fn f(p: bits[2]) -> bits[1] {
  one: bits[1] = literal(value=1)
  x: bits[2] = shll(p, one)
  ret result: bits[1] = bit_slice(x, start=0, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XAndNotXIsZero) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  np: bits[1] = not(p)
  ret result: bits[1] = and(p, np)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XNandNotXIsZero) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  np: bits[1] = not(p)
  ret result: bits[1] = nand(p, np)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XOrNotXIsNotZero) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  np: bits[1] = not(p)
  ret result: bits[1] = or(p, np)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_P(Z3ParameterizedWidthBitVectorIrTranslatorTest,
       AndReduceIsEqualToXIsAllOnes) {
  // Define a miter circuit: the implementation performs an `and_reduce` and the
  // specification checks for inequality with a bitvector of all ones. The
  // outputs should be equal across the full space of inputs.
  constexpr std::string_view program_template = R"(
fn f(p: bits[$0]) -> bits[1] {
  zero: bits[$0] = literal(value=0)
  all_ones: bits[$0] = not(zero)
  impl: bits[1] = and_reduce(p)
  spec: bits[1] = eq(p, all_ones)
  ret eq: bits[1] = eq(impl, spec)
}
)";
  const std::string program = absl::Substitute(program_template, BitWidth());
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_P(Z3ParameterizedWidthBitVectorIrTranslatorTest,
       KnownNegativeSubIsNarrowable) {
  // Verify that a subtract that is known-negative is same as narrowed sub
  // extended with 1s
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  auto full_width = BitWidth() + 4;
  BValue x = fb.Param("x", package->GetBitsType(BitWidth()));
  BValue y = fb.Param("y", package->GetBitsType(BitWidth()));
  auto x_wide = fb.ZeroExtend(x, full_width);
  // y_wide is always larger than x
  auto y_wide =
      fb.ZeroExtend(fb.Concat({fb.Literal(UBits(1, 1)), y}), full_width);
  auto full_sub = fb.Subtract(x_wide, y_wide);
  auto narrow_sub =
      fb.Concat({fb.Literal(Bits::AllOnes(3)),
                 fb.Subtract(fb.BitSlice(x_wide, 0, BitWidth() + 1),
                             fb.BitSlice(y_wide, 0, BitWidth() + 1))});
  fb.Eq(narrow_sub, full_sub);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_P(Z3ParameterizedWidthBitVectorIrTranslatorTest,
       KnownPositiveSubIsNarrowable) {
  // Verify that a subtract that is known-positive is same as narrowed sub
  // zero-extended.
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  auto full_width = BitWidth() + 4;
  BValue x = fb.Param("x", package->GetBitsType(BitWidth()));
  BValue y = fb.Param("y", package->GetBitsType(BitWidth()));
  auto x_wide = fb.ZeroExtend(x, full_width);
  // y_wide is always larger than x
  auto y_wide =
      fb.ZeroExtend(fb.Concat({fb.Literal(UBits(1, 1)), y}), full_width);
  auto full_sub = fb.Subtract(y_wide, x_wide);
  auto narrow_sub =
      fb.ZeroExtend(fb.Subtract(fb.BitSlice(y_wide, 0, BitWidth() + 1),
                                fb.BitSlice(x_wide, 0, BitWidth() + 1)),
                    full_width);
  fb.Eq(narrow_sub, full_sub);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_P(Z3ParameterizedWidthBitVectorIrTranslatorTest,
       OneBitNegativeIsSignExtendOneBit) {
  // Verify that a single bit value negated is the same as that bit
  // sign-extended to desired length.
  constexpr std::string_view program_template = R"(
fn f(p: bits[1]) -> bits[1] {
  zero: bits[$0] = literal(value=0)
  conc: bits[$1] = concat(zero, p)
  nega: bits[$1] = neg(conc)
  extn: bits[$1] = sign_ext(p, new_bit_count=$1)
  ret eq: bits[1] = eq(nega, extn)
}
)";
  const std::string program =
      absl::Substitute(program_template, BitWidth(), BitWidth() + 1);
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_P(Z3ParameterizedWidthBitVectorIrTranslatorTest,
       ThreeBitNegativeIsSignExtendOneBit) {
  // Verify that we can transform a negation of a value with many known zero
  // bits into a negation of the non-zero bits with a single zero bit then
  // sign-extended to the correct length.
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  auto full_width = BitWidth() + 3;
  auto param = fb.Param("p", package->GetBitsType(3));
  auto conc_big = fb.Concat({fb.Literal(UBits(0, BitWidth())), param});
  auto neg_big = fb.Negate(conc_big);
  auto conc_small = fb.Concat({fb.Literal(UBits(0, 1)), param});
  auto neg_small = fb.Negate(conc_small);
  auto extn_small = fb.SignExtend(neg_small, full_width);
  fb.Eq(neg_big, extn_small);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_P(Z3ParameterizedWidthBitVectorIrTranslatorTest,
       OrReduceIsEqualToXIsNotZero) {
  // Define a miter circuit: the implementation performs an `or_reduce` and the
  // specification checks for inequality with the zero bitvector. The outputs
  // should be equal across the full space of inputs.
  constexpr std::string_view program_template = R"(
fn f(p: bits[$0]) -> bits[1] {
  zero: bits[$0] = literal(value=0)
  impl: bits[1] = or_reduce(p)
  spec: bits[1] = ne(p, zero)
  ret eq: bits[1] = eq(impl, spec)
}
)";
  const std::string program = absl::Substitute(program_template, BitWidth());
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XorReduceIsEqualToXorOfBits) {
  // Define a miter circuit: the implementation performs an `xor_reduce` and the
  // specification checks for inequality with a bitvector of all ones. The
  // outputs should be equal across the full space of inputs.
  constexpr std::string_view program = R"(
fn f(p: bits[3]) -> bits[1] {
  impl: bits[1] = xor_reduce(p)
  b0: bits[1] = bit_slice(p, start=0, width=1)
  b1: bits[1] = bit_slice(p, start=1, width=1)
  b2: bits[1] = bit_slice(p, start=2, width=1)
  spec: bits[1] = xor(b0, b1, b2)
  ret eq: bits[1] = eq(impl, spec)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, SignExtendBitsAreEqual) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[1] {
  p2: bits[2] = sign_ext(p, new_bit_count=2)
  b0: bits[1] = bit_slice(p2, start=0, width=1)
  b1: bits[1] = bit_slice(p2, start=1, width=1)
  ret eq: bits[1] = eq(b0, b1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XPlusNegX) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[4] {
  np: bits[4] = neg(p)
  ret result: bits[4] = add(p, np)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, XNeX) {
  const std::string program = R"(
fn f(p: bits[4]) -> bits[1] {
  ret result: bits[1] = ne(p, p)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, OneHot) {
  const std::string program = R"(
fn f(p: bits[1]) -> bits[2] {
  ret result: bits[2] = one_hot(p, lsb_prio=true)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, DecodeZeroIsNotZero) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  z: bits[2] = xor(x, x)
  ret result: bits[1] = decode(z, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, DecodeWithOverflowedIndexIsZero) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  literal.1: bits[2] = literal(value=0b10)
  or.2: bits[2] = or(x, literal.1)
  ret result: bits[1] = decode(or.2, width=1)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, EncodeZeroIsZero) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  z: bits[2] = xor(x, x)
  ret result: bits[1] = encode(z)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, EncodeWithIndex1SetIsNotZero) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  literal.1: bits[2] = literal(value=0b10)
  or.2: bits[2] = or(x, literal.1)
  ret result: bits[1] = encode(or.2)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, SelWithDefault) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  literal.1: bits[1] = literal(value=0b1)
  literal.2: bits[1] = literal(value=0b0)
  ret sel.3: bits[1] = sel(x, cases=[literal.1], default=literal.2)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenFalse());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenFalse());
}

TEST_F(Z3IrTranslatorTest, SgeVsSlt) {
  const std::string program = R"(
fn f(x: bits[2], y: bits[2]) -> bits[1] {
  sge: bits[1] = sge(x, y)
  slt: bits[1] = slt(x, y)
  ret and: bits[1] = and(sge, slt)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_ez,
      TryProve(f, f->return_value(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_ez, IsProvenTrue());
}

// TODO(b/153195241): Re-enable these.
#ifdef NDEBUG
TEST_F(Z3IrTranslatorTest, AddToMostNegativeSge) {
  const std::string program = R"(
fn f(x: bits[2]) -> bits[1] {
  most_negative: bits[2] = literal(value=0b10)
  add: bits[2] = add(most_negative, x)
  ret result: bits[1] = sge(add, most_negative)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, SltVsMaxPositive) {
  const std::string program = R"(
fn f(x: bits[3]) -> bits[1] {
  most_positive: bits[3] = literal(value=0b011)
  most_negative: bits[3] = literal(value=0b100)
  eq_mp: bits[1] = eq(x, most_positive)
  sel: bits[3] = sel(eq_mp, cases=[x, most_negative])
  ret result: bits[1] = slt(sel, most_positive)
}
)";
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}
#endif

TEST_F(Z3IrTranslatorTest, TupleAndAccess) {
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
  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

// This test verifies that selects with tuple values can be translated.
TEST_F(Z3IrTranslatorTest, TupleSelect) {
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

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, TupleSelectsMore) {
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

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* to_compare = FindNode("literal.3", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(to_compare),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, BasicAfterAllTokenTest) {
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
  ret result: bits[32] = array_index(array.6, indices=[literal.3])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  // Check that non-token logic is not affected.
  {
    Node* eq_node = FindNode("literal.5", package.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }

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
      XLS_ASSERT_OK_AND_ASSIGN(
          ProverResult proven_eq,
          TryProve(f, token_nodes.at(l_idx),
                   Predicate::IsEqualTo(token_nodes.at(r_idx)),
                   absl::InfiniteDuration()));
      EXPECT_THAT(proven_eq, IsProvenTrue());
    }
    // Can't prove a token is 0 or non-zero because it is a non-bit type.
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::EqualToZero(),
                          absl::InfiniteDuration())
                     .status()
                     .ok());
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::NotEqualToZero(),
                          absl::InfiniteDuration())
                     .status()
                     .ok());
  }
}

TEST_F(Z3IrTranslatorTest, BasicMinDelayTokenTest) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  after_all.10: token = after_all()
  literal.2: bits[32] = literal(value=2)
  min_delay.11: token = min_delay(after_all.10, delay=1)
  literal.3: bits[32] = literal(value=4)
  min_delay.12: token = min_delay(min_delay.11, delay=2)
  literal.4: bits[32] = literal(value=8)
  min_delay.13: token = min_delay(min_delay.12, delay=4)
  literal.5: bits[32] = literal(value=16)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  ret result: bits[32] = array_index(array.6, indices=[literal.3])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  // Check that non-token logic is not affected.
  {
    Node* eq_node = FindNode("literal.5", package.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }

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
      XLS_ASSERT_OK_AND_ASSIGN(
          ProverResult proven_eq,
          TryProve(f, token_nodes.at(l_idx),
                   Predicate::IsEqualTo(token_nodes.at(r_idx)),
                   absl::InfiniteDuration()));
      EXPECT_THAT(proven_eq, IsProvenTrue());
    }
    // Can't prove a token is 0 or non-zero because it is a non-bit type.
    EXPECT_THAT(
        TryProve(f, token_nodes.at(l_idx), Predicate::EqualToZero(),
                 absl::InfiniteDuration()),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("predicate eq zero vs non-bit-vector Z3 value")));
    EXPECT_THAT(
        TryProve(f, token_nodes.at(l_idx), Predicate::NotEqualToZero(),
                 absl::InfiniteDuration()),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("predicate ne zero vs non-bit-vector Z3 value")));
  }
}

TEST_F(Z3IrTranslatorTest, TokensNotEqualToEmptyTuples) {
  const std::string program = R"(
package p

fn f(empty_tuple: ()) -> bits[32] {
  after_all.10: token = after_all()
  ret literal.1: bits[32] = literal(value=1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  Node* token_node = FindNode("after_all.10", package.get());
  Node* tuple_node = FindNode("empty_tuple", package.get());

  // Even though we represent tokens as empty tuples as a convenient hack, we
  // should not evaluate tokens == empty tuples.  Evaluation should fail because
  // an empty tuple is not a bit type.
  EXPECT_THAT(
      TryProve(f, token_node, Predicate::IsEqualTo(tuple_node),
               absl::InfiniteDuration()),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("predicate eq empty_tuple vs non-bit-vector Z3 value")));
  EXPECT_THAT(
      TryProve(f, tuple_node, Predicate::IsEqualTo(token_node),
               absl::InfiniteDuration()),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("predicate eq after_all.10 vs non-bit-vector Z3 value")));
}

TEST_F(Z3IrTranslatorTest, TokenArgsAndReturn) {
  const std::string program = R"(
package p

fn f(arr1: token, arr2: token, arr3: token) -> token {
  ret after_all.1: token = after_all(arr1, arr2, arr3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

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
                           Predicate::IsEqualTo(token_nodes.at(r_idx)),
                           absl::InfiniteDuration()),
                  IsOkAndHolds(IsProvenTrue()));
    }
    // Can't prove a token is 0 or non-zero because it is a non-bit type.
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::EqualToZero(),
                          absl::InfiniteDuration())
                     .status()
                     .ok());
    EXPECT_FALSE(TryProve(f, token_nodes.at(l_idx), Predicate::NotEqualToZero(),
                          absl::InfiniteDuration())
                     .status()
                     .ok());
  }
}

// Array test 1: Can we properly handle arrays of bits!
TEST_F(Z3IrTranslatorTest, IndexArrayOfBits) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=4)
  literal.4: bits[32] = literal(value=8)
  literal.5: bits[32] = literal(value=16)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  ret result: bits[32] = array_index(array.6, indices=[literal.3])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("literal.5", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, IndexBitsType) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  eight: bits[32] = literal(value=8)
  ret result: bits[32] = array_index(eight, indices=[])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("eight", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

// Array test 2: Can we properly handle arrays...OF ARRAYS?
TEST_F(Z3IrTranslatorTest, IndexArrayOfArrays) {
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
  ret result: bits[32] = array_index(array.11, indices=[literal.3, literal.2])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("literal.4", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, IndexArrayOfArraysWithSequentialIndexOps) {
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
  subarray: bits[32][5] = array_index(array.11, indices=[literal.3])
  ret result: bits[32] = array_index(subarray, indices=[literal.2])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("literal.4", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

// Array test 3! Arrays...OF TUPLES
TEST_F(Z3IrTranslatorTest, IndexArrayOfTuples) {
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
  element_4: (bits[32], bits[32], bits[32]) = array_index(array.11, indices=[literal.4])
  ret tuple_index.13: bits[32] = tuple_index(element_4, index=0)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("literal.5", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, IndexArrayOfTuplesOfArrays) {
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
  element_2: (bits[32][5], bits[32][5], bits[32][5]) = array_index(array.16, indices=[literal.2])
  tuple_index.18: bits[32][5] = tuple_index(element_2, index=1)
  ret result: bits[32] = array_index(tuple_index.18, indices=[literal.3])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("literal.2", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, OverflowingArrayIndex) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  array.6: bits[32][5] = array(literal.1, literal.2, literal.3, literal.4, literal.5)
  ret result: bits[32] = array_index(array.6, indices=[literal.5])
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
  Node* eq_node = FindNode("literal.5", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_eq,
      TryProve(f, f->return_value(), Predicate::IsEqualTo(eq_node),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_eq, IsProvenTrue());
}

// UpdateArray test 1: Array of bits
TEST_F(Z3IrTranslatorTest, UpdateArrayOfBits) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  zero: bits[32] = literal(value=0)
  one: bits[32] = literal(value=1)
  forty_two: bits[32] = literal(value=42)
  array: bits[32][2] = array(zero, zero)
  updated_array: bits[32][2] = array_update(array, forty_two, indices=[one])
  element_0: bits[32] = array_index(updated_array, indices=[zero])
  ret element_1: bits[32] = array_index(updated_array, indices=[one])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"zero", "forty_two"};
  std::vector<std::string> observe = {"element_0", "element_1"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

TEST_F(Z3IrTranslatorTest, UpdateArrayOfOutOfBounds) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  zero: bits[32] = literal(value=0)
  one: bits[32] = literal(value=1)
  forty_two: bits[32] = literal(value=42)
  array: bits[32][2] = array(zero, zero)
  updated_array: bits[32][2] = array_update(array, forty_two, indices=[forty_two])
  element_0: bits[32] = array_index(updated_array, indices=[zero])
  ret element_1: bits[32] = array_index(updated_array, indices=[one])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"zero", "zero"};
  std::vector<std::string> observe = {"element_0", "element_1"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

TEST_F(Z3IrTranslatorTest, UpdateBitsType) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  one: bits[32] = literal(value=1)
  forty_two: bits[32] = literal(value=42)
  ret result: bits[32] = array_update(one, forty_two, indices=[])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  EXPECT_THAT(
      TryProve(f, FindNode("result", package.get()),
               Predicate::IsEqualTo(FindNode("forty_two", package.get())),
               absl::InfiniteDuration()),
      IsOkAndHolds(IsProvenTrue()));
}

// UpdateArray test 2: Array of Arrays
TEST_F(Z3IrTranslatorTest, UpdateArrayOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret literal.2: bits[32] = literal(value=1)
  array.3: bits[32][2] = array(literal.1, literal.1)
  array.4: bits[32][2] = array(literal.2, literal.2)
  array.6: bits[32][2][2] = array(array.3, array.3)
  updated_array: bits[32][2][2] = array_update(array.6, array.4, indices=[literal.2])
  subarray_0: bits[32][2] = array_index(updated_array, indices=[literal.1])
  element_0_0: bits[32] = array_index(subarray_0, indices=[literal.1])
  element_0_1: bits[32] = array_index(subarray_0, indices=[literal.2])
  subarray_1: bits[32][2] = array_index(updated_array, indices=[literal.2])
  element_1_0: bits[32] = array_index(subarray_1, indices=[literal.1])
  element_1_1: bits[32] = array_index(subarray_1, indices=[literal.2])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"literal.1", "literal.1", "literal.2",
                                     "literal.2"};
  std::vector<std::string> observe = {"element_0_0", "element_0_1",
                                      "element_1_0", "element_1_1"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

TEST_F(Z3IrTranslatorTest, UpdateSingleElementInArrayOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  zero: bits[32] = literal(value=0)
  one: bits[32] = literal(value=1)
  array.3: bits[32][2] = array(zero, zero)
  array.6: bits[32][2][2] = array(array.3, array.3)
  forty_two: bits[32] = literal(value=42)
  updated_array: bits[32][2][2] = array_update(array.6, forty_two, indices=[one, zero])
  subarray_0: bits[32][2] = array_index(updated_array, indices=[zero])
  element_0_0: bits[32] = array_index(subarray_0, indices=[zero])
  element_0_1: bits[32] = array_index(subarray_0, indices=[one])
  subarray_1: bits[32][2] = array_index(updated_array, indices=[one])
  element_1_0: bits[32] = array_index(subarray_1, indices=[zero])
  ret element_1_1: bits[32] = array_index(subarray_1, indices=[one])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"zero", "zero", "forty_two", "zero"};
  std::vector<std::string> observe = {"element_0_0", "element_0_1",
                                      "element_1_0", "element_1_1"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

// UpdateArray test 3: Array of Tuples
TEST_F(Z3IrTranslatorTest, UpdateArrayOfTuples) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret literal.2: bits[32] = literal(value=1)
  tuple.3: (bits[32], bits[32]) = tuple(literal.1, literal.2)
  tuple.4: (bits[32], bits[32]) = tuple(literal.2, literal.1)
  array.6: (bits[32], bits[32])[2] = array(tuple.3, tuple.3)
  array_update.8:(bits[32], bits[32])[2] = array_update(array.6, tuple.4, indices=[literal.2])
  element_0: (bits[32], bits[32]) = array_index(array_update.8, indices=[literal.1])
  tuple_index.10: bits[32] = tuple_index(element_0, index=0)
  tuple_index.11: bits[32] = tuple_index(element_0, index=1)
  array_index.12: (bits[32], bits[32]) = array_index(array_update.8, indices=[literal.2])
  tuple_index.13: bits[32] = tuple_index(array_index.12, index=0)
  tuple_index.14: bits[32] = tuple_index(array_index.12, index=1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"literal.1", "literal.2", "literal.2",
                                     "literal.1"};
  std::vector<std::string> observe = {"tuple_index.10", "tuple_index.11",
                                      "tuple_index.13", "tuple_index.14"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

// UpdateArray test 4: Array of Tuples of Arrays
TEST_F(Z3IrTranslatorTest, UpdateArrayOfTuplesOfArrays) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret literal.2: bits[32] = literal(value=1)
  array.3: bits[32][2] = array(literal.1, literal.2)
  array.4: bits[32][2] = array(literal.2, literal.1)
  tuple.5: (bits[32][2], bits[32][2]) = tuple(array.3, array.4)
  tuple.6: (bits[32][2], bits[32][2]) = tuple(array.4, array.3)
  array.7: (bits[32][2], bits[32][2])[2] = array(tuple.5, tuple.5)
  array_update.8: (bits[32][2], bits[32][2])[2] = array_update(array.7, tuple.6, indices=[literal.2])
  element_0: (bits[32][2], bits[32][2]) = array_index(array_update.8, indices=[literal.1])
  tuple_index.10: bits[32][2] = tuple_index(element_0, index=0)
  tuple_index.11: bits[32][2] = tuple_index(element_0, index=1)
  array_index.12: bits[32] = array_index(tuple_index.10, indices=[literal.1])
  array_index.13: bits[32] = array_index(tuple_index.10, indices=[literal.2])
  array_index.14: bits[32] = array_index(tuple_index.11, indices=[literal.1])
  array_index.15: bits[32] = array_index(tuple_index.11, indices=[literal.2])
  array_index.16: (bits[32][2], bits[32][2]) = array_index(array_update.8, indices=[literal.2])
  tuple_index.17: bits[32][2] = tuple_index(array_index.16, index=0)
  tuple_index.18: bits[32][2] = tuple_index(array_index.16, index=1)
  array_index.19: bits[32] = array_index(tuple_index.17, indices=[literal.1])
  array_index.20: bits[32] = array_index(tuple_index.17, indices=[literal.2])
  array_index.21: bits[32] = array_index(tuple_index.18, indices=[literal.1])
  array_index.22: bits[32] = array_index(tuple_index.18, indices=[literal.2])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"literal.1", "literal.2", "literal.2",
                                     "literal.1", "literal.2", "literal.1",
                                     "literal.1", "literal.2"};
  std::vector<std::string> observe = {
      "array_index.12", "array_index.13", "array_index.14", "array_index.15",
      "array_index.19", "array_index.20", "array_index.21", "array_index.22"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

// UpdateArray test 4: Out of bounds index
TEST_F(Z3IrTranslatorTest, UpdateArrayOutOfBoundsIndex) {
  const std::string program = R"(
package p

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret literal.2: bits[32] = literal(value=1)
  literal.3: bits[32] = literal(value=99)
  array.6: bits[32][2] = array(literal.1, literal.1)
  array_update.8: bits[32][2] = array_update(array.6, literal.2, indices=[literal.3])
  element_0: bits[32] = array_index(array_update.8, indices=[literal.1])
  array_index.10: bits[32] = array_index(array_update.8, indices=[literal.2])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> expect = {"literal.1", "literal.1"};
  std::vector<std::string> observe = {"element_0", "array_index.10"};

  for (int idx = 0; idx < expect.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(expect[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(observe[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenTrue());
  }
}

// UpdateArray test 5: Unknown index
TEST_F(Z3IrTranslatorTest, UpdateArrayUnknownIndex) {
  const std::string program = R"(
package p

fn f(index: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret literal.2: bits[32] = literal(value=1)
  literal.3: bits[32] = literal(value=99)
  array.6: bits[32][2] = array(literal.1, literal.1)
  array_update.8: bits[32][2] = array_update(array.6, literal.2, indices=[index])
  element_0: bits[32] = array_index(array_update.8, indices=[literal.1])
  array_index.10: bits[32] = array_index(array_update.8, indices=[literal.2])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));

  std::vector<std::string> in_str = {"literal.1", "literal.2", "literal.1",
                                     "literal.2"};
  std::vector<std::string> out_str = {"element_0", "element_0",
                                      "array_index.10", "array_index.10"};

  // If we don't know the update index, we don't know if the final
  // value at an index is 0 or 1.
  for (int idx = 0; idx < in_str.size(); ++idx) {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult proven_eq,
        TryProve(f, FindNode(in_str[idx], package.get()),
                 Predicate::IsEqualTo(FindNode(out_str[idx], package.get())),
                 absl::InfiniteDuration()));
    EXPECT_THAT(proven_eq, IsProvenFalse());
  }
}

// Array Concat #0a - Test bits after concat are traced back to input (part a)
TEST_F(Z3IrTranslatorTest, ConcatZero) {
  const std::string program = R"(
fn f(x: bits[4][1], y: bits[4][1]) -> bits[4] {
  array_concat.3: bits[4][4] = array_concat(x, x, y, y)

  literal.4: bits[32] = literal(value=0)
  literal.5: bits[32] = literal(value=1)
  literal.6: bits[32] = literal(value=2)
  literal.7: bits[32] = literal(value=3)

  array_index.8: bits[4] = array_index(array_concat.3, indices=[literal.4])
  element_0: bits[4] = array_index(array_concat.3, indices=[literal.5])
  array_index.10: bits[4] = array_index(array_concat.3, indices=[literal.6])
  array_index.11: bits[4] = array_index(array_concat.3, indices=[literal.7])

  xor.12: bits[4] = xor(array_index.8, array_index.11)
  xor.13: bits[4] = xor(xor.12, element_0)
  ret result: bits[4] = xor(xor.13, array_index.10)
}
)";

  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  EXPECT_THAT(TryProve(f, f->return_value(), Predicate::EqualToZero(),
                       absl::InfiniteDuration()),
              IsOkAndHolds(IsProvenTrue()));
}

// Array Concat #0b - Test bits after concat are traced back to input (part b)
TEST_F(Z3IrTranslatorTest, ConcatNotZero) {
  const std::string program = R"(
fn f(x: bits[4][1], y: bits[4][1]) -> bits[1] {
  array_concat.3: bits[4][4] = array_concat(x, x, y, y)

  literal.4: bits[32] = literal(value=0)
  literal.5: bits[32] = literal(value=1)
  literal.6: bits[32] = literal(value=2)
  literal.7: bits[32] = literal(value=3)

  array_index.8: bits[4] = array_index(array_concat.3, indices=[literal.4])
  element_0: bits[4] = array_index(array_concat.3, indices=[literal.5])
  array_index.10: bits[4] = array_index(array_concat.3, indices=[literal.6])
  array_index.11: bits[4] = array_index(array_concat.3, indices=[literal.7])

  xor.12: bits[4] = xor(array_index.8, array_index.11)
  xor.13: bits[4] = xor(xor.12, element_0)

  array_index.14: bits[4] = array_index(x, indices=[literal.4])
  array_index.15: bits[4] = array_index(y, indices=[literal.4])

  ret result: bits[1] = eq(xor.13, array_index.15)
}
)";

  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  EXPECT_THAT(TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
                       absl::InfiniteDuration()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(Z3IrTranslatorTest, ParamReuse) {
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
  ret y: bits[16] = param(name=y)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto p1, ParsePackage(program_1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, p1->GetFunction("f"));
  XLS_ASSERT_OK_AND_ASSIGN(auto translator_1,
                           IrTranslator::CreateAndTranslate(f1));
  std::vector<Z3_ast> imported_params;
  for (auto* param : f1->params()) {
    imported_params.push_back(translator_1->GetTranslation(param));
  }

  Z3_context ctx = translator_1->ctx();

  XLS_ASSERT_OK_AND_ASSIGN(auto p2, ParsePackage(program_2));
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

TEST_F(Z3IrTranslatorTest, HandlesZeroOneHotSelector) {
  const std::string program = R"(
package p

fn f(selector: bits[2]) -> bits[4] {
  literal.1: bits[4] = literal(value=0xf)
  literal.2: bits[4] = literal(value=0x5)
  ret one_hot_sel.3: bits[4] = one_hot_sel(selector, cases=[literal.1, literal.2])
})";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
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

TEST_F(Z3IrTranslatorTest, HandlePrioritySelect) {
  const std::string program = R"(
fn f(idx: bits[1]) -> bits[4] {
  literal.1: bits[4] = literal(value=0xf)
  literal.2: bits[4] = literal(value=0x5)
  one_hot.4: bits[2] = one_hot(idx, lsb_prio=true)
  ret priority_sel.3: bits[4] = priority_sel(one_hot.4, cases=[literal.1, literal.2])
})";

  std::unique_ptr<Package> package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven_nez,
      TryProve(f, f->return_value(), Predicate::NotEqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven_nez, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, HandlesUMul) {
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
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    uint32_t mask = 127;
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

TEST_F(Z3IrTranslatorTest, HandlesSMul) {
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
      {-1, -1},
      {0, -0},
  });

  for (std::pair<int, int> test_case : test_cases) {
    std::string program =
        absl::Substitute(tmpl, test_case.first, test_case.second);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    Bits lhs = SBits(test_case.first, 4);
    Bits rhs = SBits(test_case.second, 8);
    Bits expected_bits = bits_ops::SMul(lhs, rhs);

    Z3_ast expected =
        Z3_mk_int(ctx, expected_bits.ToInt64().value(), Z3_mk_bv_sort(ctx, 6));
    Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
    Z3_solver_assert(ctx, solver, objective);
    Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
    EXPECT_EQ(satisfiable, Z3_L_TRUE);
    Z3_solver_dec_ref(ctx, solver);
  }
}

TEST_F(Z3IrTranslatorTest, HandlesSMulOverflow) {
  const std::string tmpl = R"(
package p

fn f() -> bits[64] {
  literal.1: bits[8] = literal(value=$0)
  literal.2: bits[8] = literal(value=$1)
  ret smul.3: bits[64] = smul(literal.1, literal.2)
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
      {-1, -1},
      {0x7f, 0x7f},
  });

  for (std::pair<int, int> test_case : test_cases) {
    std::string program =
        absl::Substitute(tmpl, test_case.first, test_case.second);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    Bits lhs = SBits(test_case.first, 8);
    Bits rhs = SBits(test_case.second, 8);
    Bits expected_bits = bits_ops::SMul(lhs, rhs);

    Z3_ast expected =
        Z3_mk_int(ctx, expected_bits.ToInt64().value(), Z3_mk_bv_sort(ctx, 64));
    Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
    Z3_solver_assert(ctx, solver, objective);
    Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
    EXPECT_EQ(satisfiable, Z3_L_TRUE);
    Z3_solver_dec_ref(ctx, solver);
  }
}

TEST_F(Z3IrTranslatorTest, HandlesUDiv) {
  constexpr std::string_view tmpl = R"(
package p

fn f() -> bits[64] {
  literal.1: bits[64] = literal(value=$0)
  literal.2: bits[64] = literal(value=$1)
  ret udiv.3: bits[64] = udiv(literal.1, literal.2)
}
)";

  std::vector<std::pair<uint64_t, uint64_t>> test_cases{
      {0, 0},
      {1, 0},
      {3, 0},
      {std::numeric_limits<uint64_t>::max(), 0},
      {0, 1},
      {1, 1},
      {3, 1},
      {std::numeric_limits<uint64_t>::max(), 1},
      {4, 2},
      {4, 3},
      {std::numeric_limits<uint64_t>::max(), 2},
      {1, std::numeric_limits<uint64_t>::max()},
  };

  for (auto [test_case_lhs, test_case_rhs] : test_cases) {
    std::string program = absl::Substitute(tmpl, test_case_lhs, test_case_rhs);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    Bits lhs = UBits(test_case_lhs, 64);
    Bits rhs = UBits(test_case_rhs, 64);
    Bits expected_bits = bits_ops::UDiv(lhs, rhs);

    Z3_ast expected = Z3_mk_int64(ctx, expected_bits.ToInt64().value(),
                                  Z3_mk_bv_sort(ctx, 64));
    Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
    Z3_solver_assert(ctx, solver, objective);
    Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
    EXPECT_EQ(satisfiable, Z3_L_TRUE);
    Z3_solver_dec_ref(ctx, solver);
  }
}

TEST_F(Z3IrTranslatorTest, HandlesSDiv) {
  constexpr std::string_view tmpl = R"(
package p

fn f() -> bits[64] {
  literal.1: bits[64] = literal(value=$0)
  literal.2: bits[64] = literal(value=$1)
  ret sdiv.3: bits[64] = sdiv(literal.1, literal.2)
}
)";

  constexpr int64_t kMinS64 = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMaxS64 = std::numeric_limits<int64_t>::max();
  const std::initializer_list<int64_t> values{
      kMinS64, kMinS64 + 1, -4, -3, -1, 0, 1, 3, 4, kMaxS64 - 1, kMaxS64};

  for (int64_t test_case_lhs : values) {
    for (int64_t test_case_rhs : values) {
      const std::string program =
          absl::Substitute(tmpl, test_case_lhs, test_case_rhs);
      XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                               Parser::ParsePackage(program));
      XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
      XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                               IrTranslator::CreateAndTranslate(f));
      Z3_context ctx = translator->ctx();
      Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
      Bits lhs = SBits(test_case_lhs, 64);
      Bits rhs = SBits(test_case_rhs, 64);
      Bits expected_bits = bits_ops::SDiv(lhs, rhs);

      Z3_ast expected = Z3_mk_int64(ctx, expected_bits.ToInt64().value(),
                                    Z3_mk_bv_sort(ctx, 64));
      Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
      Z3_solver_assert(ctx, solver, objective);
      Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
      EXPECT_EQ(satisfiable, Z3_L_TRUE)
          << test_case_lhs << " sdiv " << test_case_rhs << " -> expect "
          << BitsToRawDigits(expected_bits, FormatPreference::kSignedDecimal);
      Z3_solver_dec_ref(ctx, solver);
    }
  }
}

TEST_F(Z3IrTranslatorTest, HandlesSDiv1BitWide) {
  // 1-bit sdivs are a bit of a special case because z3 conversion special-cases
  // rhs=0 to return MIN_INT and MAX_INT.
  constexpr std::string_view tmpl = R"(
package p

fn f() -> bits[1] {
  literal.1: bits[1] = literal(value=$0)
  literal.2: bits[1] = literal(value=$1)
  ret sdiv.3: bits[1] = sdiv(literal.1, literal.2)
}
)";
  std::vector<std::pair<int64_t, int64_t>> test_cases{
      {0, 0},
      {-1, 0},
      {0, -1},
      {-1, -1},
  };

  for (auto [test_case_lhs, test_case_rhs] : test_cases) {
    std::string program = absl::Substitute(tmpl, test_case_lhs, test_case_rhs);
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(program));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
    XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                             IrTranslator::CreateAndTranslate(f));
    Z3_context ctx = translator->ctx();
    Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
    Bits lhs = SBits(test_case_lhs, 1);
    Bits rhs = SBits(test_case_rhs, 1);
    Bits expected_bits = bits_ops::SDiv(lhs, rhs);

    Z3_ast expected = Z3_mk_int64(ctx, expected_bits.ToInt64().value(),
                                  Z3_mk_bv_sort(ctx, 1));
    Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
    Z3_solver_assert(ctx, solver, objective);
    Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
    EXPECT_EQ(satisfiable, Z3_L_TRUE);
    Z3_solver_dec_ref(ctx, solver);
  }
}

TEST_F(Z3IrTranslatorTest, HandlesUMod) {
  constexpr std::string_view tmpl = R"(
package p

fn f() -> bits[64] {
  literal.1: bits[64] = literal(value=$0)
  literal.2: bits[64] = literal(value=$1)
  ret umod.3: bits[64] = umod(literal.1, literal.2)
}
)";

  constexpr uint64_t kMaxU64 = std::numeric_limits<uint64_t>::max();
  const std::initializer_list<uint64_t> values{0, 1, 3, kMaxU64 - 1, kMaxU64};
  for (uint64_t test_case_lhs : values) {
    for (uint64_t test_case_rhs : values) {
      const std::string program =
          absl::Substitute(tmpl, test_case_lhs, test_case_rhs);
      XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                               Parser::ParsePackage(program));
      XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
      XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                               IrTranslator::CreateAndTranslate(f));
      Z3_context ctx = translator->ctx();
      Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);

      // Result as coming from XLS bit ops library
      const Bits lhs = UBits(test_case_lhs, 64);
      const Bits rhs = UBits(test_case_rhs, 64);
      const Bits expected_bits = bits_ops::UMod(lhs, rhs);
      Z3_ast expected = Z3_mk_int64(ctx, expected_bits.ToInt64().value(),
                                    Z3_mk_bv_sort(ctx, 64));

      // ... compare with Z3 result
      Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
      Z3_solver_assert(ctx, solver, objective);
      Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
      EXPECT_EQ(satisfiable, Z3_L_TRUE)
          << test_case_lhs << " umod " << test_case_rhs << " -> expect "
          << BitsToRawDigits(expected_bits, FormatPreference::kUnsignedDecimal);
      Z3_solver_dec_ref(ctx, solver);
    }
  }
}

TEST_F(Z3IrTranslatorTest, HandlesSMod) {
  constexpr std::string_view tmpl = R"(
package p

fn f() -> bits[64] {
  literal.1: bits[64] = literal(value=$0)
  literal.2: bits[64] = literal(value=$1)
  ret smod.3: bits[64] = smod(literal.1, literal.2)
}
)";

  constexpr int64_t kMinS64 = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMaxS64 = std::numeric_limits<int64_t>::max();
  const std::initializer_list<int64_t> values{
      kMinS64, kMinS64 + 1, -4, -3, -1, 0, 1, 3, 4, kMaxS64 - 1, kMaxS64};

  for (int64_t test_case_lhs : values) {
    for (int64_t test_case_rhs : values) {
      const std::string program =
          absl::Substitute(tmpl, test_case_lhs, test_case_rhs);
      XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                               Parser::ParsePackage(program));
      XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("f"));
      XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                               IrTranslator::CreateAndTranslate(f));
      Z3_context ctx = translator->ctx();
      Z3_solver solver = solvers::z3::CreateSolver(ctx, /*num_threads=*/1);

      // Result as coming from XLS bit ops library
      const Bits lhs = SBits(test_case_lhs, 64);
      const Bits rhs = SBits(test_case_rhs, 64);
      const Bits expected_bits = bits_ops::SMod(lhs, rhs);
      Z3_ast expected = Z3_mk_int64(ctx, expected_bits.ToInt64().value(),
                                    Z3_mk_bv_sort(ctx, 64));

      // ... compare with Z3 result
      Z3_ast objective = Z3_mk_eq(ctx, translator->GetReturnNode(), expected);
      Z3_solver_assert(ctx, solver, objective);
      Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
      EXPECT_EQ(satisfiable, Z3_L_TRUE)
          << test_case_lhs << " smod " << test_case_rhs << " -> expect "
          << BitsToRawDigits(expected_bits, FormatPreference::kSignedDecimal);
      Z3_solver_dec_ref(ctx, solver);
    }
  }
}

TEST_F(Z3IrTranslatorTest, HandlesTupleEqAndNe) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  Type* u32 = package->GetBitsType(32);
  Type* small_tuple_tpe = package->GetTupleType({u32, u32});
  Type* tuple_tpe =
      package->GetTupleType({u32, small_tuple_tpe, small_tuple_tpe});
  BValue a = fb.Param("a", tuple_tpe);
  BValue b = fb.Param("b", tuple_tpe);
  BValue should_be_false = fb.And(fb.Eq(a, b), fb.Ne(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(should_be_false));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, should_be_false.node(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, HandlesArrayEqAndNe) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  Type* u32 = package->GetBitsType(32);
  Type* small_array_tpe = package->GetArrayType(10, u32);
  Type* array_tpe = package->GetArrayType(10, small_array_tpe);
  BValue a = fb.Param("a", array_tpe);
  BValue b = fb.Param("b", array_tpe);
  BValue should_be_false = fb.And(fb.Eq(a, b), fb.Ne(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(should_be_false));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, should_be_false.node(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

TEST_F(Z3IrTranslatorTest, HandlesComplexAggregateEqAndNe) {
  std::unique_ptr<Package> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  Type* u32 = package->GetBitsType(32);
  Type* small_array_tpe = package->GetArrayType(10, u32);
  Type* small_tuple_tpe = package->GetTupleType({u32, small_array_tpe, u32});
  Type* array_tpe = package->GetArrayType(10, small_tuple_tpe);
  Type* tuple_tpe =
      package->GetTupleType({u32, array_tpe, u32, small_tuple_tpe});
  BValue a = fb.Param("a", tuple_tpe);
  BValue b = fb.Param("b", tuple_tpe);
  BValue should_be_false = fb.And(fb.Eq(a, b), fb.Ne(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(should_be_false));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult proven,
      TryProve(f, should_be_false.node(), Predicate::EqualToZero(),
               absl::InfiniteDuration()));
  EXPECT_THAT(proven, IsProvenTrue());
}

}  // namespace
}  // namespace xls

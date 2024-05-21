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

#include "xls/passes/boolean_simplification_pass.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class BooleanSimplificationPassTest : public IrTestBase {
 protected:
  BooleanSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         BooleanSimplificationPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
            .status());
    // Return whether boolean simplification changed anything.
    return changed;
  }
};

TEST_F(BooleanSimplificationPassTest, TruthTableTestSingleVariableNegated) {
  internal::TruthTable not_x(UBits(0b100, /*bit_count=*/3),
                             UBits(0b100, /*bit_count=*/3), std::nullopt);
  EXPECT_EQ(not_x.ComputeTruthTable(),
            bits_ops::Not(not_x.GetInitialVector(0)));
  EXPECT_TRUE(not_x.MatchesVector(not_x.ComputeTruthTable()));
}

TEST_F(BooleanSimplificationPassTest, TruthTableTestAndOfNxAndNz) {
  internal::TruthTable table(UBits(0b101, /*bit_count=*/3),
                             UBits(0b101, /*bit_count=*/3), Op::kAnd);
  EXPECT_EQ(table.ComputeTruthTable(),
            bits_ops::And(bits_ops::Not(table.GetInitialVector(0)),
                          bits_ops::Not(table.GetInitialVector(2))));
}

TEST_F(BooleanSimplificationPassTest, TruthTableMatchesAndOfNxAndNy) {
  internal::TruthTable table(UBits(0b110, /*bit_count=*/3),
                             UBits(0b110, /*bit_count=*/3), Op::kAnd);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  nx: bits[42] = not(x)
  ny: bits[42] = not(y)
  ret and: bits[42] = and(nx, ny)
}
  )",
                                                       p.get()));
  EXPECT_TRUE(
      table.MatchesSymmetrical(f->return_value(), {f->param(0), f->param(1)}));
  EXPECT_TRUE(
      table.MatchesSymmetrical(f->return_value(), {f->param(1), f->param(0)}));
}

TEST_F(BooleanSimplificationPassTest, TruthTableMatchesNotX) {
  internal::TruthTable table(UBits(0b100, /*bit_count=*/3),
                             UBits(0b100, /*bit_count=*/3), std::nullopt);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  ret nx: bits[42] = not(x)
}
  )",
                                                       p.get()));
  EXPECT_TRUE(table.MatchesSymmetrical(f->return_value(), {f->param(0)}));
}

TEST_F(BooleanSimplificationPassTest, DoubleAnd) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  and.3: bits[42] = and(x, y)
  ret and.4: bits[42] = and(and.3, y)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y")));
}

TEST_F(BooleanSimplificationPassTest, NotAndOr) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  not.3: bits[42] = not(x)
  and.4: bits[42] = and(not.3, y)
  ret or.5: bits[42] = or(and.4, x)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::Param("x"), m::Param("y")));
}

TEST_F(BooleanSimplificationPassTest, TwoVarsMakingAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[7], y: bits[7]) -> bits[7] {
  nor: bits[7] = nor(x, y)
  xn: bits[7] = not(x)
  x_or_y: bits[7] = or(x, y)
  and: bits[7] = and(xn, x_or_y)
  x_nor_y: bits[7] = nor(x, y)
  ret or: bits[7] = or(x, x_nor_y, and)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(127));
}

TEST_F(BooleanSimplificationPassTest, ConvertToNand) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  and: bits[42] = and(x, y)
  ret not: bits[42] = not(and)
})",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Nand(m::Param("x"), m::Param("y")));
}

TEST_F(BooleanSimplificationPassTest, ThreeVarsZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42], z: bits[42]) -> bits[42] {
  not_z: bits[42] = not(z)
  x_or_y: bits[42] = or(x, y)
  not__x_or_y: bits[42] = not(x_or_y)
  ret and: bits[42] = and(y, not_z, not__x_or_y)
})",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(BooleanSimplificationPassTest, ThreeVarsNor) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42], z: bits[42]) -> bits[42] {
  xn: bits[42] = not(x)
  yn: bits[42] = not(y)
  zn: bits[42] = not(z)
  y_or_z: bits[42] = or(y, z)
  not_y_or_z: bits[42] = not(y_or_z)
  ret final_and: bits[42] = and(xn, yn, not_y_or_z)
})",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Nor(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(BooleanSimplificationPassTest, ConvertToNor) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  or: bits[42] = or(x, y)
  ret not: bits[42] = not(or)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Nor(m::Param("x"), m::Param("y")));
}

TEST_F(BooleanSimplificationPassTest, SimplifyToXOrNotZEquivalent) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[3], y: bits[3], z: bits[3]) -> bits[3] {
  nor: bits[3] = nor(x, y, z)
  zn: bits[3] = not(z)
  ret or: bits[3] = or(x, nor, zn)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Nand(m::Not(m::Param("x")), m::Param("z")));
}

TEST_F(BooleanSimplificationPassTest, SimplifyToNotYAndZEquivalent) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[3], y: bits[3], z: bits[3]) -> bits[3] {
  y_or_z: bits[3] = or(y, z)
  yn: bits[3] = not(y)
  ret and: bits[3] = and(yn, y_or_z)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Nor(m::Param("y"), m::Not(m::Param("z"))));
}

TEST_F(BooleanSimplificationPassTest, SimplifyRealWorld) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[3], y: bits[3], z: bits[3]) -> bits[3] {
  zn: bits[3] = not(z)
  zn_or_y: bits[3] = or(zn, y)
  nor_xyz: bits[3] = nor(x, y, z)
  ret or: bits[3] = or(zn_or_y, nor_xyz)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Nand(m::Param("z"), m::Not(m::Param("y"))));
}

TEST_F(BooleanSimplificationPassTest, ManyPaths) {
  // Verify that the pass runs quickly even if there are exponentially many
  // paths to consider.
  auto p = CreatePackage();
  Type* u1 = p->GetBitsType(1);

  FunctionBuilder b(TestName(), p.get());
  BValue x = b.Param("x", u1);
  BValue y = b.Param("y", u1);
  BValue tmp = b.And(x, y);
  for (int64_t i = 0; i < 100; ++i) {
    tmp = b.And(tmp, tmp);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y")));
}

// TODO(leary): 2019-10-11 Needs AOI21 logical function to map against.
#if 0
TEST_F(BooleanSimplificationPassTest, SimplifyRealWorld2) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[3], y: bits[3], z: bits[3]) -> bits[3] {
  yn: bits[3] = not(y)
  yn_nor_x: bits[3] = nor(yn, x)
  nor_xyz: bits[3] = nor(x, y, z)
  ret or: bits[3] = or(yn_nor_x, nor_xyz)
}
  )",
                                                   p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->DumpIr(),
            R"(fn f(x: bits[3], y: bits[3], z: bits[3]) -> bits[3] {
}
)");
}
#endif

}  // namespace
}  // namespace xls

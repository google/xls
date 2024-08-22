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

#include "xls/passes/optimization_pass_pipeline.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class OptimizationPipelineTest : public IrTestBase {
 protected:
  OptimizationPipelineTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    return RunOptimizationPassPipeline(p);
  }

  void TestAssociativeWithConstants(std::string_view xls_op, Op op,
                                    int64_t value) {
    auto p = CreatePackage();
    std::string xls_func = absl::StrFormat(R"(
     fn simple_assoc(x:bits[8]) -> bits[8] {
        lit1: bits[8] = literal(value=7)
        lit2: bits[8] = literal(value=12)
        %s1: bits[8] = %s(lit1, x)
        ret res: bits[8] = %s(%s1, lit2)
     }
  )",
                                           xls_op, xls_op, xls_op, xls_op);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(xls_func, p.get()));
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_EQ(f->return_value()->op(), op);
    EXPECT_EQ(f->return_value()->operand(1)->op(), Op::kLiteral);
    EXPECT_EQ(f->return_value()->operand(1)->As<Literal>()->value().bits(),
              UBits(value, 8));
  }

  void TestAddSubWithConstants(std::string_view xls_op1, Op op1,
                               std::string_view xls_op2, int8_t value) {
    auto p = CreatePackage();
    std::string xls_func = absl::StrFormat(R"(
     fn addsub(x:bits[8]) -> bits[8] {
        lit1: bits[8] = literal(value=7)
        lit2: bits[8] = literal(value=12)
        %s1: bits[8] = %s(x, lit1)
        ret res: bits[8] = %s(%s1, lit2)
     }
  )",
                                           xls_op1, xls_op1, xls_op2, xls_op1);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(xls_func, p.get()));
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_EQ(f->return_value()->op(), op1);
    EXPECT_EQ(f->return_value()->operand(1)->op(), Op::kLiteral);
    EXPECT_EQ(f->return_value()->operand(1)->As<Literal>()->value().bits(),
              SBits(value, 8));
  }
};

TEST_F(OptimizationPipelineTest, IdentityRemoval) {
  auto p = CreatePackage();
  std::unique_ptr<OptimizationCompoundPass> pass_mgr =
      CreateOptimizationPassPipeline();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_neg(x:bits[8]) -> bits[8] {
        one: bits[8] = literal(value=1)
        v1:  bits[8] = identity(x)
        add: bits[8] = add(v1, one)
        v2:  bits[8] = identity(add)
        v3:  bits[8] = identity(v2)
        v4:  bits[8] = identity(v3)
        v5:  bits[8] = identity(v4)
        v6:  bits[8] = identity(v5)
        v7:  bits[8] = identity(v6)
        v8:  bits[8] = identity(v7)
        v9:  bits[8] = identity(v8)
        ret add2:bits[8] = sub(v9, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
  EXPECT_EQ(f->node_count(), 1);
}

// Rrot by fixed amount.
// The goal of this test is to generate 'bitextract patterns' which
// allow to translate this directly into Verilog wire-concat constructs.
TEST_F(OptimizationPipelineTest, RrotAndBitExtract) {
  std::pair<std::unique_ptr<Package>, Function*> rrot_fixed =
      sample_packages::BuildRrot8Fixed();
  Package* p = rrot_fixed.first.get();
  ASSERT_THAT(Run(p), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetTopAsFunction());
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/3),
                        m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/5)));
}

// Double nested loop. The goal is to prepare for control generation
// using shared FSMs.
TEST_F(OptimizationPipelineTest, TwoLoops) {
  std::pair<std::unique_ptr<Package>, Function*> two_loops =
      sample_packages::BuildTwoLoops(
          /*same_trip_count=*/true, /*dependent_loops=*/true);
  Package* p = two_loops.first.get();
  ASSERT_THAT(Run(p), IsOkAndHolds(true));
}

TEST_F(OptimizationPipelineTest, SubOfLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_neg(x:bits[2]) -> bits[2] {
        one:bits[2] = literal(value=1)
        ret subval: bits[2] = sub(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Param(), m::Literal(SBits(-1, 2))));
}

TEST_F(OptimizationPipelineTest, Canonicalize) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_canon(x:bits[2]) -> bits[2] {
        one:bits[2] = literal(value=1)
        ret addval: bits[2] = add(one, x)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Param(), m::Literal(1)));
}

TEST_F(OptimizationPipelineTest, DoubleNeg) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_neg(x:bits[2]) -> bits[2] {
        neg1:bits[2] = neg(x)
        ret neg2: bits[2] = neg(neg1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param());
}

TEST_F(OptimizationPipelineTest, AssociateAdd) {
  TestAssociativeWithConstants("add", Op::kAdd, 7 + 12);
}

TEST_F(OptimizationPipelineTest, AssociateXor) {
  TestAssociativeWithConstants("xor", Op::kXor, 7 ^ 12);
}

TEST_F(OptimizationPipelineTest, SubSubTest) {
  TestAddSubWithConstants("sub", Op::kAdd, "sub", -7 - 12);
}

TEST_F(OptimizationPipelineTest, SubAddTest) {
  TestAddSubWithConstants("sub", Op::kAdd, "add", 12 - 7);
}

TEST_F(OptimizationPipelineTest, AddSubTest) {
  TestAddSubWithConstants("add", Op::kAdd, "sub", 7 - 12);
}

TEST_F(OptimizationPipelineTest, IdentityRemovalFromParam) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_param(x:bits[8]) -> bits[8] {
        ret res: bits[8] = identity(x)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param());
}

TEST_F(OptimizationPipelineTest, CompareBoolAgainstZero) {
  // This requires canonicalization to get the constant on the RHS of the eq,
  // then simplification to replace with NOT(x).
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=0)
        ret eq.2: bits[1] = eq(literal.1,x )
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<CompareOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param()));
}

TEST_F(OptimizationPipelineTest, MultiplyBy16StrengthReduction) {
  // This requires canonicalization to get the constant on the RHS of the eq,
  // then strength reduction to replace the mul with a shift left. The shift
  // left is then replaced with a concat.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42]) -> bits[42] {
       literal.1: bits[42] = literal(value=16)
       ret umul.2: bits[42] = umul(literal.1, x)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kUMul);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat());
}

TEST_F(OptimizationPipelineTest, LogicAbsorption) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
   // x & (x | y) -> x
   fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
      or.3: bits[32] = or(x, y, pos=[(0,1,10)])
      ret and.4: bits[32] = and(x, or.3, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(OptimizationPipelineTest, LogicCombining) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
   // (x & y) | (x | ~y) -> x
  fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
    not.4: bits[32] = not(y, pos=[(0,1,18)])
    and.3: bits[32] = and(x, y, pos=[(0,1,6)])
    and.5: bits[32] = and(x, not.4, pos=[(0,1,16)])
    ret or.6: bits[32] = or(and.3, and.5, pos=[(0,1,11)])
  }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

}  // namespace
}  // namespace xls

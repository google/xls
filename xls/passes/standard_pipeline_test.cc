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

#include "xls/passes/standard_pipeline.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/passes/arith_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dump_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class StandardPipelineTest : public IrTestBase {
 protected:
  StandardPipelineTest() = default;

  xabsl::StatusOr<bool> Run(Package* p) { return RunStandardPassPipeline(p); }

  void TestAssociativeWithConstants(absl::string_view xls_op, Op op,
                                    int64 value) {
    auto p = CreatePackage();
    std::string xls_func = absl::StrFormat(R"(
     fn simple_assoc(x:bits[8]) -> bits[8] {
        lit1: bits[8] = literal(value=7)
        lit2: bits[8] = literal(value=12)
        %s1: bits[8] = %s(lit1, x)
        res: bits[8] = %s(%s1, lit2)
        ret res
     }
  )",
                                           xls_op, xls_op, xls_op, xls_op);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(xls_func, p.get()));
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_EQ(f->return_value()->op(), op);
    EXPECT_EQ(f->return_value()->operand(1)->op(), OP_LITERAL);
    EXPECT_EQ(f->return_value()->operand(1)->As<Literal>()->value().bits(),
              UBits(value, 8));
  }

  void TestAddSubWithConstants(absl::string_view xls_op1, Op op1,
                               absl::string_view xls_op2, int8 value) {
    auto p = CreatePackage();
    std::string xls_func = absl::StrFormat(R"(
     fn addsub(x:bits[8]) -> bits[8] {
        lit1: bits[8] = literal(value=7)
        lit2: bits[8] = literal(value=12)
        %s1: bits[8] = %s(x, lit1)
        res: bits[8] = %s(%s1, lit2)
        ret res
     }
  )",
                                           xls_op1, xls_op1, xls_op2, xls_op1);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(xls_func, p.get()));
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_EQ(f->return_value()->op(), op1);
    EXPECT_EQ(f->return_value()->operand(1)->op(), OP_LITERAL);
    EXPECT_EQ(f->return_value()->operand(1)->As<Literal>()->value().bits(),
              SBits(value, 8));
  }
};

TEST_F(StandardPipelineTest, IdentityRemoval) {
  auto p = CreatePackage();
  std::unique_ptr<CompoundPass> pass_mgr = CreateStandardPassPipeline();
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
        add2:bits[8] = sub(v9, one)
        ret add2
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
TEST_F(StandardPipelineTest, RrotAndBitExtract) {
  std::pair<std::unique_ptr<Package>, Function*> rrot_fixed =
      sample_packages::BuildRrot8Fixed();
  Package* p = rrot_fixed.first.get();
  ASSERT_THAT(Run(p), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->EntryFunction());
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/3),
                        m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/5)));
}

// Double nested loop. The goal is to prepare for control generation
// using shared FSMs.
TEST_F(StandardPipelineTest, TwoLoops) {
  std::pair<std::unique_ptr<Package>, Function*> two_loops =
      sample_packages::BuildTwoLoops(
          /*same_trip_count=*/true, /*dependent_loops=*/true);
  Package* p = two_loops.first.get();
  ASSERT_THAT(Run(p), IsOkAndHolds(true));
}

TEST_F(StandardPipelineTest, SubOfLiteral) {
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

TEST_F(StandardPipelineTest, Canonicalize) {
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

TEST_F(StandardPipelineTest, DoubleNeg) {
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

TEST_F(StandardPipelineTest, AssociateAdd) {
  TestAssociativeWithConstants("add", OP_ADD, 7 + 12);
}

TEST_F(StandardPipelineTest, AssociateXor) {
  TestAssociativeWithConstants("xor", OP_XOR, 7 ^ 12);
}

TEST_F(StandardPipelineTest, SubSubTest) {
  TestAddSubWithConstants("sub", OP_ADD, "sub", -7 - 12);
}

TEST_F(StandardPipelineTest, SubAddTest) {
  TestAddSubWithConstants("sub", OP_ADD, "add", 12 - 7);
}

TEST_F(StandardPipelineTest, AddSubTest) {
  TestAddSubWithConstants("add", OP_ADD, "sub", 7 - 12);
}

TEST_F(StandardPipelineTest, IdentityRemovalFromParam) {
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

TEST_F(StandardPipelineTest, CompareBoolAgainstZero) {
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

TEST_F(StandardPipelineTest, MultiplyBy16StrengthReduction) {
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
  EXPECT_EQ(f->return_value()->op(), OP_UMUL);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat());
}

TEST_F(StandardPipelineTest, LogicAbsorption) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
   // x & (x | y) -> x
   fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
      or.3: bits[32] = or(x, y, pos=0,1,10)
      ret and.4: bits[32] = and(x, or.3, pos=0,1,5)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(StandardPipelineTest, LogicCombining) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
   // (x & y) | (x | ~y) -> x
  fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
    not.4: bits[32] = not(y, pos=0,1,18)
    and.3: bits[32] = and(x, y, pos=0,1,6)
    and.5: bits[32] = and(x, not.4, pos=0,1,16)
    ret or.6: bits[32] = or(and.3, and.5, pos=0,1,11)
  }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}


}  // namespace
}  // namespace xls

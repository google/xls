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

#include "xls/passes/arith_simplification_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ArithSimplificationPassTest : public IrTestBase {
 protected:
  ArithSimplificationPassTest() = default;

  xabsl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return ArithSimplificationPass().Run(p, PassOptions(), &results);
  }
};

TEST_F(ArithSimplificationPassTest, Arith1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
   fn double_shift(x:bits[32]) -> bits[32] {
     three:bits[32] = literal(value=3)
     two:bits[32] = literal(value=2)
     xshrl3:bits[32] = shrl(x, three)
     xshrl3_shrl2:bits[32] = shrl(xshrl3, two)
     ret add: bits[32] = add(xshrl3_shrl2, xshrl3_shrl2)
   }
   )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Concat(m::Literal(0), m::BitSlice()),
                     m::Concat(m::Literal(0), m::BitSlice())));
}

TEST_F(ArithSimplificationPassTest, DoubleNeg) {
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

TEST_F(ArithSimplificationPassTest, MulBy0) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        zero:bits[8] = literal(value=0)
        ret ret_mul: bits[8] = umul(x, zero)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, MulBy1SignExtendedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[16] {
        one: bits[8] = literal(value=1)
        ret ret_mul: bits[16] = smul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::SignExt(m::Param("x")));
}

TEST_F(ArithSimplificationPassTest, MulBy1ZeroExtendedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[16] {
        one: bits[8] = literal(value=1)
        ret ret_mul: bits[16] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Param("x")));
}

TEST_F(ArithSimplificationPassTest, MulBy1NarrowedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[3] {
        one: bits[8] = literal(value=1)
        ret ret_mul: bits[3] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/3));
}

TEST_F(ArithSimplificationPassTest, SMulByMinusOne) {
  // A single-bit value of 1 is a -1 when interpreted as a signed number. The
  // Mul-by-1 optimization should not kick in in this case.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[3] {
        one: bits[1] = literal(value=1)
        ret ret_mul: bits[3] = smul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::SMul());
}

TEST_F(ArithSimplificationPassTest, UDivBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn div_one(x:bits[8]) -> bits[8] {
        one:bits[8] = literal(value=1)
        ret ret_mul: bits[8] = udiv(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param());
}

TEST_F(ArithSimplificationPassTest, MulBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        one:bits[8] = literal(value=1)
        ret ret_mul: bits[8] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param());
}

TEST_F(ArithSimplificationPassTest, CanonicalizeXorAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[2]) -> bits[2] {
        literal.1: bits[2] = literal(value=3)
        ret xor.2: bits[2] = xor(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param()));
}

TEST_F(ArithSimplificationPassTest, OverlargeShiftAfterSimp) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[2]) -> bits[2] {
        literal.1: bits[2] = literal(value=2)
        shrl.2: bits[2] = shrl(x, literal.1)
        ret shrl.3: bits[2] = shrl(shrl.2, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, ShiftRightArithmeticByLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=13)
        ret shra.3: bits[16] = shra(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::SignExt(), m::BitSlice()));
}

TEST_F(ArithSimplificationPassTest, OverlargeShiftRightArithmeticByLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=1234)
        ret shra.3: bits[16] = shra(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(m::BitSlice(/*start=*/15, /*width=*/1)));
}

TEST_F(ArithSimplificationPassTest, CompareBoolAgainstOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=1)
        ret eq.2: bits[1] = eq(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<CompareOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_TRUE(f->return_value()->Is<Param>());
}

TEST_F(ArithSimplificationPassTest, CompareBoolAgainstZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=0)
        ret eq.2: bits[1] = eq(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<CompareOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param()));
}

TEST_F(ArithSimplificationPassTest, DoubleNegation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42]) -> bits[42] {
        not.2: bits[42] = not(x)
        ret not.3: bits[42] = not(not.2)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<UnOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, NaryOrEliminateSeveralZeros) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8]) -> bits[8] {
        literal.3: bits[8] = literal(value=0)
        literal.4: bits[8] = literal(value=0)
        ret or.5: bits[8] = or(x, literal.3, y, literal.4)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::Param("x"), m::Param("y")));
}

TEST_F(ArithSimplificationPassTest, NaryAndEliminateSeveralOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8]) -> bits[8] {
        literal.3: bits[8] = literal(value=0xff)
        literal.4: bits[8] = literal(value=0xff)
        ret and.5: bits[8] = and(x, literal.3, y, literal.4)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y")));
}

TEST_F(ArithSimplificationPassTest, NaryAndEliminateAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[8] {
        literal.1: bits[8] = literal(value=0xff)
        literal.2: bits[8] = literal(value=0xff)
        ret and.3: bits[8] = and(literal.1, literal.2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(255));
}

TEST_F(ArithSimplificationPassTest, NaryFlattening) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
        literal.3: bits[8] = literal(value=0x1f)
        literal.4: bits[8] = literal(value=0x0f)
        and.5: bits[8] = and(z, literal.3)
        ret and.6: bits[8] = and(x, y, literal.4, and.5)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"), m::Param("y"), m::Literal(15),
                     m::Param("z"), m::Literal(31)));
}

TEST_F(ArithSimplificationPassTest, NaryLiteralConsolidation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
  literal.4: bits[8] = literal(value=15)
  literal.5: bits[8] = literal(value=31)
  ret and.6: bits[8] = and(x, y, literal.4, z, literal.5)
}
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y"),
                                        m::Param("z"), m::Literal(15)));
}

TEST_F(ArithSimplificationPassTest, XAndNotX) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[32]) -> bits[32] {
        not.2: bits[32] = not(x)
        ret and.3: bits[32] = and(x, not.2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, NotXAndX) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[32]) -> bits[32] {
        not.2: bits[32] = not(x)
        ret and.3: bits[32] = and(not.2, x)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, SignExtTwice) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8]) -> bits[32] {
        sign_ext.2: bits[24] = sign_ext(x, new_bit_count=24)
        ret sign_ext.3: bits[32] = sign_ext(sign_ext.2, new_bit_count=32)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::SignExt(m::Param()));
}

TEST_F(ArithSimplificationPassTest, CollapseToNaryOr) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(w: bits[8], x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    or.5: bits[8] = or(w, x)
    or.6: bits[8] = or(y, z)
    ret or.7: bits[8] = or(or.5, or.6)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::Param("w"), m::Param("x"),
                                       m::Param("y"), m::Param("z")));
}

TEST_F(ArithSimplificationPassTest, CollapseOneSideToNaryOr) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    or.4: bits[8] = or(y, z)
    ret or.5: bits[8] = or(x, or.4)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(ArithSimplificationPassTest, NorWithLiteralZeroOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    literal.1: bits[8] = literal(value=0)
    literal.2: bits[8] = literal(value=0)
    ret nor.3: bits[8] = nor(x, literal.1, y, literal.2, z)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Nor(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(ArithSimplificationPassTest, NandWithLiteralAllOnesOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    literal.1: bits[8] = literal(value=255)
    ret nand.2: bits[8] = nand(x, literal.1, y, z)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Nand(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(ArithSimplificationPassTest, SingleOperandNand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddNaryOp(OP_NAND, {fb.Param("x", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param("x")));
}

TEST_F(ArithSimplificationPassTest, SingleOperandOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddNaryOp(OP_OR, {fb.Param("x", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, RedundantAnd) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
       ret and.3: bits[32] = and(x, x, pos=0,1,5)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, RedundantOr) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[32], y: bits[32]) -> bits[32] {
       ret and.3: bits[32] = or(x, x, pos=0,1,5)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, AddWithZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn add_zero(x: bits[32]) -> bits[32] {
      literal.1: bits[32] = literal(value=0)
      ret slt.2: bits[32] = add(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, ZeroWidthMulOperand) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[0], y: bits[32]) -> bits[32] {
       ret smul.1: bits[32] = smul(x, y, pos=0,1,5)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 32)));
}

TEST_F(ArithSimplificationPassTest, SltZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn slt_zero(x: bits[32]) -> bits[1] {
      literal.1: bits[32] = literal(value=0)
      ret slt.2: bits[1] = slt(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::BitSlice(m::Param("x"), /*start=*/31, /*width=*/1));
}

TEST_F(ArithSimplificationPassTest, SGeZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn sge_zero(x: bits[32]) -> bits[1] {
      literal.1: bits[32] = literal(value=0)
      ret sge.2: bits[1] = sge(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::BitSlice(m::Param("x"), /*start=*/31, /*width=*/1)));
}

TEST_F(ArithSimplificationPassTest, InvertedComparison) {
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    Type* u32 = p->GetBitsType(32);
    fb.Not(fb.ULt(fb.Param("x", u32), fb.Param("y", u32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::UGe(m::Param("x"), m::Param("y")));
  }
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    Type* u32 = p->GetBitsType(32);
    fb.Not(fb.SGe(fb.Param("x", u32), fb.Param("y", u32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::SLt(m::Param("x"), m::Param("y")));
  }
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    Type* u32 = p->GetBitsType(32);
    fb.Not(fb.Eq(fb.Param("x", u32), fb.Param("y", u32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Ne(m::Param("x"), m::Param("y")));
  }
}

TEST_F(ArithSimplificationPassTest, ULtMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b0011)
      ret ult.2: bits[1] = ult(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Nor(m::Or(m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/1),
                   m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1)),
             m::And(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/1),
                    m::BitSlice(m::Param("x"), /*start=*/1, /*width=*/1))));
}

TEST_F(ArithSimplificationPassTest, UGtMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b0011)
      ret res: bits[1] = ugt(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/1),
                    m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1)));
}

TEST_F(ArithSimplificationPassTest, UGtMaskAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b1111)
      ret res: bits[1] = ugt(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

}  // namespace
}  // namespace xls

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

#include "xls/passes/basic_simplification_pass.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::xls::solvers::z3::ScopedVerifyEquivalence;

class BasicSimplificationPassTest : public IrTestBase {
 protected:
  BasicSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return BasicSimplificationPass().Run(p, OptimizationPassOptions(),
                                         &results);
  }
};

TEST_F(BasicSimplificationPassTest, DoubleNeg) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_neg(x:bits[2]) -> bits[2] {
        neg1:bits[2] = neg(x)
        ret result: bits[2] = neg(neg1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param());
}

TEST_F(BasicSimplificationPassTest, SubFallsToZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto param = fb.Param("x", p->GetBitsType(32));
  fb.Subtract(param, param);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(BasicSimplificationPassTest, MulBy0) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        zero:bits[8] = literal(value=0)
        ret result: bits[8] = umul(x, zero)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(BasicSimplificationPassTest, CanonicalizeXorAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[2]) -> bits[2] {
        literal.1: bits[2] = literal(value=3)
        ret result: bits[2] = xor(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param()));
}

TEST_F(BasicSimplificationPassTest, CompareBoolAgainstOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=1)
        ret result: bits[1] = eq(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<CompareOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_TRUE(f->return_value()->Is<Param>());
}

TEST_F(BasicSimplificationPassTest, CompareBoolAgainstZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=0)
        ret result: bits[1] = eq(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<CompareOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param()));
}

TEST_F(BasicSimplificationPassTest, DoubleNot) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42]) -> bits[42] {
        not.2: bits[42] = not(x)
        ret result: bits[42] = not(not.2)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<UnOp>());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, NaryOrEliminateSeveralZeros) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8]) -> bits[8] {
        literal.3: bits[8] = literal(value=0)
        literal.4: bits[8] = literal(value=0)
        ret result: bits[8] = or(x, literal.3, y, literal.4)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::Param("x"), m::Param("y")));
}

TEST_F(BasicSimplificationPassTest, NaryAndEliminateSeveralOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8]) -> bits[8] {
        literal.3: bits[8] = literal(value=0xff)
        literal.4: bits[8] = literal(value=0xff)
        ret result: bits[8] = and(x, literal.3, y, literal.4)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y")));
}

TEST_F(BasicSimplificationPassTest, NaryAndEliminateAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[8] {
        literal.1: bits[8] = literal(value=0xff)
        literal.2: bits[8] = literal(value=0xff)
        ret result: bits[8] = and(literal.1, literal.2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(255));
}

TEST_F(BasicSimplificationPassTest, NaryFlattening) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
        literal.3: bits[8] = literal(value=0x1f)
        literal.4: bits[8] = literal(value=0x0f)
        and.5: bits[8] = and(z, literal.3)
        ret result: bits[8] = and(x, y, literal.4, and.5)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y"),
                                        m::Param("z"), m::Literal(15)));
}

TEST_F(BasicSimplificationPassTest, NaryLiteralConsolidation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
  literal.4: bits[8] = literal(value=15)
  literal.5: bits[8] = literal(value=31)
  ret result: bits[8] = and(x, y, literal.4, z, literal.5)
}
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y"),
                                        m::Param("z"), m::Literal(15)));
}

TEST_F(BasicSimplificationPassTest, XAndNotX) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[32]) -> bits[32] {
        not.2: bits[32] = not(x)
        ret result: bits[32] = and(x, not.2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(BasicSimplificationPassTest, NotXAndX) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[32]) -> bits[32] {
        not.2: bits[32] = not(x)
        ret result: bits[32] = and(not.2, x)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(BasicSimplificationPassTest, CollapseToNaryOr) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(w: bits[8], x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    or.5: bits[8] = or(w, x)
    or.6: bits[8] = or(y, z)
    ret result: bits[8] = or(or.5, or.6)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::Param("w"), m::Param("x"),
                                       m::Param("y"), m::Param("z")));
}

TEST_F(BasicSimplificationPassTest, CollapseToNaryOrWithOtherUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(w: bits[8], x: bits[8], y: bits[8], z: bits[8]) -> (bits[8], bits[8]) {
    w_or_x: bits[8] = or(w, x)
    y_or_z: bits[8] = or(y, z)
    tmp0: bits[8] = or(w_or_x, y_or_z)
    ret result: (bits[8], bits[8]) = tuple(w_or_x, tmp0)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // `Or(w, x)` should not be folded because it has more than one use.
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Or(m::Param("w"), m::Param("x")),
                       m::Or(m::Or(), m::Param("y"), m::Param("z"))));
}

TEST_F(BasicSimplificationPassTest, CollapseOneSideToNaryOr) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    or.4: bits[8] = or(y, z)
    ret result: bits[8] = or(x, or.4)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(BasicSimplificationPassTest, NorWithLiteralZeroOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    literal.1: bits[8] = literal(value=0)
    literal.2: bits[8] = literal(value=0)
    ret result: bits[8] = nor(x, literal.1, y, literal.2, z)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Nor(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(BasicSimplificationPassTest, NandWithLiteralAllOnesOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    literal.1: bits[8] = literal(value=255)
    ret result: bits[8] = nand(x, literal.1, y, z)
 }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Nand(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(BasicSimplificationPassTest, SingleOperandNand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddNaryOp(Op::kNand, {fb.Param("x", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param("x")));
}

TEST_F(BasicSimplificationPassTest, SingleOperandOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddNaryOp(Op::kOr, {fb.Param("x", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, SingleBitAndReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddBitwiseReductionOp(Op::kAndReduce, fb.Param("x", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, SingleBitOrReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddBitwiseReductionOp(Op::kOrReduce, fb.Param("x", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, SingleBitXorReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddBitwiseReductionOp(Op::kXorReduce, fb.Param("x", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, EmptyAndReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddBitwiseReductionOp(Op::kAndReduce, fb.Param("x", p->GetBitsType(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(1, 1));
}

TEST_F(BasicSimplificationPassTest, EmptyOrReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddBitwiseReductionOp(Op::kOrReduce, fb.Param("x", p->GetBitsType(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0, 1));
}

TEST_F(BasicSimplificationPassTest, EmptyXorReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.AddBitwiseReductionOp(Op::kXorReduce, fb.Param("x", p->GetBitsType(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0, 1));
}

TEST_F(BasicSimplificationPassTest, AndWithDuplicateOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = and(x, y, y, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("x"), m::Param("y")));
}

TEST_F(BasicSimplificationPassTest, AndWithSameOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_and(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = and(x, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, OrWithSameOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = or(x, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, NandWithDuplicateOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = nand(x, x, y, y, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Nand(m::Param("x"), m::Param("y")));
}

TEST_F(BasicSimplificationPassTest, NandWithSameOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = nand(x, x, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param("x")));
}

TEST_F(BasicSimplificationPassTest, XorWithSameOperandsEven) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = xor(x, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(BasicSimplificationPassTest, XorWithSameOperandsOdd) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[32], y: bits[32]) -> bits[32] {
       ret result: bits[32] = xor(x, x, x, pos=[(0,1,5)])
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, AddWithZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn add_zero(x: bits[32]) -> bits[32] {
      zero: bits[32] = literal(value=0)
      ret result: bits[32] = add(x, zero)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BasicSimplificationPassTest, NeAggregateType) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue x = fb.Param("x", p->GetTupleType({u32, u32}));
  BValue y = fb.Param("y", p->GetTupleType({u32, u32}));
  BValue x_eq_x = fb.Ne(x, x);
  BValue x_eq_y = fb.Ne(x, y);
  fb.Tuple({x_eq_x, x_eq_y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Literal(0), m::Ne(m::Param("x"), m::Param("y"))));
}

TEST_F(BasicSimplificationPassTest, EqNeZeroWidthTypes) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetTupleType({}));
  BValue x_eq_y = fb.Eq(x, y);
  BValue x_ne_y = fb.Ne(x, y);
  fb.Tuple({x_eq_y, x_ne_y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Tuple(m::Literal(1), m::Literal(0)));
}

TEST_F(BasicSimplificationPassTest, OrInverses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue p1 = fb.Param("p1", p->GetBitsType(32));
  BValue p2 = fb.Param("p2", p->GetBitsType(32));
  fb.Tuple({
      fb.Or(p1, fb.Not(p1)),
      fb.Or(fb.Not(p1), p1),
      fb.Or({p1, p2, fb.Not(p1)}),
      fb.Or({p1, p2, fb.Not(p2)}),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::Literal(Bits::AllOnes(32)), m::Literal(Bits::AllOnes(32)),
               m::Literal(Bits::AllOnes(32)), m::Literal(Bits::AllOnes(32))));
}

}  // namespace
}  // namespace xls

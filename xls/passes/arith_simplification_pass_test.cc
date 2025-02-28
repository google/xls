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

#include "xls/passes/arith_simplification_pass.h"

#include <stdint.h>  // NOLINT(modernize-deprecated-headers) needed for UINT64_C

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

constexpr absl::Duration kProverTimeout = absl::Seconds(10);

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::z3::ScopedVerifyEquivalence;

using ::testing::_;
using ::testing::AllOf;

class ArithSimplificationPassTest : public IrTestBase {
 protected:
  ArithSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return ArithSimplificationPass().Run(
        p, OptimizationPassOptions().WithOptLevel(kMaxOptLevel), &results,
        context);
  }

  void CheckUnsignedDivide(int n, int divisor);
  void CheckSignedDividePowerOfTwo(int n, int divisor);
  void CheckSignedDivideNotPowerOfTwo(int n, int divisor);
};

TEST_F(ArithSimplificationPassTest, Arith1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
   fn double_shift(x:bits[32]) -> bits[32] {
     three:bits[32] = literal(value=3)
     two:bits[32] = literal(value=2)
     xshrl3:bits[32] = shrl(x, three)
     xshrl3_shrl2:bits[32] = shrl(xshrl3, two)
     ret result: bits[32] = add(xshrl3_shrl2, xshrl3_shrl2)
   }
   )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Concat(m::Literal(0), m::BitSlice()),
                     m::Concat(m::Literal(0), m::BitSlice())));
}

TEST_F(ArithSimplificationPassTest, CompareEqNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        ret result: bits[1] = eq(neg1, neg2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(), m::Eq(m::Neg(), m::Neg()));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Eq(m::Param("x"), m::Param("y")));
}

TEST_F(ArithSimplificationPassTest, CompareEqNegatedConstant) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[1] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        ret result: bits[1] = eq(neg_x, k)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Eq(m::Neg(), m::Literal(3)));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Eq(m::Param("x"), m::Literal(253)));
}

TEST_F(ArithSimplificationPassTest, CompareEqNegatedWithOneUsedElsewhere) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[9] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        cmp:bits[1] = eq(neg1, neg2)
        ret result: bits[9] = concat(cmp, neg1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::Eq(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x"))));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::Eq(m::Param("x"), m::Param("y")),
                                           m::Neg(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, CompareEqNegatedWithBothUsedElsewhere) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[17] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        cmp:bits[1] = eq(neg1, neg2)
        ret result: bits[17] = concat(cmp, neg1, neg2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::Eq(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x")), m::Neg(m::Param("y"))));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::Eq(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x")), m::Neg(m::Param("y"))));
}

TEST_F(ArithSimplificationPassTest, CompareEqNegatedConstantWithOtherUse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[9] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        cmp:bits[1] = eq(neg_x, k)
        ret result: bits[9] = concat(cmp, neg_x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Eq(m::Neg(m::Param("x")), m::Literal(3)),
                        m::Neg(m::Param("x"))));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Eq(m::Neg(m::Param("x")), m::Literal(3)),
                        m::Neg(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, CompareNeNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        ret result: bits[1] = ne(neg1, neg2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Ne(m::Neg(), m::Neg()));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Ne(m::Param("x"), m::Param("y")));
}

TEST_F(ArithSimplificationPassTest, CompareNeNegatedConstant) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[1] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        ret result: bits[1] = ne(neg_x, k)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Ne(m::Neg(), m::Literal(3)));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Ne(m::Param("x"), m::Literal(253)));
}

TEST_F(ArithSimplificationPassTest, CompareNeNegatedWithOneUsedElsewhere) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[9] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        cmp:bits[1] = ne(neg1, neg2)
        ret result: bits[9] = concat(cmp, neg1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::Ne(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x"))));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::Ne(m::Param("x"), m::Param("y")),
                                           m::Neg(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, CompareNeNegatedWithBothUsedElsewhere) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[17] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        cmp:bits[1] = ne(neg1, neg2)
        ret result: bits[17] = concat(cmp, neg1, neg2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::Ne(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x")), m::Neg(m::Param("y"))));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::Ne(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x")), m::Neg(m::Param("y"))));
}

TEST_F(ArithSimplificationPassTest, CompareNeNegatedConstantWithOtherUse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[9] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        cmp:bits[1] = ne(neg_x, k)
        ret result: bits[9] = concat(cmp, neg_x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Ne(m::Neg(m::Param("x")), m::Literal(3)),
                        m::Neg(m::Param("x"))));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Ne(m::Neg(m::Param("x")), m::Literal(3)),
                        m::Neg(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedLtNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        ret result: bits[1] = slt(neg1, neg2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SLt(m::Neg(), m::Neg()));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SGt(m::Param("x"), m::Param("y")),
                                        m::Ne(m::Param(), m::Literal(128)),
                                        m::Ne(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedLtNegatedConstant) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        ret result: bits[1] = slt(neg_x, k)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SLt(m::Neg(), m::Literal(3)));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SGt(m::Param("x"), m::Literal(253)),
                                        m::Eq(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest,
       CompareSignedLtNegatedWithOneUsedElsewhere) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[9] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        cmp:bits[1] = slt(neg1, neg2)
        ret result: bits[9] = concat(cmp, neg1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::SLt(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x"))));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Xor(m::SGt(m::Param("x"), m::Param("y")),
                               m::Ne(m::Param("x"), m::Literal(128)),
                               m::Ne(m::Param("y"), m::Literal(128))),
                        m::Neg(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest,
       CompareSignedLtNegatedWithBothUsedElsewhere) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[17] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        cmp:bits[1] = slt(neg1, neg2)
        ret result: bits[17] = concat(cmp, neg1, neg2)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::SLt(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x")), m::Neg(m::Param("y"))));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  ASSERT_THAT(f->return_value(),
              m::Concat(m::SLt(m::Neg(m::Param("x")), m::Neg(m::Param("y"))),
                        m::Neg(m::Param("x")), m::Neg(m::Param("y"))));
}

TEST_F(ArithSimplificationPassTest,
       CompareSignedLtNegatedConstantWithOtherUse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[9] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        cmp:bits[1] = slt(neg_x, k)
        ret result: bits[9] = concat(cmp, neg_x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::SLt(m::Neg(m::Param("x")), m::Literal(3)),
                        m::Neg(m::Param("x"))));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::SLt(m::Neg(m::Param("x")), m::Literal(3)),
                        m::Neg(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedGtNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        ret result: bits[1] = sgt(neg1, neg2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SGt(m::Neg(), m::Neg()));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SLt(m::Param("x"), m::Param("y")),
                                        m::Ne(m::Param(), m::Literal(128)),
                                        m::Ne(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedGtNegatedConstant) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        ret result: bits[1] = sgt(neg_x, k)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SGt(m::Neg(), m::Literal(3)));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SLt(m::Param("x"), m::Literal(253)),
                                        m::Eq(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedLeNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        ret result: bits[1] = sle(neg1, neg2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SLe(m::Neg(), m::Neg()));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SGe(m::Param("x"), m::Param("y")),
                                        m::Ne(m::Param(), m::Literal(128)),
                                        m::Ne(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedLeNegatedConstant) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[1] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        ret result: bits[1] = sle(neg_x, k)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SLe(m::Neg(), m::Literal(3)));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SGe(m::Param("x"), m::Literal(253)),
                                        m::Eq(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedGeNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8], y:bits[8]) -> bits[1] {
        neg1:bits[8] = neg(x)
        neg2:bits[8] = neg(y)
        ret result: bits[1] = sge(neg1, neg2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SGe(m::Neg(), m::Neg()));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SLe(m::Param("x"), m::Param("y")),
                                        m::Ne(m::Param(), m::Literal(128)),
                                        m::Ne(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, CompareSignedGeNegatedConstant) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn compare_neg(x:bits[8]) -> bits[1] {
        neg_x:bits[8] = neg(x)
        k:bits[8] = literal(value=3)
        ret result: bits[1] = sge(neg_x, k)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::SGe(m::Neg(), m::Literal(3)));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::SLe(m::Param("x"), m::Literal(253)),
                                        m::Eq(m::Param(), m::Literal(128))));
}

TEST_F(ArithSimplificationPassTest, MulBy42) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=42)
        ret result: bits[8] = umul(x, literal.1)
     }
  )",
                              p.get())
                    .status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ArithSimplificationPassTest, MulBy5) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=5)
        ret result: bits[8] = umul(x, literal.1)
     }
  )",
                                                       p.get()));
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Param("x"),
                     m::Concat(m::BitSlice(m::Param("x")), m::Literal(0, 2))));
}

TEST_F(ArithSimplificationPassTest, MulBy6) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=6)
        ret result: bits[8] = umul(x, literal.1)
     }
  )",
                                                       p.get()));
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Concat(m::BitSlice(m::Param("x")), m::Literal(0, 1)),
                     m::Concat(m::BitSlice(m::Param("x")), m::Literal(0, 2))));
}

TEST_F(ArithSimplificationPassTest, MulBy7) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=7)
        ret result: bits[8] = umul(x, literal.1)
     }
  )",
                                                       p.get()));
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Sub(m::Concat(m::BitSlice(m::Param("x")), m::Literal(0, 3)),
                     m::Param("x")));
}

TEST_F(ArithSimplificationPassTest, MulBy11) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=11)
        ret result: bits[8] = umul(x, literal.1)
     }
  )",
                              p.get())
                    .status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ArithSimplificationPassTest, SMulBy1SignExtendedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[16] {
        one: bits[8] = literal(value=1)
        ret result: bits[16] = smul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::SignExt(m::Param("x")));
}

TEST_F(ArithSimplificationPassTest, SMulBy16SignExtendedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[16] {
        one: bits[8] = literal(value=16)
        ret result: bits[16] = smul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::SignExt(m::Param("x"))),
                        m::Literal(UBits(0, 4))));
}

TEST_F(ArithSimplificationPassTest, UMulBy1ZeroExtendedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[16] {
        one: bits[8] = literal(value=1)
        ret result: bits[16] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Param("x")));
}

TEST_F(ArithSimplificationPassTest, UMulBy256ZeroExtendedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[16] {
        one: bits[12] = literal(value=256)
        ret result: bits[16] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::ZeroExt(m::Param("x"))),
                        m::Literal(UBits(0, 8))));
}

TEST_F(ArithSimplificationPassTest, UModBy4) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn umod_power_of_two(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=4)
        ret result: bits[16] = umod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::ZeroExt(m::BitSlice(0, 2)));
}

TEST_F(ArithSimplificationPassTest, UModBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn umod_power_of_two(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=1)
        ret result: bits[16] = umod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, UModBy512) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn umod_power_of_two(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=512)
        ret result: bits[16] = umod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::ZeroExt(m::BitSlice(m::Param("x"), 0, 9)));
}

TEST_F(ArithSimplificationPassTest, UModBy42) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn umod_non_power_of_two(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=42)
        ret result: bits[8] = umod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Sub(m::Param("x"), m::UMul(_, m::Literal(42))));
}

TEST_F(ArithSimplificationPassTest, UModBy0) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn umod_zero(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=0)
        ret result: bits[16] = umod(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, UModByVariable) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn umod(x:bits[16], y:bits[16]) -> bits[16] {
        ret result: bits[16] = umod(x, y)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::UMod());
}

TEST_F(ArithSimplificationPassTest, SModBy4) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn smod_power_of_two(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=4)
        ret result: bits[16] = smod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  auto nonnegative_result = m::ZeroExt(m::BitSlice(0, 2));
  EXPECT_THAT(f->return_value(),
              m::Select(m::And(m::Ne(m::BitSlice(0, 2), m::Literal(0)),
                               m::BitSlice(m::Param(), 15, 1)),
                        {nonnegative_result,
                         m::Sub(nonnegative_result, m::Literal(4))}));
}

TEST_F(ArithSimplificationPassTest, SModBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn smod_power_of_two(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=1)
        ret result: bits[16] = smod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, SModBy512) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn smod_power_of_two(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=512)
        ret result: bits[16] = smod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  auto nonnegative_result = m::ZeroExt(m::BitSlice(0, 9));
  EXPECT_THAT(f->return_value(),
              m::Select(m::And(m::Ne(m::BitSlice(0, 9), m::Literal(0)),
                               m::BitSlice(m::Param(), 15, 1)),
                        {nonnegative_result,
                         m::Sub(nonnegative_result, m::Literal(512))}));
}

TEST_F(ArithSimplificationPassTest, SModBy42) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn smod_non_power_of_two(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=42)
        ret result: bits[8] = smod(x, literal.1)
     }
  )",
                                                       p.get()));

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Sub(m::Param("x"), m::SMul(_, m::Literal(42))));
}

TEST_F(ArithSimplificationPassTest, SModBy0) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn smod_zero(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=0)
        ret result: bits[16] = smod(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, SModByVariable) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn smod(x:bits[16], y:bits[16]) -> bits[16] {
        ret result: bits[16] = smod(x, y)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::SMod());
}

TEST_F(ArithSimplificationPassTest, UDivBy4) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=4)
        ret result: bits[16] = udiv(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 2)), m::BitSlice(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, SDivBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn sdiv_by_1(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=1)
        ret result: bits[16] = sdiv(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, OneBitSDivByMinus1) {
  // 0b1 is -1 for a bits[1] type so sdiv by literal one should not apply.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn one_bit_sdiv(x:bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=1)
        ret result: bits[1] = sdiv(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Neg());

  auto interp_and_check = [&f](int x, int expected) {
    constexpr int N = 1;
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                             InterpretFunction(f, {Value(SBits(x, N))}));
    EXPECT_EQ(r.value, Value(SBits(expected, N)));
  };
  // Even though the operation is negate, a 1-bit value can't be negated so the
  // overall effect is no change.
  interp_and_check(0, 0);
  interp_and_check(-1, -1);
}

TEST_F(ArithSimplificationPassTest, SDivWithLiteralDivisorSweep) {
  // Sweep all possible values for a Bits[3] SDIV with a literal divisor.
  for (int64_t divisor = -4; divisor <= 3; ++divisor) {
    for (int64_t dividend = -4; dividend <= 3; ++dividend) {
      auto p = CreatePackage();
      Type* u3 = p->GetBitsType(3);
      FunctionBuilder fb(TestName(), p.get());
      fb.SDiv(fb.Param("dividend", u3), fb.Literal(SBits(divisor, 3)));
      XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

      Value expected;
      // XLS divide by zero semantics are to return the maximal
      // positive/negative value.
      if (divisor == 0) {
        expected = Value(SBits(dividend >= 0 ? 3 : -4, 3));
      } else if (dividend == -4 && divisor == -1) {
        // Overflow. In this case we just return the truncated twos-complement
        // result (0b0100 => 0b100).
        expected = Value(SBits(-4, 3));
      } else {
        expected = Value(SBits(dividend / divisor, 3));
      }
      VLOG(1) << absl::StreamFormat("%d / %d = %d (%s)", dividend, divisor,
                                    expected.bits().ToInt64().value(),
                                    expected.ToString());
      XLS_ASSERT_OK_AND_ASSIGN(
          InterpreterResult<Value> before_result,
          InterpretFunction(f, {Value(SBits(dividend, 3))}));
      XLS_ASSERT_OK_AND_ASSIGN(Value before_value,
                               InterpreterResultToStatusOrValue(before_result));
      EXPECT_EQ(before_value, expected)
          << absl::StreamFormat("Before: %d / %d", dividend, divisor);

      XLS_ASSERT_OK(Run(p.get()));

      XLS_ASSERT_OK_AND_ASSIGN(
          InterpreterResult<Value> after_result,
          InterpretFunction(f, {Value(SBits(dividend, 3))}));
      XLS_ASSERT_OK_AND_ASSIGN(Value after_value,
                               InterpreterResultToStatusOrValue(after_result));
      EXPECT_EQ(after_result.value, expected)
          << absl::StreamFormat("After: %d / %d", dividend, divisor);
    }
  }
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// Optimizes the IR for an unsigned divide by the given non-power of two. Checks
// that optimized IR matches expected IR. Numerically validates (via
// interpretation) the optimized IR, exhaustively up to 2^n.
void ArithSimplificationPassTest::CheckUnsignedDivide(int n, int divisor) {
  auto p = CreatePackage();
  FunctionBuilder fb("UnsignedDivideBy" + std::to_string(divisor), p.get());
  fb.UDiv(fb.Param("x", p->GetBitsType(n)),
          fb.Literal(Value(UBits(divisor, n))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  // Clean up the dumped IR
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(DeadCodeEliminationPass().Run(p.get(), OptimizationPassOptions(),
                                            &results, context),
              IsOkAndHolds(true));

  std::string optimized_ir = f->DumpIr();

  // A non-power of two divisor will be rewritten to shifts (which are further
  // rewritten to slices), mul. No div, add, or sub.
  EXPECT_TRUE(Contains(optimized_ir, "bit_slice"));
  EXPECT_TRUE(Contains(optimized_ir, "umul"));
  EXPECT_FALSE(Contains(optimized_ir, "div"));
  EXPECT_FALSE(Contains(optimized_ir, "add"));
  EXPECT_FALSE(Contains(optimized_ir, "sub"));

  // compute x/divisor. assert result == expected
  auto interp_and_check = [n, &f](int x, int expected) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                             InterpretFunction(f, {Value(UBits(x, n))}));
    EXPECT_EQ(r.value, Value(UBits(expected, n)));
  };

  // compute RoundToZero(i/divisor)
  auto div_rt0 = [=](int i) { return i / divisor; };

  for (int i = 0; i < Exp2<int>(n); ++i) {
    interp_and_check(i, div_rt0(i));
  }
}

// Exhaustively test unsigned division. Vary n and divisor (excluding powers of
// two).
TEST_F(ArithSimplificationPassTest, UnsignedDivideAllNonPowersOfTwoExhaustive) {
  constexpr int kTestUpToN = 10;
  for (int divisor = 1; divisor < Exp2<int>(kTestUpToN); ++divisor) {
    if (!IsPowerOfTwo(static_cast<unsigned int>(divisor))) {
      for (int n = Bits::MinBitCountUnsigned(divisor); n <= kTestUpToN; ++n) {
        CheckUnsignedDivide(n, divisor);
      }
    }
  }
}

// Regression test for
// https://github.com/google/xls/issues/736
TEST_F(ArithSimplificationPassTest, UDivWrongIssue736) {
  constexpr uint64_t nBits = 43;
  constexpr uint64_t divisor = UINT64_C(1876853526877);

  auto p = CreatePackage();
  FunctionBuilder fb("UnsignedDivideBy" + std::to_string(divisor), p.get());
  fb.UDiv(fb.Param("x", p->GetBitsType(nBits)),
          fb.Literal(Value(UBits(divisor, nBits))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  // Clean up the dumped IR
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(DeadCodeEliminationPass().Run(p.get(), OptimizationPassOptions(),
                                            &results, context),
              IsOkAndHolds(true));

  std::string optimized_ir = f->DumpIr();

  // A non-power of two divisor will be rewritten to shifts (which are further
  // rewritten to slices), mul. No div, add, or sub.
  EXPECT_TRUE(Contains(optimized_ir, "bit_slice"));
  EXPECT_TRUE(Contains(optimized_ir, "umul"));
  EXPECT_FALSE(Contains(optimized_ir, "div"));
  EXPECT_FALSE(Contains(optimized_ir, "add"));
  EXPECT_FALSE(Contains(optimized_ir, "sub"));

  // compute x/divisor. assert result == expected
  auto interp_and_check = [&f](uint64_t x, uint64_t expected) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                             InterpretFunction(f, {Value(UBits(x, nBits))}));
    EXPECT_EQ(r.value, Value(UBits(expected, nBits)));
  };

  // compute RoundToZero(i/divisor)
  auto div_rt0 = [=](uint64_t i) { return i / divisor; };

  // The input that triggered the bug
  uint64_t x = UINT64_C(5864062014805);  // 0x555_5555_5555
  interp_and_check(x, div_rt0(x));
}

// Regression test for
// https://github.com/google/xls/issues/736
TEST_F(ArithSimplificationPassTest, SDivWrongIssue736) {
  constexpr uint64_t nBits = 66;
  constexpr int64_t divisor =
      INT64_C(2305843009213693950);  // floor(dividend/2 - 1) = 2^61 - 2

  auto p = CreatePackage();
  FunctionBuilder fb("UnsignedDivideBy" + std::to_string(divisor), p.get());
  fb.SDiv(fb.Param("x", p->GetBitsType(nBits)),
          fb.Literal(Value(SBits(divisor, nBits))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  // Clean up the dumped IR
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(DeadCodeEliminationPass().Run(p.get(), OptimizationPassOptions(),
                                            &results, context),
              IsOkAndHolds(true));

  std::string optimized_ir = f->DumpIr();

  // A non-power of two divisor will be rewritten to shifts (which are further
  // rewritten to slices), mul, sub. No div or add.
  EXPECT_TRUE(Contains(optimized_ir, "bit_slice"));
  EXPECT_TRUE(Contains(optimized_ir, "smul"));
  EXPECT_TRUE(Contains(optimized_ir, "sub"));
  EXPECT_FALSE(Contains(optimized_ir, "div"));
  EXPECT_FALSE(Contains(optimized_ir, "add"));

  // compute x/divisor. assert result == expected
  auto interp_and_check = [&f](int64_t x, int64_t expected) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                             InterpretFunction(f, {Value(UBits(x, nBits))}));
    EXPECT_EQ(r.value, Value(UBits(expected, nBits)));
  };

  // compute RoundToZero(i/divisor)
  auto div_rt0 = [=](int64_t i) { return i / divisor; };

  // The input that triggered the bug
  int64_t x = INT64_C(4611686018427387903);  // 2^62 - 1

  // obviously, this should be 2
  interp_and_check(x, div_rt0(x));
}

// Optimizes the IR for a divide by a power of two (which may be negative or
// positive). Checks that optimized IR matches expected IR. Numerically
// validates (via interpretation) the optimized IR, exhaustively up to 2^N.
void ArithSimplificationPassTest::CheckSignedDividePowerOfTwo(int n,
                                                              int divisor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SDiv(fb.Param("x", p->GetBitsType(n)),
          fb.Literal(Value(SBits(divisor, n))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  // Clean up the dumped IR
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(DeadCodeEliminationPass().Run(p.get(), OptimizationPassOptions(),
                                            &results, context),
              IsOkAndHolds(true));

  std::string optimized_ir = f->DumpIr();

  if (std::abs(divisor) > 1) {
    // A power of two divisor will be rewritten to shifts (which are further
    // rewritten to slices) and an add. There will be no multiply (unlike other
    // divisors).
    EXPECT_TRUE(Contains(optimized_ir, "bit_slice"));
    EXPECT_TRUE(Contains(optimized_ir, "add"));
    EXPECT_FALSE(Contains(optimized_ir, "div"));
    EXPECT_FALSE(Contains(optimized_ir, "mul"));
  } else if (divisor == 1) {
    EXPECT_FALSE(Contains(optimized_ir, "neg"));
    EXPECT_FALSE(Contains(optimized_ir, "bit_slice"));
    EXPECT_FALSE(Contains(optimized_ir, "add"));
    EXPECT_FALSE(Contains(optimized_ir, "div"));
    EXPECT_FALSE(Contains(optimized_ir, "mul"));
  } else if (divisor == -1) {
    EXPECT_TRUE(Contains(optimized_ir, "neg"));
    EXPECT_FALSE(Contains(optimized_ir, "bit_slice"));
    EXPECT_FALSE(Contains(optimized_ir, "add"));
    EXPECT_FALSE(Contains(optimized_ir, "div"));
    EXPECT_FALSE(Contains(optimized_ir, "mul"));
  }

  // compute x/divisor. assert result == expected
  auto interp_and_check = [n, &f](int x, int expected) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                             InterpretFunction(f, {Value(SBits(x, n))}));
    EXPECT_EQ(r.value, Value(SBits(expected, n)));
  };
  // compute RoundToZero(i/divisor)
  auto div_rt0 = [=](int i) {
    const int q = std::abs(i) / std::abs(divisor);
    const bool exactly_one_negative =
        (i < 0 || divisor < 0) && !(i < 0 && divisor < 0);
    return exactly_one_negative ? -q : q;
  };

  // N-1 because we create signed values
  for (int i = 0; i < Exp2<unsigned int>(n - 1); ++i) {
    interp_and_check(i, div_rt0(i));
    interp_and_check(-i, div_rt0(-i));
  }

  // Avoid overflow: -2^(N-1) * -1 = 2^(N-1) which won't fit in a signed N-bit
  // integer.
  if (divisor != -1) {
    const int last = -Exp2<unsigned int>(n - 1);
    interp_and_check(last, div_rt0(last));
  }
}

// Exhaustively test signed division by power of two. Vary N and divisor.
TEST_F(ArithSimplificationPassTest, SignedDivideAllPowersOfTwoExhaustive) {
  const int kTestUpToN = 14;
  // first divisor = 2^0 = 1
  for (int exp = 0; exp <= kTestUpToN; ++exp) {
    const int divisor = Exp2<int>(exp);
    for (int n = Bits::MinBitCountSigned(divisor); n <= kTestUpToN; ++n) {
      CheckSignedDividePowerOfTwo(n, divisor);
      CheckSignedDividePowerOfTwo(n, -divisor);
    }
  }
}

void ArithSimplificationPassTest::CheckSignedDivideNotPowerOfTwo(int n,
                                                                 int divisor) {
  auto p = CreatePackage();
  FunctionBuilder fb("SignedDivideBy" + std::to_string(divisor), p.get());
  fb.SDiv(fb.Param("x", p->GetBitsType(n)),
          fb.Literal(Value(SBits(divisor, n))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  // Clean up the dumped IR
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(DeadCodeEliminationPass().Run(p.get(), OptimizationPassOptions(),
                                            &results, context),
              IsOkAndHolds(true));

  std::string optimized_ir = f->DumpIr();

  // A non-power of two divisor will be rewritten to shifts (which are further
  // rewritten to slices), mul, sub. No div or add.
  EXPECT_TRUE(Contains(optimized_ir, "bit_slice"));
  EXPECT_TRUE(Contains(optimized_ir, "smul"));
  EXPECT_TRUE(Contains(optimized_ir, "sub"));
  EXPECT_FALSE(Contains(optimized_ir, "div"));
  EXPECT_FALSE(Contains(optimized_ir, "add"));

  // compute x/divisor. assert result == expected
  auto interp_and_check = [n, &f](int x, int expected) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                             InterpretFunction(f, {Value(SBits(x, n))}));
    EXPECT_EQ(r.value, Value(SBits(expected, n)));
  };

  // compute RoundToZero(i/divisor)
  auto div_rt0 = [=](int i) {
    const int q = std::abs(i) / std::abs(divisor);
    const bool exactly_one_negative =
        (i < 0 || divisor < 0) && !(i < 0 && divisor < 0);
    return exactly_one_negative ? -q : q;
  };

  // N-1 because we create signed values
  for (int i = 0; i < Exp2<unsigned int>(n - 1); ++i) {
    interp_and_check(i, div_rt0(i));
    interp_and_check(-i, div_rt0(-i));
  }
  const int last = -Exp2<unsigned int>(n - 1);
  interp_and_check(last, div_rt0(last));
}

// Exhaustively test signed division. For divisor, test all non-powers of 2, up
// to...
TEST_F(ArithSimplificationPassTest, SignedDivideAllNonPowersOfTwoExhaustive) {
  constexpr int kTestUpToN = 10;
  for (int divisor = 1; divisor < Exp2<int>(kTestUpToN - 1); ++divisor) {
    if (!IsPowerOfTwo(static_cast<unsigned int>(divisor))) {
      for (int n = Bits::MinBitCountSigned(divisor); n <= kTestUpToN; ++n) {
        CheckSignedDivideNotPowerOfTwo(n, divisor);
        CheckSignedDivideNotPowerOfTwo(n, -divisor);
      }
    }
  }
}

TEST_F(ArithSimplificationPassTest, MulBy1NarrowedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[3] {
        one: bits[8] = literal(value=1)
        ret result: bits[3] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/3));
}

TEST_F(ArithSimplificationPassTest, UMulByMaxPowerOfTwo) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[16]) -> bits[16] {
        literal.1: bits[8] = literal(value=128)
        ret result: bits[16] = umul(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x")), m::Literal(UBits(0, 7))));
}

TEST_F(ArithSimplificationPassTest, UMulWithSlicedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[16]) -> (bits[8], bits[8]) {
        literal.1: bits[8] = literal(value=25)
        umul.2: bits[16] = umul(x, literal.1)
        lo_bits: bits[8] = bit_slice(umul.2, start=0, width=8)
        mid_bits: bits[8] = bit_slice(umul.2, start=4, width=8)
        ret tuple.10: (bits[8], bits[8]) = tuple(lo_bits, mid_bits)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::BitSlice(AllOf(m::UMul(), m::Type("bits[12]"))),
                       m::BitSlice(AllOf(m::UMul(), m::Type("bits[12]")))));
}

TEST_F(ArithSimplificationPassTest, SMulByMinNegative) {
  // The minimal negative number has only one bit set like powers of two do, but
  // the mul-by-power-of-two optimization should not kick in.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=128)
        ret result: bits[8] = smul(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::SMul());
}

TEST_F(ArithSimplificationPassTest, SMulByMinusOne) {
  // A single-bit value of 1 is a -1 when interpreted as a signed number. The
  // Mul-by-power-of-two optimization should not kick in this case.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[3] {
        one: bits[1] = literal(value=1)
        ret result: bits[3] = smul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::SMul());
}

TEST_F(ArithSimplificationPassTest, SMulWithSlicedResult) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[16]) -> (bits[8], bits[8]) {
        literal.1: bits[8] = literal(value=25)
        smul.2: bits[16] = smul(x, literal.1)
        lo_bits: bits[8] = bit_slice(smul.2, start=0, width=8)
        mid_bits: bits[8] = bit_slice(smul.2, start=4, width=8)
        ret tuple.10: (bits[8], bits[8]) = tuple(lo_bits, mid_bits)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::BitSlice(AllOf(m::SMul(), m::Type("bits[12]"))),
                       m::BitSlice(AllOf(m::SMul(), m::Type("bits[12]")))));
}

TEST_F(ArithSimplificationPassTest, UDivBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn div_one(x:bits[8]) -> bits[8] {
        one:bits[8] = literal(value=1)
        ret result: bits[8] = udiv(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, MulBy1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn mul_zero(x:bits[8]) -> bits[8] {
        one:bits[8] = literal(value=1)
        ret result: bits[8] = umul(x, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArithSimplificationPassTest, OverlargeShiftAfterSimp) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[2]) -> bits[2] {
        literal.1: bits[2] = literal(value=2)
        shrl.2: bits[2] = shrl(x, literal.1)
        ret result: bits[2] = shrl(shrl.2, literal.1)
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
        ret result: bits[16] = shra(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SignExt(m::BitSlice(m::Param("x"), /*start=*/13, /*width=*/3)));
}

TEST_F(ArithSimplificationPassTest, OverlargeShiftRightArithmeticByLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[16]) -> bits[16] {
        literal.1: bits[16] = literal(value=1234)
        ret result: bits[16] = shra(x, literal.1)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(m::BitSlice(/*start=*/15, /*width=*/1)));
}

TEST_F(ArithSimplificationPassTest, ArithmeticShiftRightOfConstantOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[10]) -> bits[42] {
        literal.1: bits[42] = literal(value=1)
        ret result: bits[42] = shra(literal.1, x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Shra());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Eq(m::Param(), m::Literal(0))));
}

TEST_F(ArithSimplificationPassTest, LogicalShiftRightOfConstantOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[10]) -> bits[42] {
        literal.1: bits[42] = literal(value=1)
        ret result: bits[42] = shrl(literal.1, x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Shrl());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Eq(m::Param(), m::Literal(0))));
}

TEST_F(ArithSimplificationPassTest, ArithmeticShiftRightOfOneBitUnknown) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[10], y:bits[1]) -> bits[1] {
        ret result: bits[1] = shra(y, x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Shra());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("y"));
}

TEST_F(ArithSimplificationPassTest, ArithmeticShiftRightOfOneBitOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[10]) -> bits[1] {
        literal.1: bits[1] = literal(value=1)
        ret result: bits[1] = shra(literal.1, x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Shra());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(1));
}

TEST_F(ArithSimplificationPassTest, OneBitDecode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[10]) -> bits[1] {
        ret result: bits[1] = decode(x, width=1)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Decode());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Eq(m::Param("x"), m::Literal(0)));
}

TEST_F(ArithSimplificationPassTest, DecodeOfOneBit) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x:bits[1]) -> bits[2] {
        ret result: bits[2] = decode(x, width=2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Decode());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Param("x"), m::Not(m::Param("x"))));
}

TEST_F(ArithSimplificationPassTest, SignExtTwice) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8]) -> bits[32] {
        sign_ext.2: bits[24] = sign_ext(x, new_bit_count=24)
        ret result: bits[32] = sign_ext(sign_ext.2, new_bit_count=32)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::SignExt(m::Param()));
}

TEST_F(ArithSimplificationPassTest, ZeroWidthMulOperand) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn id_or(x: bits[0], y: bits[32]) -> bits[32] {
       ret result: bits[32] = smul(x, y, pos=[(0,1,5)])
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
      zero: bits[32] = literal(value=0)
      ret result: bits[1] = slt(x, zero)
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
      zero: bits[32] = literal(value=0)
      ret result: bits[1] = sge(x, zero)
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

TEST_F(ArithSimplificationPassTest, InvertedComparisonWithMultipleUsers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue lt = fb.ULt(fb.Param("x", u32), fb.Param("y", u32));
  BValue not_lt = fb.Not(lt);
  fb.Tuple({lt, not_lt});
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ArithSimplificationPassTest, ULtMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b0011)
      ret result: bits[1] = ult(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Nor(
          m::OrReduce(m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/2)),
          m::AndReduce(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/2))));
}

TEST_F(ArithSimplificationPassTest, ULeMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b0011)
      ret result: bits[1] = ule(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::OrReduce(
                  m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/2))));
}

TEST_F(ArithSimplificationPassTest, UGtMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b0011)
      ret result: bits[1] = ugt(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::OrReduce(m::BitSlice(
                                     m::Param("x"), /*start=*/2, /*width=*/2)));
}

TEST_F(ArithSimplificationPassTest, UGeMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b0011)
      ret result: bits[1] = uge(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Or(
          m::OrReduce(m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/2)),
          m::AndReduce(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/2))));
}

TEST_F(ArithSimplificationPassTest, UGtMaskAllOnes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn f(x: bits[4]) -> bits[1] {
      literal.1: bits[4] = literal(value=0b1111)
      ret result: bits[1] = ugt(x, literal.1)
    }
)",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ArithSimplificationPassTest, ShiftByConcatWithZeroValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("x", p->GetBitsType(32)),
          fb.Concat({fb.Literal(Value(UBits(0, 24))),
                     fb.Param("y", p->GetBitsType(8))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("x"), m::Param("y")));
}

TEST_F(ArithSimplificationPassTest, ShiftByConcatWithZeroValueAndOthers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("x", p->GetBitsType(32)),
          fb.Concat({fb.Literal(Value(UBits(0, 16))),
                     fb.Param("y", p->GetBitsType(8)),
                     fb.Param("z", p->GetBitsType(8))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Shll(m::Param("x"), m::Concat(m::Param("y"), m::Param("z"))));
}

TEST_F(ArithSimplificationPassTest, DecodeTrivialConcat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Concat({fb.Literal(Value(UBits(0, 32)))}),
            /*width=*/128);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::Decode(m::Literal(/*value=*/0, /*width=*/32)),
                    m::Type("bits[128]")));
}

TEST_F(ArithSimplificationPassTest, DecodeConcatWithZeroValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Concat({fb.Literal(Value(UBits(0, 24))),
                       fb.Param("y", p->GetBitsType(8))}),
            /*width=*/128);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::Decode(m::Param("y")), m::Type("bits[128]")));
}

TEST_F(ArithSimplificationPassTest, DecodeConcatWithZeroValueAndOthers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Concat({fb.Literal(Value(UBits(0, 16))),
                       fb.Param("y", p->GetBitsType(8)),
                       fb.Param("z", p->GetBitsType(8))}),
            /*width=*/128);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::Decode(m::Concat(m::Param("y"), m::Param("z"))),
                    m::Type("bits[128]")));
}

TEST_F(ArithSimplificationPassTest, DecodeNarrowedConcatWithZeroValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Concat({fb.Literal(Value(UBits(0, 24))),
                       fb.Param("y", p->GetBitsType(8))}),
            /*width=*/1024);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      AllOf(m::ZeroExt(AllOf(m::Decode(m::Param("y")), m::Type("bits[256]"))),
            m::Type("bits[1024]")));
}

TEST_F(ArithSimplificationPassTest, GuardedShiftOperation) {
  // Test that a shift amount clamped to the shift's width is removed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(100));
  BValue amt = fb.Param("amt", p->GetBitsType(32));
  BValue limit = fb.Literal(Value(UBits(100, 32)));
  BValue clamped_amt =
      fb.Select(fb.UGt(amt, limit), /*on_true=*/limit, /*on_false=*/amt);
  fb.Shll(x, clamped_amt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("x"), m::Param("amt")));
}

TEST_F(ArithSimplificationPassTest, GuardedShiftOperationWithDefault) {
  // Test that a shift amount clamped to the shift's width is removed.
  // Uses the default element of a sel.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(100));
  BValue amt = fb.Param("amt", p->GetBitsType(32));
  BValue limit = fb.Literal(Value(UBits(100, 32)));
  BValue clamped_amt =
      fb.Select(fb.UGt(amt, limit), /*cases=*/absl::MakeConstSpan({amt}),
                /*default_value=*/limit);
  fb.Shll(x, clamped_amt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(
      f->return_value(),
      m::Shll(m::Param("x"), m::Select(m::UGt(), /*cases=*/{m::Param("amt")},
                                       /*default_value=*/m::Literal())));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("x"), m::Param("amt")));
}

TEST_F(ArithSimplificationPassTest, GuardedShiftOperationWithPrioritySelect) {
  // Test that a shift amount clamped to the shift's width is removed.
  // Uses a priority select.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(100));
  BValue amt = fb.Param("amt", p->GetBitsType(32));
  BValue limit = fb.Literal(Value(UBits(100, 32)));
  BValue clamped_amt = fb.PrioritySelect(fb.UGt(amt, limit), /*cases=*/{limit},
                                         /*default_value=*/amt);
  fb.Shll(x, clamped_amt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("x"), m::Param("amt")));
}

TEST_F(ArithSimplificationPassTest, GuardedArithShiftOperation) {
  // Test that a shift amount clamped to the shift's width is removed for
  // arithmetic shift right.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(100));
  BValue amt = fb.Param("amt", p->GetBitsType(32));
  BValue limit = fb.Literal(Value(UBits(100, 32)));
  BValue clamped_amt =
      fb.Select(fb.UGt(amt, limit), /*on_true=*/limit, /*on_false=*/amt);
  fb.Shra(x, clamped_amt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Shra(m::Param("x"), m::Param("amt")));
}

TEST_F(ArithSimplificationPassTest, GuardedShiftOperationHighLimit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(100));
  BValue amt = fb.Param("amt", p->GetBitsType(32));
  // Set a clamp limit higher than the width of the shift, the end result is the
  // same as if the limit were the shift amount.
  BValue limit = fb.Literal(Value(UBits(1234, 32)));
  BValue clamped_amt =
      fb.Select(fb.UGt(amt, limit), /*on_true=*/limit, /*on_false=*/amt);
  fb.Shrl(x, clamped_amt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Shrl(m::Param("x"), m::Param("amt")));
}

TEST_F(ArithSimplificationPassTest, GuardedShiftOperationLowLimit) {
  // Test that a shift amount clamped to a value less that the shift's width is
  // not removed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(100));
  BValue amt = fb.Param("amt", p->GetBitsType(32));
  BValue limit = fb.Literal(Value(UBits(99, 32)));
  BValue clamped_amt =
      fb.Select(fb.UGt(amt, limit), /*on_true=*/limit, /*on_false=*/amt);
  fb.Shll(x, clamped_amt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("x"), m::Select()));
}

TEST_F(ArithSimplificationPassTest, UMulCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Eq(fb.Literal(UBits(1000, 50)),
        fb.UMul(x, fb.Literal(UBits(100, 32)), 50));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Eq(m::ZeroExt(x.node()), m::Literal(UBits(10, 50))));
}

TEST_F(ArithSimplificationPassTest, UMulCompareOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  // This can overflow which makes getting the inverse more difficult.
  // TODO(allight): It might be nice to do this when we can.
  fb.Eq(fb.Literal(UBits(1, 8)), fb.UMul(x, fb.Literal(UBits(170, 8)), 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ArithSimplificationPassTest, UMulCompareZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  // Make sure that x*<foo> == 0 is x == 0
  fb.Eq(fb.Literal(UBits(0, 6)), fb.UMul(x, fb.Literal(UBits(7, 3)), 6));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Eq(x.node(), m::Literal(UBits(0, 3))))
      << f->DumpIr();
}

TEST_F(ArithSimplificationPassTest, UMulMulAndCompareZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  fb.Eq(fb.Literal(UBits(0, 3)), fb.UMul(x, fb.Literal(UBits(0, 3)), 3));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // Just make sure that whatever we create for a vacuously true x*0 == 0 is
  // consistent.
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOk());
}

TEST_F(ArithSimplificationPassTest, UMulCompareImpossible) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Eq(fb.Literal(UBits(1001, 50)),
        fb.UMul(x, fb.Literal(UBits(100, 32)), 50));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(Value::Bool(false)));
}

void UmulFuzz(const Bits& multiplicand, const Bits& result, int64_t var_width,
              bool const_on_right, bool var_on_right) {
  VerifiedPackage p("umul_fuzz");
  FunctionBuilder fb("umul_fuzz", &p);
  BValue eq_const = fb.Literal(result);
  BValue mul_const = fb.Literal(multiplicand);
  BValue var = fb.Param("param_val", p.GetBitsType(var_width));
  BValue mul;
  if (var_on_right) {
    mul = fb.UMul(mul_const, var, result.bit_count());
  } else {
    mul = fb.UMul(var, mul_const, result.bit_count());
  }
  if (const_on_right) {
    fb.Eq(mul, eq_const);
  } else {
    fb.Eq(eq_const, mul);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(&p);
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(ArithSimplificationPass().Run(
                  &p, OptimizationPassOptions().WithOptLevel(kMaxOptLevel),
                  &results, context),
              absl_testing::IsOk());
}

FUZZ_TEST(ArithSimplificationPassFuzzTest, UmulFuzz)
    .WithDomains(ArbitraryBits(16), ArbitraryBits(16), fuzztest::InRange(1, 40),
                 fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());

}  // namespace
}  // namespace xls

// Copyright 2021 The XLS Authors
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

#include "xls/passes/comparison_simplification_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "third_party/googlefuzztest/fuzztest_macros.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class ComparisonSimplificationPassTest : public IrTestBase {
 protected:
  ComparisonSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return ComparisonSimplificationPass().Run(p, OptimizationPassOptions(),
                                              &results, context);
  }
};

TEST_F(ComparisonSimplificationPassTest, OrOfEqAndNe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_ne_42 = fb.Ne(x, fb.Literal(UBits(42, 32)));
  BValue x_eq_37 = fb.Eq(x, fb.Literal(UBits(37, 32)));
  fb.And(x_ne_42, x_eq_37);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Eq(x.node(), m::Literal(37)));
}

TEST_F(ComparisonSimplificationPassTest, EqWithNonliterals) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue x_ne_42 = fb.Ne(x, fb.Literal(UBits(42, 32)));
  BValue x_eq_y = fb.Eq(x, y);
  fb.And(x_ne_42, x_eq_y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_THAT(f->return_value(), m::And(m::Ne(), m::Eq()));
}

TEST_F(ComparisonSimplificationPassTest, ComparisonsWithDifferentVariables) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue x_ne_42 = fb.Ne(fb.Literal(UBits(42, 32)), x);
  BValue y_eq_37 = fb.Eq(y, fb.Literal(UBits(37, 32)));
  fb.And(x_ne_42, y_eq_37);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // The comparisons are between different variables `x` and `y` so should not
  // be transformed.
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_THAT(f->return_value(), m::And());
}

TEST_F(ComparisonSimplificationPassTest, EmptyRange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 32)));
  BValue x_eq_37 = fb.Eq(fb.Literal(UBits(37, 32)), x);
  fb.And(x_eq_42, x_eq_37);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ComparisonSimplificationPassTest, MaximalRange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_ne_42 = fb.Ne(x, fb.Literal(UBits(42, 32)));
  BValue x_ne_37 = fb.Ne(fb.Literal(UBits(37, 32)), x);
  fb.Or(x_ne_42, x_ne_37);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(1));
}

TEST_F(ComparisonSimplificationPassTest, NotNeLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Not(fb.Ne(x, fb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Eq(x.node(), m::Literal(42)));
}

TEST_F(ComparisonSimplificationPassTest, EqsWithPreciseGap) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 2)));
  BValue x_eq_1 = fb.Eq(x, fb.Literal(UBits(1, 2)));
  BValue x_eq_3 = fb.Eq(x, fb.Literal(UBits(3, 2)));
  fb.Not(fb.Or({x_eq_0, x_eq_1, x_eq_3}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Eq(x.node(), m::Literal(2)));
}

TEST_F(ComparisonSimplificationPassTest, NotAndOfNeqsPrecise) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue x_ne_0 = fb.Ne(x, fb.Literal(UBits(0, 2)));
  BValue x_ne_1 = fb.Ne(x, fb.Literal(UBits(1, 2)));
  BValue x_ne_3 = fb.Ne(x, fb.Literal(UBits(3, 2)));
  fb.Not(fb.And({x_ne_0, x_ne_1, x_ne_3}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Ne(x.node(), m::Literal(2)));
}

TEST_F(ComparisonSimplificationPassTest, NotAndOfNeqsButNot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue x_ne_0 = fb.Ne(x, fb.Literal(UBits(0, 3)));
  BValue x_ne_1 = fb.Ne(x, fb.Literal(UBits(1, 3)));
  BValue x_ne_3 = fb.Ne(x, fb.Literal(UBits(3, 3)));
  fb.Not(fb.And({x_ne_0, x_ne_1, x_ne_3}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_THAT(f->return_value(), m::Not(m::And()));
}

TEST_F(ComparisonSimplificationPassTest, RedundantULt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_lt_42 = fb.ULt(x, fb.Literal(UBits(42, 8)));
  BValue x_lt_123 = fb.ULt(x, fb.Literal(UBits(123, 8)));
  fb.And({x_lt_42, x_lt_123});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::ULt(x.node(), m::Literal(42)));
}

TEST_F(ComparisonSimplificationPassTest, UltOrEq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_lt_42 = fb.ULt(x, fb.Literal(UBits(42, 8)));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 8)));
  fb.Or({x_lt_42, x_eq_42});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::ULt(x.node(), m::Literal(43)));
}

TEST_F(ComparisonSimplificationPassTest, UltAndUgt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_lt_42 = fb.ULt(x, fb.Literal(UBits(42, 8)));
  BValue x_gt_12 = fb.UGt(x, fb.Literal(UBits(12, 8)));
  fb.And({x_lt_42, x_gt_12});
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ComparisonSimplificationPassTest, RedundantUGt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_gt_42 = fb.UGt(x, fb.Literal(UBits(42, 8)));
  BValue x_gt_123 = fb.UGt(x, fb.Literal(UBits(123, 8)));
  fb.And({x_gt_42, x_gt_123});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::UGt(x.node(), m::Literal(123)));
}

TEST_F(ComparisonSimplificationPassTest, ULtZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  fb.ULt(x, fb.Literal(UBits(0, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ComparisonSimplificationPassTest, ULtMax) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  fb.ULt(x, fb.Literal(UBits(255, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Ne(m::Param("x"), m::Literal(255)));
}

TEST_F(ComparisonSimplificationPassTest, UGtZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  fb.UGt(x, fb.Literal(UBits(0, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Ne(m::Param("x"), m::Literal(0)));
}

TEST_F(ComparisonSimplificationPassTest, UGtMax) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  fb.UGt(x, fb.Literal(UBits(255, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(ComparisonSimplificationPassTest, EqAndNe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  fb.Tuple({fb.Eq(x, y), fb.Ne(x, y)});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Eq(m::Param("x"), m::Param("y")),
                       m::Not(m::Eq(m::Param("x"), m::Param("y")))));
}

TEST_F(ComparisonSimplificationPassTest, LtAndCommutedGe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  fb.Tuple({fb.ULt(x, y), fb.UGt(y, x)});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Tuple(m::ULt(m::Param("x"), m::Param("y")),
                       m::ULt(m::Param("x"), m::Param("y"))));
}

TEST_F(ComparisonSimplificationPassTest, SltAndCommutedSgt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  fb.Tuple({fb.SLt(x, y), fb.SGt(y, x)});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Tuple(m::SLt(m::Param("x"), m::Param("y")),
                       m::SLt(m::Param("x"), m::Param("y"))));
}

TEST_F(ComparisonSimplificationPassTest, UltAndCommutedSle) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  fb.Tuple({fb.ULt(x, y), fb.ULe(y, x)});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Tuple(m::ULt(m::Param("x"), m::Param("y")),
                       m::Not(m::ULt(m::Param("x"), m::Param("y")))));
}

void IrFuzzComparisonSimplificationPassTest(
    const PackageAndTestParams& paramaterized_package) {
  ComparisonSimplificationPass pass;
  OptimizationPassChangesOutputs(paramaterized_package, pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzComparisonSimplificationPassTest)
    .WithDomains(IrFuzzDomainWithParams(/*param_set_count=*/10));

}  // namespace
}  // namespace xls

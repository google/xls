// Copyright 2026 The XLS Authors
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

#include "xls/passes/value_set_simplification_pass.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

using ::xls::solvers::z3::ScopedVerifyEquivalence;

class FakeAreaEstimator : public AreaEstimator {
 public:
  FakeAreaEstimator() : AreaEstimator("fake") {}
  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override {
    if (node->op() == Op::kUMul || node->op() == Op::kSMul) {
      return 10.0;
    }
    if (node->op() == Op::kUDiv || node->op() == Op::kSDiv) {
      return 100.0;
    }
    return 1.0;
  }
  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override {
    return 1.0;
  }
};

class ValueSetSimplificationPassTest : public IrTestBase {
 protected:
  ValueSetSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         ValueSetSimplificationPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results, context));
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results, context)
            .status());
    return changed;
  }
};

TEST_F(ValueSetSimplificationPassTest, UDivByPowerOfTwoVarShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue divisor = fb.OneHot(s, LsbOrMsb::kLsb);
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Shrl(m::Param("dividend"),
                      m::Encode(m::OneHot(m::Param("s"), LsbOrMsb::kLsb))));
}

TEST_F(ValueSetSimplificationPassTest, UDivByPowerOfTwoOrZeroVarShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue divisor = fb.Decode(s, /*width=*/4);
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Eq(m::Decode(), m::Literal(0)),
                        {m::Shrl(m::Param("dividend"), m::Encode(m::Decode())),
                         m::Literal(Bits::AllOnes(4))}));
}

TEST_F(ValueSetSimplificationPassTest, UDivByPowerOfTwoOrZeroConstShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Concat({fb.Decode(s, /*width=*/2), fb.Literal(UBits(0, 2))});
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Concat(m::Eq(m::Concat(m::Decode(m::Param("s")),
                                                  m::Literal(UBits(0, 2))),
                                        m::Literal(8)),
                                  m::Eq(m::Concat(m::Decode(m::Param("s")),
                                                  m::Literal(UBits(0, 2))),
                                        m::Literal(4))),
                        {m::UDiv(m::Param("dividend"), m::Literal(4)),
                         m::UDiv(m::Param("dividend"), m::Literal(8))},
                        /*default_value=*/m::Literal(Bits::AllOnes(4))));
}

TEST_F(ValueSetSimplificationPassTest, UDivByFewConstantsFallback) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Select(s, {fb.Literal(UBits(3, 4)), fb.Literal(UBits(5, 4)),
                    fb.Literal(UBits(3, 4)), fb.Literal(UBits(5, 4))});
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::Eq(m::Select(), m::Literal(5)),
                                          m::Eq(m::Select(), m::Literal(3))),
                                {m::UDiv(m::Param("dividend"), m::Literal(3)),
                                 m::UDiv(m::Param("dividend"), m::Literal(5))},
                                /*default_value=*/m::Literal(0)));
}

TEST_F(ValueSetSimplificationPassTest, UDivByMoreConstantsWithAreaEstimator) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Select(s, {fb.Literal(UBits(3, 4)), fb.Literal(UBits(5, 4)),
                    fb.Literal(UBits(7, 4)), fb.Literal(UBits(3, 4))});
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  FakeAreaEstimator fake_ae;
  OptimizationPassOptions options;
  options.area_estimator = &fake_ae;
  PassResults results;
  OptimizationContext context;

  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(ValueSetSimplificationPass().RunOnFunctionBase(f, options,
                                                             &results, context),
              IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::Eq(m::Select(), m::Literal(7)),
                                          m::Eq(m::Select(), m::Literal(5)),
                                          m::Eq(m::Select(), m::Literal(3))),
                                {m::UDiv(m::Param("dividend"), m::Literal(3)),
                                 m::UDiv(m::Param("dividend"), m::Literal(5)),
                                 m::UDiv(m::Param("dividend"), m::Literal(7))},
                                /*default_value=*/m::Literal(0)));
}

TEST_F(ValueSetSimplificationPassTest, UDivByMoreConstantsHybridCase) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue divisor =
      fb.Select(s, {fb.Literal(UBits(1, 4)), fb.Literal(UBits(2, 4)),
                    fb.Literal(UBits(4, 4)), fb.Literal(UBits(8, 4)),
                    fb.Literal(UBits(3, 4)), fb.Literal(UBits(3, 4)),
                    fb.Literal(UBits(3, 4)), fb.Literal(UBits(3, 4))});
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Concat(m::Eq(m::Select(), m::Literal(3))),
                        {m::UDiv(m::Param("dividend"), m::Literal(3))},
                        /*default_value=*/
                        m::Shrl(m::Param("dividend"), m::Encode(m::Select()))));
}

TEST_F(ValueSetSimplificationPassTest, SDivByFewConstantsFallback) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Select(s, {fb.Literal(SBits(3, 4)), fb.Literal(SBits(5, 4)),
                    fb.Literal(SBits(3, 4)), fb.Literal(SBits(5, 4))});
  fb.SDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::Eq(m::Select(), m::Literal(5)),
                                          m::Eq(m::Select(), m::Literal(3))),
                                {m::SDiv(m::Param("dividend"), m::Literal(3)),
                                 m::SDiv(m::Param("dividend"), m::Literal(5))},
                                /*default_value=*/m::Literal(0)));
}

TEST_F(ValueSetSimplificationPassTest, SDivByMoreConstantsHybridCase) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue divisor =
      fb.Select(s, {fb.Literal(SBits(1, 4)), fb.Literal(SBits(2, 4)),
                    fb.Literal(SBits(4, 4)), fb.Literal(SBits(3, 4)),
                    fb.Literal(SBits(3, 4)), fb.Literal(SBits(3, 4)),
                    fb.Literal(SBits(3, 4)), fb.Literal(SBits(3, 4))});
  fb.SDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::Eq(m::Select(), m::Literal(4)),
                                          m::Eq(m::Select(), m::Literal(3)),
                                          m::Eq(m::Select(), m::Literal(2)),
                                          m::Eq(m::Select(), m::Literal(1))),
                                {m::SDiv(m::Param("dividend"), m::Literal(1)),
                                 m::SDiv(m::Param("dividend"), m::Literal(2)),
                                 m::SDiv(m::Param("dividend"), m::Literal(3)),
                                 m::SDiv(m::Param("dividend"), m::Literal(4))},
                                /*default_value=*/m::Literal(0)));
}

TEST_F(ValueSetSimplificationPassTest, SDivByZeroFallback) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Select(s, {fb.Literal(SBits(3, 4)), fb.Literal(SBits(5, 4)),
                    fb.Literal(SBits(0, 4)), fb.Literal(SBits(0, 4))});
  fb.SDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Concat(m::Eq(m::Select(), m::Literal(5)),
                                  m::Eq(m::Select(), m::Literal(3))),
                        {m::SDiv(m::Param("dividend"), m::Literal(3)),
                         m::SDiv(m::Param("dividend"), m::Literal(5))},
                        /*default_value=*/
                        m::Select(m::BitSlice(m::Param("dividend"), 3, 1),
                                  {m::Literal(Bits::MaxSigned(4)),
                                   m::Literal(Bits::MinSigned(4))})));
}

TEST_F(ValueSetSimplificationPassTest,
       UDivByManyConstantsFallbackUnprofitable) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Select(s, {fb.Literal(UBits(3, 4)), fb.Literal(UBits(5, 4)),
                    fb.Literal(UBits(7, 4)), fb.Literal(UBits(3, 4))});
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

class UnprofitableFakeAreaEstimator : public AreaEstimator {
 public:
  UnprofitableFakeAreaEstimator() : AreaEstimator("unprofitable_fake") {}
  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override {
    if (node->op() == Op::kUMul || node->op() == Op::kSMul) {
      return 1000.0;
    }
    if (node->op() == Op::kUDiv || node->op() == Op::kSDiv) {
      return 10.0;
    }
    return 1.0;
  }
  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override {
    return 1.0;
  }
};

TEST_F(ValueSetSimplificationPassTest, UDivUnprofitableWithAreaEstimator) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue dividend = fb.Param("dividend", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue divisor =
      fb.Select(s, {fb.Literal(UBits(3, 4)), fb.Literal(UBits(5, 4)),
                    fb.Literal(UBits(3, 4)), fb.Literal(UBits(5, 4))});
  fb.UDiv(dividend, divisor);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  UnprofitableFakeAreaEstimator fake_ae;
  OptimizationPassOptions options;
  options.area_estimator = &fake_ae;
  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(ValueSetSimplificationPass().RunOnFunctionBase(f, options,
                                                             &results, context),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls

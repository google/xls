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

#include "xls/passes/bdd_simplification_pass.h"

#include <cstdint>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class BddSimplificationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Function* f, int64_t opt_level = kMaxOptLevel) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(
        bool changed, BddSimplificationPass().RunOnFunctionBase(
                          f, OptimizationPassOptions().WithOptLevel(opt_level),
                          &results, &context));
    return changed;
  }
};

TEST_F(BddSimplificationPassTest, ReplaceAllKnownValues) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue x_or_not_x = fb.Or(x, fb.Not(x));
  BValue y_and_not_y = fb.And(y, fb.Not(y));
  fb.Concat({x_or_not_x, y_and_not_y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal(0b11110000));
}

TEST_F(BddSimplificationPassTest, ReplaceKnownPrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(9));
  fb.And(x, fb.Concat({fb.Literal(UBits(0, 7)), y}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(0), m::BitSlice(m::And())));
}

TEST_F(BddSimplificationPassTest, ReplaceKnownSuffix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(31));
  fb.Or(x, fb.Concat({y, fb.Literal(UBits(1, 1))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Or()), m::Literal(1)));
}

TEST_F(BddSimplificationPassTest, KnownSuffixButNotReplaced) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  // The suffix (least-significant bits) of the expression is known the
  // expression is not simplified because the "simplification" is the same as
  // the expression itself (concat of a literal).
  fb.Concat({x, fb.Literal(UBits(123, 10))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));

  EXPECT_THAT(f->return_value(), m::Concat(m::Param("x"), m::Literal(123)));
}

TEST_F(BddSimplificationPassTest, RemoveRedundantOneHot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 8)));
  BValue x_gt_123 = fb.UGt(x, fb.Literal(UBits(123, 8)));
  fb.OneHot(fb.Concat({x_eq_0, x_eq_42, x_gt_123}), LsbOrMsb::kLsb);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::Eq(), m::Concat()));
}

TEST_F(BddSimplificationPassTest, RemoveRedundantInputsToAnd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 8)));
  BValue x_ne_0 = fb.Not(x_eq_0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x_ne_0, x_eq_42)));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Literal(1), m::Eq(m::Param("x"), m::Literal(42))));
}

TEST_F(BddSimplificationPassTest, RemoveRedundantInputsToNand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 8)));
  BValue x_ne_0 = fb.Not(x_eq_0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Nand(x_ne_0, x_eq_42)));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Nand(m::Literal(1), m::Eq(m::Param("x"), m::Literal(42))));
}

TEST_F(BddSimplificationPassTest, RemoveRedundantInputsToOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 8)));
  BValue x_ne_0 = fb.Not(x_eq_0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or(x_ne_0, x_eq_42)));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Or(m::Not(m::Eq(m::Param("x"), m::Literal(0))), m::Literal(0)));
}

TEST_F(BddSimplificationPassTest, RemoveRedundantInputsToNor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 8)));
  BValue x_ne_0 = fb.Not(x_eq_0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Nor(x_ne_0, x_eq_42)));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Nor(m::Not(m::Eq(m::Param("x"), m::Literal(0))), m::Literal(0)));
}

TEST_F(BddSimplificationPassTest, RemoveRedundantPrioritySelectCases) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue a = fb.Param("a", p->GetBitsType(24));
  BValue b = fb.Param("b", p->GetBitsType(24));
  BValue c = fb.Param("c", p->GetBitsType(24));
  BValue d = fb.Param("d", p->GetBitsType(24));
  BValue x_ge_5 = fb.UGe(x, fb.Literal(UBits(5, 8)));
  BValue x_le_42 = fb.ULe(x, fb.Literal(UBits(42, 8)));
  BValue x_eq_8 = fb.Eq(x, fb.Literal(UBits(8, 8)));
  fb.PrioritySelect(fb.Concat({x_eq_8, x_ge_5, x_le_42}), /*cases=*/{a, b, c},
                    /*default_value=*/d);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::BitSlice(m::Concat()), {m::Param("a")},
                                m::Param("b")));
}

TEST_F(BddSimplificationPassTest, PreserveNonRedundantPrioritySelectCases) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue a = fb.Param("a", p->GetBitsType(24));
  BValue b = fb.Param("b", p->GetBitsType(24));
  BValue c = fb.Param("c", p->GetBitsType(24));
  BValue d = fb.Param("d", p->GetBitsType(24));
  BValue x_ge_5 = fb.UGe(x, fb.Literal(UBits(5, 8)));
  BValue x_lt_3 = fb.ULt(x, fb.Literal(UBits(3, 8)));
  BValue x_eq_3 = fb.Eq(x, fb.Literal(UBits(3, 8)));
  fb.PrioritySelect(fb.Concat({x_eq_3, x_ge_5, x_lt_3}), /*cases=*/{a, b, c},
                    /*default_value=*/d);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BddSimplificationPassTest, ConvertTwoWayOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], x: bits[32], y: bits[32]) -> bits[32] {
       not.1: bits[1] = not(p)
       concat.2: bits[2] = concat(p, not.1)
       ret one_hot_sel.3: bits[32] = one_hot_sel(concat.2, cases=[x, y])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::BitSlice(), /*cases=*/{
                                               m::Param("y"), m::Param("x")}));
}

TEST_F(BddSimplificationPassTest, SelectChainOneHot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 3)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 3)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 3)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 3)));
  BValue pred4 = fb.Eq(s, fb.Literal(UBits(4, 3)));
  auto param = [&](std::string_view s) {
    return fb.Param(s, p->GetBitsType(8));
  };
  fb.Select(
      pred4, param("x4"),
      fb.Select(
          pred3, param("x3"),
          fb.Select(pred2, param("x2"),
                    fb.Select(pred1, param("x1"),
                              fb.Select(pred0, param("x0"), param("y"))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                                m::Eq(m::Param("s"), m::Literal(3)),
                                m::Eq(m::Param("s"), m::Literal(2)),
                                m::Eq(m::Param("s"), m::Literal(1)),
                                m::Eq(m::Param("s"), m::Literal(0)), m::Nor()),
                      {m::Param("y"), m::Param("x0"), m::Param("x1"),
                       m::Param("x2"), m::Param("x3"), m::Param("x4")}));
}

TEST_F(BddSimplificationPassTest, SelectChainOneHotTooShort) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 2)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 2)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 2)));
  auto param = [&](std::string_view s) {
    return fb.Param(s, p->GetBitsType(8));
  };
  fb.Select(
      pred2, param("x2"),
      fb.Select(pred1, param("x1"), fb.Select(pred0, param("x0"), param("y"))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Select());
}

TEST_F(BddSimplificationPassTest, SelectChainOneHotArray) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 2)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 2)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 2)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 2)));
  auto param = [&](std::string_view s) {
    return fb.Param(s, p->GetArrayType(8, p->GetBitsType(32)));
  };
  fb.Select(pred3, param("x3"),
            fb.Select(pred2, param("x2"),
                      fb.Select(pred1, param("x1"),
                                fb.Select(pred0, param("x0"), param("y")))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // Transform was not applied to an array type.
  // This should not be done because it's not supported in codegen.
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BddSimplificationPassTest, SelectChainOneHotOrZeroSelectors) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(8));
  BValue pred0 = fb.UGt(s, fb.Literal(UBits(42, 8)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(11, 8)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(12, 8)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(13, 8)));
  BValue pred4 = fb.ULt(s, fb.Literal(UBits(7, 8)));
  auto param = [&](std::string_view s) {
    return fb.Param(s, p->GetBitsType(8));
  };
  fb.Select(
      pred4, param("x4"),
      fb.Select(
          pred3, param("x3"),
          fb.Select(pred2, param("x2"),
                    fb.Select(pred1, param("x1"),
                              fb.Select(pred0, param("x0"), param("y"))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(
          m::Concat(m::ULt(m::Param("s"), m::Literal(7)),
                    m::Eq(m::Param("s"), m::Literal(13)),
                    m::Eq(m::Param("s"), m::Literal(12)),
                    m::Eq(m::Param("s"), m::Literal(11)),
                    m::UGt(m::Param("s"), m::Literal(42)),
                    m::Nor(m::ULt(), m::Eq(), m::Eq(), m::Eq(), m::UGt())),
          {m::Param("y"), m::Param("x0"), m::Param("x1"), m::Param("x2"),
           m::Param("x3"), m::Param("x4")}));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest, DISABLED_OneHotMsbTypical) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue encode = fb.Encode(hot);
  BValue literal_zero3 = fb.Literal(UBits(0, 3));
  BValue literal_comp = fb.Literal(UBits(7, 3));
  BValue eq_test = fb.Eq(encode, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero3, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::Encode(
                  m::Concat(m::Literal(UBits(0, 1)),
                            m::BitSlice(m::OneHot(m::Param("input")), 0, 7)))));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest, DISABLED_OneHotMsbAlternateForm) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue hot_msb = fb.BitSlice(hot, hot.node()->BitCountOrDie() - 1, 1);
  BValue encode = fb.Encode(hot);
  BValue literal_zero3 = fb.Literal(UBits(0, 3));
  BValue literal_comp = fb.Literal(UBits(1, 1));
  BValue eq_test = fb.Eq(hot_msb, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero3, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::Encode(
                  m::Concat(m::Literal(UBits(0, 1)),
                            m::BitSlice(m::OneHot(m::Param("input")), 0, 7)))));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest,
       DISABLED_OneHotMsbRequireFullBddToAnalyzeOneMsbCase) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue reduce = fb.OrReduce(input);
  BValue extend = fb.SignExtend(reduce, 8);
  fb.And(hot, extend);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 1)),
                        m::BitSlice(m::OneHot(m::Param("input")), 0, 7)));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest, DISABLED_OneHotMsbMsbLeaks) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue hot_msb = fb.BitSlice(hot, hot.node()->BitCountOrDie() - 1, 1);
  BValue encode = fb.Encode(hot);
  BValue literal_zero3 = fb.Literal(UBits(0, 3));
  BValue literal_comp = fb.Literal(UBits(1, 1));
  BValue eq_test = fb.Eq(hot_msb, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero3, encode);
  fb.Concat({hot_msb, select});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest, DISABLED_OneHotMsbNonMsbOneComparison) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue encode = fb.Encode(hot);
  BValue literal_zero3 = fb.Literal(UBits(0, 3));
  BValue literal_comp = fb.Literal(UBits(4, 3));
  BValue eq_test = fb.Eq(encode, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero3, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest, DISABLED_OneHotMsbNonZeroReplacementValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue encode = fb.Encode(hot);
  BValue literal_zero3 = fb.Literal(UBits(4, 3));
  BValue literal_comp = fb.Literal(UBits(7, 3));
  BValue eq_test = fb.Eq(encode, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero3, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest, DISABLED_OneHotMsbNoRecursion) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue encode = fb.Encode(hot);
  BValue literal_zero3 = fb.Literal(UBits(0, 3));
  BValue literal_comp = fb.Literal(UBits(7, 3));
  BValue eq_test = fb.Eq(encode, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero3, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::Encode(
                  m::Concat(m::Literal(UBits(0, 1)),
                            m::BitSlice(m::OneHot(m::Param("input")), 0, 7)))));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest,
       DISABLED_OneHotMsbNoRecursionExistingSliceIncludesMsb) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue hot_slice = fb.BitSlice(hot, 4, 4);
  BValue encode = fb.Encode(hot_slice);
  BValue literal_zero2 = fb.Literal(UBits(0, 2));
  BValue literal_comp = fb.Literal(UBits(3, 2));
  BValue eq_test = fb.Eq(encode, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero2, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::Encode(
                  m::Concat(m::Literal(UBits(0, 1)),
                            m::BitSlice(m::OneHot(m::Param("input")), 4, 3)))));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest,
       DISABLED_OneHotMsbNoRecursionExistingSliceExcludesMsb) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(7));
  BValue hot = fb.OneHot(input, LsbOrMsb::kLsb);
  BValue hot_slice = fb.BitSlice(hot, 0, 4);
  BValue encode = fb.Encode(hot_slice);
  BValue literal_zero2 = fb.Literal(UBits(0, 2));
  BValue literal_comp = fb.Literal(UBits(3, 2));
  BValue eq_test = fb.Eq(encode, literal_comp);
  BValue select = fb.Select(eq_test, literal_zero2, encode);
  // Extra instruction just to add another post-dominator.
  fb.Not(select);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(
    BddSimplificationPassTest,
    DISABLED_OneHotMsbPostponeOneHotNativeOneHotDetectionUntilAfterOneHotMsb) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(2));
  BValue one_val = fb.Literal(UBits(1, 2));
  BValue three_val = fb.Literal(UBits(3, 2));
  BValue one_eq = fb.Eq(one_val, input);
  BValue three_eq = fb.Eq(three_val, input);
  BValue cat = fb.Concat({one_eq, three_eq});
  BValue hot = fb.OneHot(cat, LsbOrMsb::kLsb);
  BValue encode = fb.Encode(hot);
  BValue literal_zero2 = fb.Literal(UBits(0, 2));
  BValue literal_comp = fb.Literal(UBits(2, 2));
  BValue eq_test = fb.Eq(encode, literal_comp);
  fb.Select(eq_test, literal_zero2, encode);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  ASSERT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Encode(m::Concat(
          m::Literal(UBits(0, 1)),
          m::Concat(m::Eq(m::Literal(UBits(1, 2)), m::Param("input")),
                    m::Eq(m::Literal(UBits(3, 2)), m::Param("input"))))));
}

// TODO(https://github.com/google/xls/issues/423) 2021/04/05 Renable when
// sensitivity analysis added.
TEST_F(BddSimplificationPassTest,
       DISABLED_BddNotStaleAfterOneHotSimplification) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("input", p->GetBitsType(1));
  BValue literal_zero2 = fb.Literal(UBits(0, 2));
  BValue concat = fb.Concat({input, literal_zero2});
  BValue hot = fb.OneHot(concat, LsbOrMsb::kLsb);
  BValue encode = fb.Encode(hot);
  fb.XorReduce(encode);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("input"));
}

}  // namespace
}  // namespace xls

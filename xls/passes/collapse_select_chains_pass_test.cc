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

#include "xls/passes/collapse_select_chains_pass.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
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
#include "xls/solvers/ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::ScopedVerifyEquivalence;

class CollapseSelectChainsPassTest : public IrTestBase {
 protected:
  CollapseSelectChainsPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f, int64_t opt_level = kMaxOptLevel) {
    PassResults results;
    OptimizationContext context;
    return CollapseSelectChainsPass().RunOnFunctionBase(
        f, OptimizationPassOptions().WithOptLevel(opt_level), &results,
        context);
  }
};

TEST_F(CollapseSelectChainsPassTest, SelectChainOneHot) {
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
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                                m::Eq(m::Param("s"), m::Literal(3)),
                                m::Eq(m::Param("s"), m::Literal(2)),
                                m::Eq(m::Param("s"), m::Literal(1)),
                                m::Eq(m::Param("s"), m::Literal(0)), m::And()),
                      {m::Param("y"), m::Param("x0"), m::Param("x1"),
                       m::Param("x2"), m::Param("x3"), m::Param("x4")}));
}

TEST_F(CollapseSelectChainsPassTest, SelectChainOneHotTooShort) {
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
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Select());
}

TEST_F(CollapseSelectChainsPassTest, SelectChainOneHotArray) {
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
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(CollapseSelectChainsPassTest, SelectChainOneHotOrZeroSelectors) {
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
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(
          m::Concat(m::ULt(m::Param("s"), m::Literal(7)),
                    m::Eq(m::Param("s"), m::Literal(13)),
                    m::Eq(m::Param("s"), m::Literal(12)),
                    m::Eq(m::Param("s"), m::Literal(11)),
                    m::UGt(m::Param("s"), m::Literal(42)),
                    m::And(m::Eq(), m::Eq(), m::Eq(), m::Eq(), m::Eq())),
          {m::Param("y"), m::Param("x0"), m::Param("x1"), m::Param("x2"),
           m::Param("x3"), m::Param("x4")}));
}

TEST_F(CollapseSelectChainsPassTest, SelectChainWithPrioritySelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 3)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 3)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 3)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 3)));
  BValue pred4 = fb.Eq(s, fb.Literal(UBits(4, 3)));
  auto param = [&](std::string_view name) {
    return fb.Param(name, p->GetBitsType(8));
  };
  // PrioritySelect with 2 cases at the bottom of the chain.
  BValue prio = fb.PrioritySelect(fb.Concat({pred1, pred0}),
                                  {param("x0"), param("x1")}, param("y"));
  fb.Select(pred4, param("x4"),
            fb.Select(pred3, param("x3"), fb.Select(pred2, param("x2"), prio)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                                m::Eq(m::Param("s"), m::Literal(3)),
                                m::Eq(m::Param("s"), m::Literal(2)),
                                m::Concat(m::Eq(m::Param("s"), m::Literal(1)),
                                          m::Eq(m::Param("s"), m::Literal(0))),
                                m::And()),
                      {m::Param("y"), m::Param("x0"), m::Param("x1"),
                       m::Param("x2"), m::Param("x3"), m::Param("x4")}));
}

TEST_F(CollapseSelectChainsPassTest, SelectChainWithOneHotSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 3)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 3)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 3)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 3)));
  BValue pred4 = fb.Eq(s, fb.Literal(UBits(4, 3)));
  auto param = [&](std::string_view name) {
    return fb.Param(name, p->GetBitsType(8));
  };
  BValue ohs =
      fb.OneHotSelect(fb.Concat({pred1, pred0}), {param("x0"), param("x1")});
  fb.Select(pred4, param("x4"),
            fb.Select(pred3, param("x3"), fb.Select(pred2, param("x2"), ohs)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                                m::Eq(m::Param("s"), m::Literal(3)),
                                m::Eq(m::Param("s"), m::Literal(2)),
                                m::Concat(m::Eq(m::Param("s"), m::Literal(1)),
                                          m::Eq(m::Param("s"), m::Literal(0)))),
                      {m::Param("x0"), m::Param("x1"), m::Param("x2"),
                       m::Param("x3"), m::Param("x4")}));
}

TEST_F(CollapseSelectChainsPassTest, PrioritySelectChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 3)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 3)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 3)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 3)));
  BValue pred4 = fb.Eq(s, fb.Literal(UBits(4, 3)));
  auto param = [&](std::string_view name) {
    return fb.Param(name, p->GetBitsType(8));
  };
  BValue prio1 = fb.PrioritySelect(fb.Concat({pred1, pred0}),
                                   {param("x0"), param("x1")}, param("y"));
  fb.PrioritySelect(fb.Concat({pred4, pred3, pred2}),
                    {param("x2"), param("x3"), param("x4")}, prio1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                                          m::Eq(m::Param("s"), m::Literal(3)),
                                          m::Eq(m::Param("s"), m::Literal(2))),
                                m::Concat(m::Eq(m::Param("s"), m::Literal(1)),
                                          m::Eq(m::Param("s"), m::Literal(0))),
                                m::And()),
                      {m::Param("y"), m::Param("x0"), m::Param("x1"),
                       m::Param("x2"), m::Param("x3"), m::Param("x4")}));
}

TEST_F(CollapseSelectChainsPassTest, SelectChainWithDefaultValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 3)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 3)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 3)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 3)));
  BValue pred4 = fb.Eq(s, fb.Literal(UBits(4, 3)));
  auto param = [&](std::string_view name) {
    return fb.Param(name, p->GetBitsType(8));
  };
  // Build chain using 1-case select with default value:
  // sel(pred, cases=[fallthrough], default=x) means:
  // if pred==1 (default) => x, if pred==0 (case 0) => fallthrough.
  BValue s0 = fb.Select(pred0, std::vector<BValue>{param("y")}, param("x0"));
  BValue s1 = fb.Select(pred1, std::vector<BValue>{s0}, param("x1"));
  BValue s2 = fb.Select(pred2, std::vector<BValue>{s1}, param("x2"));
  BValue s3 = fb.Select(pred3, std::vector<BValue>{s2}, param("x3"));
  fb.Select(pred4, std::vector<BValue>{s3}, param("x4"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                                m::Eq(m::Param("s"), m::Literal(3)),
                                m::Eq(m::Param("s"), m::Literal(2)),
                                m::Eq(m::Param("s"), m::Literal(1)),
                                m::Eq(m::Param("s"), m::Literal(0)), m::And()),
                      {m::Param("y"), m::Param("x0"), m::Param("x1"),
                       m::Param("x2"), m::Param("x3"), m::Param("x4")}));
}

TEST_F(CollapseSelectChainsPassTest, SelectChainHeterogeneousMixed) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue pred0 = fb.Eq(s, fb.Literal(UBits(0, 3)));
  BValue pred1 = fb.Eq(s, fb.Literal(UBits(1, 3)));
  BValue pred2 = fb.Eq(s, fb.Literal(UBits(2, 3)));
  BValue pred3 = fb.Eq(s, fb.Literal(UBits(3, 3)));
  BValue pred4 = fb.Eq(s, fb.Literal(UBits(4, 3)));
  BValue pred5 = fb.Eq(s, fb.Literal(UBits(5, 3)));
  BValue pred6 = fb.Eq(s, fb.Literal(UBits(6, 3)));
  auto param = [&](std::string_view name) {
    return fb.Param(name, p->GetBitsType(8));
  };
  // Leaf: OneHotSelect (pred0 -> x0, pred1 -> x1)
  BValue ohs =
      fb.OneHotSelect(fb.Concat({pred1, pred0}), {param("x0"), param("x1")});
  // Child 3: PrioritySelect with 1 case (pred2 -> x2, else ohs)
  BValue ps1 = fb.PrioritySelect(pred2, {param("x2")}, ohs);
  // Child 2: PrioritySelect with 2 cases (pred3 -> x3, pred4 -> x4, else ps1)
  BValue ps2 = fb.PrioritySelect(fb.Concat({pred4, pred3}),
                                 {param("x3"), param("x4")}, ps1);
  // Child 1: Select with 1 case + default (pred5 == 1 -> x5, pred5 == 0 -> ps2)
  BValue sel1 = fb.Select(pred5, std::vector<BValue>{ps2}, param("x5"));
  // Root: Select with 2 cases (pred6 == 1 -> x6, pred6 == 0 -> sel1)
  fb.Select(pred6, param("x6"), sel1);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(
          m::Concat(m::Eq(m::Param("s"), m::Literal(6)),
                    m::Eq(m::Param("s"), m::Literal(5)),
                    m::Concat(m::Eq(m::Param("s"), m::Literal(4)),
                              m::Eq(m::Param("s"), m::Literal(3))),
                    m::Eq(m::Param("s"), m::Literal(2)),
                    m::Concat(m::Eq(m::Param("s"), m::Literal(1)),
                              m::Eq(m::Param("s"), m::Literal(0)))),
          {m::Param("x0"), m::Param("x1"), m::Param("x2"), m::Param("x3"),
           m::Param("x4"), m::Param("x5"), m::Param("x6")}));
}

void IrFuzzCollapseSelectChains(FuzzPackageWithArgs fuzz_package_with_args) {
  CollapseSelectChainsPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzCollapseSelectChains)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls

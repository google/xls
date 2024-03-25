// Copyright 2023 The XLS Authors
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

#include "xls/passes/predicate_dominator_analysis.h"

#include <cstdint>
#include <memory>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/benchmark_support.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/predicate_state.h"

namespace xls {

namespace {
class PredicateDominatorAnalysisTest : public IrTestBase {};

TEST_F(PredicateDominatorAnalysisTest, NoPredicates) {
  // No predicates everything goes to base predicate state.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue w = fb.Param("w", p->GetBitsType(8));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Not(w);
  BValue z = fb.Not(x);
  BValue wxyz = fb.Add(fb.Add(w, x), fb.Add(y, z));

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  EXPECT_EQ(analysis.GetSingleNearestPredicate(w.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(z.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxyz.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // dslx:
  // ```
  // fn foo(s: u8, x: u8, y: u8, z: u8, d: u8) {
  //   ~(match ~s {
  //     case 0 => ~x,
  //     case 1 => ~y,
  //     case 2 => ~z,
  //     _ => ~d,
  //   })
  // }
  // Selector
  BValue s = fb.Param("s", p->GetBitsType(8));
  BValue ss = fb.Not(s);
  // case 0
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue xx = fb.Not(x);
  // case 1
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue yy = fb.Not(y);
  // case 2
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue zz = fb.Not(z);
  // case _
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue dd = fb.Not(d);
  // result
  BValue wxyz = fb.Select(ss, {xx, yy, zz}, dd);
  BValue nwxyz = fb.Not(wxyz);

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  auto* select = wxyz.node()->As<Select>();
  // condition
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(ss.node()), PredicateState());
  // case 0
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(select, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(xx.node()),
            PredicateState(select, 0));
  // case 1
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()),
            PredicateState(select, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(yy.node()),
            PredicateState(select, 1));
  // case 2
  EXPECT_EQ(analysis.GetSingleNearestPredicate(z.node()),
            PredicateState(select, 2));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(zz.node()),
            PredicateState(select, 2));
  // case _
  EXPECT_EQ(analysis.GetSingleNearestPredicate(d.node()),
            PredicateState(select, PredicateState::kDefaultArm));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(dd.node()),
            PredicateState(select, PredicateState::kDefaultArm));
  // match
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxyz.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(nwxyz.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, ValueInSelectorAndArm) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let v1 = match x { 0 => a, 1 => x, _ => b };
  // let v2 = match s { 0 => v1, 1 => c };
  // v2
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue c = fb.Param("c", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue v1 = fb.Select(x, {a, x}, b);
  BValue v2 = fb.Select(s, {v1, c});

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  auto s_v1 = v1.node()->As<Select>();
  auto s_v2 = v2.node()->As<Select>();
  EXPECT_EQ(analysis.GetSingleNearestPredicate(v2.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(v1.node()),
            PredicateState(s_v2, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(c.node()),
            PredicateState(s_v2, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(b.node()),
            PredicateState(s_v1, DefaultArm{}));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(a.node()),
            PredicateState(s_v1, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_v2, 0));
}

TEST_F(PredicateDominatorAnalysisTest, ValueInSelectorAndDefaultArm) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let v1 = match x { 0 => a, 1 => b, _ => x };
  // let v2 = match s { 0 => v1, 1 => c };
  // v2
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue c = fb.Param("c", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue v1 = fb.Select(x, {a, b}, x);
  BValue v2 = fb.Select(s, {v1, c});

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  auto s_v1 = v1.node()->As<Select>();
  auto s_v2 = v2.node()->As<Select>();
  EXPECT_EQ(analysis.GetSingleNearestPredicate(v2.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(v1.node()),
            PredicateState(s_v2, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(c.node()),
            PredicateState(s_v2, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(b.node()),
            PredicateState(s_v1, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(a.node()),
            PredicateState(s_v1, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_v2, 0));
}

TEST_F(PredicateDominatorAnalysisTest, MultipleIndependentSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let wx = match s1 { 0 => w, 1 => x };
  // let yz = match s2 { 0 => y, 1 => z };
  // wx + yz
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue s2 = fb.Param("s2", p->GetBitsType(1));
  BValue w = fb.Param("w", p->GetBitsType(8));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue wx = fb.Select(s1, {w, x});
  BValue yz = fb.Select(s2, {y, z});
  BValue wxyz = fb.Add(wx, yz);

  auto s_wx = wx.node()->As<Select>();
  auto s_yz = yz.node()->As<Select>();

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  EXPECT_EQ(analysis.GetSingleNearestPredicate(s1.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s2.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wx.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(yz.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(w.node()),
            PredicateState(s_wx, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_wx, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()),
            PredicateState(s_yz, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(z.node()),
            PredicateState(s_yz, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxyz.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, NestedSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let wx = match s1 { 0 => w, 1 => x }; -- whole select guarded by wxy[b0]
  // let wxy = match s2 { 0 => wx, 1 => y }; -- guarded by wxyz[b0]
  // let wxyz = match s3 { 0 => wxy, 1 => z };
  // wxyz
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue s2 = fb.Param("s2", p->GetBitsType(1));
  BValue s3 = fb.Param("s3", p->GetBitsType(1));
  BValue w = fb.Param("w", p->GetBitsType(8));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue wx = fb.Select(s1, {w, x});
  BValue wxy = fb.Select(s2, {wx, y});
  BValue wxyz = fb.Select(s3, {wxy, z});

  auto* s_wx = wx.node()->As<Select>();
  auto* s_wxy = wxy.node()->As<Select>();
  auto* s_wxyz = wxyz.node()->As<Select>();

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  EXPECT_EQ(analysis.GetSingleNearestPredicate(s1.node()),
            PredicateState(s_wxy, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s2.node()),
            PredicateState(s_wxyz, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s3.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(w.node()),
            PredicateState(s_wx, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_wx, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wx.node()),
            PredicateState(s_wxy, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()),
            PredicateState(s_wxy, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxy.node()),
            PredicateState(s_wxyz, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(z.node()),
            PredicateState(s_wxyz, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxyz.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, SimpleCovering) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // x + match s1 { 0 => y, 1 => x }
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue yx = fb.Select(s, {y, x});
  BValue xyx = fb.Add(x, yx);

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  auto* select = yx.node()->As<Select>();
  // condition
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()),
            PredicateState(select, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(xyx.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(yx.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, DisjointCovering) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let bt = a + t
  // let yt = x + t
  // let abt = match s1 { 0 => a, 1 => bt };
  // let xyt = match s2 { 0 => x, 1 => yt };
  // abt + xyt
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue s2 = fb.Param("s2", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue t = fb.Param("t", p->GetBitsType(8));
  BValue bt = fb.Add(b, t);
  BValue yt = fb.Add(y, t);
  BValue abt = fb.Select(s1, {a, bt});
  BValue xyt = fb.Select(s2, {x, yt});

  auto* s_abt = abt.node()->As<Select>();
  auto* s_xyt = xyt.node()->As<Select>();

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  EXPECT_EQ(analysis.GetSingleNearestPredicate(s1.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s2.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(a.node()),
            PredicateState(s_abt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(b.node()),
            PredicateState(s_abt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(bt.node()),
            PredicateState(s_abt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_xyt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()),
            PredicateState(s_xyt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(yt.node()),
            PredicateState(s_xyt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(t.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, NestedCovering) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let xt = x + t;
  // let zt = z + t;
  // let wxt = match s1 { 0 => w, 1 => xt };
  // let wxty = match s2 { 0 => wxt, 1 => y };
  // let wxtyzt = match s3 { 0 => wxty, 1 => zt };  -- t is on both sides
  // wxtyzt
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue s2 = fb.Param("s2", p->GetBitsType(1));
  BValue s3 = fb.Param("s3", p->GetBitsType(1));
  BValue w = fb.Param("w", p->GetBitsType(8));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue t = fb.Param("t", p->GetBitsType(8));
  BValue xt = fb.Add(x, t);
  BValue wxt = fb.Select(s1, {w, xt});
  BValue wxty = fb.Select(s2, {wxt, y});
  BValue zt = fb.Add(z, t);
  BValue wxtyzt = fb.Select(s3, {wxty, zt});

  auto* s_wxt = wxt.node()->As<Select>();
  auto* s_wxty = wxty.node()->As<Select>();
  auto* s_wxtyzt = wxtyzt.node()->As<Select>();

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  EXPECT_EQ(analysis.GetSingleNearestPredicate(s1.node()),
            PredicateState(s_wxty, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s2.node()),
            PredicateState(s_wxtyzt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s3.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(w.node()),
            PredicateState(s_wxt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(xt.node()),
            PredicateState(s_wxt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_wxt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxt.node()),
            PredicateState(s_wxty, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(y.node()),
            PredicateState(s_wxty, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxty.node()),
            PredicateState(s_wxtyzt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(zt.node()),
            PredicateState(s_wxtyzt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(z.node()),
            PredicateState(s_wxtyzt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(wxtyzt.node()),
            PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(t.node()), PredicateState());
}

TEST_F(PredicateDominatorAnalysisTest, NestedPartialDisjointCovering) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // let at = match s2 { 0 => a, 1 => t };
  // let bt = match s2 { 0 => b, 1 => t };
  // let atbt = match s3 { 0 => at, 1 => bt }; -- t on both sides
  // let xatbt = match s4 { 0 => x, 1 => atbt }; -- t on one side
  // xatbt
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue s2 = fb.Param("s2", p->GetBitsType(1));
  BValue s3 = fb.Param("s3", p->GetBitsType(1));
  BValue s4 = fb.Param("s4", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue t = fb.Param("t", p->GetBitsType(8));
  BValue at = fb.Select(s1, {a, t});
  BValue bt = fb.Select(s2, {b, t});
  BValue atbt = fb.Select(s3, {at, bt});
  BValue xatbt = fb.Select(s4, {x, atbt});

  auto* s_at = at.node()->As<Select>();
  auto* s_bt = bt.node()->As<Select>();
  auto* s_atbt = atbt.node()->As<Select>();
  auto* s_xatbt = xatbt.node()->As<Select>();

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  auto analysis = PredicateDominatorAnalysis::Run(f);

  EXPECT_EQ(analysis.GetSingleNearestPredicate(s1.node()),
            PredicateState(s_atbt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s2.node()),
            PredicateState(s_atbt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s3.node()),
            PredicateState(s_xatbt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(s4.node()), PredicateState());
  EXPECT_EQ(analysis.GetSingleNearestPredicate(x.node()),
            PredicateState(s_xatbt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(a.node()),
            PredicateState(s_at, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(b.node()),
            PredicateState(s_bt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(at.node()),
            PredicateState(s_atbt, 0));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(bt.node()),
            PredicateState(s_atbt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(atbt.node()),
            PredicateState(s_xatbt, 1));
  EXPECT_EQ(analysis.GetSingleNearestPredicate(t.node()),
            PredicateState(s_xatbt, 1));
}

// A balanced tree with all leaf nodes and all selector nodes fully independent
// of one another. NB They are all Literals. Binary trees with deep predicate
// chains is the near-worst-case for the analysis performance.
void BM_NoShareBalancedTree(benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("balanced_tree_pkg");
  benchmark_support::strategy::DistinctLiteral selectors(UBits(1, 1));
  benchmark_support::strategy::DistinctLiteral leaf(UBits(42, 8));
  benchmark_support::strategy::CaseSelect csts(selectors);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, benchmark_support::GenerateBalancedTree(
                                        p.get(), /*depth=*/state.range(0),
                                        /*fan_out=*/2, csts, leaf));
  for (auto _ : state) {
    auto v = PredicateDominatorAnalysis::Run(f);
    benchmark::DoNotOptimize(v);
  }
}

// A balanced tree with all leaf nodes fully independent of
// one another. All selectors share a single Param node. Binary trees with deep
// predicate chains is the near-worst-case for the analysis performance.
void BM_ShareSelectorBalancedTree(benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("balanced_tree_pkg");
  benchmark_support::strategy::SharedLiteral selectors(UBits(1, 1));
  benchmark_support::strategy::DistinctLiteral leaf(UBits(42, 8));
  benchmark_support::strategy::CaseSelect csts(selectors);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, benchmark_support::GenerateBalancedTree(
                                        p.get(), /*depth=*/state.range(0),
                                        /*fan_out=*/2, csts, leaf));
  for (auto _ : state) {
    auto v = PredicateDominatorAnalysis::Run(f);
    benchmark::DoNotOptimize(v);
  }
}

// A balanced tree with all selector nodes fully independent of
// one another. All leaf nodes share a single Param node. Binary trees with deep
// predicate chains is the near-worst-case for the analysis performance.
void BM_ShareReturnBalancedTree(benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("balanced_tree_pkg");
  benchmark_support::strategy::DistinctLiteral selectors(UBits(1, 1));
  benchmark_support::strategy::SharedLiteral leaf(UBits(42, 8));
  benchmark_support::strategy::CaseSelect csts(selectors);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, benchmark_support::GenerateBalancedTree(
                                        p.get(), /*depth=*/state.range(0),
                                        /*fan_out=*/2, csts, leaf));
  for (auto _ : state) {
    auto v = PredicateDominatorAnalysis::Run(f);
    benchmark::DoNotOptimize(v);
  }
}

// A balanced tree with all leaf and selector nodes sharing a param node each.
// Binary trees with deep predicate chains is the near-worst-case for the
// analysis performance.
void BM_ShareAllBalancedTree(benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("balanced_tree_pkg");
  benchmark_support::strategy::SharedLiteral selectors(UBits(1, 1));
  benchmark_support::strategy::SharedLiteral leaf(UBits(42, 8));
  benchmark_support::strategy::CaseSelect csts(selectors);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, benchmark_support::GenerateBalancedTree(
                                        p.get(), /*depth=*/state.range(0),
                                        /*fan_out=*/2, csts, leaf));
  for (auto _ : state) {
    auto v = PredicateDominatorAnalysis::Run(f);
    benchmark::DoNotOptimize(v);
  }
}

BENCHMARK(BM_NoShareBalancedTree)->DenseRange(2, 20, 2);
BENCHMARK(BM_ShareReturnBalancedTree)->DenseRange(2, 20, 2);
BENCHMARK(BM_ShareSelectorBalancedTree)->DenseRange(2, 20, 2);
BENCHMARK(BM_ShareAllBalancedTree)->DenseRange(2, 20, 2);

}  // namespace
}  // namespace xls

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

#include "xls/passes/cached_state_element_query_engine.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/proc_state_range_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

class CachedStateElementQueryEngineTest : public IrTestBase {};

TEST_F(CachedStateElementQueryEngineTest, ComputePopulatesStateElementBounds) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  BValue res = pb.Add(st, pb.Literal(UBits(12, 32)));
  pb.Next(st, pb.ZeroExtend(
                  pb.Add(pb.Literal(UBits(1, 3)), pb.BitSlice(st, 0, 3)), 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  OptimizationContext ctx;
  auto* partial_qe =
      CachedStateElementQueryEngine::FromContext<PartialInfoQueryEngine>(ctx,
                                                                         proc);
  auto* ternary_qe =
      CachedStateElementQueryEngine::FromContext<LazyTernaryQueryEngine>(ctx,
                                                                         proc);

  EXPECT_FALSE(partial_qe->has_givens());
  EXPECT_FALSE(ternary_qe->has_givens());
  EXPECT_EQ(partial_qe->GetIntervals(st.node()).Get({}),
            IntervalSet::Maximal(32));
  EXPECT_THAT(ternary_qe->GetTernary(st.node()),
              testing::Optional(LttIs<TernaryVector>(testing::ElementsAre(
                  TernaryIs("0bXXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX")))));

  XLS_ASSERT_OK(partial_qe->Compute());
  XLS_ASSERT_OK(ternary_qe->Compute());

  EXPECT_TRUE(partial_qe->has_givens());
  EXPECT_TRUE(ternary_qe->has_givens());
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 7]]");
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(res.node())),
            "[[12, 19]]");
  EXPECT_THAT(ternary_qe->GetTernary(st.node()),
              testing::Optional(LttIs<TernaryVector>(testing::ElementsAre(
                  TernaryIs("0b0000_0000_0000_0000_0000_0000_0000_0XXX")))));

  // Re-populating a UnionQueryEngine containing the cached engine on the same
  // proc must preserve the cached state element bounds.
  auto union_qe = UnionQueryEngine::Of(partial_qe);
  XLS_ASSERT_OK_AND_ASSIGN(ReachedFixpoint rf, union_qe.Populate(proc));
  EXPECT_EQ(rf, ReachedFixpoint::Unchanged);
  EXPECT_TRUE(partial_qe->has_givens());
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 7]]");
}

TEST_F(CachedStateElementQueryEngineTest, ComputeWithStateInfoAndClearGivens) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  BValue res = pb.Add(st, pb.Literal(UBits(12, 32)));
  pb.Next(st, pb.ZeroExtend(
                  pb.Add(pb.Literal(UBits(1, 3)), pb.BitSlice(st, 0, 3)), 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  OptimizationContext ctx;
  auto* partial_qe =
      CachedStateElementQueryEngine::FromContext<PartialInfoQueryEngine>(ctx,
                                                                         proc);

  ProcStateRangeQueryEngine range_qe;
  XLS_ASSERT_OK(range_qe.Populate(proc).status());

  XLS_ASSERT_OK(partial_qe->ComputeWithStateInfo(range_qe));
  EXPECT_TRUE(partial_qe->has_givens());
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(res.node())),
            "[[12, 19]]");
  EXPECT_EQ(partial_qe->GetIntervals(res.node()).Get({}).LowerBound(),
            UBits(12, 32));
  EXPECT_EQ(partial_qe->GetIntervals(res.node()).Get({}).UpperBound(),
            UBits(19, 32));

  XLS_ASSERT_OK(partial_qe->ClearGivens());
  EXPECT_FALSE(partial_qe->has_givens());
  EXPECT_EQ(partial_qe->GetIntervals(st.node()).Get({}),
            IntervalSet::Maximal(32));
}

TEST_F(CachedStateElementQueryEngineTest,
       FromContextAndComputeOnNonProcFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  fb.Add(x, fb.Literal(UBits(1, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  OptimizationContext ctx;
  QueryEngine* from_ctx =
      CachedStateElementQueryEngine::FromContext<LazyTernaryQueryEngine>(
          ctx, static_cast<FunctionBase*>(f));
  EXPECT_EQ(from_ctx, ctx.SharedQueryEngine<LazyTernaryQueryEngine>(f));

  CachedStateElementQueryEngine direct_qe(
      ctx.SharedQueryEngine<LazyTernaryQueryEngine>(f));
  XLS_ASSERT_OK(direct_qe.Populate(f).status());
  XLS_ASSERT_OK(direct_qe.Compute());
  EXPECT_FALSE(direct_qe.has_givens());
}

}  // namespace
}  // namespace xls

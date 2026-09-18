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

#include "xls/passes/invalidate_cached_proc_state_info_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/passes/cached_state_element_query_engine.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/repopulate_cached_proc_state_info_pass.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class InvalidateCachedProcStateInfoPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> RunInvalidate(Package* p, OptimizationContext& ctx) {
    PassResults results;
    InvalidateCachedProcStateInfoPass pass;
    ScopedRecordIr sri(p);
    return pass.Run(p, {}, &results, ctx);
  }
};

TEST_F(InvalidateCachedProcStateInfoPassTest,
       InvalidatesPopulatedPartialInfoAndLazyTernaryQueryEngines) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  pb.Next(st, pb.ZeroExtend(
                  pb.Add(pb.Literal(UBits(1, 3)), pb.BitSlice(st, 0, 3)), 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  OptimizationContext ctx;
  PassResults results;
  RepopulateCachedProcStateInfoPass repopulate;
  EXPECT_THAT(repopulate.Run(p.get(), {}, &results, ctx), IsOkAndHolds(false));

  auto* partial_qe =
      CachedStateElementQueryEngine::FromContext<PartialInfoQueryEngine>(ctx,
                                                                         proc);
  auto* ternary_qe =
      CachedStateElementQueryEngine::FromContext<LazyTernaryQueryEngine>(ctx,
                                                                         proc);
  EXPECT_TRUE(partial_qe->has_givens());
  EXPECT_TRUE(ternary_qe->has_givens());
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 7]]");
  EXPECT_THAT(ternary_qe->GetTernary(st.node()),
              testing::Optional(LttIs<TernaryVector>(testing::ElementsAre(
                  TernaryIs("0b0000_0000_0000_0000_0000_0000_0000_0XXX")))));

  EXPECT_THAT(RunInvalidate(p.get(), ctx), IsOkAndHolds(false));

  EXPECT_FALSE(partial_qe->has_givens());
  EXPECT_FALSE(ternary_qe->has_givens());
  EXPECT_EQ(partial_qe->GetIntervals(st.node()).Get({}),
            IntervalSet::Maximal(32));
  EXPECT_THAT(ternary_qe->GetTernary(st.node()),
              testing::Optional(LttIs<TernaryVector>(testing::ElementsAre(
                  TernaryIs("0bXXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX")))));
}

TEST_F(InvalidateCachedProcStateInfoPassTest, NoopWhenUninitialized) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  pb.Next(st, pb.ZeroExtend(
                  pb.Add(pb.Literal(UBits(1, 3)), pb.BitSlice(st, 0, 3)), 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  OptimizationContext ctx;
  EXPECT_THAT(RunInvalidate(p.get(), ctx), IsOkAndHolds(false));

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
}

TEST_F(InvalidateCachedProcStateInfoPassTest,
       InvalidateThenRepopulateCompoundPipeline) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  pb.Next(st, pb.ZeroExtend(
                  pb.Add(pb.Literal(UBits(1, 3)), pb.BitSlice(st, 0, 3)), 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  OptimizationContext ctx;
  PassResults results;

  OptimizationCompoundPass repopulate_then_invalidate(
      "repop_then_inval", "Repopulate then invalidate");
  repopulate_then_invalidate.Add<RepopulateCachedProcStateInfoPass>();
  repopulate_then_invalidate.Add<InvalidateCachedProcStateInfoPass>();
  EXPECT_THAT(repopulate_then_invalidate.Run(p.get(), {}, &results, ctx),
              IsOkAndHolds(false));

  auto* partial_qe =
      CachedStateElementQueryEngine::FromContext<PartialInfoQueryEngine>(ctx,
                                                                         proc);
  EXPECT_FALSE(partial_qe->has_givens());
  EXPECT_EQ(partial_qe->GetIntervals(st.node()).Get({}),
            IntervalSet::Maximal(32));

  OptimizationCompoundPass invalidate_then_repopulate(
      "inval_then_repop", "Invalidate then repopulate");
  invalidate_then_repopulate.Add<InvalidateCachedProcStateInfoPass>();
  invalidate_then_repopulate.Add<RepopulateCachedProcStateInfoPass>();
  EXPECT_THAT(invalidate_then_repopulate.Run(p.get(), {}, &results, ctx),
              IsOkAndHolds(false));

  EXPECT_TRUE(partial_qe->has_givens());
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 7]]");
}

}  // namespace
}  // namespace xls

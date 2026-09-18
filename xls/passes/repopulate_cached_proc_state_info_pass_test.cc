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

#include "xls/passes/repopulate_cached_proc_state_info_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/passes/cached_state_element_query_engine.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/range_query_engine.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class RepopulateCachedProcStateInfoPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> RunPass(Package* p, OptimizationContext& ctx) {
    PassResults results;
    RepopulateCachedProcStateInfoPass pass;
    ScopedRecordIr sri(p);
    return pass.Run(p, {}, &results, ctx);
  }
};

TEST_F(RepopulateCachedProcStateInfoPassTest,
       PopulatesPartialInfoAndLazyTernaryQueryEngines) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  BValue sum = pb.Add(st, pb.Literal(UBits(16, 32)));
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

  EXPECT_THAT(RunPass(p.get(), ctx), IsOkAndHolds(false));

  EXPECT_TRUE(partial_qe->has_givens());
  EXPECT_TRUE(ternary_qe->has_givens());
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 7]]");
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(sum.node())),
            "[[16, 23]]");
  EXPECT_THAT(ternary_qe->GetTernary(st.node()),
              testing::Optional(LttIs<TernaryVector>(testing::ElementsAre(
                  TernaryIs("0b0000_0000_0000_0000_0000_0000_0000_0XXX")))));
}

TEST_F(RepopulateCachedProcStateInfoPassTest,
       RepopulatesWhenProcStateNextChanges) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.ReadStateElement("foo", UBits(0, 32));
  BValue next_val =
      pb.ZeroExtend(pb.Add(pb.Literal(UBits(1, 3)), pb.BitSlice(st, 0, 3)), 32);
  BValue next_node = pb.Next(st, next_val);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  OptimizationContext ctx;
  EXPECT_THAT(RunPass(p.get(), ctx), IsOkAndHolds(false));

  auto* partial_qe =
      CachedStateElementQueryEngine::FromContext<PartialInfoQueryEngine>(ctx,
                                                                         proc);
  auto* ternary_qe =
      CachedStateElementQueryEngine::FromContext<LazyTernaryQueryEngine>(ctx,
                                                                         proc);
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 7]]");

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * slice_2bit,
      proc->MakeNode<BitSlice>(SourceInfo(), st.node(), /*start=*/0,
                               /*width=*/2));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * one_2bit,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 2))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * add_2bit,
      proc->MakeNode<BinOp>(SourceInfo(), one_2bit, slice_2bit, Op::kAdd));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * narrower_next,
      proc->MakeNode<ExtendOp>(SourceInfo(), add_2bit, /*new_bit_count=*/32,
                               Op::kZeroExt));
  XLS_ASSERT_OK(next_node.node()->ReplaceOperandNumber(Next::kValueOperand,
                                                       narrower_next));

  EXPECT_THAT(RunPass(p.get(), ctx), IsOkAndHolds(false));
  EXPECT_EQ(IntervalSetTreeToString(partial_qe->GetIntervals(st.node())),
            "[[0, 3]]");
  EXPECT_THAT(ternary_qe->GetTernary(st.node()),
              testing::Optional(LttIs<TernaryVector>(testing::ElementsAre(
                  TernaryIs("0b0000_0000_0000_0000_0000_0000_0000_00XX")))));
}

}  // namespace
}  // namespace xls

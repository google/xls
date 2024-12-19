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

#include "xls/passes/post_dominator_analysis.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::testing::ElementsAre;

class PostDominatorAnalysisTest : public IrTestBase {};

TEST_F(PostDominatorAnalysisTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Not(x);
  BValue z = fb.Not(y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node(), y.node(), z.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(y.node()),
              ElementsAre(y.node(), z.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(y.node()),
              ElementsAre(x.node(), y.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(z.node()),
              ElementsAre(z.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(y.node()),
              ElementsAre(x.node(), y.node()));

  EXPECT_TRUE(analysis->NodeIsPostDominatedBy(x.node(), z.node()));
  EXPECT_FALSE(analysis->NodePostDominates(x.node(), z.node()));
  EXPECT_FALSE(analysis->NodeIsPostDominatedBy(z.node(), x.node()));
  EXPECT_TRUE(analysis->NodePostDominates(z.node(), x.node()));
}

TEST_F(PostDominatorAnalysisTest, VShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue and_op = fb.And(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node(), and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(y.node()),
              ElementsAre(y.node(), and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(y.node()),
              ElementsAre(y.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(and_op.node()),
              ElementsAre(and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(and_op.node()),
              ElementsAre(x.node(), y.node(), and_op.node()));
}

TEST_F(PostDominatorAnalysisTest, DiamondShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Not(x);
  BValue b = fb.Not(x);
  BValue and_op = fb.And(a, b);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node(), and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(a.node()),
              ElementsAre(a.node(), and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(a.node()),
              ElementsAre(a.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(b.node()),
              ElementsAre(b.node(), and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(b.node()),
              ElementsAre(b.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(and_op.node()),
              ElementsAre(and_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(and_op.node()),
              ElementsAre(x.node(), a.node(), b.node(), and_op.node()));
}

TEST_F(PostDominatorAnalysisTest, DoubleDiamondShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Not(x);
  BValue b = fb.Not(x);
  BValue and_op = fb.And(a, b);
  BValue or_op = fb.Or(a, and_op);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node(), or_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(a.node()),
              ElementsAre(a.node(), or_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(a.node()),
              ElementsAre(a.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(b.node()),
              ElementsAre(b.node(), and_op.node(), or_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(b.node()),
              ElementsAre(b.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(and_op.node()),
              ElementsAre(and_op.node(), or_op.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(and_op.node()),
              ElementsAre(b.node(), and_op.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(or_op.node()),
              ElementsAre(or_op.node()));
  EXPECT_THAT(
      analysis->GetNodesPostDominatedByNode(or_op.node()),
      ElementsAre(x.node(), a.node(), b.node(), and_op.node(), or_op.node()));
}

TEST_F(PostDominatorAnalysisTest, DanglingNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue dangling_intermediate = fb.Not(x);
  BValue dangling_tail = fb.Not(dangling_intermediate);
  BValue z = fb.Not(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(dangling_intermediate.node()),
              ElementsAre(dangling_intermediate.node(), dangling_tail.node()));
  EXPECT_THAT(
      analysis->GetNodesPostDominatedByNode(dangling_intermediate.node()),
      ElementsAre(dangling_intermediate.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(dangling_tail.node()),
              ElementsAre(dangling_tail.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(dangling_tail.node()),
              ElementsAre(dangling_intermediate.node(), dangling_tail.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(z.node()),
              ElementsAre(z.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(z.node()),
              ElementsAre(z.node()));
}

TEST_F(PostDominatorAnalysisTest, DisconnectedNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Not(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node(), z.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(y.node()),
              ElementsAre(y.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(y.node()),
              ElementsAre(y.node()));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(z.node()),
              ElementsAre(z.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(z.node()),
              ElementsAre(x.node(), z.node()));
}

TEST_F(PostDominatorAnalysisTest, MultipleOutputs) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 1)));
  BValue y = pb.StateElement("y", Value(UBits(0, 1)));
  BValue z = pb.And(x, y);
  BValue next_y = pb.Next(y, z);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PostDominatorAnalysis> analysis,
                           PostDominatorAnalysis::Run(proc));

  EXPECT_THAT(analysis->GetPostDominatorsOfNode(x.node()),
              ElementsAre(x.node()));
  EXPECT_THAT(analysis->GetPostDominatorsOfNode(y.node()),
              ElementsAre(y.node(), z.node(), next_y.node()));
  EXPECT_THAT(analysis->GetPostDominatorsOfNode(z.node()),
              ElementsAre(z.node(), next_y.node()));

  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(x.node()),
              ElementsAre(x.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(y.node()),
              ElementsAre(y.node()));
  EXPECT_THAT(analysis->GetNodesPostDominatedByNode(z.node()),
              ElementsAre(y.node(), z.node()));
}

}  // namespace
}  // namespace xls

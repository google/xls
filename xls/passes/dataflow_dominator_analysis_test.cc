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

#include "xls/passes/dataflow_dominator_analysis.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class DataflowDominatorAnalysisTest : public IrTestBase {};

TEST_F(DataflowDominatorAnalysisTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Not(x);
  BValue z = fb.Not(y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node(), y.node(), z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(y.node()),
              ElementsAre(x.node(), y.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(y.node()),
              ElementsAre(y.node(), z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(z.node()),
              ElementsAre(x.node(), y.node(), z.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(z.node()),
              ElementsAre(z.node()));

  EXPECT_FALSE(analysis.NodeIsDominatedBy(x.node(), z.node()));
  EXPECT_TRUE(analysis.NodeDominates(x.node(), z.node()));
  EXPECT_TRUE(analysis.NodeIsDominatedBy(z.node(), x.node()));
  EXPECT_FALSE(analysis.NodeDominates(z.node(), x.node()));
}

TEST_F(DataflowDominatorAnalysisTest, SimpleWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue one = fb.Literal(UBits(1, 1));
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Xor(x, one);
  BValue z = fb.Xor(y, one);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node(), y.node(), z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(y.node()),
              ElementsAre(x.node(), y.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(y.node()),
              ElementsAre(y.node(), z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(z.node()),
              ElementsAre(x.node(), y.node(), z.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(z.node()),
              ElementsAre(z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(one.node()), IsEmpty());
  EXPECT_THAT(analysis.GetNodesDominatedByNode(one.node()), IsEmpty());
}

TEST_F(DataflowDominatorAnalysisTest, VShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue and_op = fb.And(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(y.node()), ElementsAre(y.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(y.node()),
              ElementsAre(y.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(and_op.node()),
              ElementsAre(and_op.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(and_op.node()),
              ElementsAre(and_op.node()));
}

TEST_F(DataflowDominatorAnalysisTest, VShapeWithReceive) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * input,
      p->CreateStreamingChannel("input", ChannelOps::kReceiveOnly,
                                p->GetBitsType(1)));

  ProcBuilder pb(TestName(), p.get());
  BValue tok = pb.StateElement("tok", Value::Token());
  BValue x = pb.StateElement("x", UBits(1, 1));
  BValue recv = pb.Receive(input, tok);
  BValue recv_tok = pb.TupleIndex(recv, 0);
  BValue y = pb.TupleIndex(recv, 1);
  BValue and_op = pb.And(x, y);
  pb.Next(tok, recv_tok);
  pb.Next(x, and_op);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * f, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(y.node()),
              ElementsAre(recv.node(), y.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(y.node()),
              ElementsAre(y.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(and_op.node()),
              ElementsAre(and_op.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(and_op.node()),
              ElementsAre(and_op.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(recv.node()),
              ElementsAre(recv.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(recv.node()),
              ElementsAre(recv.node(), recv_tok.node(), y.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(recv_tok.node()),
              ElementsAre(recv.node(), recv_tok.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(recv_tok.node()),
              ElementsAre(recv_tok.node()));
}

TEST_F(DataflowDominatorAnalysisTest, AShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue x1 = fb.Identity(x);
  BValue x2 = fb.Identity(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node(), x1.node(), x2.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x1.node()),
              ElementsAre(x.node(), x1.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x1.node()),
              ElementsAre(x1.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x2.node()),
              ElementsAre(x.node(), x2.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x2.node()),
              ElementsAre(x2.node()));
}

TEST_F(DataflowDominatorAnalysisTest, DiamondShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Not(x);
  BValue b = fb.Not(x);
  BValue and_op = fb.And(a, b);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node(), a.node(), b.node(), and_op.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(a.node()),
              ElementsAre(x.node(), a.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(a.node()),
              ElementsAre(a.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(b.node()),
              ElementsAre(x.node(), b.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(b.node()),
              ElementsAre(b.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(and_op.node()),
              ElementsAre(x.node(), and_op.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(and_op.node()),
              ElementsAre(and_op.node()));
}

TEST_F(DataflowDominatorAnalysisTest, DoubleDiamondShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Not(x);
  BValue b = fb.Not(x);
  BValue and_op = fb.And(a, b);
  BValue or_op = fb.Or(a, and_op);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(
      analysis.GetNodesDominatedByNode(x.node()),
      ElementsAre(x.node(), a.node(), b.node(), and_op.node(), or_op.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(a.node()),
              ElementsAre(x.node(), a.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(a.node()),
              ElementsAre(a.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(b.node()),
              ElementsAre(x.node(), b.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(b.node()),
              ElementsAre(b.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(and_op.node()),
              ElementsAre(x.node(), and_op.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(and_op.node()),
              ElementsAre(and_op.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(or_op.node()),
              ElementsAre(x.node(), or_op.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(or_op.node()),
              ElementsAre(or_op.node()));
}

TEST_F(DataflowDominatorAnalysisTest, DanglingNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue dangling_intermediate = fb.Not(x);
  BValue dangling_tail = fb.Not(dangling_intermediate);
  BValue z = fb.Not(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node(), dangling_intermediate.node(),
                          dangling_tail.node(), z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(dangling_intermediate.node()),
              ElementsAre(x.node(), dangling_intermediate.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(dangling_intermediate.node()),
              ElementsAre(dangling_intermediate.node(), dangling_tail.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(dangling_tail.node()),
              ElementsAre(x.node(), dangling_intermediate.node(),
                          dangling_tail.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(dangling_tail.node()),
              ElementsAre(dangling_tail.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(z.node()),
              ElementsAre(x.node(), z.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(z.node()),
              ElementsAre(z.node()));
}

TEST_F(DataflowDominatorAnalysisTest, DisconnectedNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Not(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(f));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              ElementsAre(x.node(), z.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(y.node()), ElementsAre(y.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(y.node()),
              ElementsAre(y.node()));

  EXPECT_THAT(analysis.GetDominatorsOfNode(z.node()),
              ElementsAre(x.node(), z.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(z.node()),
              ElementsAre(z.node()));
}

TEST_F(DataflowDominatorAnalysisTest, MultipleOutputs) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 1)));
  BValue y = pb.StateElement("y", Value(UBits(0, 1)));
  BValue z = pb.And(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({x, z}));
  XLS_ASSERT_OK_AND_ASSIGN(DataflowDominatorAnalysis analysis,
                           DataflowDominatorAnalysis::Run(proc));

  EXPECT_THAT(analysis.GetDominatorsOfNode(x.node()), ElementsAre(x.node()));
  EXPECT_THAT(analysis.GetDominatorsOfNode(y.node()), ElementsAre(y.node()));
  EXPECT_THAT(analysis.GetDominatorsOfNode(z.node()), ElementsAre(z.node()));

  EXPECT_THAT(analysis.GetNodesDominatedByNode(x.node()),
              UnorderedElementsAre(
                  x.node(), m::Next(m::StateRead("x"), m::StateRead("x"))));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(y.node()),
              ElementsAre(y.node()));
  EXPECT_THAT(analysis.GetNodesDominatedByNode(z.node()),
              ElementsAre(z.node()));
}

}  // namespace
}  // namespace xls

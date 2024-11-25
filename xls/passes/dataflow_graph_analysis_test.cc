// Copyright 2024 The XLS Authors
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

#include "xls/passes/dataflow_graph_analysis.h"

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
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace m = ::xls::op_matchers;

class DataflowGraphAnalysisTest : public IrTestBase {};

TEST_F(DataflowGraphAnalysisTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Not(x);
  BValue z = fb.Not(y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(y.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(z.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, SimpleWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue one = fb.Literal(UBits(1, 1));
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Xor(x, one);
  BValue z = fb.Xor(y, one);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(y.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(z.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, VShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue and_op = fb.And(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(y.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(and_op.node()),
              IsOkAndHolds(ElementsAre(x.node(), y.node())));
}

TEST_F(DataflowGraphAnalysisTest, VShapeWithReceive) {
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
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(y.node()),
              IsOkAndHolds(ElementsAre(recv.node())));
  EXPECT_THAT(analysis.GetMinCutFor(and_op.node()),
              IsOkAndHolds(ElementsAre(x.node(), recv.node())));
  EXPECT_THAT(analysis.GetMinCutFor(recv.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(recv_tok.node()),
              IsOkAndHolds(ElementsAre(recv.node())));
}

TEST_F(DataflowGraphAnalysisTest, AShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue x1 = fb.Identity(x);
  BValue x2 = fb.Identity(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(x1.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(x2.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, DiamondShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Not(x);
  BValue b = fb.Not(x);
  BValue and_op = fb.And(a, b);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(a.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(b.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(and_op.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, DoubleDiamondShape) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Not(x);
  BValue b = fb.Not(x);
  BValue and_op = fb.And(a, b);
  BValue or_op = fb.Or(a, and_op);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(a.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(b.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(and_op.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(or_op.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, DanglingNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue dangling_intermediate = fb.Not(x);
  BValue dangling_tail = fb.Not(dangling_intermediate);
  BValue z = fb.Not(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(dangling_intermediate.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(dangling_tail.node()),
              IsOkAndHolds(ElementsAre(x.node())));
  EXPECT_THAT(analysis.GetMinCutFor(z.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, DisconnectedNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Not(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DataflowGraphAnalysis analysis(f);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(y.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(z.node()),
              IsOkAndHolds(ElementsAre(x.node())));
}

TEST_F(DataflowGraphAnalysisTest, MultipleOutputs) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 1)));
  BValue y = pb.StateElement("y", Value(UBits(0, 1)));
  BValue z = pb.And(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({x, z}));
  DataflowGraphAnalysis analysis(proc);

  EXPECT_THAT(analysis.GetMinCutFor(x.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(y.node()), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(analysis.GetMinCutFor(z.node()),
              IsOkAndHolds(ElementsAre(x.node(), y.node())));
}

// Found by minimizing a real example during development.
TEST_F(DataflowGraphAnalysisTest, ComplexDataflowExample) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn test(x: bits[2]) -> bits[1] {
      bit_slice.2: bits[1] = bit_slice(x, start=0, width=1)
      bit_slice.3: bits[1] = bit_slice(x, start=1, width=1)
      literal.4: bits[2] = literal(value=0)
      zero_ext.5: bits[2] = zero_ext(bit_slice.2, new_bit_count=2)
      sel.6: bits[2] = sel(bit_slice.3, cases=[literal.4, zero_ext.5])
      literal.7: bits[1] = literal(value=0)
      ret sel.8: bits[1] = sel(sel.6, cases=[literal.7, literal.7], default=literal.7)
    }
  )",
                                                       p.get()));

  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f));
  DataflowGraphAnalysis analysis(f, &query_engine);

  XLS_ASSERT_OK_AND_ASSIGN(Node * sel_6, f->GetNode("sel.6"));
  EXPECT_THAT(analysis.GetMinCutFor(sel_6),
              IsOkAndHolds(ElementsAre(m::Param("x"))));
}

}  // namespace
}  // namespace xls

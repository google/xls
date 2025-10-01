// Copyright 2022 The XLS Authors
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

#include "xls/estimators/delay_model/analyze_critical_path.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::FieldsAre;

class AnalyzeCriticalPathTest : public IrTestBase {
 protected:
  const DelayEstimator* delay_estimator_ = GetDelayEstimator("unit").value();
};

TEST_F(AnalyzeCriticalPathTest, TrivialFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto neg = fb.Negate(x);
  auto rev = fb.Reverse(neg);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> cp,
      AnalyzeCriticalPath(f, /*clock_period_ps=*/std::nullopt,
                          *delay_estimator_));

  ASSERT_EQ(cp.size(), 3);
  EXPECT_EQ(cp[0].node, rev.node());
  EXPECT_EQ(cp[0].node_delay_ps, 1);
  EXPECT_EQ(cp[0].path_delay_ps, 2);
  EXPECT_FALSE(cp[0].delayed_by_cycle_boundary);

  EXPECT_EQ(cp[1].node, neg.node());
  EXPECT_EQ(cp[1].node_delay_ps, 1);
  EXPECT_EQ(cp[1].path_delay_ps, 1);
  EXPECT_FALSE(cp[1].delayed_by_cycle_boundary);

  EXPECT_EQ(cp[2].node, x.node());
  EXPECT_EQ(cp[2].node_delay_ps, 0);
  EXPECT_EQ(cp[2].path_delay_ps, 0);
  EXPECT_FALSE(cp[2].delayed_by_cycle_boundary);
}

TEST_F(AnalyzeCriticalPathTest, MultipathFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto neg_x = fb.Negate(x);
  auto rev_neg_x = fb.Reverse(neg_x);
  auto neg_y = fb.Negate(y);
  auto sum = fb.Add(rev_neg_x, neg_y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> cp,
      AnalyzeCriticalPath(f, /*clock_period_ps=*/std::nullopt,
                          *delay_estimator_));

  ASSERT_EQ(cp.size(), 4);
  EXPECT_EQ(cp[0].node, sum.node());
  EXPECT_EQ(cp[0].node_delay_ps, 1);
  EXPECT_EQ(cp[0].path_delay_ps, 3);
  EXPECT_FALSE(cp[0].delayed_by_cycle_boundary);

  EXPECT_EQ(cp[1].node, rev_neg_x.node());
  EXPECT_EQ(cp[1].path_delay_ps, 2);

  EXPECT_EQ(cp[2].node, neg_x.node());
  EXPECT_EQ(cp[2].path_delay_ps, 1);

  EXPECT_EQ(cp[3].node, x.node());
  EXPECT_EQ(cp[3].path_delay_ps, 0);
}

TEST_F(AnalyzeCriticalPathTest, ProcWithState) {
  auto p = CreatePackage();
  TokenlessProcBuilder b(TestName(), "tkn", p.get());
  auto st = b.StateElement("st", Value(UBits(0, 32)));
  auto neg = b.Negate(st);
  auto rev = b.Reverse(neg);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({rev}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> cp,
      AnalyzeCriticalPath(proc, /*clock_period_ps=*/std::nullopt,
                          *delay_estimator_));

  EXPECT_THAT(cp, ElementsAre(FieldsAre(m::Next(), _, 2, _),
                              FieldsAre(rev.node(), _, 2, _),
                              FieldsAre(neg.node(), _, 1, _),
                              FieldsAre(st.node(), _, 0, _)));
}

TEST_F(AnalyzeCriticalPathTest, ProcWithSendReceive) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder b(TestName(), "tkn", p.get());
  auto in = b.Receive(ch_in);
  auto neg = b.Negate(in);
  auto rev = b.Reverse(neg);
  auto send = b.Send(ch_out, rev);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> cp,
      AnalyzeCriticalPath(proc, /*clock_period_ps=*/std::nullopt,
                          *delay_estimator_));
  EXPECT_THAT(cp, ElementsAre(FieldsAre(send.node(), _, _, _),
                              FieldsAre(rev.node(), _, _, _),
                              FieldsAre(neg.node(), _, _, _),
                              FieldsAre(m::TupleIndex(), _, _, _),
                              FieldsAre(m::Receive(), _, _, _),
                              FieldsAre(m::Literal(Value::Token()), _, _, _)));
}

TEST_F(AnalyzeCriticalPathTest, EmptyProc) {
  auto p = CreatePackage();
  ProcBuilder b(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> cp,
      AnalyzeCriticalPath(proc, /*clock_period_ps=*/std::nullopt,
                          *delay_estimator_));
  EXPECT_TRUE(cp.empty());
}

TEST_F(AnalyzeCriticalPathTest, SlackFromCriticalPathFromExampleComment) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  // Path 'a' contributes 2 to delay.
  auto a1 = fb.Negate(x, SourceInfo(), "a1");
  auto a2 = fb.Reverse(a1, SourceInfo(), "a2");
  // Path 'b' contributes 3 to delay.
  auto b1 = fb.Add(a2, y, SourceInfo(), "b1");
  auto b2 = fb.Negate(b1, SourceInfo(), "b2");
  auto b3 = fb.Reverse(b2, SourceInfo(), "b3");
  // Path 'c' contributes 1 to delay.
  BValue c1 = fb.Add(a2, z, SourceInfo(), "c1");
  // Path 'd' contributes 5 to delay.
  auto d1 = fb.And(b3, c1, SourceInfo(), "d1");
  auto d2 = fb.Negate(d1, SourceInfo(), "d2");
  auto d3 = fb.Reverse(d2, SourceInfo(), "d3");
  auto d4 = fb.Add(d3, z, SourceInfo(), "d4");
  auto d5 = fb.Negate(d4, SourceInfo(), "d5");
  // Path 'e' contributes 2 to delay.
  auto e1 = fb.And(a2, y, SourceInfo(), "e1");
  auto e2 = fb.And(e1, z, SourceInfo(), "e2");
  auto return_val = fb.Tuple({d5, e2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      (absl::flat_hash_map<Node*, int64_t> slacks),
      SlackFromCriticalPath(f, /*clock_period_ps=*/std::nullopt,
                            *delay_estimator_));
  EXPECT_EQ(slacks.at(return_val.node()), 0);
  EXPECT_EQ(slacks.at(c1.node()), 2);
  EXPECT_EQ(slacks.at(e1.node()), 6);
  EXPECT_EQ(slacks.at(e2.node()), 6);
  for (Node* node : {a1.node(), a2.node(), b1.node(), b2.node(), b3.node(),
                     d1.node(), d2.node(), d3.node(), d4.node(), d5.node()}) {
    EXPECT_EQ(slacks.at(node), 0);
  }
}

TEST_F(AnalyzeCriticalPathTest, SlackFromCriticalPathWithPartialView) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  // global critical path, should be ignored by test:
  auto a1 = fb.Add(x, y, SourceInfo(), "a1");
  auto a2 = fb.Negate(a1, SourceInfo(), "a2");
  auto a3 = fb.Reverse(a2, SourceInfo(), "a3");
  auto a4 = fb.And(a3, y, SourceInfo(), "a4");
  auto a5 = fb.Or(a4, x, SourceInfo(), "a5");
  // other nodes on shorter path:
  auto b = fb.Subtract(x, y, SourceInfo(), "b");
  auto c1 = fb.SDiv(x, y, SourceInfo(), "c1");
  auto c2 = fb.Negate(c1, SourceInfo(), "c");
  auto d = fb.And({b, c2, a5}, SourceInfo(), "d");
  fb.Tuple({a5, d});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      (absl::flat_hash_map<Node*, int64_t> slacks),
      SlackFromCriticalPath(f, /*clock_period_ps=*/std::nullopt,
                            *delay_estimator_, [b, c1, c2, d](Node* n) {
                              return n == b.node() || n == c1.node() ||
                                     n == c2.node() || n == d.node();
                            }));

  EXPECT_EQ(slacks.at(b.node()), 1);
  EXPECT_EQ(slacks.at(c1.node()), 0);
  EXPECT_EQ(slacks.at(c2.node()), 0);
  EXPECT_EQ(slacks.at(d.node()), 0);
  for (Node* node : {a1.node(), a2.node(), a3.node(), a4.node(), a5.node()}) {
    EXPECT_FALSE(slacks.contains(node));
  }
}

}  // namespace
}  // namespace xls

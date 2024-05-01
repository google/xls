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

#include "xls/delay_model/analyze_critical_path.h"

#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

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

  ASSERT_EQ(cp.size(), 3);
  EXPECT_EQ(cp[0].node, rev.node());
  EXPECT_EQ(cp[0].path_delay_ps, 2);
  EXPECT_EQ(cp[1].node, neg.node());
  EXPECT_EQ(cp[1].path_delay_ps, 1);
  EXPECT_EQ(cp[2].node, proc->GetStateParam(0));
  EXPECT_EQ(cp[2].path_delay_ps, 0);
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
  ASSERT_EQ(cp.size(), 6);
  EXPECT_EQ(cp[0].node, send.node());
  EXPECT_EQ(cp[1].node, rev.node());
  EXPECT_EQ(cp[2].node, neg.node());
  EXPECT_EQ(cp[3].node->op(), Op::kTupleIndex);
  EXPECT_EQ(cp[4].node->op(), Op::kReceive);
  EXPECT_EQ(cp[5].node, proc->TokenParam());
}

}  // namespace
}  // namespace xls

// Copyright 2025 The XLS Authors
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

#include "xls/passes/critical_path_slack_analysis.h"

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/passes/critical_path_delay_analysis.h"

namespace xls {
namespace {

class CriticalPathSlackAnalysisTest : public IrTestBase {};

TEST_F(CriticalPathSlackAnalysisTest, SlackFromCriticalPathSimple) {
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

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  CriticalPathDelayAnalysis critical_path_delay(delay_estimator);
  CriticalPathSlackAnalysis critical_path_slack(&critical_path_delay);
  XLS_ASSERT_OK(critical_path_slack.Attach(f));
  XLS_ASSERT_OK(critical_path_delay.Attach(f));

  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(return_val.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(c1.node()), 2);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(e1.node()), 6);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(e2.node()), 6);
  for (Node* node : {a1.node(), a2.node(), b1.node(), b2.node(), b3.node(),
                     d1.node(), d2.node(), d3.node(), d4.node(), d5.node()}) {
    EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(node), 0);
  }
}

TEST_F(CriticalPathSlackAnalysisTest, IndependentPathsStillInvalidateTheOther) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto one = fb.Literal(UBits(1, 32));
  auto a1 = fb.Add(x, one);
  auto a2 = fb.Add(a1, one);
  auto other_one = fb.Literal(UBits(1, 32));
  auto b1 = fb.Subtract(x, other_one);
  auto b2 = fb.Subtract(b1, other_one);
  auto b3 = fb.Subtract(b2, other_one);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  CriticalPathDelayAnalysis critical_path_delay(delay_estimator);
  CriticalPathSlackAnalysis critical_path_slack(&critical_path_delay);
  XLS_ASSERT_OK(critical_path_slack.Attach(f));
  XLS_ASSERT_OK(critical_path_delay.Attach(f));

  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a1.node()), 1);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a2.node()), 1);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b1.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b2.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b3.node()), 0);

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * a3,
      f->MakeNode<BinOp>(a2.node()->loc(), a2.node(), one.node(), Op::kAdd));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * a4, f->MakeNode<BinOp>(a3->loc(), a3, one.node(), Op::kAdd));

  // Path 'b' should now have slack and be recalculated correctly.
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a4), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a1.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a2.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b1.node()), 1);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b2.node()), 1);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b3.node()), 1);

  XLS_ASSERT_OK(f->RemoveNode(a4));
  XLS_ASSERT_OK(f->RemoveNode(a3));

  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a1.node()), 1);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(a2.node()), 1);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b1.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b2.node()), 0);
  EXPECT_EQ(critical_path_slack.SlackFromCriticalPath(b3.node()), 0);
}

}  // namespace
}  // namespace xls

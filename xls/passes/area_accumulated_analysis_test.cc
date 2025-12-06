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

#include "xls/passes/area_accumulated_analysis.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

class AreaAccumulatedAnalysisTest : public IrTestBase {};

TEST_F(AreaAccumulatedAnalysisTest, DoNotDoubleCountNodes) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue ab = fb.And(a, b);
  BValue abc = fb.And(ab, c);
  BValue ret = fb.Or({abc, a, b, c, d});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  AreaAccumulatedAnalysis analysis(ae);
  XLS_ASSERT_OK(analysis.Attach(f));
  EXPECT_EQ(analysis.GetAreaThroughToNode(a.node()), 0.0);
  EXPECT_EQ(analysis.GetAreaThroughToNode(ab.node()), 1.0);
  EXPECT_EQ(analysis.GetAreaThroughToNode(abc.node()), 2.0);
  EXPECT_EQ(analysis.GetAreaThroughToNode(ret.node()), 3.0);
}

TEST_F(AreaAccumulatedAnalysisTest, InvalidateCache) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue ab = fb.And(a, b);
  BValue a_or_b = fb.Or(a, b);
  BValue abc = fb.And(ab, c);
  BValue ret = fb.Or({abc, a, b, c, d});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  AreaAccumulatedAnalysis analysis(ae);
  XLS_ASSERT_OK(analysis.Attach(f));
  EXPECT_EQ(analysis.GetAreaThroughToNode(ret.node()), 3.0);

  EXPECT_TRUE(abc.node()->ReplaceOperand(c.node(), a_or_b.node()));
  EXPECT_EQ(analysis.GetAreaThroughToNode(ret.node()), 4.0);
  EXPECT_EQ(analysis.GetAreaThroughToNode(abc.node()), 3.0);
  EXPECT_TRUE(abc.node()->ReplaceOperand(a_or_b.node(), c.node()));
  EXPECT_EQ(analysis.GetAreaThroughToNode(ret.node()), 3.0);
  EXPECT_EQ(analysis.GetAreaThroughToNode(abc.node()), 2.0);
}

}  // namespace
}  // namespace xls

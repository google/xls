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

#include "xls/passes/reachability_analysis.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

class ReachabilityAnalysisTest : public IrTestBase {
 protected:
  bool IsReachableFrom(const ReachabilityAnalysis& reachability, Node* end,
                       Node* start) {
    return reachability.IsReachableFrom(end, start);
  }
};

TEST_F(ReachabilityAnalysisTest, SuccessorsAreReachableButPredecessorsAreNot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Add(x, x);
  BValue z = fb.UMul(x, y);
  BValue w = fb.Subtract(z, x);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(w));
  ReachabilityAnalysis reachability;

  EXPECT_TRUE(IsReachableFrom(reachability, x.node(), x.node()));
  EXPECT_TRUE(IsReachableFrom(reachability, y.node(), x.node()));
  EXPECT_TRUE(IsReachableFrom(reachability, z.node(), x.node()));
  EXPECT_TRUE(IsReachableFrom(reachability, w.node(), x.node()));
  EXPECT_TRUE(IsReachableFrom(reachability, z.node(), y.node()));
  EXPECT_TRUE(IsReachableFrom(reachability, w.node(), y.node()));
  EXPECT_TRUE(IsReachableFrom(reachability, z.node(), y.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, x.node(), y.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, x.node(), z.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, x.node(), w.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, y.node(), z.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, y.node(), w.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, z.node(), w.node()));
}

TEST_F(ReachabilityAnalysisTest, SiblingsCannotReachEachOther) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue x = fb.Or(a, b);
  BValue y1 = fb.Add(x, a);
  BValue y2 = fb.Subtract(x, b);
  BValue z = fb.UMul(y1, y2);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(z));
  ReachabilityAnalysis reachability;

  EXPECT_FALSE(IsReachableFrom(reachability, y1.node(), y2.node()));
  EXPECT_FALSE(IsReachableFrom(reachability, y2.node(), y1.node()));
}

}  // namespace
}  // namespace xls

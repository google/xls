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

#include "xls/passes/critical_path_delay_analysis.h"

#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::Pointee;

class FakeDelayEstimator : public DelayEstimator {
 public:
  FakeDelayEstimator() : DelayEstimator("fake") {}
  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kLiteral:
        return 0;
      case Op::kAdd:
        return 100;
      case Op::kUMul:
        return 300;
      case Op::kSel:
        return 30;
      default:
        return 10;
    }
  }
};

class CriticalPathDelayAnalysisTest : public IrTestBase {
 public:
  static void SetUpTestSuite() {
    XLS_ASSERT_OK(GetDelayEstimatorManagerSingleton().RegisterDelayEstimator(
        std::make_unique<FakeDelayEstimator>(),
        DelayEstimatorPrecedence::kLow));
  }
};

TEST_F(CriticalPathDelayAnalysisTest, SimpleChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  CriticalPathDelayAnalysis analysis(*GetDelayEstimator("fake"));
  XLS_ASSERT_OK(analysis.Attach(f));

  EXPECT_THAT(analysis.GetInfo(x.node()), Pointee(0));
  EXPECT_THAT(analysis.GetInfo(y.node()), Pointee(0));
  EXPECT_THAT(analysis.GetInfo(add.node()), Pointee(100));
}

TEST_F(CriticalPathDelayAnalysisTest, Diamond) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add1 = fb.Add(x, y);        // 100
  BValue mul1 = fb.UMul(x, y);       // 300
  BValue add2 = fb.Add(add1, mul1);  // max(100, 300) + 100 = 400
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  CriticalPathDelayAnalysis analysis(*GetDelayEstimator("fake"));
  XLS_ASSERT_OK(analysis.Attach(f));

  EXPECT_THAT(analysis.GetInfo(x.node()), Pointee(0));
  EXPECT_THAT(analysis.GetInfo(y.node()), Pointee(0));
  EXPECT_THAT(analysis.GetInfo(add1.node()), Pointee(100));
  EXPECT_THAT(analysis.GetInfo(mul1.node()), Pointee(300));
  EXPECT_THAT(analysis.GetInfo(add2.node()), Pointee(400));
}

TEST_F(CriticalPathDelayAnalysisTest, Select) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue pred = fb.Param("pred", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sel = fb.Select(pred, x, y);  // max(0, 0, 0) + 30 = 30
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  CriticalPathDelayAnalysis analysis(*GetDelayEstimator("fake"));
  XLS_ASSERT_OK(analysis.Attach(f));
  EXPECT_THAT(analysis.GetInfo(sel.node()), Pointee(30));
}

TEST_F(CriticalPathDelayAnalysisTest, SelectWithNonZeroOperand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue pred = fb.Param("pred", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue mul = fb.UMul(x, y);            // 300
  BValue sel = fb.Select(pred, mul, y);  // max(0, 300, 0) + 30 = 330
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  CriticalPathDelayAnalysis analysis(*GetDelayEstimator("fake"));
  XLS_ASSERT_OK(analysis.Attach(f));
  EXPECT_THAT(analysis.GetInfo(sel.node()), Pointee(330));
}

}  // namespace
}  // namespace xls

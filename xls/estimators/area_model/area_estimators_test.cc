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

#include "xls/estimators/area_model/area_estimators.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class DelayEstimatorsTest : public IrTestBase {};

TEST_F(DelayEstimatorsTest, UnitDelayModel) {
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * estimator, GetAreaEstimator("unit"));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue add = fb.Add(x, x);
  BValue negate = fb.Negate(x);
  BValue mul = fb.UMul(x, x);
  BValue and_op = fb.And(x, x);
  BValue tuple = fb.Tuple({x, add, mul});

  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(x.node()),
              IsOkAndHolds(0));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(add.node()),
              IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(negate.node()),
              IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(mul.node()),
              IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(and_op.node()),
              IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(tuple.node()),
              IsOkAndHolds(1));
}

TEST_F(DelayEstimatorsTest, RecurseNonRecursiveOps) {
  XLS_ASSERT_OK_AND_ASSIGN(
      AreaEstimator * estimator,
      GetAreaEstimator("unit_with_default_recursive_op_handling"));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue add = fb.Add(x, x);
  BValue negate = fb.Negate(x);

  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(x.node()),
              IsOkAndHolds(0));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(add.node()),
              IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(negate.node()),
              IsOkAndHolds(1));
}

TEST_F(DelayEstimatorsTest, RecurseFunctionBase) {
  XLS_ASSERT_OK_AND_ASSIGN(
      AreaEstimator * estimator,
      GetAreaEstimator("unit_with_default_recursive_op_handling"));
  auto p = CreatePackage();

  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(x, x);
  fb.Negate(x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(estimator->GetFunctionBaseAreaInSquareMicrons(f),
              IsOkAndHolds(2));
}

TEST_F(DelayEstimatorsTest, RecurseInvoke) {
  XLS_ASSERT_OK_AND_ASSIGN(
      AreaEstimator * estimator,
      GetAreaEstimator("unit_with_default_recursive_op_handling"));
  auto p = CreatePackage();

  Function* f;
  {
    FunctionBuilder fb("f", p.get());
    BValue x = fb.Param("x", p->GetBitsType(32));
    fb.Add(x, x);
    fb.Negate(x);
    XLS_ASSERT_OK_AND_ASSIGN(f, fb.Build());
  }

  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue invoke = fb.Invoke({x}, f);

  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(invoke.node()),
              IsOkAndHolds(2));
}

TEST_F(DelayEstimatorsTest, RecurseCountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(
      AreaEstimator * estimator,
      GetAreaEstimator("unit_with_default_recursive_op_handling"));
  auto p = CreatePackage();

  Function* g;
  {
    FunctionBuilder fb("g", p.get());
    fb.Param("i", p->GetBitsType(32));
    BValue arg = fb.Param("arg", p->GetBitsType(32));
    fb.Add(arg, arg);
    fb.Negate(arg);
    XLS_ASSERT_OK_AND_ASSIGN(g, fb.Build());
  }

  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue counted_for = fb.CountedFor(x, /*trip_count=*/5, /*stride=*/1, g);

  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(counted_for.node()),
              IsOkAndHolds(10));
}

TEST_F(DelayEstimatorsTest, RecurseDynamicCountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(
      AreaEstimator * estimator,
      GetAreaEstimator("unit_with_default_recursive_op_handling"));
  auto p = CreatePackage();

  Function* g;
  {
    FunctionBuilder fb("g", p.get());
    fb.Param("i", p->GetBitsType(32));
    BValue arg = fb.Param("arg", p->GetBitsType(32));
    fb.Add(arg, arg);
    fb.Negate(arg);
    XLS_ASSERT_OK_AND_ASSIGN(g, fb.Build());
  }

  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue trip_count = fb.Param("trip_count", p->GetBitsType(16));
  BValue stride = fb.Param("stride", p->GetBitsType(16));
  BValue dynamic_counted_for = fb.DynamicCountedFor(x, trip_count, stride, g);

  // DynamicCountedFor estimated area is body area (regardless of trip count).
  // Body "g" has one add and one negate which cost 2.
  EXPECT_THAT(
      estimator->GetOperationAreaInSquareMicrons(dynamic_counted_for.node()),
      IsOkAndHolds(2));
}

TEST_F(DelayEstimatorsTest, RecurseDeeplyNested) {
  XLS_ASSERT_OK_AND_ASSIGN(
      AreaEstimator * estimator,
      GetAreaEstimator("unit_with_default_recursive_op_handling"));
  auto p = CreatePackage();

  Function* f1;
  {
    FunctionBuilder fb("f1", p.get());
    BValue x = fb.Param("x", p->GetBitsType(32));
    fb.Add(x, x);
    fb.Negate(x);
    XLS_ASSERT_OK_AND_ASSIGN(f1, fb.Build());
  }

  Function* f2;
  {
    FunctionBuilder fb("f2", p.get());
    fb.Param("i", p->GetBitsType(32));
    BValue x = fb.Param("x", p->GetBitsType(32));
    fb.Negate(fb.Invoke({x}, f1));
    XLS_ASSERT_OK_AND_ASSIGN(f2, fb.Build());
  }

  Function* f3;
  {
    FunctionBuilder fb("f3", p.get());
    BValue x = fb.Param("x", p->GetBitsType(32));
    BValue cf = fb.CountedFor(x, /*trip_count=*/3, /*stride=*/1, f2);
    fb.UMul(cf, x);
    XLS_ASSERT_OK_AND_ASSIGN(f3, fb.Build());
  }

  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue invoke_f3 = fb.Invoke({x}, f3);

  // Cost: 3 (trip count) * 3 (f2 cost) + 1 (umul) = 10
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(invoke_f3.node()),
              IsOkAndHolds(10));
}

}  // namespace
}  // namespace xls

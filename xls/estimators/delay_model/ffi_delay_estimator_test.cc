// Copyright 2026 The XLS Authors
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

#include "xls/estimators/delay_model/ffi_delay_estimator.h"

#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

class FfiDelayEstimatorTest : public IrTestBase {};

TEST_F(FfiDelayEstimatorTest, DelayFromCalleeForeignFunctionData) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* callee;
  {
    FunctionBuilder fb("callee", p.get());
    BValue a = fb.Param("a", u32);
    ForeignFunctionData ffd;
    ffd.set_code_template("my_ffi {fn} (.a({a}), .out({return}))");
    ffd.set_delay_ps(1234);
    fb.SetForeignFunctionData(ffd);
    XLS_ASSERT_OK_AND_ASSIGN(callee, fb.BuildWithReturnValue(a));
  }

  BValue invoke;
  {
    FunctionBuilder fb("caller", p.get());
    BValue x = fb.Param("x", u32);
    invoke = fb.Invoke({x}, callee);
    XLS_ASSERT_OK(fb.BuildWithReturnValue(invoke).status());
  }

  FfiDelayEstimator estimator(/*fallback_delay_estimate=*/std::nullopt);
  EXPECT_THAT(estimator.GetOperationDelayInPs(invoke.node()),
              IsOkAndHolds(1234));
}

TEST_F(FfiDelayEstimatorTest,
       CalleeDelayOverridesCallerForeignFunctionDataAndFallback) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* callee;
  {
    FunctionBuilder fb("callee", p.get());
    BValue a = fb.Param("a", u32);
    ForeignFunctionData ffd;
    ffd.set_delay_ps(100);
    fb.SetForeignFunctionData(ffd);
    XLS_ASSERT_OK_AND_ASSIGN(callee, fb.BuildWithReturnValue(a));
  }

  BValue invoke;
  {
    FunctionBuilder fb("caller", p.get());
    BValue x = fb.Param("x", u32);
    invoke = fb.Invoke({x}, callee);
    ForeignFunctionData caller_ffd;
    caller_ffd.set_delay_ps(999);
    fb.SetForeignFunctionData(caller_ffd);
    XLS_ASSERT_OK(fb.BuildWithReturnValue(invoke).status());
  }

  FfiDelayEstimator estimator(/*fallback_delay_estimate=*/50);
  EXPECT_THAT(estimator.GetOperationDelayInPs(invoke.node()),
              IsOkAndHolds(100));
}

TEST_F(FfiDelayEstimatorTest, FallbackUsedWhenNoDelayPs) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* callee;
  {
    FunctionBuilder fb("callee", p.get());
    BValue a = fb.Param("a", u32);
    ForeignFunctionData ffd;
    ffd.set_code_template("my_ffi");
    fb.SetForeignFunctionData(ffd);
    XLS_ASSERT_OK_AND_ASSIGN(callee, fb.BuildWithReturnValue(a));
  }

  BValue invoke;
  {
    FunctionBuilder fb("caller", p.get());
    BValue x = fb.Param("x", u32);
    invoke = fb.Invoke({x}, callee);
    XLS_ASSERT_OK(fb.BuildWithReturnValue(invoke).status());
  }

  FfiDelayEstimator with_fallback(/*fallback_delay_estimate=*/500);
  EXPECT_THAT(with_fallback.GetOperationDelayInPs(invoke.node()),
              IsOkAndHolds(500));

  FfiDelayEstimator no_fallback(/*fallback_delay_estimate=*/std::nullopt);
  EXPECT_THAT(no_fallback.GetOperationDelayInPs(invoke.node()),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(FfiDelayEstimatorTest, NonInvokeNodeReturnsUnimplemented) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue add = fb.Add(x, x);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(add).status());

  FfiDelayEstimator estimator(/*fallback_delay_estimate=*/100);
  EXPECT_THAT(estimator.GetOperationDelayInPs(add.node()),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xls

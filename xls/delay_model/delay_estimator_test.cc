// Copyright 2020 Google LLC
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

#include "xls/delay_model/delay_estimator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;

// A test delay estimator that returns a fixed delay for every node.
class TestDelayEstimator : public DelayEstimator {
 public:
  explicit TestDelayEstimator(int64 delay) : delay_(delay) {}

  xabsl::StatusOr<int64> GetOperationDelayInPs(Node* node) const override {
    return delay_;
  }

 private:
  int64 delay_;
};

class DelayEstimatorTest : public IrTestBase {};

TEST_F(DelayEstimatorTest, DelayEstimatorManager) {
  DelayEstimatorManager manager;
  EXPECT_THAT(manager.estimator_names(), ElementsAre());

  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      "forty_two", absl::make_unique<TestDelayEstimator>(42),
      DelayEstimatorPrecedence::kLow));
  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      "one", absl::make_unique<TestDelayEstimator>(1),
      DelayEstimatorPrecedence::kLow));

  EXPECT_THAT(manager.estimator_names(), ElementsAre("forty_two", "one"));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * forty_two,
                           manager.GetDelayEstimator("forty_two"));
  EXPECT_THAT(forty_two->GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(42));

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * one,
                           manager.GetDelayEstimator("one"));
  EXPECT_THAT(one->GetOperationDelayInPs(f->return_value()), IsOkAndHolds(1));

  EXPECT_THAT(manager.GetDelayEstimator("foo"),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls

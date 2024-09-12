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

#include "xls/estimators/area_model/area_estimator.h"

#include <memory>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"

namespace xls {
namespace {

class FakeAreaEstimator : public AreaEstimator {
 public:
  explicit FakeAreaEstimator(std::string_view name, double area)
      : AreaEstimator(name), area_(area) {}
  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override {
    return area_;
  }
  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override {
    return area_;
  }

 private:
  double area_;
};

class AreaEstimatorTest : public IrTestBase {};

TEST_F(AreaEstimatorTest, AreaEstimatorManager) {
  AreaEstimatorManager manager;
  XLS_ASSERT_OK(manager.AddAreaEstimator(
      std::make_unique<FakeAreaEstimator>("unit_area", 1.0)));
  XLS_ASSERT_OK(manager.AddAreaEstimator(
      std::make_unique<FakeAreaEstimator>("ten_area", 10.0)));
  EXPECT_THAT(manager.GetAreaEstimator("unit_area"),
              status_testing::StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(manager.GetAreaEstimator("ten_area"),
              status_testing::StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(manager.estimator_names(),
              testing::ElementsAre("ten_area", "unit_area"));
  EXPECT_THAT(manager.GetAreaEstimator("negative_area"),
              status_testing::StatusIs(absl::StatusCode::kNotFound));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * unit_estimator,
                           manager.GetAreaEstimator("unit_area"));
  EXPECT_THAT(
      unit_estimator->GetOperationAreaInSquareMicrons(f->return_value()),
      status_testing::IsOkAndHolds(1.0));
  EXPECT_THAT(unit_estimator->GetRegisterAreaInSquareMicrons(42),
              status_testing::IsOkAndHolds(42.0));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ten_estimator,
                           manager.GetAreaEstimator("ten_area"));
  EXPECT_THAT(ten_estimator->GetOperationAreaInSquareMicrons(f->return_value()),
              status_testing::IsOkAndHolds(10.0));
  EXPECT_THAT(ten_estimator->GetRegisterAreaInSquareMicrons(42),
              status_testing::IsOkAndHolds(420.0));
}

}  // namespace
}  // namespace xls

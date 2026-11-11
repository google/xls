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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/estimator_model.pb.h"
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
              absl_testing::StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(manager.GetAreaEstimator("ten_area"),
              absl_testing::StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(manager.estimator_names(),
              testing::ElementsAre("ten_area", "unit_area"));
  EXPECT_THAT(manager.GetAreaEstimator("negative_area"),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * unit_estimator,
                           manager.GetAreaEstimator("unit_area"));
  EXPECT_THAT(
      unit_estimator->GetOperationAreaInSquareMicrons(f->return_value()),
      absl_testing::IsOkAndHolds(1.0));
  EXPECT_THAT(unit_estimator->GetRegisterAreaInSquareMicrons(42),
              absl_testing::IsOkAndHolds(42.0));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ten_estimator,
                           manager.GetAreaEstimator("ten_area"));
  EXPECT_THAT(ten_estimator->GetOperationAreaInSquareMicrons(f->return_value()),
              absl_testing::IsOkAndHolds(10.0));
  EXPECT_THAT(ten_estimator->GetRegisterAreaInSquareMicrons(42),
              absl_testing::IsOkAndHolds(420.0));
}

TEST_F(AreaEstimatorTest, ProtoAreaEstimator) {
  estimator_model::CompiledEstimatorModel model_proto;
  QCHECK(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        metric: AREA_METRIC
        one_bit_register_area: 7.5
        op_models {
          op: "kAdd"
          estimator {
            area_regression {
              constant: 1.0
              terms {
                expression { factor { source: RESULT_BIT_COUNT } }
                square: 0.0
                linear_log2: 0.0
                linear: 2.0
                log2: 0.0
              }
            }
          }
          specializations {
            kind: OPERANDS_IDENTICAL
            estimator { fixed: 3 }
          }
        }
        op_models {
          op: "kSub"
          estimator { alias_op: "kAdd" }
        }
        op_models {
          op: "kUMul"
          estimator {
            bounding_box {
              points {
                bounds {
                  factor { source: RESULT_BIT_COUNT }
                  max_value: 8
                }
                area_sq_microns: 42.5
              }
            }
          }
        }
      )pb",
      &model_proto));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProtoAreaEstimator> estimator,
      ProtoAreaEstimator::Create("test_area", model_proto));
  EXPECT_THAT(estimator->GetOneBitRegisterAreaInSquareMicrons(),
              absl_testing::IsOkAndHolds(7.5));
  EXPECT_THAT(estimator->GetRegisterAreaInSquareMicrons(4),
              absl_testing::IsOkAndHolds(30.0));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue add_diff = fb.Add(a, b);
  BValue add_same = fb.Add(a, a);
  BValue sub_diff = fb.Subtract(a, b);
  BValue sub_same = fb.Subtract(a, a);
  BValue umul_8 = fb.UMul(a, b, 8);
  BValue neg = fb.Negate(a);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(add_diff.node()),
              absl_testing::IsOkAndHolds(17.0));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(add_same.node()),
              absl_testing::IsOkAndHolds(3.0));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(sub_diff.node()),
              absl_testing::IsOkAndHolds(17.0));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(sub_same.node()),
              absl_testing::IsOkAndHolds(3.0));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(umul_8.node()),
              absl_testing::IsOkAndHolds(42.5));
  EXPECT_THAT(estimator->GetOperationAreaInSquareMicrons(neg.node()),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xls

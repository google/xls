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

#include <cstdint>
#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/integrator/area_model/area_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class AreaEstimatorTest : public IrTestBase {};

TEST_F(AreaEstimatorTest, AreaModelTesting2Point5MuxPerNode) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  auto sel_in = fb.Param("sel_in", p->GetBitsType(1));
  fb.Select(sel_in, {in1, in2}, /*default_value=*/std::nullopt, SourceInfo(),
            "mux");
  fb.Add(in1, in2, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  Node* add_node = FindNode("add", func);
  Node* mux_node = FindNode("mux", func);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<integrator::AreaEstimator> area_estimator,
      integrator::GetAreaEstimatorByName(
          "area_model_testing_2_point_5_mux_per_node"));
  EXPECT_THAT(area_estimator->GetOperationArea(add_node), IsOkAndHolds(5));
  EXPECT_THAT(area_estimator->GetOperationArea(mux_node), IsOkAndHolds(2));
}

TEST_F(AreaEstimatorTest, AreaModelIce40Multiply) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(16));
  auto in2 = fb.Param("in2", p->GetBitsType(32));
  fb.UMul(in1, in2, /*result_width=*/16, SourceInfo(), "mul_c");
  fb.UMul(in1, in2, /*result_width=*/32, SourceInfo(), "mul_b");
  fb.UMul(in1, in2, /*result_width=*/100, SourceInfo(), "mul_a");

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  Node* mul_c = FindNode("mul_c", func);
  Node* mul_b = FindNode("mul_b", func);
  Node* mul_a = FindNode("mul_a", func);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<integrator::AreaEstimator> area_estimator,
      integrator::GetAreaEstimatorByName("ice40_lut4"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t mul_c_area,
                           area_estimator->GetOperationArea(mul_c));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t mul_b_area,
                           area_estimator->GetOperationArea(mul_b));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t mul_a_area,
                           area_estimator->GetOperationArea(mul_a));

  // Nearly all cpp delay expression components must work to
  // get multiplier area (and these ratios) correct.
  EXPECT_NEAR(mul_c_area * 3, mul_b_area, 0.01 * mul_b_area);
  EXPECT_NEAR(mul_c_area * 4, mul_a_area, 0.01 * mul_a_area);
}

}  // namespace
}  // namespace xls

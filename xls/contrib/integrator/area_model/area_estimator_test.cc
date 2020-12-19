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

#include "xls/contrib/integrator/area_model/area_estimator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class AreaEstimatorTest : public IrTestBase {};

TEST_F(AreaEstimatorTest, GetAndUseModelByName) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  auto sel_in = fb.Param("sel_in", p->GetBitsType(1));
  fb.Select(sel_in, {in1, in2}, /*default_value=*/absl::nullopt, /*loc=*/absl::nullopt, "mux");
  fb.Add(in1, in2, /*loc=*/absl::nullopt, "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  Node* add_node = FindNode("add", func);
  Node* mux_node = FindNode("mux", func);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AreaEstimator> area_estimator, GetAreaEstimatorByName("area_model_testing_2_point_5_mux_per_node"));
  EXPECT_THAT(area_estimator->GetOperationArea(add_node), IsOkAndHolds(5));
  EXPECT_THAT(area_estimator->GetOperationArea(mux_node), IsOkAndHolds(2));
}

}  // namespace
}  // namespace xls

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

#include "xls/tools/schedule.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

class ScheduleTest : public IrTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<VerifiedPackage>> CreateAddXYPackage() {
    std::unique_ptr<VerifiedPackage> p = CreatePackage();
    FunctionBuilder fb("simple", p.get());
    fb.Add(fb.Param("x", p->GetBitsType(8)), fb.Param("y", p->GetBitsType(8)));
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    XLS_RETURN_IF_ERROR(p->SetTop(f));
    return p;
  }
};

TEST_F(ScheduleTest, FailsEarlyWithoutPipelineStagesOrClockPeriod) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  // Enable FDO with a nonexistent synthesizer to confirm missing timing options
  // are rejected before synthesizer setup runs.
  SchedulingOptions options;
  options.use_fdo(true).fdo_synthesizer_name("nonexistent_synthesizer");
  EXPECT_THAT(Schedule(p.get(), options, delay_estimator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--pipeline_stages"),
                             HasSubstr("--clock_period_ps"))));
}

TEST_F(ScheduleTest, RejectsNonPositivePipelineStages) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  EXPECT_THAT(Schedule(p.get(), SchedulingOptions().pipeline_stages(0),
                       delay_estimator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages must be positive")));
  EXPECT_THAT(
      Schedule(p.get(),
               SchedulingOptions().pipeline_stages(-1).clock_period_ps(100),
               delay_estimator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages must be positive")));
}

TEST_F(ScheduleTest, RejectsNonPositiveClockPeriodPs) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  EXPECT_THAT(Schedule(p.get(), SchedulingOptions().clock_period_ps(0),
                       delay_estimator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps must be positive")));
  EXPECT_THAT(
      Schedule(p.get(),
               SchedulingOptions().pipeline_stages(1).clock_period_ps(-5),
               delay_estimator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--clock_period_ps must be positive")));
}

TEST_F(ScheduleTest, SucceedsWithValidPipelineStages) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  XLS_EXPECT_OK(Schedule(p.get(), SchedulingOptions().pipeline_stages(2),
                         delay_estimator));
}

}  // namespace
}  // namespace xls

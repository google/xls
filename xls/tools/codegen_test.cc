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

#include "xls/tools/codegen.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

class CodegenTest : public IrTestBase {
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

TEST_F(CodegenTest, PipelineFailsEarlyWithoutPipelineStagesOrClockPeriod) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  CodegenFlagsProto codegen_flags;
  codegen_flags.set_generator(GENERATOR_KIND_PIPELINE);
  // Leave delay_model unset too so this verifies timing options are validated
  // before delay model setup.
  SchedulingOptionsFlagsProto scheduling_flags;

  EXPECT_THAT(ScheduleAndCodegen(p.get(), scheduling_flags, codegen_flags,
                                 /*with_delay_model=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--pipeline_stages"),
                             HasSubstr("--clock_period_ps"))));
}

TEST_F(CodegenTest, PipelineRejectsNegativeTimingOptions) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  CodegenFlagsProto codegen_flags;
  codegen_flags.set_generator(GENERATOR_KIND_PIPELINE);

  SchedulingOptionsFlagsProto bad_stages;
  bad_stages.set_delay_model("unit");
  bad_stages.set_pipeline_stages(-1);
  EXPECT_THAT(ScheduleAndCodegen(p.get(), bad_stages, codegen_flags,
                                 /*with_delay_model=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages must be positive")));

  SchedulingOptionsFlagsProto bad_clock;
  bad_clock.set_delay_model("unit");
  bad_clock.set_clock_period_ps(-100);
  EXPECT_THAT(ScheduleAndCodegen(p.get(), bad_clock, codegen_flags,
                                 /*with_delay_model=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps must be positive")));
}

TEST_F(CodegenTest, CombinationalDoesNotRequireTimingOptions) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateAddXYPackage());
  CodegenFlagsProto codegen_flags;
  codegen_flags.set_generator(GENERATOR_KIND_COMBINATIONAL);
  SchedulingOptionsFlagsProto scheduling_flags;

  XLS_EXPECT_OK(ScheduleAndCodegen(p.get(), scheduling_flags, codegen_flags,
                                   /*with_delay_model=*/false));
}

}  // namespace
}  // namespace xls

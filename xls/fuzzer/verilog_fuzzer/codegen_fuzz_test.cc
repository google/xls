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

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/verilog_fuzzer/verilog_fuzz_domain.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr std::string_view kTop = "dut";

MATCHER_P(HasVerilogText, text,
          absl::StrCat("Expected verilog text to match ",
                       testing::DescribeMatcher<std::string>(text))) {
  return testing::ExplainMatchResult(text, arg.codegen_result.verilog_text,
                                     result_listener);
}

SchedulingOptionsFlagsProto DefaultSchedulingOptions() {
  SchedulingOptionsFlagsProto scheduling_options;
  scheduling_options.set_delay_model("sky130");
  scheduling_options.set_pipeline_stages(10);

  return scheduling_options;
}

CodegenFlagsProto DefaultCodegenOptions() {
  CodegenFlagsProto codegen_options = CodegenFlagsProto::default_instance();
  codegen_options.clear_top();
  codegen_options.set_generator(GENERATOR_KIND_PIPELINE);
  codegen_options.set_module_name(kTop);
  codegen_options.set_flop_inputs(false);
  codegen_options.set_flop_outputs(false);
  return codegen_options;
}

void CodegenSucceedsForEveryFunctionWithNoPipelineLimit(
    VerilogGenerator verilog) {
  auto result = verilog.GenerateVerilog();
  EXPECT_THAT(result, IsOkAndHolds(HasVerilogText(Not(IsEmpty()))))
      << result.status();
}

FUZZ_TEST(CodegenFuzzTest, CodegenSucceedsForEveryFunctionWithNoPipelineLimit)
    .WithDomains(VerilogGeneratorDomain(
        IrFuzzDomain(), fuzztest::Just(DefaultSchedulingOptions()),
        fuzztest::Just(DefaultCodegenOptions())));

void CodegenSucceedsOrThrowsReasonableError(VerilogGenerator verilog) {
  // We check for success or expected errors here. The expected errors are:
  // - absl::NotFoundError("No delay estimator found named")- an
  //   incorrectly-named delay estimator in the scheduling options.
  // - absl::InternalError("Schedule does not meet timing")- the schedule does
  //   not meet the timing constraints.
  // - absl::ResourceExhaustedError("schedule")- one of a number of errors
  //   indicating that we couldn't produce a schedule.
  // - absl::InvalidArgumentError()- any of a number of invalid arguments
  //   (e.g. invalid IO constraints).
  // - absl::UnimplementedError()- when XLS is complete we'll remove this one :)
  // We don't expect:
  // - CHECK-fails, OOMs, etc.
  // - Random absl::InternalError(), or other "unexpected" errors.
  EXPECT_THAT(verilog.GenerateVerilog(),
              AnyOf(IsOkAndHolds(HasVerilogText(Not(IsEmpty()))),
                    StatusIs(absl::StatusCode::kNotFound,
                             HasSubstr("No delay estimator found named")),
                    StatusIs(absl::StatusCode::kInternal,
                             HasSubstr("Schedule does not meet timing")),
                    StatusIs(absl::StatusCode::kResourceExhausted,
                             HasSubstr("schedule")),
                    StatusIs(AnyOf(absl::StatusCode::kInvalidArgument,
                                   absl::StatusCode::kUnimplemented))));
}
FUZZ_TEST(CodegenFuzzTest, CodegenSucceedsOrThrowsReasonableError)
    .WithDomains(VerilogGeneratorDomain(IrFuzzDomain(),
                                        NoFdoSchedulingOptionsFlagsDomain(),
                                        CodegenFlagsDomain()));

}  // namespace
}  // namespace xls

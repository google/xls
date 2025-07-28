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

#include "gmock/gmock.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/fuzzer/verilog_fuzzer/verilog_fuzz_domain.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr std::string_view kTop = "dut";

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
  codegen_options.set_register_merge_strategy(STRATEGY_DONT_MERGE);
  return codegen_options;
}

void CodegenSucceedsForEveryFunctionWithNoPipelineLimit(
    absl::StatusOr<std::string> verilog) {
  EXPECT_THAT(verilog, IsOkAndHolds(Not(IsEmpty())));
}
FUZZ_TEST(CodegenFuzzTest, CodegenSucceedsForEveryFunctionWithNoPipelineLimit)
    .WithDomains(StatusOrVerilogFuzzDomain(DefaultSchedulingOptions(),
                                           DefaultCodegenOptions()));

}  // namespace
}  // namespace xls

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

#include "xls/fuzzer/verilog_fuzzer/verilog_fuzz_domain.h"

#include <cstdint>
#include <memory>
#include <string>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/ir/package.h"
#include "xls/public/runtime_build_actions.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

fuzztest::Domain<absl::StatusOr<std::string>> StatusOrVerilogFuzzDomain(
    SchedulingOptionsFlagsProto scheduling_options,
    CodegenFlagsProto codegen_options) {
  return fuzztest::Map(
      [scheduling_options, codegen_options](
          std::shared_ptr<Package> package) -> absl::StatusOr<std::string> {
        XLS_RET_CHECK_EQ(package->functions().size(), 1)
            << "We expect IrFuzzDomain() to return a package with exactly one "
               "function.\n"
            << package->DumpIr();
        // Make the function the top-level module.
        XLS_RETURN_IF_ERROR(package->SetTop(package->functions().front().get()))
            << "Failed to set top-level module in:\n"
            << package->DumpIr();

        XLS_ASSIGN_OR_RETURN(
            (auto [scheduling_result, codegen_result]),
            ScheduleAndCodegenPackage(package.get(), scheduling_options,
                                      codegen_options,
                                      /*with_delay_model=*/true),
            _ << " while scheduling and codegening package:\n"
              << package->DumpIr());

        XLS_LOG_LINES(INFO, package->DumpIr());
        XLS_LOG_LINES(INFO, codegen_result.verilog_text);

        return codegen_result.verilog_text;
      },
      IrFuzzDomain());
}

fuzztest::Domain<std::string> VerilogFuzzDomain(
    SchedulingOptionsFlagsProto scheduling_options,
    CodegenFlagsProto codegen_options) {
  return fuzztest::Map(
      [](absl::StatusOr<std::string> verilog) -> std::string {
        // Should be filtered
        CHECK_OK(verilog.status());
        return *verilog;
      },
      fuzztest::Filter(
          [](const absl::StatusOr<std::string> &verilog) {
            return verilog.ok();
          },
          StatusOrVerilogFuzzDomain(scheduling_options, codegen_options)));
}

fuzztest::Domain<CodegenFlagsProto> CodegenFlagsDomain() {
  return fuzztest::Arbitrary<CodegenFlagsProto>()
      .WithInt64Field("codegen_version",
                      fuzztest::ElementOf<int64_t>({0, 1, 2}))
      .WithEnumFieldAlwaysSet(
          "generator", fuzztest::ElementOf<int>(
                           {GeneratorKind::GENERATOR_KIND_PIPELINE,
                            GeneratorKind::GENERATOR_KIND_COMBINATIONAL}))
      // We don't set this field because we set "top" via the IR domain.
      .WithFieldUnset("top");
}

fuzztest::Domain<SchedulingOptionsFlagsProto>
NoFdoSchedulingOptionsFlagsDomain() {
  return fuzztest::Arbitrary<SchedulingOptionsFlagsProto>()
      .WithFieldUnset("use_fdo")
      .WithFieldUnset("fdo_iteration_number")
      .WithFieldUnset("fdo_delay_driven_path_number")
      .WithFieldUnset("fdo_fanout_driven_path_number")
      .WithFieldUnset("fdo_refinement_stochastic_ratio")
      .WithFieldUnset("fdo_path_evaluate_strategy")
      .WithFieldUnset("fdo_synthesizer_name")
      .WithFieldUnset("fdo_yosys_path")
      .WithFieldUnset("fdo_sta_path")
      .WithFieldUnset("fdo_synthesis_libraries")
      .WithFieldUnset("fdo_default_driver_cell")
      .WithFieldUnset("fdo_default_load");
}

}  // namespace xls

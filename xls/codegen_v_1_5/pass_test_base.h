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

#ifndef XLS_CODEGEN_V_1_5_PASS_TEST_BASE_H_
#define XLS_CODEGEN_V_1_5_PASS_TEST_BASE_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/golden_files.h"
#include "xls/common/source_location.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"

namespace xls::codegen {

// A base class for tests for a single pass in the block conversion pass
// pipeline.
template <typename PassType>
class PassTestBase : public IrTestBase {
 protected:
  static constexpr std::string_view kTestDataPath =
      "xls/codegen_v_1_5/testdata";

  explicit PassTestBase(std::string_view test_suite_name)
      : test_suite_name_(test_suite_name) {}

  absl::StatusOr<std::string> RunPassAndRoundTripIrText(
      std::string_view input_ir, bool expect_change = true,
      std::optional<verilog::CodegenOptions> codegen_options = std::nullopt,
      std::optional<SchedulingOptions> scheduling_options = std::nullopt) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         Parser::ParsePackageNoVerify(input_ir));

    PassType pass;
    PassResults results;

    BlockConversionPassOptions options;
    if (codegen_options.has_value()) {
      options.codegen_options = *codegen_options;
    } else {
      options.codegen_options.clock_name("clk").reset("rst", false, false,
                                                      false);
    }
    if (scheduling_options.has_value()) {
      TestDelayEstimator delay_estimator;
      XLS_ASSIGN_OR_RETURN(
          SchedulingResult scheduling_result,
          Schedule(package.get(), *scheduling_options, &delay_estimator));
      options.package_schedule = std::move(scheduling_result.package_schedule);
    }

    XLS_ASSIGN_OR_RETURN(bool result,
                         pass.Run(package.get(), options, &results));

    XLS_RET_CHECK(expect_change == result);

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> round_tripped_package,
                         Parser::ParsePackageNoVerify(package->DumpIr()));
    XLS_RETURN_IF_ERROR(VerifyPackage(round_tripped_package.get(),
                                      {.incomplete_lowering = true}));
    return round_tripped_package->DumpIr();
  }

  void ExpectEqualToGoldenFile(
      std::string_view actual_ir,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
    ::xls::ExpectEqualToGoldenFile(
        absl::Substitute("$0/$1_$2.ir", kTestDataPath, test_suite_name_,
                         TestName()),
        actual_ir, loc);
  }

 private:
  std::string test_suite_name_;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_PASS_TEST_BASE_H_

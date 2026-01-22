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

#include "xls/codegen_v_1_5/convert_to_block.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_pass_pipeline.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/schedule.h"

namespace xls::codegen {

absl::Status ConvertToBlock(
    Package* p, verilog::CodegenOptions codegen_options,
    SchedulingOptions scheduling_options, const DelayEstimator* delay_estimator,
    std::optional<PackageScheduleProto> schedule_override,
    OptimizationContext* opt_context) {
  XLS_RET_CHECK(!schedule_override.has_value() ||
                schedule_override->schedules_size() == 0 ||
                !codegen_options.generate_combinational())
      << "A schedule must not be specified for combinational block conversion.";

  PackageScheduleProto schedule;
  if (!codegen_options.generate_combinational()) {
    if (schedule_override.has_value()) {
      schedule = std::move(*schedule_override);
    } else {
      XLS_ASSIGN_OR_RETURN(SchedulingResult scheduling_result,
                           Schedule(p, scheduling_options, delay_estimator));
      schedule = scheduling_result.package_schedule;
    }
  }

  BlockConversionPassOptions options{
      .codegen_options = std::move(codegen_options),
      .package_schedule = std::move(schedule),
  };

  std::optional<OptimizationContext> opt_context_local;
  if (opt_context == nullptr) {
    opt_context_local.emplace();
    opt_context = &*opt_context_local;
  }
  std::unique_ptr<BlockConversionCompoundPass> pipeline =
      CreateBlockConversionPassPipeline(*opt_context);
  PassResults results;
  XLS_ASSIGN_OR_RETURN(bool result, pipeline->Run(p, options, &results));
  XLS_RET_CHECK(result);
  return absl::OkStatus();
}

}  // namespace xls::codegen

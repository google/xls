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
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_pass_pipeline.h"
#include "xls/scheduling/scheduling_result.h"

namespace xls {
namespace {

absl::StatusOr<SchedulingResult> RunSchedulingPipeline(
    FunctionBase* main, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator,
    synthesis::Synthesizer* synthesizer) {
  PassResults results;
  SchedulingPassOptions sched_options;
  sched_options.scheduling_options = scheduling_options;
  sched_options.delay_estimator = delay_estimator;
  sched_options.synthesizer = synthesizer;
  OptimizationContext optimization_context;
  std::unique_ptr<SchedulingCompoundPass> scheduling_pipeline =
      CreateSchedulingPassPipeline(optimization_context, scheduling_options);
  XLS_RETURN_IF_ERROR(main->package()->SetTop(main));
  auto scheduling_context =
      (scheduling_options.schedule_all_procs())
          ? SchedulingContext::CreateForWholePackage(main->package())
          : SchedulingContext::CreateForSingleFunction(main);
  absl::Status scheduling_status =
      scheduling_pipeline
          ->Run(main->package(), sched_options, &results, scheduling_context)
          .status();
  if (!scheduling_status.ok()) {
    if (absl::IsResourceExhausted(scheduling_status)) {
      // Resource exhausted error indicates that the schedule was
      // infeasible. Emit a meaningful error in this case.
      std::string error_message = "Design cannot be scheduled";
      if (scheduling_options.pipeline_stages().has_value()) {
        absl::StrAppendFormat(&error_message, " in %d stages",
                              scheduling_options.pipeline_stages().value());
      }
      if (scheduling_options.clock_period_ps().has_value()) {
        absl::StrAppendFormat(&error_message, " with a %dps clock",
                              scheduling_options.clock_period_ps().value());
      }
      return xabsl::StatusBuilder(scheduling_status).SetPrepend()
             << error_message << ": ";
    }
    return scheduling_status;
  }
  XLS_RET_CHECK(scheduling_context.package_schedule().HasSchedule(main));
  return SchedulingResult{
      .package_schedule =
          scheduling_context.package_schedule().ToProto(*delay_estimator),
      .pass_pipeline_metrics = results.ToProto()};
}

}  // namespace

absl::StatusOr<SchedulingResult> Schedule(
    Package* p, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator) {
  QCHECK(scheduling_options.pipeline_stages() != 0 ||
         scheduling_options.clock_period_ps() != 0)
      << "Must specify --pipeline_stages or --clock_period_ps (or both).";
  synthesis::Synthesizer* synthesizer = nullptr;
  if (scheduling_options.use_fdo() &&
      !scheduling_options.fdo_synthesizer_name().empty()) {
    XLS_ASSIGN_OR_RETURN(synthesizer, SetUpSynthesizer(scheduling_options));
  }
  return RunSchedulingPipeline(*p->GetTop(), scheduling_options,
                               delay_estimator, synthesizer);
}

}  // namespace xls

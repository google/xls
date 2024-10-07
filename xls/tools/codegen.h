// Copyright 2024 The XLS Authors
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

#include <optional>
#include <variant>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

#ifndef XLS_TOOLS_CODEGEN_H_
#define XLS_TOOLS_CODEGEN_H_

namespace xls {

using PipelineScheduleOrGroup =
    std::variant<PipelineSchedule, PackagePipelineSchedules>;

absl::StatusOr<PipelineScheduleOrGroup> Schedule(
    Package* p, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator,
    absl::Duration* scheduling_time = nullptr);

struct CodegenResult {
  verilog::ModuleGeneratorResult module_generator_result;
  std::optional<PackagePipelineSchedulesProto>
      package_pipeline_schedules_proto = std::nullopt;
};

absl::StatusOr<CodegenResult> CodegenPipeline(
    Package* p, PipelineScheduleOrGroup schedules,
    const verilog::CodegenOptions& codegen_options,
    const DelayEstimator* delay_estimator,
    absl::Duration* codegen_time = nullptr);

absl::StatusOr<CodegenResult> CodegenCombinational(
    Package* p, const verilog::CodegenOptions& codegen_options,
    const DelayEstimator* delay_estimator,
    absl::Duration* codegen_time = nullptr);

absl::StatusOr<verilog::CodegenOptions> CodegenOptionsFromProto(
    const CodegenFlagsProto& p);

struct TimingReport {
  absl::Duration scheduling_time;
  absl::Duration codegen_time;
};

// Run scheduling and/or codegen based on options from the given flags protos.
absl::StatusOr<PipelineScheduleOrGroup> Schedule(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto,
    absl::Duration* scheduling_time);
absl::StatusOr<CodegenResult> Codegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model,
    const PipelineScheduleOrGroup* schedules, absl::Duration* codegen_time);
absl::StatusOr<CodegenResult> ScheduleAndCodegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model,
    TimingReport* timing_report = nullptr);

}  // namespace xls

#endif  // XLS_TOOLS_CODEGEN_H_

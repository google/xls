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

#include <utility>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

#ifndef XLS_TOOLS_CODEGEN_H_
#define XLS_TOOLS_CODEGEN_H_

namespace xls {

absl::StatusOr<SchedulingResult> Schedule(
    Package* p, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator);

absl::StatusOr<verilog::CodegenOptions> CodegenOptionsFromProto(
    const CodegenFlagsProto& p);

// Run scheduling and/or codegen based on options from the given flags protos.
absl::StatusOr<SchedulingResult> Schedule(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto);
absl::StatusOr<SchedulingResult> Schedule(
    Package* p, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator);
absl::StatusOr<verilog::CodegenResult> Codegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model,
    const PackageSchedule* packge_schedule);
absl::StatusOr<std::pair<SchedulingResult, verilog::CodegenResult>>
ScheduleAndCodegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model);

}  // namespace xls

#endif  // XLS_TOOLS_CODEGEN_H_

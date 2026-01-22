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

#ifndef XLS_CODEGEN_V_1_5_CODEGEN_H_
#define XLS_CODEGEN_V_1_5_CODEGEN_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls::codegen {

absl::StatusOr<verilog::CodegenResult> Codegen(
    Package* package, const verilog::CodegenOptions& codegen_options,
    const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator,
    std::optional<PackageScheduleProto> schedule = std::nullopt);

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_CODEGEN_H_

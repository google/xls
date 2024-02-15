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

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

#ifndef XLS_TOOLS_CODEGEN_H_
#define XLS_TOOLS_CODEGEN_H_

namespace xls {

struct CodegenResult {
  verilog::ModuleGeneratorResult module_generator_result;
  std::optional<PackagePipelineSchedulesProto>
      package_pipeline_schedules_proto = std::nullopt;
};

absl::StatusOr<verilog::CodegenOptions> CodegenOptionsFromProto(
    const CodegenFlagsProto& p);

absl::StatusOr<CodegenResult> ScheduleAndCodegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model);

}  // namespace xls

#endif  // XLS_TOOLS_CODEGEN_H_

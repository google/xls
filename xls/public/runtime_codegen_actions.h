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

#ifndef XLS_PUBLIC_RUNTIME_CODEGEN_ACTIONS_H_
#define XLS_PUBLIC_RUNTIME_CODEGEN_ACTIONS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_result.h"
#include "xls/public/ir.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

struct ScheduleAndCodegenResult {
  SchedulingResult scheduling_result;
  verilog::CodegenResult codegen_result;
};

absl::StatusOr<ScheduleAndCodegenResult> ScheduleAndCodegenPackage(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model);
}  // namespace xls

#endif  // XLS_PUBLIC_RUNTIME_CODEGEN_ACTIONS_H_

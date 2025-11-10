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

#include "xls/public/runtime_codegen_actions.h"

#include <utility>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_result.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

absl::StatusOr<ScheduleAndCodegenResult> ScheduleAndCodegenPackage(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model) {
  std::pair<SchedulingResult, verilog::CodegenResult> result;
  XLS_ASSIGN_OR_RETURN(
      result, ScheduleAndCodegen(p, scheduling_options_flags_proto,
                                 codegen_flags_proto, with_delay_model));
  return ScheduleAndCodegenResult{.scheduling_result = result.first,
                                  .codegen_result = result.second};
}
}  // namespace xls

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

#include "xls/codegen/passes_ng/stage_conversion_pass.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

absl::StatusOr<bool> StageConversionPass::RunOnFunctionInternal(
    CodegenPassUnit* unit, Function* f, const PipelineSchedule& schedule,
    const CodegenPassOptions& options, CodegenPassResults* results) const {
  VLOG(3) << "Converting function ir to stage ir:";
  XLS_VLOG_LINES(3, f->DumpIr());

  XLS_RETURN_IF_ERROR(SingleFunctionBaseToPipelinedStages(
      f->name(), schedule, options.codegen_options,
      unit->stage_conversion_metadata()));

  return true;
}

}  // namespace xls::verilog

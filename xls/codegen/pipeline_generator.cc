// Copyright 2020 The XLS Authors
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

#include "xls/codegen/pipeline_generator.h"

#include <algorithm>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const CodegenOptions& options) {
  return ToPipelineModuleText(schedule, static_cast<FunctionBase*>(func),
                              options);
}

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, FunctionBase* module,
    const CodegenOptions& options) {
  XLS_VLOG(2) << "Generating pipelined module for module:";
  XLS_VLOG_LINES(2, module->DumpIr());
  XLS_VLOG_LINES(2, schedule.ToString());

  Block* block;

  CodegenPassOptions pass_options;
  pass_options.codegen_options = options;
  pass_options.schedule = schedule;

  XLS_RET_CHECK(module->IsProc() || module->IsFunction());
  // Convert to block and add in pipe stages according to schedule.
  if (module->IsFunction()) {
    Function* func = module->AsFunctionOrDie();
    XLS_ASSIGN_OR_RETURN(block,
                         FunctionToPipelinedBlock(schedule, options, func));
  } else {
    Proc* proc = module->AsProcOrDie();
    XLS_ASSIGN_OR_RETURN(block, ProcToPipelinedBlock(schedule, options, proc));

    // Force using non-pretty printed codegen when generating procs.
    // TODO(tedhong): 2021-09-25 - Update pretty-printer to support
    //  blocks with flow control.
    pass_options.codegen_options.emit_as_pipeline(false);
  }

  CodegenPassUnit unit(module->package(), block);
  PassResults results;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline()->Run(&unit, pass_options, &results).status());
  XLS_RET_CHECK(unit.signature.has_value());
  VerilogLineMap verilog_line_map;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(block, pass_options.codegen_options, &verilog_line_map));

  return ModuleGeneratorResult{verilog, verilog_line_map,
                               unit.signature.value()};
}

}  // namespace verilog
}  // namespace xls

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

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const CodegenOptions& options) {
  XLS_VLOG(2) << "Generating pipelined module for function:";
  XLS_VLOG_LINES(2, func->DumpIr());
  XLS_VLOG_LINES(2, schedule.ToString());

  XLS_ASSIGN_OR_RETURN(Block * block,
                       FunctionToPipelinedBlock(schedule, options, func));

  CodegenPassUnit unit(func->package(), block);
  CodegenPassOptions pass_options;
  pass_options.codegen_options = options;
  pass_options.schedule = schedule;
  PassResults results;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline()->Run(&unit, pass_options, &results).status());
  XLS_RET_CHECK(unit.signature.has_value());
  XLS_ASSIGN_OR_RETURN(std::string verilog,
                       GenerateVerilog(block, pass_options.codegen_options));

  return ModuleGeneratorResult{verilog, unit.signature.value()};
}

}  // namespace verilog
}  // namespace xls

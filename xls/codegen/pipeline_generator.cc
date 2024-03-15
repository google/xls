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
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const CodegenOptions& options, const DelayEstimator* delay_estimator) {
  return ToPipelineModuleText(schedule, static_cast<FunctionBase*>(func),
                              options, delay_estimator);
}

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, FunctionBase* module,
    const CodegenOptions& options, const DelayEstimator* delay_estimator) {
  XLS_VLOG(2) << "Generating pipelined module for module:";
  XLS_VLOG_LINES(2, module->DumpIr());
  XLS_VLOG_LINES(2, schedule.ToString());

  CodegenPassOptions pass_options;
  pass_options.codegen_options = options;
  pass_options.schedule = schedule;
  pass_options.delay_estimator = delay_estimator;

  // Convert to block and add in pipe stages according to schedule.
  XLS_ASSIGN_OR_RETURN(CodegenPassUnit unit,
                       FunctionBaseToPipelinedBlock(schedule, options, module));
  if (module->IsProc()) {
    // Force using non-pretty printed codegen when generating procs.
    // TODO: google/xls#1331 - Update pretty-printer to support blocks with flow
    // control.
    pass_options.codegen_options.emit_as_pipeline(false);
  }

  PassResults results;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline()->Run(&unit, pass_options, &results).status());
  XLS_RET_CHECK(unit.top_block != nullptr &&
                unit.metadata.contains(unit.top_block) &&
                unit.metadata.at(unit.top_block).signature.has_value());
  VerilogLineMap verilog_line_map;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(unit.top_block, pass_options.codegen_options,
                      &verilog_line_map));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return ModuleGeneratorResult{
      verilog, verilog_line_map,
      unit.metadata.at(unit.top_block).signature.value()};
}

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PackagePipelineSchedules& schedules, Package* package,
    const CodegenOptions& options, const DelayEstimator* delay_estimator) {
  XLS_VLOG(2) << "Generating pipelined module for module:";
  XLS_VLOG_LINES(2, package->DumpIr());
  if (VLOG_IS_ON(2)) {
    for (const auto& [_, schedule] : schedules) {
      XLS_VLOG_LINES(2, schedule.ToString());
    }
  }

  CodegenPassOptions pass_options;
  pass_options.codegen_options = options;
  pass_options.delay_estimator = delay_estimator;

  // Convert to block and add in pipe stages according to schedule.
  XLS_ASSIGN_OR_RETURN(CodegenPassUnit unit,
                       PackageToPipelinedBlocks(schedules, options, package));
  if (std::any_of(
          schedules.begin(), schedules.end(),
          [](const std::pair<FunctionBase*, PipelineSchedule>& element) {
            return element.first->IsProc();
          })) {
    // Force using non-pretty printed codegen when generating procs.
    // TODO: google/xls#1331 - Update pretty-printer to support blocks with flow
    // control.
    // TODO: google/xls#1332 - Update this setting per-block.
    pass_options.codegen_options.emit_as_pipeline(false);
  }

  PassResults results;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline()->Run(&unit, pass_options, &results).status());
  XLS_RET_CHECK(unit.top_block != nullptr &&
                unit.metadata.contains(unit.top_block) &&
                unit.metadata.at(unit.top_block).signature.has_value());
  VerilogLineMap verilog_line_map;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(unit.top_block, options, &verilog_line_map));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return ModuleGeneratorResult{
      verilog, verilog_line_map,
      unit.metadata.at(unit.top_block).signature.value()};
}

}  // namespace verilog
}  // namespace xls

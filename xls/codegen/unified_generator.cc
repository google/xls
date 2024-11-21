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

#include "xls/codegen/unified_generator.h"

#include <algorithm>
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
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> GenerateModuleText(
    const PackagePipelineSchedules& schedules, Package* package,
    const CodegenOptions& options, const DelayEstimator* delay_estimator) {
  VLOG(2) << "Generating module for package:";
  XLS_VLOG_LINES(2, package->DumpIr());
  if (VLOG_IS_ON(2)) {
    for (const auto& [_, schedule] : schedules) {
      XLS_VLOG_LINES(2, schedule.ToString());
    }
  }

  CodegenPassOptions pass_options;
  pass_options.codegen_options = options;
  pass_options.delay_estimator = delay_estimator;

  // TODO(tedhong): 2024-11-15 - Make passes that can be done at the IR level
  // into to IR to IR codegen passes.  Passes that will be needed to unify
  // codegen are:
  //   1. Combinational function to proc w/ single-value channels.
  //   2. Pipelined function to proc w/ single-value or valid-only streaming
  //   channels.
  //
  // For now, these passes call the existing generators and go directly from IR
  // to Block IR.
  XLS_ASSIGN_OR_RETURN(
      CodegenPassUnit unit,
      options.generate_combinational()
          ? FunctionBaseToCombinationalBlock(*package->GetTop(), options)
          : PackageToPipelinedBlocks(schedules, options, package));

  // Block to Block codegen passes.
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

  CodegenPassResults results;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline()->Run(&unit, pass_options, &results).status());
  XLS_RET_CHECK(unit.top_block != nullptr &&
                unit.metadata.contains(unit.top_block) &&
                unit.metadata.at(unit.top_block).signature.has_value());

  // VAST Generation: Block to Verilog codegen pass.
  VerilogLineMap verilog_line_map;
  const auto& pipeline =
      unit.metadata[unit.top_block].streaming_io_and_pipeline;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(unit.top_block, options, &verilog_line_map,
                      pipeline.input_port_sv_type,
                      pipeline.output_port_sv_type));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return ModuleGeneratorResult{
      verilog, verilog_line_map,
      unit.metadata.at(unit.top_block).signature.value()};
}

}  // namespace verilog
}  // namespace xls

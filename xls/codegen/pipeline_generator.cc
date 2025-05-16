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
#include "absl/log/vlog_is_on.h"
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
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const CodegenOptions& options, const DelayEstimator* delay_estimator,
    PassPipelineMetricsProto* metrics) {
  return ToPipelineModuleText(schedule, static_cast<FunctionBase*>(func),
                              options, delay_estimator, metrics);
}

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, FunctionBase* module,
    const CodegenOptions& options, const DelayEstimator* delay_estimator,
    PassPipelineMetricsProto* metrics) {
  VLOG(2) << "Generating pipelined module for module:";
  XLS_VLOG_LINES(2, module->DumpIr());
  XLS_VLOG_LINES(2, schedule.ToString());

  CodegenPassOptions pass_options;
  pass_options.codegen_options = options;
  pass_options.schedule = schedule;
  pass_options.delay_estimator = delay_estimator;

  // Convert to block and add in pipe stages according to schedule.
  XLS_ASSIGN_OR_RETURN(CodegenContext context,
                       FunctionBaseToPipelinedBlock(schedule, options, module));
  if (module->IsProc()) {
    // Force using non-pretty printed codegen when generating procs.
    // TODO: google/xls#1331 - Update pretty-printer to support blocks with flow
    // control.
    pass_options.codegen_options.emit_as_pipeline(false);
  }

  PassResults results;
  OptimizationContext opt_context;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline(opt_context)
          ->Run(module->package(), pass_options, &results, context)
          .status());
  if (metrics != nullptr) {
    *metrics = results.ToProto();
  }
  XLS_RET_CHECK(context.top_block() != nullptr &&
                context.HasMetadataForBlock(context.top_block()) &&
                context.top_block()->GetSignature().has_value());

  VerilogLineMap verilog_line_map;
  const auto& pipeline = context.GetMetadataForBlock(context.top_block())
                             .streaming_io_and_pipeline;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(context.top_block(), pass_options.codegen_options,
                      &verilog_line_map, pipeline.input_port_sv_type,
                      pipeline.output_port_sv_type));

  XLS_ASSIGN_OR_RETURN(
      ModuleSignature signature,
      ModuleSignature::FromProto(*context.top_block()->GetSignature()));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return ModuleGeneratorResult{verilog, verilog_line_map, std::move(signature)};
}

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PackagePipelineSchedules& schedules, Package* package,
    const CodegenOptions& options, const DelayEstimator* delay_estimator,
    PassPipelineMetricsProto* metrics) {
  VLOG(2) << "Generating pipelined module for module:";
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
  XLS_ASSIGN_OR_RETURN(CodegenContext context,
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
  OptimizationContext opt_context;
  XLS_RETURN_IF_ERROR(CreateCodegenPassPipeline(opt_context)
                          ->Run(package, pass_options, &results, context)
                          .status());
  if (metrics != nullptr) {
    *metrics = results.ToProto();
  }

  XLS_RET_CHECK(context.top_block() != nullptr &&
                context.HasMetadataForBlock(context.top_block()) &&
                context.top_block()->GetSignature().has_value());
  VerilogLineMap verilog_line_map;
  const auto& pipeline = context.GetMetadataForBlock(context.top_block())
                             .streaming_io_and_pipeline;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(context.top_block(), options, &verilog_line_map,
                      pipeline.input_port_sv_type,
                      pipeline.output_port_sv_type));

  XLS_ASSIGN_OR_RETURN(
      ModuleSignature signature,
      ModuleSignature::FromProto(*context.top_block()->GetSignature()));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return ModuleGeneratorResult{verilog, verilog_line_map, std::move(signature)};
}

}  // namespace verilog
}  // namespace xls

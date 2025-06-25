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

#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/block_conversion_pass_pipeline.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/block_metrics.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

absl::StatusOr<CodegenResult> GenerateModuleText(
    const PackageSchedule& package_schedule, Package* package,
    const CodegenOptions& options, const DelayEstimator* delay_estimator) {
  VLOG(2) << "Generating module for package:";
  XLS_VLOG_LINES(2, package->DumpIr());
  if (VLOG_IS_ON(2)) {
    XLS_VLOG_LINES(2, package_schedule.ToString());
  }

  // TODO(tedhong): 2024-11-15 - Make passes that can be done at the IR level
  // into to IR to IR codegen passes.  Passes that will be needed to unify
  // codegen are:
  //   1. Combinational function to proc w/ single-value channels.
  //   2. Pipelined function to proc w/ single-value or valid-only streaming
  //   channels.
  //
  // For now, these passes call the existing generators and go directly from IR
  // to Block IR.
  XLS_ASSIGN_OR_RETURN(CodegenContext codegen_context,
                       CreateBlocksFor(package_schedule, options, package));

  // Note: this is mutated below so cannot be const. It would be nice to
  // refactor this so it could be.
  CodegenPassOptions pass_options = {
      .codegen_options = options,
      .delay_estimator = delay_estimator,
  };

  PassResults results;
  OptimizationContext opt_context;
  XLS_RETURN_IF_ERROR(
      CreateBlockConversionPassPipeline(options, opt_context)
          ->Run(package, pass_options, &results, codegen_context)
          .status());

  // Block to Block codegen passes.
  if (absl::c_any_of(
          package_schedule.GetSchedules(),
          [](const std::pair<FunctionBase*, PipelineSchedule>& element) {
            return element.first->IsProc();
          })) {
    // Force using non-pretty printed codegen when generating procs.
    // TODO: google/xls#1331 - Update pretty-printer to support blocks with flow
    // control.
    // TODO: google/xls#1332 - Update this setting per-block.
    pass_options.codegen_options.emit_as_pipeline(false);
  }

  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline(opt_context)
          ->Run(package, pass_options, &results, codegen_context)
          .status());
  XLS_RET_CHECK(
      codegen_context.HasTopBlock() &&
      codegen_context.HasMetadataForBlock(codegen_context.top_block()) &&
      codegen_context.top_block()->GetSignature().has_value());

  // VAST Generation: Block to Verilog codegen pass.
  VerilogLineMap verilog_line_map;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      GenerateVerilog(codegen_context.top_block(), options, &verilog_line_map));

  XLS_ASSIGN_OR_RETURN(
      ModuleSignature signature,
      ModuleSignature::FromProto(*codegen_context.top_block()->GetSignature()));

  XlsMetricsProto metrics;
  XLS_ASSIGN_OR_RETURN(
      *metrics.mutable_block_metrics(),
      GenerateBlockMetrics(codegen_context.top_block(), delay_estimator));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return CodegenResult{.verilog_text = verilog,
                       .verilog_line_map = verilog_line_map,
                       .signature = signature,
                       .block_metrics = metrics,
                       .pass_pipeline_metrics = results.ToProto()};
}

}  // namespace verilog
}  // namespace xls

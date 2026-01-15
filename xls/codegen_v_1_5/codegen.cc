// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/codegen.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xls/codegen/block_metrics.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_residual_data.pb.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/codegen_wrapper_pass.h"
#include "xls/codegen/ffi_instantiation_pass.h"
#include "xls/codegen/maybe_materialize_fifos_pass.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/mulp_combining_pass.h"
#include "xls/codegen/name_legalization_pass.h"
#include "xls/codegen/port_legalization_pass.h"
#include "xls/codegen/priority_select_reduction_pass.h"
#include "xls/codegen/ram_rewrite_pass.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/codegen/trace_verbosity_pass.h"
#include "xls/codegen/verilog_conversion.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/codegen_v_1_5/convert_to_block.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/block.h"
#include "xls/ir/package.h"
#include "xls/passes/basic_simplification_pass.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls::codegen {

namespace {

std::unique_ptr<verilog::CodegenCompoundPass> CreatePostBlockConversionPipeline(
    OptimizationContext& context) {
  // Reuses a subset of the pipeline from codegen 1.0 for several operations
  // that can take place after block conversion.
  // For reference, see: xls/codegen/codegen_pass_pipeline.cc

  auto top = std::make_unique<verilog::CodegenCompoundPass>(
      "post_block_conversion", "Post-block-conversion pass pipeline");

  // Rewrite channels that codegen options have labeled as to/from a RAM. This
  // removes ready+valid ports, instead AND-ing the request valid signal with
  // the read- and write-enable signals, as well as adding a skid buffer on the
  // response channel.
  top->Add<verilog::RamRewritePass>();

  // Remove zero-width input/output ports.
  // TODO(meheff): 2021/04/29 Also flatten ports with types here.
  top->Add<verilog::PortLegalizationPass>();

  // Remove zero-width registers.
  top->Add<verilog::RegisterLegalizationPass>();

  // Eliminate no-longer-needed partial product operations by turning them into
  // normal multiplies.
  top->Add<verilog::MulpCombiningPass>();

  // Create instantiations from ffi invocations.
  top->Add<verilog::FfiInstantiationPass>();

  // Filter out traces filtered by verbosity config.
  top->Add<verilog::TraceVerbosityPass>();

  // Replace provably-unneeded priority-select operations with simpler selects.
  top->Add<verilog::PrioritySelectReductionPass>();

  // Deduplicate registers across mutually exclusive stages.
  // TODO(epastor): Reimplement to avoid the need for metadata support.
  // top->Add<RegisterCombiningPass>();

  // Remove any identity ops which might have been added earlier in the
  // pipeline.
  top->Add<verilog::CodegenWrapperPass>(std::make_unique<IdentityRemovalPass>(),
                                        context);

  // Do some trivial simplifications to any flow control logic added during code
  // generation.
  top->Add<verilog::CodegenWrapperPass>(
      std::make_unique<CsePass>(/*common_literals=*/false), context);
  top->Add<verilog::CodegenWrapperPass>(
      std::make_unique<BasicSimplificationPass>(), context);

  // Swap out fifo instantiations with materialized fifos where required by
  // codegen options.
  top->Add<verilog::MaybeMaterializeFifosPass>();

  // Final dead-code elimination pass to remove cruft left from earlier passes.
  top->Add<verilog::CodegenWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>(), context);

  // Legalize names.
  top->Add<verilog::NameLegalizationPass>();

  return top;
}

absl::StatusOr<verilog::CodegenResult> ConvertToVerilog(
    Block* top_block, const verilog::CodegenOptions& options,
    const DelayEstimator* delay_estimator) {
  XLS_RET_CHECK(top_block->GetSignature().has_value());
  verilog::VerilogLineMap verilog_line_map;
  verilog::CodegenResidualData residual_data;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      verilog::GenerateVerilog(top_block, options, &verilog_line_map,
                               &residual_data));

  XLS_ASSIGN_OR_RETURN(auto signature, verilog::ModuleSignature::FromProto(
                                           *top_block->GetSignature()));

  verilog::XlsMetricsProto metrics;
  XLS_ASSIGN_OR_RETURN(
      *metrics.mutable_block_metrics(),
      verilog::GenerateBlockMetrics(top_block, delay_estimator));

  // TODO: google/xls#1323 - add all block signatures to ModuleGeneratorResult,
  // not just top.
  return verilog::CodegenResult{
      .verilog_text = verilog,
      .verilog_line_map = std::move(verilog_line_map),
      .signature = signature,
      .block_metrics = metrics,
      .pass_pipeline_metrics = {},
  };
}

}  // namespace

absl::StatusOr<verilog::CodegenResult> Codegen(
    Package* package, const verilog::CodegenOptions& codegen_options,
    const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator,
    std::optional<PackageScheduleProto> schedule) {
  verilog::CodegenOptions options = codegen_options;
  if (package->GetTop().has_value() && package->GetTop().value()->IsProc()) {
    // Force using non-pretty printed Verilog when generating procs.
    // TODO - Update pretty-printer to support blocks with flow control.
    options.emit_as_pipeline(false);
  }

  OptimizationContext opt_context;
  XLS_RETURN_IF_ERROR(ConvertToBlock(package, options, scheduling_options,
                                     delay_estimator, schedule, &opt_context));

  // Now that we've finished block conversion for the package, we can run the
  // remaining passes required before Verilog conversion.
  XLS_RET_CHECK(package->GetTop().has_value() && package->GetTop() != nullptr);
  XLS_ASSIGN_OR_RETURN(Block * top_block, package->GetTopAsBlock());
  verilog::CodegenContext compatibility_context(top_block);
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    compatibility_context.metadata().insert(
        {block.get(), verilog::CodegenMetadata{}});
  }
  PassResults pass_results;
  verilog::CodegenPassOptions pass_options{
      .codegen_options = options,
  };
  XLS_RETURN_IF_ERROR(
      CreatePostBlockConversionPipeline(opt_context)
          ->Run(package, pass_options, &pass_results, compatibility_context)
          .status());

  // Finally, we convert the block to Verilog.
  return ConvertToVerilog(top_block, options, delay_estimator);
}

}  // namespace xls::codegen

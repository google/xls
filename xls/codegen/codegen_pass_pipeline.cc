// Copyright 2021 The XLS Authors
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

// Create a standard pipeline of passes. This pipeline should
// be used in the main driver as well as in testing.

#include "xls/codegen/codegen_pass_pipeline.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xls/codegen/block_metrics_generation_pass.h"
#include "xls/codegen/block_stitching_pass.h"
#include "xls/codegen/codegen_checker.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_wrapper_pass.h"
#include "xls/codegen/ffi_instantiation_pass.h"
#include "xls/codegen/materialize_fifos_pass.h"
#include "xls/codegen/mulp_combining_pass.h"
#include "xls/codegen/name_legalization_pass.h"
#include "xls/codegen/port_legalization_pass.h"
#include "xls/codegen/priority_select_reduction_pass.h"
#include "xls/codegen/ram_rewrite_pass.h"
#include "xls/codegen/register_combining_pass.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/codegen/side_effect_condition_pass.h"
#include "xls/codegen/signature_generation_pass.h"
#include "xls/codegen/trace_verbosity_pass.h"
#include "xls/ir/block.h"
#include "xls/passes/basic_simplification_pass.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

std::unique_ptr<CodegenCompoundPass> CreateCodegenPassPipeline(
    OptimizationContext* context) {
  auto top = std::make_unique<CodegenCompoundPass>(
      "codegen", "Top level codegen pass pipeline");
  top->AddInvariantChecker<CodegenChecker>();

  // Stitch multi-block designs together in a top-level block that instantiates
  // and stitches the others.
  top->Add<BlockStitchingPass>();

  // Generate the signature from the initial proc and options prior to any
  // transformations. If necessary the signature can be mutated later if the
  // proc is transformed in a way which affects its externally visible
  // interface.
  top->Add<SignatureGenerationPass>();

  // Rewrite channels that codegen options have labeled as to/from a RAM. This
  // removes ready+valid ports, instead AND-ing the request valid signal with
  // the read- and write-enable signals, as well as adding a skid buffer on the
  // response channel.
  top->Add<RamRewritePass>();

  // TODO(meheff): 2021/04/29. Add the following passes:
  // * pass to optionally generate pipeline.
  // * pass to optionally flop inputs and outputs.

  // Remove zero-width input/output ports.
  // TODO(meheff): 2021/04/29 Also flatten ports with types here.
  top->Add<PortLegalizationPass>();

  // Remove zero-width registers.
  top->Add<RegisterLegalizationPass>();

  // Eliminate no-longer-needed partial product operations by turning them into
  // normal multiplies.
  top->Add<MulpCombiningPass>();

  // Create instantiations from ffi invocations.
  top->Add<FfiInstantiationPass>();

  // Filter out traces filtered by verbosity config.
  top->Add<TraceVerbosityPass>();

  // Replace provably-unneeded priority-select operations with simpler selects.
  top->Add<PrioritySelectReductionPass>();

  // Update assert conditions to be guarded by pipeline_valid signals.
  top->Add<SideEffectConditionPass>();

  // Deduplicate registers across mutually exclusive stages.
  top->Add<RegisterCombiningPass>();

  // Remove any identity ops which might have been added earlier in the
  // pipeline.
  top->Add<CodegenWrapperPass>(std::make_unique<IdentityRemovalPass>(),
                               context);

  // Do some trivial simplifications to any flow control logic added during code
  // generation.
  top->Add<CodegenWrapperPass>(
      std::make_unique<CsePass>(/*common_literals=*/false), context);
  top->Add<CodegenWrapperPass>(std::make_unique<BasicSimplificationPass>(),
                               context);

  // Swap out fifo instantiations with materialized fifos if required by codegen
  // options.
  top->Add<MaybeMaterializeInternalFifoPass>();

  // Final dead-code elimination pass to remove cruft left from earlier passes.
  top->Add<CodegenWrapperPass>(std::make_unique<DeadCodeEliminationPass>(),
                               context);

  // Legalize names.
  top->Add<NameLegalizationPass>();

  // Final metrics collection for the final block.
  top->Add<BlockMetricsGenerationPass>();

  return top;
}

absl::StatusOr<bool> RunCodegenPassPipeline(const CodegenPassOptions& options,
                                            Block* block,
                                            OptimizationContext* context) {
  std::unique_ptr<CodegenCompoundPass> pipeline = CreateCodegenPassPipeline();
  CodegenPassUnit unit(block->package(), block);
  CodegenPassResults results;
  return pipeline->Run(&unit, options, &results);
}

}  // namespace xls::verilog

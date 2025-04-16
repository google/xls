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

// Create a standard pipeline of passes. This pipeline should
// be used in the main driver as well as in testing.

#include "xls/passes/optimization_pass_pipeline.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/passes/arith_simplification_pass.h"
#include "xls/passes/array_simplification_pass.h"
#include "xls/passes/array_untuple_pass.h"
#include "xls/passes/basic_simplification_pass.h"
#include "xls/passes/bdd_cse_pass.h"
#include "xls/passes/bdd_simplification_pass.h"
#include "xls/passes/bit_slice_simplification_pass.h"
#include "xls/passes/boolean_simplification_pass.h"
#include "xls/passes/canonicalization_pass.h"
#include "xls/passes/channel_legalization_pass.h"
#include "xls/passes/comparison_simplification_pass.h"
#include "xls/passes/concat_simplification_pass.h"
#include "xls/passes/conditional_specialization_pass.h"
#include "xls/passes/constant_folding_pass.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/label_recovery_pass.h"
#include "xls/passes/lut_conversion_pass.h"
#include "xls/passes/map_inlining_pass.h"
#include "xls/passes/narrowing_pass.h"
#include "xls/passes/next_value_optimization_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/proc_state_array_flattening_pass.h"
#include "xls/passes/proc_state_narrowing_pass.h"
#include "xls/passes/proc_state_optimization_pass.h"
#include "xls/passes/proc_state_provenance_narrowing_pass.h"
#include "xls/passes/proc_state_tuple_flattening_pass.h"
#include "xls/passes/query_engine_checker.h"
#include "xls/passes/ram_rewrite_pass.h"
#include "xls/passes/reassociation_pass.h"
#include "xls/passes/receive_default_value_simplification_pass.h"
#include "xls/passes/resource_sharing_pass.h"
#include "xls/passes/select_lifting_pass.h"
#include "xls/passes/select_merging_pass.h"
#include "xls/passes/select_simplification_pass.h"
#include "xls/passes/sparsify_select_pass.h"
#include "xls/passes/strength_reduction_pass.h"
#include "xls/passes/table_switch_pass.h"
#include "xls/passes/token_dependency_pass.h"
#include "xls/passes/token_simplification_pass.h"
#include "xls/passes/unroll_pass.h"
#include "xls/passes/useless_assert_removal_pass.h"
#include "xls/passes/useless_io_removal_pass.h"
#include "xls/passes/verifier_checker.h"

namespace xls {

namespace {

void AddSimplificationPasses(OptimizationCompoundPass& pass) {
  pass.Add<IdentityRemovalPass>();
  pass.Add<ConstantFoldingPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<CanonicalizationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<BasicSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArithSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ComparisonSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<TableSwitchPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ReceiveDefaultValueSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<SelectSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<DataflowSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ConditionalSpecializationPass>(/*use_bdd=*/false);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ReassociationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ConstantFoldingPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<BitSliceSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ConcatSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArrayUntuplePass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<DataflowSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<StrengthReductionPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArraySimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<CsePass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<BasicSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArithSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<NarrowingPass>(/*analysis=*/NarrowingPass::AnalysisType::kTernary);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<BooleanSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<TokenSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
}

}  // namespace

SimplificationPass::SimplificationPass()
    : OptimizationCompoundPass("simp", "Simplification") {
  AddSimplificationPasses(*this);
}

FixedPointSimplificationPass::FixedPointSimplificationPass()
    : OptimizationFixedPointCompoundPass("fixedpoint_simp",
                                         "Fixed-point Simplification") {
  AddSimplificationPasses(*this);
}

PreInliningPassGroup::PreInliningPassGroup()
    : OptimizationCompoundPass(PreInliningPassGroup::kName,
                               "pre-inlining passes") {
  Add<DeadFunctionEliminationPass>();
  Add<DeadCodeEliminationPass>();
  // At this stage in the pipeline only optimizations up to level 2 should
  // run. 'opt_level' is the maximum level of optimization which should be run
  // in the entire pipeline so set the level of the simplification pass to the
  // minimum of the two values. Same below.
  using Inner = CapOptLevel<2, SimplificationPass>;
  Add<IfOptLevelAtLeast<1, Inner>>();
}

UnrollingAndInliningPassGroup::UnrollingAndInliningPassGroup()
    : OptimizationCompoundPass(UnrollingAndInliningPassGroup::kName,
                               "full function inlining passes") {
  Add<UnrollPass>();
  Add<MapInliningPass>();
  Add<InliningPass>();
  Add<DeadFunctionEliminationPass>();
}

ProcStateFlatteningFixedPointPass::ProcStateFlatteningFixedPointPass()
    : OptimizationFixedPointCompoundPass(
          ProcStateFlatteningFixedPointPass::kName, "Proc State Flattening") {
  Add<ProcStateArrayFlatteningPass>();
  Add<ProcStateTupleFlatteningPass>();
}

namespace {
class PostInliningOptPassGroup : public OptimizationCompoundPass {
 public:
  PostInliningOptPassGroup()
      : OptimizationCompoundPass("post-inlining optimization passes",
                                 "post-inlining-opt") {
    Add<CapOptLevel<2, FixedPointSimplificationPass>>();

    Add<CapOptLevel<2, BddSimplificationPass>>();
    Add<DeadCodeEliminationPass>();
    Add<BddCsePass>();
    // TODO(https://github.com/google/xls/issues/274): 2022/01/20 Remove this
    // extra conditional specialization pass when the pipeline has been
    // reorganized better follow a high level of abstraction down to low level.
    Add<DeadCodeEliminationPass>();
    Add<ConditionalSpecializationPass>(/*use_bdd=*/true);

    Add<DeadCodeEliminationPass>();
    Add<CapOptLevel<2, FixedPointSimplificationPass>>();

    Add<NarrowingPass>(
        /*analysis=*/NarrowingPass::AnalysisType::kRangeWithOptionalContext);
    Add<DeadCodeEliminationPass>();
    Add<BasicSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<ArithSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<CsePass>();
    Add<SparsifySelectPass>();
    Add<DeadCodeEliminationPass>();
    Add<UselessAssertRemovalPass>();
    Add<RamRewritePass>();
    Add<UselessIORemovalPass>();
    Add<DeadCodeEliminationPass>();

    // Run ConditionalSpecializationPass before TokenDependencyPass to remove
    // false data dependencies
    Add<ConditionalSpecializationPass>(/*use_bdd=*/true);
    // Legalize multiple channel operations before proc inlining. The
    // legalization can add an adapter proc that should be inlined.
    Add<ChannelLegalizationPass>();
    Add<TokenDependencyPass>();
    // Simplify the adapter procs before inlining.
    Add<CapOptLevel<2, FixedPointSimplificationPass>>();

    // After proc inlining flatten and optimize the proc state. Run tuple
    // simplification to simplify tuple structures left over from flattening.
    // TODO(meheff): Consider running proc state optimization more than once.
    Add<ProcStateFlatteningFixedPointPass>();
    Add<IdentityRemovalPass>();
    Add<DataflowSimplificationPass>();
    Add<CapOptLevel<3, NextValueOptimizationPass>>();

    Add<ProcStateNarrowingPass>();
    Add<DeadCodeEliminationPass>();
    Add<ProcStateOptimizationPass>();
    Add<DeadCodeEliminationPass>();

    Add<ProcStateProvenanceNarrowingPass>();
    Add<DeadCodeEliminationPass>();
    Add<ProcStateOptimizationPass>();
    Add<DeadCodeEliminationPass>();

    Add<CapOptLevel<3, BddSimplificationPass>>();
    Add<DeadCodeEliminationPass>();
    Add<BddCsePass>();
    Add<SelectLiftingPass>();
    Add<DeadCodeEliminationPass>();

    Add<LutConversionPass>();
    Add<DeadCodeEliminationPass>();

    Add<ConditionalSpecializationPass>(/*use_bdd=*/true);
    Add<DeadCodeEliminationPass>();

    Add<CapOptLevel<3, FixedPointSimplificationPass>>();

    // Range based select simplification is heavier so we only do it once.
    Add<SelectRangeSimplificationPass>();
    Add<DeadCodeEliminationPass>();

    Add<CapOptLevel<3, FixedPointSimplificationPass>>();

    Add<CapOptLevel<3, BddSimplificationPass>>();
    Add<DeadCodeEliminationPass>();
    Add<BddCsePass>();
    Add<DeadCodeEliminationPass>();

    Add<CapOptLevel<3, FixedPointSimplificationPass>>();

    Add<UselessAssertRemovalPass>();
    Add<UselessIORemovalPass>();
    Add<CapOptLevel<3, NextValueOptimizationPass>>();
    // TODO(allight): We might want another proc-narrowing pass here but it's
    // not clear if it will be likely to find anything and we'd need more
    // cleanup passes if we did to take advantage of the narrower state.
    Add<ProcStateOptimizationPass>();
    Add<DeadCodeEliminationPass>();

    Add<ConditionalSpecializationPass>(/*use_bdd=*/true);
    Add<DeadCodeEliminationPass>();
    Add<SelectMergingPass>();
    Add<DeadCodeEliminationPass>();
    Add<CapOptLevel<3, FixedPointSimplificationPass>>();
  }
};
}  // namespace

PostInliningPassGroup::PostInliningPassGroup()
    : OptimizationCompoundPass(PostInliningPassGroup::kName,
                               "Post-inlining passes") {
  Add<IfOptLevelAtLeast<1, PostInliningOptPassGroup>>();
  Add<DeadCodeEliminationPass>();
  Add<LabelRecoveryPass>();
  Add<ResourceSharingPass>();
}
std::unique_ptr<OptimizationCompoundPass> CreateOptimizationPassPipeline(
    bool debug_optimizations) {
  auto top = std::make_unique<OptimizationCompoundPass>(
      "ir", "Top level pass pipeline");
  top->AddInvariantChecker<VerifierChecker>();
  if (debug_optimizations) {
    top->AddInvariantChecker<QueryEngineChecker>();
  }

  top->Add<PreInliningPassGroup>();
  top->Add<UnrollingAndInliningPassGroup>();
  top->Add<PostInliningPassGroup>();

  return top;
}

absl::StatusOr<bool> RunOptimizationPassPipeline(Package* package,
                                                 int64_t opt_level,
                                                 bool debug_optimizations) {
  std::unique_ptr<OptimizationCompoundPass> pipeline =
      CreateOptimizationPassPipeline(debug_optimizations);
  PassResults results;
  OptimizationContext context;
  return pipeline->Run(package,
                       OptimizationPassOptions().WithOptLevel(opt_level),
                       &results, context);
}

absl::Status OptimizationPassPipelineGenerator::AddPassToPipeline(
    OptimizationCompoundPass* pass, std::string_view pass_name,
    const PassPipelineProto::PassOptions& options) const {
  XLS_ASSIGN_OR_RETURN(auto* generator,
                       GetOptimizationRegistry().Generator(pass_name));
  return generator->AddToPipeline(pass, options);
}

absl::StatusOr<std::unique_ptr<OptimizationPass>>
OptimizationPassPipelineGenerator::FinalizeWithOptions(
    std::unique_ptr<OptimizationCompoundPass>&& cur,
    const PassPipelineProto::PassOptions& options) const {
  std::unique_ptr<OptimizationPass> base = std::move(cur);
  if (options.has_max_opt_level()) {
    base = std::make_unique<
        xls::internal::DynamicCapOptLevel<OptimizationWrapperPass>>(
        options.max_opt_level(), std::move(base));
  }
  if (options.has_min_opt_level()) {
    base = std::make_unique<
        xls::internal::DynamicIfOptLevelAtLeast<OptimizationWrapperPass>>(
        options.min_opt_level(), std::move(base));
  }
  return std::move(base);
}

std::string OptimizationPassPipelineGenerator::GetAvailablePassesStr() const {
  std::ostringstream oss;
  oss << "[";
  bool first = true;
  auto all_passes = GetAvailablePasses();
  absl::c_sort(all_passes);
  for (auto v : all_passes) {
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << v;
  }
  oss << "]";
  return oss.str();
}

std::vector<std::string_view>
OptimizationPassPipelineGenerator::GetAvailablePasses() const {
  return GetOptimizationRegistry().GetRegisteredNames();
}

XLS_REGISTER_MODULE_INITIALIZER(simp_pass, {
  CHECK_OK(RegisterOptimizationPass<FixedPointSimplificationPass>(
      "fixedpoint_simp"));
  CHECK_OK(
      (RegisterOptimizationPass<CapOptLevel<2, FixedPointSimplificationPass>>(
          "fixedpoint_simp(2)")));
  CHECK_OK(
      (RegisterOptimizationPass<CapOptLevel<3, FixedPointSimplificationPass>>(
          "fixedpoint_simp(3)")));
  CHECK_OK(RegisterOptimizationPass<SimplificationPass>("simp"));
  CHECK_OK((
      RegisterOptimizationPass<CapOptLevel<2, SimplificationPass>>("simp(2)")));
  CHECK_OK((
      RegisterOptimizationPass<CapOptLevel<3, SimplificationPass>>("simp(3)")));
});

REGISTER_OPT_PASS(PreInliningPassGroup);
REGISTER_OPT_PASS(UnrollingAndInliningPassGroup);
REGISTER_OPT_PASS(ProcStateFlatteningFixedPointPass);
REGISTER_OPT_PASS(PostInliningPassGroup);

}  // namespace xls

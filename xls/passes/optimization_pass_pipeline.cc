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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/package.h"

// LINT.IfChange(pass_includes)
// Every pass we include in this file should be accessible to the opt --passes
// flag by adding it with an appropriate name to the pass map.
#include "xls/passes/arith_simplification_pass.h"
#include "xls/passes/array_simplification_pass.h"
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
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/label_recovery_pass.h"
#include "xls/passes/literal_uncommoning_pass.h"
#include "xls/passes/map_inlining_pass.h"
#include "xls/passes/narrowing_pass.h"
#include "xls/passes/next_value_optimization_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/proc_inlining_pass.h"
#include "xls/passes/proc_state_flattening_pass.h"
#include "xls/passes/proc_state_optimization_pass.h"
#include "xls/passes/ram_rewrite_pass.h"
#include "xls/passes/reassociation_pass.h"
#include "xls/passes/receive_default_value_simplification_pass.h"
#include "xls/passes/select_simplification_pass.h"
#include "xls/passes/sparsify_select_pass.h"
#include "xls/passes/strength_reduction_pass.h"
#include "xls/passes/table_switch_pass.h"
#include "xls/passes/token_dependency_pass.h"
#include "xls/passes/token_simplification_pass.h"
#include "xls/passes/tuple_simplification_pass.h"
#include "xls/passes/unroll_pass.h"
#include "xls/passes/useless_assert_removal_pass.h"
#include "xls/passes/useless_io_removal_pass.h"
#include "xls/passes/verifier_checker.h"
// LINT.ThenChange(:pass_maps)

namespace xls {

namespace {

void AddSimplificationPasses(OptimizationCompoundPass& pass,
                             int64_t opt_level) {
  pass.Add<IdentityRemovalPass>();
  pass.Add<ConstantFoldingPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<CanonicalizationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArithSimplificationPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ComparisonSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<TableSwitchPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ReceiveDefaultValueSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<SelectSimplificationPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ConditionalSpecializationPass>(/*use_bdd=*/false);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ReassociationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ConstantFoldingPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<BitSliceSimplificationPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ConcatSimplificationPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<TupleSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<StrengthReductionPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArraySimplificationPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<CsePass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<ArithSimplificationPass>(opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<NarrowingPass>(/*analysis=*/NarrowingPass::AnalysisType::kTernary,
                          opt_level);
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<BooleanSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
  pass.Add<TokenSimplificationPass>();
  pass.Add<DeadCodeEliminationPass>();
}

}  // namespace

SimplificationPass::SimplificationPass(int64_t opt_level)
    : OptimizationCompoundPass("simp", "Simplification") {
  AddSimplificationPasses(*this, opt_level);
}

FixedPointSimplificationPass::FixedPointSimplificationPass(int64_t opt_level)
    : OptimizationFixedPointCompoundPass("fixedpoint_simp",
                                         "Fixed-point Simplification") {
  AddSimplificationPasses(*this, opt_level);
}

std::unique_ptr<OptimizationCompoundPass> CreateOptimizationPassPipeline(
    int64_t opt_level) {
  auto top = std::make_unique<OptimizationCompoundPass>(
      "ir", "Top level pass pipeline");
  top->AddInvariantChecker<VerifierChecker>();

  top->Add<DeadFunctionEliminationPass>();
  top->Add<DeadCodeEliminationPass>();
  // At this stage in the pipeline only optimizations up to level 2 should
  // run. 'opt_level' is the maximum level of optimization which should be run
  // in the entire pipeline so set the level of the simplification pass to the
  // minimum of the two values. Same below.
  top->Add<SimplificationPass>(std::min(int64_t{2}, opt_level));
  top->Add<UnrollPass>();
  top->Add<MapInliningPass>();
  top->Add<InliningPass>();
  top->Add<DeadFunctionEliminationPass>();

  top->Add<FixedPointSimplificationPass>(std::min(int64_t{2}, opt_level));

  top->Add<BddSimplificationPass>(std::min(int64_t{2}, opt_level));
  top->Add<DeadCodeEliminationPass>();
  top->Add<BddCsePass>();
  // TODO(https://github.com/google/xls/issues/274): 2022/01/20 Remove this
  // extra conditional specialization pass when the pipeline has been
  // reorganized better follow a high level of abstraction down to low level.
  top->Add<DeadCodeEliminationPass>();
  top->Add<ConditionalSpecializationPass>(/*use_bdd=*/true);

  top->Add<DeadCodeEliminationPass>();
  top->Add<FixedPointSimplificationPass>(std::min(int64_t{2}, opt_level));

  top->Add<NarrowingPass>(
      /*analysis=*/NarrowingPass::AnalysisType::kRangeWithOptionalContext,
      opt_level);
  top->Add<DeadCodeEliminationPass>();
  top->Add<ArithSimplificationPass>(opt_level);
  top->Add<DeadCodeEliminationPass>();
  top->Add<CsePass>();
  top->Add<SparsifySelectPass>();
  top->Add<DeadCodeEliminationPass>();
  top->Add<UselessAssertRemovalPass>();
  top->Add<RamRewritePass>();
  top->Add<UselessIORemovalPass>();
  top->Add<DeadCodeEliminationPass>();

  // Run ConditionalSpecializationPass before TokenDependencyPass to remove
  // false data dependencies
  top->Add<ConditionalSpecializationPass>(/*use_bdd=*/true);
  // Legalize multiple channel operations before proc inlining. The legalization
  // can add an adapter proc that should be inlined.
  top->Add<ChannelLegalizationPass>();
  top->Add<TokenDependencyPass>();
  // Simplify the adapter procs before inlining.
  top->Add<FixedPointSimplificationPass>(std::min(int64_t{2}, opt_level));
  top->Add<ProcInliningPass>();

  // After proc inlining flatten and optimize the proc state. Run tuple
  // simplification to simplify tuple structures left over from flattening.
  // TODO(meheff): Consider running proc state optimization more than once.
  top->Add<ProcStateFlatteningPass>();
  top->Add<IdentityRemovalPass>();
  top->Add<TupleSimplificationPass>();
  top->Add<NextValueOptimizationPass>();
  top->Add<ProcStateOptimizationPass>();
  top->Add<DeadCodeEliminationPass>();

  top->Add<BddSimplificationPass>(std::min(int64_t{3}, opt_level));
  top->Add<DeadCodeEliminationPass>();
  top->Add<BddCsePass>();
  top->Add<DeadCodeEliminationPass>();

  top->Add<ConditionalSpecializationPass>(/*use_bdd=*/true);
  top->Add<DeadCodeEliminationPass>();

  top->Add<FixedPointSimplificationPass>(std::min(int64_t{3}, opt_level));

  top->Add<UselessAssertRemovalPass>();
  top->Add<UselessIORemovalPass>();
  top->Add<NextValueOptimizationPass>();
  top->Add<ProcStateOptimizationPass>();
  top->Add<DeadCodeEliminationPass>();

  top->Add<LiteralUncommoningPass>();
  top->Add<DeadFunctionEliminationPass>();
  top->Add<LabelRecoveryPass>();
  return top;
}

absl::StatusOr<bool> RunOptimizationPassPipeline(Package* package,
                                                 int64_t opt_level) {
  std::unique_ptr<OptimizationCompoundPass> pipeline =
      CreateOptimizationPassPipeline();
  PassResults results;
  return pipeline->Run(package, OptimizationPassOptions(), &results);
}

namespace {
class BaseAdd {
 public:
  virtual ~BaseAdd() = default;
  virtual absl::Status Add(OptimizationCompoundPass* pass) const = 0;
};
template <typename PassClass, typename... Args>
class Adder final : public BaseAdd {
 public:
  explicit Adder(Args... args) : args_(std::forward_as_tuple(args...)) {}
  absl::Status Add(OptimizationCompoundPass* pass) const final {
    auto function = [&](auto... args) {
      pass->Add<PassClass>(std::forward<decltype(args)>(args)...);
    };
    std::apply(function, args_);
    return absl::OkStatus();
  }

 private:
  std::tuple<Args...> args_;
};
template <typename PassType, typename... Args>
std::unique_ptr<BaseAdd> Pass(Args... args) {
  return std::make_unique<Adder<PassType, Args...>>(
      std::forward<Args>(args)...);
}

absl::flat_hash_map<std::string_view, std::unique_ptr<BaseAdd>> MakeOptMap(
    int64_t opt_level) {
  // TODO(https://github.com/google/xls/issues/1254): 2024-1-8: This should
  // really be done by the pass libraries themselves registering their passes to
  // a central location.
  absl::flat_hash_map<std::string_view, std::unique_ptr<BaseAdd>> passes;

  // LINT.IfChange(pass_maps)
  // This map should include every pass in every configuration needed to
  // recreate at a minimum the standard optimization pipeline.
  passes["arith_simp"] = Pass<ArithSimplificationPass>(opt_level);
  passes["array_simp"] = Pass<ArraySimplificationPass>(opt_level);
  passes["bdd_cse"] = Pass<BddCsePass>();
  passes["bdd_simp"] =
      Pass<BddSimplificationPass>(std::min(int64_t{2}, opt_level));
  passes["bdd_simp(2)"] =
      Pass<BddSimplificationPass>(std::min(int64_t{2}, opt_level));
  passes["bdd_simp(3)"] =
      Pass<BddSimplificationPass>(std::min(int64_t{3}, opt_level));
  passes["bitslice_simp"] = Pass<BitSliceSimplificationPass>(opt_level);
  passes["bool_simp"] = Pass<BooleanSimplificationPass>();
  passes["canon"] = Pass<CanonicalizationPass>();
  passes["channel_legalization"] = Pass<ChannelLegalizationPass>();
  passes["comparison_simp"] = Pass<ComparisonSimplificationPass>();
  passes["concat_simp"] = Pass<ConcatSimplificationPass>(opt_level);
  passes["cond_spec"] = Pass<ConditionalSpecializationPass>(/*use_bdd=*/true);
  passes["cond_spec(false)"] =
      Pass<ConditionalSpecializationPass>(/*use_bdd=*/false);
  passes["cond_spec(true)"] =
      Pass<ConditionalSpecializationPass>(/*use_bdd=*/true);
  passes["const_fold"] = Pass<ConstantFoldingPass>();
  passes["cse"] = Pass<CsePass>();
  passes["dce"] = Pass<DeadCodeEliminationPass>();
  passes["dfe"] = Pass<DeadFunctionEliminationPass>();
  passes["ident_remove"] = Pass<IdentityRemovalPass>();
  passes["ident_remove"] = Pass<IdentityRemovalPass>();
  passes["inlining"] = Pass<InliningPass>();
  passes["label-recovery"] = Pass<LabelRecoveryPass>();
  passes["label_recovery"] = Pass<LabelRecoveryPass>();
  passes["literal_uncommon"] = Pass<LiteralUncommoningPass>();
  passes["loop_unroll"] = Pass<UnrollPass>();
  passes["map_inlining"] = Pass<MapInliningPass>();
  passes["narrow"] =
      Pass<NarrowingPass>(NarrowingPass::AnalysisType::kRange, opt_level);
  passes["narrow(Ternary)"] =
      Pass<NarrowingPass>(NarrowingPass::AnalysisType::kTernary, opt_level);
  passes["narrow(Context)"] = Pass<NarrowingPass>(
      NarrowingPass::AnalysisType::kRangeWithContext, opt_level);
  passes["narrow(OptionalContext)"] = Pass<NarrowingPass>(
      NarrowingPass::AnalysisType::kRangeWithOptionalContext, opt_level);
  passes["narrow(Range)"] =
      Pass<NarrowingPass>(NarrowingPass::AnalysisType::kRange, opt_level);
  passes["next_value_opt"] = Pass<NextValueOptimizationPass>();
  passes["proc_inlining"] = Pass<ProcInliningPass>();
  passes["proc_state_flat"] = Pass<ProcStateFlatteningPass>();
  passes["proc_state_opt"] = Pass<ProcStateOptimizationPass>();
  passes["ram_rewrite"] = Pass<RamRewritePass>();
  passes["reassociation"] = Pass<ReassociationPass>();
  passes["recv_default"] = Pass<ReceiveDefaultValueSimplificationPass>();
  passes["select_simp"] = Pass<SelectSimplificationPass>();
  passes["simp"] = Pass<SimplificationPass>(std::min(int64_t{2}, opt_level));
  passes["simp(2)"] = Pass<SimplificationPass>(std::min(int64_t{2}, opt_level));
  passes["simp(3)"] = Pass<SimplificationPass>(std::min(int64_t{3}, opt_level));
  passes["fixedpoint_simp"] =
      Pass<FixedPointSimplificationPass>(std::min(int64_t{2}, opt_level));
  passes["fixedpoint_simp(2)"] =
      Pass<FixedPointSimplificationPass>(std::min(int64_t{2}, opt_level));
  passes["fixedpoint_simp(3)"] =
      Pass<FixedPointSimplificationPass>(std::min(int64_t{3}, opt_level));
  passes["sparsify_select"] = Pass<SparsifySelectPass>();
  passes["strength_red"] = Pass<StrengthReductionPass>(opt_level);
  passes["table_switch"] = Pass<TableSwitchPass>();
  passes["token_dependency"] = Pass<TokenDependencyPass>();
  passes["token_simp"] = Pass<TokenSimplificationPass>();
  passes["tuple_simp"] = Pass<TupleSimplificationPass>();
  passes["useless_assert_remove"] = Pass<UselessAssertRemovalPass>();
  passes["useless_io_remove"] = Pass<UselessIORemovalPass>();
  // LINT.ThenChange(:pass_includes)
  return passes;
}
std::vector<absl::flat_hash_map<std::string_view, std::unique_ptr<BaseAdd>>>
BuildPassMaps() {
  std::vector<absl::flat_hash_map<std::string_view, std::unique_ptr<BaseAdd>>>
      result;

  result.reserve(kMaxOptLevel + 1);
  for (int i = 0; i < kMaxOptLevel + 1; ++i) {
    result.emplace_back(MakeOptMap(i));
  }

  return result;
}

// Generate the static maps for pass names to constructors.
const absl::flat_hash_map<std::string_view, std::unique_ptr<BaseAdd>>& GetMap(
    int64_t opt_level) {
  // Avoid potential issues with destructors running during shutdown.
  static std::vector<absl::flat_hash_map<std::string_view,
                                         std::unique_ptr<BaseAdd>>>* kPassMaps =
      new std::vector<
          absl::flat_hash_map<std::string_view, std::unique_ptr<BaseAdd>>>(
          BuildPassMaps());
  XLS_CHECK_GE(opt_level, 0);
  XLS_CHECK_LE(opt_level, kMaxOptLevel);
  return (*kPassMaps)[opt_level];
}

}  // namespace

absl::Status OptimizationPassPipelineGenerator::AddPassToPipeline(
    OptimizationCompoundPass* pass, std::string_view pass_name) const {
  const auto& map = GetMap(opt_level_);
  if (map.contains(pass_name)) {
    return map.at(pass_name)->Add(pass);
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("'%v' is not a valid pass name!", pass_name));
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
  std::vector<std::string_view> results;
  results.reserve(GetMap(opt_level_).size());
  for (const auto& [k, _] : GetMap(opt_level_)) {
    results.push_back(k);
  }
  return results;
}

}  // namespace xls

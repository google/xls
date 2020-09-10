// Copyright 2020 Google LLC
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

#include "xls/passes/standard_pipeline.h"

#include "xls/passes/arith_simplification_pass.h"
#include "xls/passes/array_simplification_pass.h"
#include "xls/passes/bdd_cse_pass.h"
#include "xls/passes/bdd_simplification_pass.h"
#include "xls/passes/bit_slice_simplification_pass.h"
#include "xls/passes/boolean_simplification_pass.h"
#include "xls/passes/canonicalization_pass.h"
#include "xls/passes/concat_simplification_pass.h"
#include "xls/passes/constant_folding_pass.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/literal_uncommoning_pass.h"
#include "xls/passes/map_inlining_pass.h"
#include "xls/passes/narrowing_pass.h"
#include "xls/passes/reassociation_pass.h"
#include "xls/passes/select_simplification_pass.h"
#include "xls/passes/strength_reduction_pass.h"
#include "xls/passes/tuple_simplification_pass.h"
#include "xls/passes/unroll_pass.h"
#include "xls/passes/verifier_checker.h"
#include "xls/scheduling/pipeline_scheduling_pass.h"
#include "xls/scheduling/scheduling_checker.h"

namespace xls {

class SimplificationPass : public FixedPointCompoundPass {
 public:
  explicit SimplificationPass(bool split_ops)
      : FixedPointCompoundPass("simp", "Simplification") {
    Add<ConstantFoldingPass>();
    Add<DeadCodeEliminationPass>();
    Add<CanonicalizationPass>();
    Add<DeadCodeEliminationPass>();
    Add<SelectSimplificationPass>(split_ops);
    Add<DeadCodeEliminationPass>();
    Add<ArithSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<ReassociationPass>();
    Add<DeadCodeEliminationPass>();
    Add<ConstantFoldingPass>();
    Add<DeadCodeEliminationPass>();
    Add<BitSliceSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<ConcatSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<TupleSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<StrengthReductionPass>(split_ops);
    Add<DeadCodeEliminationPass>();
    Add<ArraySimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<NarrowingPass>();
    Add<DeadCodeEliminationPass>();
    Add<BooleanSimplificationPass>();
    Add<DeadCodeEliminationPass>();
    Add<CsePass>();
  }
};

std::unique_ptr<CompoundPass> CreateStandardPassPipeline() {
  auto top = absl::make_unique<CompoundPass>("ir", "Top level pass pipeline");
  top->AddInvariantChecker<VerifierChecker>();

  top->Add<DeadFunctionEliminationPass>();
  top->Add<DeadCodeEliminationPass>();
  top->Add<IdentityRemovalPass>();
  top->Add<SimplificationPass>(/*split_ops=*/false);
  top->Add<UnrollPass>();
  top->Add<MapInliningPass>();
  top->Add<InliningPass>();
  top->Add<DeadFunctionEliminationPass>();
  top->Add<BddSimplificationPass>(/*split_ops=*/false);
  top->Add<DeadCodeEliminationPass>();
  top->Add<BddCsePass>();
  top->Add<DeadCodeEliminationPass>();
  top->Add<SimplificationPass>(/*split_ops=*/false);

  top->Add<BddSimplificationPass>(/*split_ops=*/true);
  top->Add<DeadCodeEliminationPass>();
  top->Add<BddCsePass>();
  top->Add<DeadCodeEliminationPass>();
  top->Add<SimplificationPass>(/*split_ops=*/true);
  top->Add<LiteralUncommoningPass>();
  top->Add<DeadFunctionEliminationPass>();
  return top;
}

xabsl::StatusOr<bool> RunStandardPassPipeline(Package* package) {
  std::unique_ptr<CompoundPass> pipeline = CreateStandardPassPipeline();
  PassResults results;
  return pipeline->Run(package, PassOptions(), &results);
}

std::unique_ptr<SchedulingCompoundPass> CreateStandardSchedulingPassPipeline() {
  auto top = absl::make_unique<SchedulingCompoundPass>(
      "sched", "Top level scheduling pass pipeline");
  top->AddInvariantChecker<SchedulingChecker>();
  top->Add<PipelineSchedulingPass>();
  return top;
}

}  // namespace xls

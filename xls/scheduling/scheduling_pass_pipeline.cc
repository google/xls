// Copyright 2022 The XLS Authors
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

#include "xls/scheduling/scheduling_pass_pipeline.h"

#include <memory>

#include "xls/passes/dce_pass.h"
#include "xls/passes/literal_uncommoning_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/scheduling/mutual_exclusion_pass.h"
#include "xls/scheduling/pipeline_scheduling_pass.h"
#include "xls/scheduling/proc_state_legalization_pass.h"
#include "xls/scheduling/scheduling_checker.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_wrapper_pass.h"

namespace xls {

std::unique_ptr<SchedulingCompoundPass> CreateSchedulingPassPipeline() {
  auto top = std::make_unique<SchedulingCompoundPass>(
      "scheduling", "Top level scheduling pass pipeline");
  top->AddInvariantChecker<SchedulingChecker>();

  // Make sure we have all of our state in the form of `next_value` nodes before
  // scheduling.
  top->Add<ProcStateLegalizationPass>();

  top->Add<MutualExclusionPass>();
  top->Add<SchedulingWrapperPass>(
      std::make_unique<FixedPointSimplificationPass>(3));
  top->Add<SchedulingWrapperPass>(std::make_unique<LiteralUncommoningPass>());
  top->Add<PipelineSchedulingPass>();
  top->Add<SchedulingWrapperPass>(std::make_unique<DeadCodeEliminationPass>());
  top->Add<MutualExclusionPass>();
  top->Add<SchedulingWrapperPass>(std::make_unique<DeadCodeEliminationPass>());
  top->Add<PipelineSchedulingPass>();

  return top;
}

}  // namespace xls

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
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pipeline_generator.h"
#include "xls/passes/query_engine_checker.h"
#include "xls/passes/verifier_checker.h"

namespace xls {

absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
TryCreateOptimizationPassPipeline(bool debug_optimizations) {
  XLS_ASSIGN_OR_RETURN(auto generator, GetOptimizationRegistry().Generator(
                                           kDefaultPassPipelineName));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> pipeline,
                       generator->Generate());
  auto top = std::make_unique<OptimizationCompoundPass>(
      "ir", "Top level pass pipeline");
  top->AddOwned(std::move(pipeline));
  if (debug_optimizations) {
    top->AddInvariantChecker<VerifierChecker>();
    top->AddInvariantChecker<QueryEngineChecker>();
  } else {
    top->AddWeakInvariantChecker<VerifierChecker>();
  }

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
    const BasicPipelineOptions& options) const {
  XLS_ASSIGN_OR_RETURN(auto* generator,
                       GetOptimizationRegistry().Generator(pass_name));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> req_pass,
                       generator->Generate());
  XLS_ASSIGN_OR_RETURN(req_pass,
                       FinalizeWithOptions(std::move(req_pass), options));
  pass->AddOwned(std::move(req_pass));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<OptimizationPass>>
OptimizationPassPipelineGenerator::FinalizeWithOptions(
    std::unique_ptr<OptimizationPass>&& cur,
    const BasicPipelineOptions& options) const {
  return WrapPassWithOptions(std::move(cur), options);
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

}  // namespace xls

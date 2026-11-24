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

#include "xls/tools/pass_pipeline_config.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional_ref.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/query_engine_checker.h"
#include "xls/passes/verifier_checker.h"

namespace xls {
namespace {

void AttachInvariantCheckers(OptimizationCompoundPass& pass,
                             bool debug_optimizations) {
  if (debug_optimizations) {
    pass.AddInvariantChecker<VerifierChecker>();
    pass.AddInvariantChecker<QueryEngineChecker>();
  } else {
    pass.AddWeakInvariantChecker<VerifierChecker>();
  }
}

}  // namespace

absl::StatusOr<std::optional<OptimizationPassRegistry>>
MaybeCreateCustomOptimizationRegistry(
    absl::optional_ref<const OptimizationPipelineProto> custom_registry,
    const OptimizationPassRegistry& base_registry) {
  if (!custom_registry.has_value()) {
    return std::nullopt;
  }
  OptimizationPassRegistry registry = base_registry.OverridableClone();
  XLS_RETURN_IF_ERROR(
      registry.RegisterPipelineProto(*custom_registry, "custom-registry"));
  return registry;
}

absl::StatusOr<PassPipelineProto> ParsePassPipelineFromString(
    std::string_view pass_list, const OptimizationPassRegistry& registry) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<OptimizationCompoundPass> pipeline,
      GetOptimizationPipelineGenerator(registry).GeneratePipeline(pass_list));
  PassPipelineProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_top(), pipeline->ToProto());
  return proto;
}

absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
BuildOptimizationPassPipelineWithRegistry(
    const OptimizationPipelineConfig& config,
    const OptimizationPassRegistry& registry,
    DefaultPassPipeline default_pipeline) {
  if (config.pipeline.has_value()) {
    if (!config.skip_passes.empty()) {
      return absl::InvalidArgumentError(
          "Skipping passes via --skip_passes cannot be combined with a custom "
          "pass pipeline (--passes, --passes_proto, or --passes_textproto).");
    }
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<OptimizationCompoundPass> pipeline,
        GetOptimizationPipelineGenerator(registry).GeneratePipeline(
            *config.pipeline));
    AttachInvariantCheckers(*pipeline, config.debug_optimizations);
    return pipeline;
  }
  if (default_pipeline == DefaultPassPipeline::kNone) {
    return nullptr;
  }
  return TryCreateOptimizationPassPipeline(config.debug_optimizations,
                                           registry);
}

absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
BuildOptimizationPassPipeline(const OptimizationPipelineConfig& config,
                              DefaultPassPipeline default_pipeline,
                              const OptimizationPassRegistry& base_registry) {
  if (default_pipeline == DefaultPassPipeline::kNone &&
      !config.HasExplicitPipeline()) {
    return nullptr;
  }
  XLS_ASSIGN_OR_RETURN(std::optional<OptimizationPassRegistry> custom_registry,
                       MaybeCreateCustomOptimizationRegistry(
                           config.custom_registry, base_registry));
  const OptimizationPassRegistry& chosen_registry =
      custom_registry.has_value() ? *custom_registry : base_registry;
  return BuildOptimizationPassPipelineWithRegistry(
      config, chosen_registry, DefaultPassPipeline::kDefault);
}

void ApplyOptimizationPipelineOptions(const OptimizationPipelineConfig& config,
                                      OptimizationPassOptions& options) {
  options.skip_passes = config.skip_passes;
  if (config.passes_bisect_limit.has_value()) {
    options.bisect_limit = *config.passes_bisect_limit;
  }
}

absl::Status CheckPassesBisectLimit(const OptimizationPipelineConfig& config,
                                    int64_t total_passes) {
  if (config.passes_bisect_limit_is_error &&
      config.passes_bisect_limit.has_value() &&
      total_passes >= *config.passes_bisect_limit) {
    return absl::InternalError("passes bisect limit was reached.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GetAvailablePassesStr(
    absl::optional_ref<const OptimizationPipelineProto> custom_registry,
    const OptimizationPassRegistry& base_registry) {
  XLS_ASSIGN_OR_RETURN(
      std::optional<OptimizationPassRegistry> cloned,
      MaybeCreateCustomOptimizationRegistry(custom_registry, base_registry));
  const OptimizationPassRegistry& chosen_registry =
      cloned.has_value() ? *cloned : base_registry;
  return GetOptimizationPipelineGenerator(chosen_registry)
      .GetAvailablePassesStr();
}

}  // namespace xls

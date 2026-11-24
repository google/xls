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

#ifndef XLS_TOOLS_PASS_PIPELINE_CONFIG_H_
#define XLS_TOOLS_PASS_PIPELINE_CONFIG_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional_ref.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_pipeline.pb.h"

namespace xls {

// Policy for selecting a pipeline when `OptimizationPipelineConfig::pipeline`
// is not explicitly set.
enum class DefaultPassPipeline {
  // Run the default optimization pass pipeline from the active registry
  // (honoring `custom_registry` if set).
  kDefault,
  // Run no optimization passes unless `pipeline` or `custom_registry` is set.
  kNone,
};

// Flag-independent configuration describing an optimization pass pipeline,
// optional custom pass registry, skip list, bisect limit, and invariant checks.
struct OptimizationPipelineConfig {
  // Custom compound-pass definitions and default pipeline (`--pipeline_proto` /
  // `--pipeline_textproto`).
  std::optional<OptimizationPipelineProto> custom_registry;
  // Explicit pass pipeline (`--passes` / `--passes_proto` /
  // `--passes_textproto`). Unset selects the caller's `DefaultPassPipeline`.
  std::optional<PassPipelineProto> pipeline;
  // Short names of passes to skip (`--skip_passes`). Cannot be combined with an
  // explicit `pipeline`.
  std::vector<std::string> skip_passes;
  // Maximum number of passes to run (`--passes_bisect_limit`).
  std::optional<int64_t> passes_bisect_limit;
  // Whether hitting `passes_bisect_limit` should be reported as an error.
  bool passes_bisect_limit_is_error = false;
  // Whether to attach strict invariant checkers (Verifier + QueryEngine)
  // instead of only the weak verifier checker.
  bool debug_optimizations = false;

  // True if an explicit pipeline or custom pass registry has been configured.
  bool HasExplicitPipeline() const {
    return pipeline.has_value() || custom_registry.has_value();
  }
};

// Returns an overridable clone of `base_registry` with `custom_registry`
// registered when `custom_registry` is provided, or `std::nullopt` when no
// cloning is needed (so callers can reference `base_registry` directly without
// copying it).
absl::StatusOr<std::optional<OptimizationPassRegistry>>
MaybeCreateCustomOptimizationRegistry(
    absl::optional_ref<const OptimizationPipelineProto> custom_registry,
    const OptimizationPassRegistry& base_registry = GetOptimizationRegistry());

// Parses a textual pass pipeline specification (`--passes` syntax) into a
// `PassPipelineProto` using `registry`.
absl::StatusOr<PassPipelineProto> ParsePassPipelineFromString(
    std::string_view pass_list,
    const OptimizationPassRegistry& registry = GetOptimizationRegistry());

// Builds an `OptimizationCompoundPass` from `config` using `base_registry`
// (applying `config.custom_registry` if set) and attaches invariant checkers
// per `config.debug_optimizations`.
//
// Returns `nullptr` if `default_pipeline == DefaultPassPipeline::kNone` and
// `!config.HasExplicitPipeline()`. Returns an invalid-argument error if both
// `config.pipeline` and non-empty `config.skip_passes` are provided.
absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
BuildOptimizationPassPipeline(
    const OptimizationPipelineConfig& config,
    DefaultPassPipeline default_pipeline = DefaultPassPipeline::kDefault,
    const OptimizationPassRegistry& base_registry = GetOptimizationRegistry());

// Same as `BuildOptimizationPassPipeline`, except it builds directly against
// an already-constructed/decorated `registry` (ignoring
// `config.custom_registry`). When `config.pipeline` is unset, returns `nullptr`
// if `default_pipeline == DefaultPassPipeline::kNone`, or the default pipeline
// registered in `registry` if `default_pipeline ==
// DefaultPassPipeline::kDefault`. Useful when a pass-registry decorator
// requires `registry` to outlive `Run()`.
absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
BuildOptimizationPassPipelineWithRegistry(
    const OptimizationPipelineConfig& config,
    const OptimizationPassRegistry& registry,
    DefaultPassPipeline default_pipeline = DefaultPassPipeline::kDefault);

// Applies `config.skip_passes` and `config.passes_bisect_limit` to `options`.
void ApplyOptimizationPipelineOptions(const OptimizationPipelineConfig& config,
                                      OptimizationPassOptions& options);

// Returns an error if `config.passes_bisect_limit_is_error` is enabled and
// `total_passes` reached `config.passes_bisect_limit`.
absl::Status CheckPassesBisectLimit(const OptimizationPipelineConfig& config,
                                    int64_t total_passes);

// Returns a formatted list of registered pass names, including any compound
// passes declared in `custom_registry`.
absl::StatusOr<std::string> GetAvailablePassesStr(
    absl::optional_ref<const OptimizationPipelineProto> custom_registry =
        std::nullopt,
    const OptimizationPassRegistry& base_registry = GetOptimizationRegistry());

}  // namespace xls

#endif  // XLS_TOOLS_PASS_PIPELINE_CONFIG_H_

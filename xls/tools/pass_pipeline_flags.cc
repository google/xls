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

#include "xls/tools/pass_pipeline_flags.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/tools/pass_pipeline_config.h"

// LINT.IfChange
ABSL_FLAG(std::vector<std::string>, skip_passes, {},
          "If specified, passes in this comma-separated list of (short) "
          "pass names are skipped.");
ABSL_FLAG(
    std::optional<std::string>, passes, std::nullopt,
    "Explicit list of passes to run in a specific order. Passes are named "
    "by 'short_name' and if they have non-opt-level arguments these are "
    "placed in (). Fixed point sets of passes can be put within []. Pass "
    "names are separated based on spaces. For example a simple pipeline "
    "might be \"dfe dce [ ident_remove const_fold dce canon dce arith_simp "
    "dce comparison_simp ] loop_unroll map_inlining\". This should not be "
    "used with --skip_passes. If this is given the standard optimization "
    "pipeline is ignored entirely; care should be taken to ensure the "
    "given pipeline will run in reasonable amount of time. Pass \"\" to run "
    "no passes. Available passes are shown by running `opt_main "
    "--list_passes`.");
// TODO(allight): Remove this flag, and the old passes proto.
ABSL_FLAG(std::optional<std::string>, passes_proto, std::nullopt,
          "A file containing binary PassPipelineProto defining a pipeline "
          "of passes to run. --pipeline_proto should be preferred instead.");
// TODO(allight): Remove this flag, and the old passes proto.
ABSL_FLAG(std::optional<std::string>, passes_textproto, std::nullopt,
          "A file containing textproto PassPipelineProto defining a "
          "pipeline of passes to run. --pipeline_textproto should be "
          "preferred instead.");
ABSL_FLAG(std::optional<std::string>, pipeline_proto, std::nullopt,
          "A file containing binary OptimizationPipelineProto defining a "
          "pipeline of passes to run.");
ABSL_FLAG(std::optional<std::string>, pipeline_textproto, std::nullopt,
          "A file containing textproto OptimizationPipelineProto defining a "
          "pipeline of passes to run.");
ABSL_FLAG(std::optional<int64_t>, passes_bisect_limit, std::nullopt,
          "Number of passes to allow to execute. This can be used as compiler "
          "fuel to ensure the compiler finishes at a particular point.");
ABSL_FLAG(bool, passes_bisect_limit_is_error, false,
          "If set then reaching passes bisect limit is considered an error.");
ABSL_FLAG(bool, debug_optimizations, false,
          "If passed, run additional strict correctness-checking passes; this "
          "slows down the optimization significantly, and is mostly intended "
          "for internal XLS debugging.");
// LINT.ThenChange(//xls/build_rules/xls_providers.bzl)

namespace xls {

absl::StatusOr<PassPipelineFlagsResult> GetPassPipelineFlags() {
  PassPipelineFlagsResult result;
  OptimizationPipelineConfig& config = result.config;

  config.skip_passes = absl::GetFlag(FLAGS_skip_passes);
  result.any_flags_set |= FLAGS_skip_passes.IsSpecifiedOnCommandLine() ||
                          !config.skip_passes.empty();

  config.passes_bisect_limit = absl::GetFlag(FLAGS_passes_bisect_limit);
  result.any_flags_set |=
      FLAGS_passes_bisect_limit.IsSpecifiedOnCommandLine() ||
      config.passes_bisect_limit.has_value();

  config.passes_bisect_limit_is_error =
      absl::GetFlag(FLAGS_passes_bisect_limit_is_error);
  result.any_flags_set |=
      FLAGS_passes_bisect_limit_is_error.IsSpecifiedOnCommandLine() ||
      config.passes_bisect_limit_is_error;

  config.debug_optimizations = absl::GetFlag(FLAGS_debug_optimizations);
  result.any_flags_set |=
      FLAGS_debug_optimizations.IsSpecifiedOnCommandLine() ||
      config.debug_optimizations;

  const std::optional<std::string> pipeline_proto_path =
      absl::GetFlag(FLAGS_pipeline_proto);
  const std::optional<std::string> pipeline_textproto_path =
      absl::GetFlag(FLAGS_pipeline_textproto);
  result.any_flags_set |= FLAGS_pipeline_proto.IsSpecifiedOnCommandLine() ||
                          FLAGS_pipeline_textproto.IsSpecifiedOnCommandLine();
  if (pipeline_proto_path.has_value() && pipeline_textproto_path.has_value()) {
    return absl::InvalidArgumentError(
        "At most one of --pipeline_proto and --pipeline_textproto is allowed.");
  }
  if (pipeline_proto_path.has_value()) {
    result.any_flags_set = true;
    OptimizationPipelineProto custom_registry;
    XLS_RETURN_IF_ERROR(
        ParseProtobinFile(*pipeline_proto_path, &custom_registry));
    config.custom_registry = std::move(custom_registry);
  }
  if (pipeline_textproto_path.has_value()) {
    result.any_flags_set = true;
    OptimizationPipelineProto custom_registry;
    XLS_RETURN_IF_ERROR(
        ParseTextProtoFile(*pipeline_textproto_path, &custom_registry));
    config.custom_registry = std::move(custom_registry);
  }

  const std::optional<std::string> passes_proto_path =
      absl::GetFlag(FLAGS_passes_proto);
  const std::optional<std::string> passes_textproto_path =
      absl::GetFlag(FLAGS_passes_textproto);
  std::optional<std::string> passes = absl::GetFlag(FLAGS_passes);
  // Abseil parses `--passes=` on an `std::optional<std::string>` flag as
  // `std::nullopt` while setting `IsSpecifiedOnCommandLine()`. Treat an
  // explicit `--passes=` as an empty (no-op) pass pipeline rather than falling
  // back to the default pipeline.
  if (!passes.has_value() && FLAGS_passes.IsSpecifiedOnCommandLine()) {
    passes = "";
  }
  result.any_flags_set |= FLAGS_passes_proto.IsSpecifiedOnCommandLine() ||
                          FLAGS_passes_textproto.IsSpecifiedOnCommandLine() ||
                          FLAGS_passes.IsSpecifiedOnCommandLine();
  const int explicit_pass_flag_count =
      (passes_proto_path.has_value() ? 1 : 0) +
      (passes_textproto_path.has_value() ? 1 : 0) +
      (passes.has_value() ? 1 : 0);
  if (explicit_pass_flag_count > 1) {
    return absl::InvalidArgumentError(
        "At most one of --passes_proto, --passes_textproto, or --passes is "
        "allowed.");
  }

  if (explicit_pass_flag_count > 0 && !config.skip_passes.empty()) {
    return absl::InvalidArgumentError(
        "Skipping passes via --skip_passes cannot be combined with a custom "
        "pass pipeline (--passes, --passes_proto, or --passes_textproto).");
  }

  if (passes_proto_path.has_value()) {
    result.any_flags_set = true;
    PassPipelineProto pipeline;
    XLS_RETURN_IF_ERROR(ParseProtobinFile(*passes_proto_path, &pipeline));
    config.pipeline = std::move(pipeline);
  } else if (passes_textproto_path.has_value()) {
    result.any_flags_set = true;
    PassPipelineProto pipeline;
    XLS_RETURN_IF_ERROR(ParseTextProtoFile(*passes_textproto_path, &pipeline));
    config.pipeline = std::move(pipeline);
  } else if (passes.has_value()) {
    result.any_flags_set = true;
    XLS_ASSIGN_OR_RETURN(
        std::optional<OptimizationPassRegistry> custom_registry,
        MaybeCreateCustomOptimizationRegistry(config.custom_registry));
    const OptimizationPassRegistry& chosen_registry =
        custom_registry.has_value() ? *custom_registry
                                    : GetOptimizationRegistry();
    XLS_ASSIGN_OR_RETURN(config.pipeline,
                         ParsePassPipelineFromString(*passes, chosen_registry));
  }

  return result;
}

}  // namespace xls

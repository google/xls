// Copyright 2025 The XLS Authors
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

#include "xls/tools/opt_flags.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xls/tools/opt_flags.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/query_engine_checker.h"
#include "xls/passes/verifier_checker.h"

// LINT.IfChange
ABSL_FLAG(std::string, top, "", "Top entity to optimize.");
ABSL_FLAG(std::string, ir_dump_path, "",
          "Dump all intermediate IR files to the given directory");
ABSL_FLAG(std::vector<std::string>, skip_passes, {},
          "If specified, passes in this comma-separated list of (short) "
          "pass names are skipped.");
ABSL_FLAG(std::optional<int64_t>, convert_array_index_to_select, std::nullopt,
          "If specified, convert array indexes with fewer than or "
          "equal to the given number of possible indices (by range analysis) "
          "into chains of selects. Otherwise, this optimization is skipped, "
          "since it can sometimes reduce output quality.");
ABSL_FLAG(
    std::optional<int64_t>, split_next_value_selects, 4,
    "If positive, split `next_value`s that assign `sel`s to state params if "
    "they have fewer than the given number of cases. This optimization is "
    "skipped for selects with more cases, since it can sometimes reduce output "
    "quality by replacing MUX trees with separate equality checks.");
ABSL_FLAG(int64_t, opt_level, xls::kMaxOptLevel, []() -> const std::string& {
  static const std::string kDescription = absl::StrFormat(
      "Optimization level. Ranges from 1 to %d.", xls::kMaxOptLevel);
  return kDescription;
}());
ABSL_FLAG(std::string, ram_rewrites_pb, "",
          "Path to protobuf describing ram rewrites.");
ABSL_FLAG(bool, use_context_narrowing_analysis, false,
          "Use context sensitive narrowing analysis. This is somewhat slower "
          "but might produce better results in some circumstances by using "
          "usage context to narrow values more aggressively.");
ABSL_FLAG(
    bool, optimize_for_best_case_throughput, false,
    "Optimize for best case throughput, even at the cost of area. This will "
    "aggressively optimize to create opportunities for improved throughput, "
    "but at the cost of constraining the schedule and thus increasing area.");
ABSL_FLAG(bool, enable_resource_sharing, false,
          "Enable the resource sharing optimization to save area.");
ABSL_FLAG(bool, force_resource_sharing, false,
          "Force the resource sharing pass to apply the transformation where "
          "it is legal to do so, overriding therefore the profitability "
          "heuristic of such pass. This option is only used when the resource "
          "sharing pass is enabled.");
ABSL_FLAG(std::string, area_model, "asap7",
          "Area model to use for optimizations.");
ABSL_FLAG(
    std::optional<std::string>, passes, std::nullopt,
    "Explicit list of passes to run in a specific order. Passes are named "
    "by 'short_name' and if they have non-opt-level arguments these are "
    "placed in (). Fixed point sets of passes can be put within []. Pass "
    "names are separated based on spaces. For example a simple pipeline "
    "might be \"dfe dce [ ident_remove const_fold dce canon dce arith dce "
    "comparison_simp ] loop_unroll map_inline\". This should not be used "
    "with --skip_passes. If this is given the standard optimization "
    "pipeline is ignored entirely, care should be taken to ensure the "
    "given pipeline will run in reasonable amount of time. See the map in "
    "passes/optimization_pass_pipeline.cc for pass mappings. Available "
    "passes shown by running with --list_passes");
// TODO(allight): Remove this flag, and the old passes proto.
ABSL_FLAG(std::optional<std::string>, passes_proto, std::nullopt,
          "A file containing binary PipelinePassList proto defining a pipeline "
          "of passes to run. The pipeline_proto should be preferred instead.");
// TODO(allight): Remove this flag, and the old passes proto.
ABSL_FLAG(std::optional<std::string>, passes_textproto, std::nullopt,
          "A file containing textproto PipelinePassList proto defining a "
          "pipeline of passes to run. The pipeline_textproto should be "
          "preferred instead.");
ABSL_FLAG(std::optional<std::string>, pipeline_proto, std::nullopt,
          "A file containing binary OptimizationPipelineProto proto defining a "
          "pipeline of passes to run");
ABSL_FLAG(
    std::optional<std::string>, pipeline_textproto, std::nullopt,
    "A file containing textproto OptimizationPipelineProto proto defining a "
    "pipeline of passes to run");
ABSL_FLAG(std::optional<int64_t>, passes_bisect_limit, std::nullopt,
          "Number of passes to allow to execute. This can be used as compiler "
          "fuel to ensure the compiler finishes at a particular point.");
ABSL_FLAG(bool, passes_bisect_limit_is_error, false,
          "If set then reaching passes bisect limit is considered an error.");
ABSL_FLAG(
    std::optional<std::string>, pass_metrics_path, std::nullopt,
    "Output path for the pass pipeline metrics as a PassPipelineMetricsProto.");
ABSL_FLAG(bool, debug_optimizations, false,
          "If passed, run additional strict correctness-checking passes; this "
          "slows down the optimization significantly, and is mostly intended "
          "for internal XLS debugging.");

ABSL_FLAG(std::string, opt_options_proto, "",
          "Path to a protobuf containing all opt args.");
ABSL_FLAG(std::optional<std::string>, opt_options_used_textproto_file,
          std::nullopt,
          "If present, path to write a protobuf recording all opt args "
          "used (including those set on the cmd line).");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

namespace xls {
namespace {

absl::Status PopulatePassPipelineProtoFromPassList(std::string_view list,
                                                   OptFlagsProto& proto) {
  XLS_RET_CHECK_EQ(proto.skip_passes_size(), 0)
      << "Skipping/restricting passes while running a custom "
         "pipeline is probably not something you want to do.";

  std::optional<OptimizationPassRegistry> registry;
  if (proto.has_custom_registry()) {
    registry.emplace(GetOptimizationRegistry().OverridableClone());
    XLS_RETURN_IF_ERROR(registry->RegisterPipelineProto(proto.custom_registry(),
                                                        "custom-registry"));
  }
  const OptimizationPassRegistry& chosen_registry =
      registry.value_or(GetOptimizationRegistry());

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<OptimizationCompoundPass> res,
      GetOptimizationPipelineGenerator(chosen_registry).GeneratePipeline(list));
  if (proto.debug_optimizations()) {
    res->AddInvariantChecker<VerifierChecker>();
    res->AddInvariantChecker<QueryEngineChecker>();
  } else {
    res->AddWeakInvariantChecker<VerifierChecker>();
  }
  PassPipelineProto::Element pipeline_proto;

  XLS_ASSIGN_OR_RETURN(pipeline_proto, res->ToProto());
  *proto.mutable_pipeline()->mutable_top() = std::move(pipeline_proto);
  return absl::OkStatus();
}

absl::StatusOr<bool> SetOptionsFromFlags(OptFlagsProto& proto) {
#define POPULATE_FLAG(__x)                         \
  {                                                \
    if (FLAGS_##__x.IsSpecifiedOnCommandLine()) {  \
      any_flags_set |= true;                       \
      proto.set_##__x(absl::GetFlag(FLAGS_##__x)); \
    }                                              \
  }
#define POPULATE_OPTIONAL_FLAG(__x)                       \
  {                                                       \
    if (auto optional_value = absl::GetFlag(FLAGS_##__x); \
        optional_value.has_value()) {                     \
      any_flags_set |= true;                              \
      proto.set_##__x(*optional_value);                   \
    }                                                     \
  }
  bool any_flags_set = false;

  POPULATE_FLAG(opt_level)
  POPULATE_FLAG(top)
  POPULATE_FLAG(ir_dump_path)
  // skip_passes is a repeated flag.
  {
    any_flags_set |= FLAGS_skip_passes.IsSpecifiedOnCommandLine();
    proto.mutable_skip_passes()->Clear();
    auto repeated_flag = absl::GetFlag(FLAGS_skip_passes);
    proto.mutable_skip_passes()->Add(repeated_flag.begin(),
                                     repeated_flag.end());
  }
  POPULATE_OPTIONAL_FLAG(convert_array_index_to_select)
  POPULATE_OPTIONAL_FLAG(split_next_value_selects)
  if (FLAGS_ram_rewrites_pb.IsSpecifiedOnCommandLine()) {
    XLS_RETURN_IF_ERROR(ParseTextProtoFile(absl::GetFlag(FLAGS_ram_rewrites_pb),
                                           proto.mutable_ram_rewrites()));
  }
  POPULATE_FLAG(use_context_narrowing_analysis)
  POPULATE_FLAG(optimize_for_best_case_throughput)
  POPULATE_FLAG(enable_resource_sharing)
  POPULATE_FLAG(force_resource_sharing)
  POPULATE_FLAG(area_model)
  // pipeline proto flags
  {
    std::optional<std::string> protobin_path =
        absl::GetFlag(FLAGS_pipeline_proto);
    std::optional<std::string> textproto_path =
        absl::GetFlag(FLAGS_pipeline_textproto);
    if (protobin_path.has_value() && textproto_path.has_value()) {
      return absl::InvalidArgumentError(
          "At most one of --pipeline_proto and "
          "--pipeline__textproto is allowed.");
    }
    if (protobin_path.has_value()) {
      any_flags_set |= true;
      XLS_RETURN_IF_ERROR(
          ParseProtobinFile(*protobin_path, proto.mutable_pipeline()));
    }
    if (textproto_path.has_value()) {
      any_flags_set |= true;
      XLS_RETURN_IF_ERROR(
          ParseTextProtoFile(*textproto_path, proto.mutable_pipeline()));
    }
  }
  POPULATE_OPTIONAL_FLAG(passes_bisect_limit)
  POPULATE_FLAG(passes_bisect_limit_is_error)
  POPULATE_OPTIONAL_FLAG(pass_metrics_path)
  POPULATE_FLAG(debug_optimizations)
  std::optional<std::string> passes_binproto =
      absl::GetFlag(FLAGS_passes_proto);
  std::optional<std::string> passes_textproto =
      absl::GetFlag(FLAGS_passes_textproto);
  std::optional<std::string> passes = absl::GetFlag(FLAGS_passes);
  if (absl::c_count_if<std::initializer_list<std::optional<std::string>>>(
          {passes_binproto, passes_textproto, passes},
          [](const auto& v) -> bool { return v.has_value(); }) > 1) {
    return absl::InvalidArgumentError(
        "At most one of --passes_proto, --passes_textproto, or --passes is "
        "allowed.");
  }
  if (passes_binproto.has_value()) {
    any_flags_set |= true;
    XLS_RETURN_IF_ERROR(
        ParseProtobinFile(*passes_binproto, proto.mutable_pipeline()));
  }
  if (passes_textproto.has_value()) {
    any_flags_set |= true;
    XLS_RETURN_IF_ERROR(
        ParseTextProtoFile(*passes_textproto, proto.mutable_pipeline()));
  }
  if (passes.has_value()) {
    any_flags_set |= true;
    XLS_RETURN_IF_ERROR(PopulatePassPipelineProtoFromPassList(*passes, proto));
  }

#undef POPULATE_FLAG
#undef POPULATE_OPTIONAL_FLAG

  return any_flags_set;
}  // namespace

}  // namespace

absl::StatusOr<OptFlagsProto> GetOptFlags(
    std::optional<std::string_view> ir_path) {
  OptFlagsProto proto;
  if (ir_path.has_value()) {
    proto.set_ir_path(*ir_path);
  }
  XLS_ASSIGN_OR_RETURN(bool any_individual_flags_set,
                       SetOptionsFromFlags(proto));
  if (any_individual_flags_set) {
    if (FLAGS_opt_options_proto.IsSpecifiedOnCommandLine()) {
      return absl::InvalidArgumentError(
          "Cannot combine 'opt_options_proto' and opt arguments");
    }
  } else if (FLAGS_opt_options_proto.IsSpecifiedOnCommandLine()) {
    XLS_RETURN_IF_ERROR(xls::ParseTextProtoFile(
        absl::GetFlag(FLAGS_opt_options_proto), &proto));
  }
  if (absl::GetFlag(FLAGS_opt_options_used_textproto_file)) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        *absl::GetFlag(FLAGS_opt_options_used_textproto_file), proto));
  }
  return proto;
}

}  // namespace xls

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
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/tools/area_model_flags.h"
#include "xls/tools/delay_model_flags.h"
#include "xls/tools/opt_flags.pb.h"
#include "xls/tools/opt_flags_handler_registry.h"
#include "xls/tools/pass_pipeline_config.h"
#include "xls/tools/pass_pipeline_flags.h"

// LINT.IfChange
ABSL_FLAG(std::string, top, "", "Top entity to optimize.");
ABSL_FLAG(std::string, ir_dump_path, "",
          "Dump all intermediate IR files to the given directory");
ABSL_FLAG(std::optional<int64_t>, convert_array_index_to_select, std::nullopt,
          "If specified, convert array indexes with fewer than or "
          "equal to the given number of possible indices (by range analysis) "
          "into chains of selects. Otherwise, this optimization is skipped, "
          "since it can sometimes reduce output quality.");
ABSL_FLAG(
    std::optional<int64_t>, split_next_value_selects,
    xls::kDefaultSplitNextValueSelects,
    "If positive, split `next_value`s that assign `sel`s to state params if "
    "they have fewer than the given number of cases. This optimization is "
    "skipped for selects with more cases, since it can sometimes reduce output "
    "quality by replacing MUX trees with separate equality checks.");
ABSL_FLAG(int64_t, opt_level, xls::kMaxOptLevel, []() -> const std::string& {
  static const std::string kDescription = absl::StrFormat(
      "Optimization level. Ranges from 0 to %d.", xls::kMaxOptLevel);
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
// TODO(google/xls#5510): Remove once legacy coupled proc state lowering is
// retired.
ABSL_FLAG(
    bool, decouple_state_access, false,
    "Whether proc StateRead and Next operations are decoupled (allowing writes "
    "without an active StateRead, StateRead predicate specialization purely "
    "from data uses, and removal of always-false StateReads).");
ABSL_FLAG(bool, enable_resource_sharing, false,
          "Enable the resource sharing optimization to save area.");
ABSL_FLAG(bool, force_resource_sharing, false,
          "Force the resource sharing pass to apply the transformation where "
          "it is legal to do so, overriding therefore the profitability "
          "heuristic of such pass. This option is only used when the resource "
          "sharing pass is enabled.");
ABSL_FLAG(
    std::optional<std::string>, pass_metrics_path, std::nullopt,
    "Output path for the pass pipeline metrics as a PassPipelineMetricsProto.");

ABSL_FLAG(std::string, opt_options_proto, "",
          "Path to a protobuf containing all opt args.");
ABSL_FLAG(std::optional<std::string>, opt_options_used_textproto_file,
          std::nullopt,
          "If present, path to write a protobuf recording all opt args "
          "used (including those set on the cmd line).");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

namespace xls {
namespace {

void PopulateOptFlagsProtoFromPipelineConfig(OptimizationPipelineConfig config,
                                             OptFlagsProto& proto) {
  proto.mutable_skip_passes()->Clear();
  proto.mutable_skip_passes()->Add(config.skip_passes.begin(),
                                   config.skip_passes.end());
  if (config.custom_registry.has_value()) {
    *proto.mutable_custom_registry() = std::move(*config.custom_registry);
  }
  if (config.pipeline.has_value()) {
    *proto.mutable_pipeline() = std::move(*config.pipeline);
  }
  if (config.passes_bisect_limit.has_value()) {
    proto.set_passes_bisect_limit(*config.passes_bisect_limit);
  }
  if (config.passes_bisect_limit_is_error) {
    proto.set_passes_bisect_limit_is_error(true);
  }
  if (config.debug_optimizations) {
    proto.set_debug_optimizations(true);
  }
}

absl::StatusOr<bool> SetOptionsFromFlags(OptFlagsProto& proto) {
#define POPULATE_FLAG(__x)                         \
  {                                                \
    if (FLAGS_##__x.IsSpecifiedOnCommandLine()) {  \
      any_flags_set |= true;                       \
      proto.set_##__x(absl::GetFlag(FLAGS_##__x)); \
    }                                              \
  }
#define POPULATE_OPTIONAL_FLAG(__x)                         \
  {                                                         \
    if (FLAGS_##__x.IsSpecifiedOnCommandLine()) {           \
      any_flags_set |= true;                                \
      if (auto optional_value = absl::GetFlag(FLAGS_##__x); \
          optional_value.has_value()) {                     \
        proto.set_##__x(*optional_value);                   \
      } else {                                              \
        proto.clear_##__x();                                \
      }                                                     \
    }                                                       \
  }
  bool any_flags_set = false;

  POPULATE_FLAG(opt_level)
  POPULATE_FLAG(top)
  POPULATE_FLAG(ir_dump_path)
  POPULATE_OPTIONAL_FLAG(convert_array_index_to_select)
  if (FLAGS_split_next_value_selects.IsSpecifiedOnCommandLine()) {
    any_flags_set = true;
    proto.set_split_next_value_selects(
        absl::GetFlag(FLAGS_split_next_value_selects).value_or(0));
  }
  if (FLAGS_ram_rewrites_pb.IsSpecifiedOnCommandLine()) {
    any_flags_set = true;
    XLS_RETURN_IF_ERROR(ParseTextProtoFile(absl::GetFlag(FLAGS_ram_rewrites_pb),
                                           proto.mutable_ram_rewrites()));
  }
  POPULATE_FLAG(use_context_narrowing_analysis)
  POPULATE_FLAG(optimize_for_best_case_throughput)
  POPULATE_FLAG(decouple_state_access)
  POPULATE_FLAG(enable_resource_sharing)
  POPULATE_FLAG(force_resource_sharing)
  if (IsAreaModelSpecifiedOnCommandLine()) {
    any_flags_set = true;
    proto.set_area_model(GetAreaModelNameFromFlags());
  }
  if (IsDelayModelSpecifiedOnCommandLine()) {
    any_flags_set = true;
    proto.set_delay_model(GetDelayModelNameFromFlags());
  }
  POPULATE_OPTIONAL_FLAG(pass_metrics_path)

  XLS_ASSIGN_OR_RETURN(PassPipelineFlagsResult pipeline_flags,
                       GetPassPipelineFlags());
  any_flags_set |= pipeline_flags.any_flags_set;
  PopulateOptFlagsProtoFromPipelineConfig(std::move(pipeline_flags.config),
                                          proto);

#undef POPULATE_FLAG
#undef POPULATE_OPTIONAL_FLAG

  return any_flags_set;
}

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
  XLS_RETURN_IF_ERROR(OptFlagsHandlerRegistry::ParseFlags(proto));
  if (absl::GetFlag(FLAGS_opt_options_used_textproto_file)) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        *absl::GetFlag(FLAGS_opt_options_used_textproto_file), proto));
  }
  return proto;
}

}  // namespace xls

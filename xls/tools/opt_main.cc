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

// Takes in an IR file and produces an IR file that has been run through the
// standard optimization pipeline.

#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/tool_timeout.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/tools/opt.h"

static constexpr std::string_view kUsage = R"(
Takes in an IR file and produces an IR file that has been run through the
standard optimization pipeline.

Successfully optimized IR is printed to stdout.

Expected invocation:
  opt_main <IR file>
where:
  - <IR file> is the path to the input IR file. '-' denotes stdin as input.

Example invocation:
  opt_main path/to/file.ir
)";

ABSL_FLAG(std::string, output_path, "-",
          "Output path for the optimized IR file; '-' denotes stdout.");
ABSL_FLAG(std::optional<std::string>, alsologto, std::nullopt,
          "Path to write logs to, in addition to stderr.");
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
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)
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
ABSL_FLAG(std::optional<std::string>, passes_proto, std::nullopt,
          "A file containing binary PipelinePassList proto defining a pipeline "
          "of passes to run");
ABSL_FLAG(std::optional<std::string>, passes_textproto, std::nullopt,
          "A file containing textproto PipelinePassList proto defining a "
          "pipeline of passes to run");
ABSL_FLAG(std::optional<int64_t>, passes_bisect_limit, std::nullopt,
          "Number of passes to allow to execute. This can be used as compiler "
          "fuel to ensure the compiler finishes at a particular point.");
ABSL_FLAG(bool, passes_bisect_limit_is_error, false,
          "If set then reaching passes bisect limit is considered an error.");
ABSL_FLAG(bool, list_passes, false,
          "If passed list the names of all passes and exit.");
ABSL_FLAG(
    std::optional<std::string>, pass_metrics_path, std::nullopt,
    "Output path for the pass pipeline metrics as a PassPipelineMetricsProto.");
ABSL_FLAG(bool, debug_optimizations, false,
          "If passed, run additional strict correctness-checking passes; this "
          "slows down the optimization significantly, and is mostly intended "
          "for internal XLS debugging.");

namespace xls::tools {
namespace {

class FileStderrLogSink final : public absl::LogSink {
 public:
  explicit FileStderrLogSink(std::filesystem::path path)
      : path_(std::move(path)) {
    CHECK_OK(SetFileContents(path_, ""));
  }

  ~FileStderrLogSink() override = default;

  void Send(const absl::LogEntry& entry) override {
    if (entry.log_severity() < absl::StderrThreshold()) {
      return;
    }

    if (!entry.stacktrace().empty()) {
      CHECK_OK(AppendStringToFile(path_, entry.stacktrace()));
    } else {
      CHECK_OK(AppendStringToFile(
          path_, entry.text_message_with_prefix_and_newline()));
    }
  }

 private:
  const std::filesystem::path path_;
};

template <typename T>
  requires(std::is_integral_v<T>)
std::optional<T> NegativeIsNullopt(std::optional<T> v) {
  if (v && *v < 0) {
    return std::nullopt;
  }
  return v;
}

absl::Status RealMain(std::string_view input_path) {
  auto timeout = StartTimeoutTimer();
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }

  std::string output_path = absl::GetFlag(FLAGS_output_path);

  std::optional<std::string> alsologto = absl::GetFlag(FLAGS_alsologto);
  std::unique_ptr<absl::LogSink> log_file_sink;
  if (alsologto.has_value()) {
    log_file_sink = std::make_unique<FileStderrLogSink>(*alsologto);
    absl::AddLogSink(log_file_sink.get());
  }
  absl::Cleanup log_file_sink_cleanup = [&log_file_sink] {
    if (log_file_sink) {
      absl::RemoveLogSink(log_file_sink.get());
    }
  };

  int64_t opt_level = absl::GetFlag(FLAGS_opt_level);
  std::string top = absl::GetFlag(FLAGS_top);
  std::string ir_dump_path = absl::GetFlag(FLAGS_ir_dump_path);
  std::vector<std::string> skip_passes = absl::GetFlag(FLAGS_skip_passes);
  std::optional<int64_t> convert_array_index_to_select =
      NegativeIsNullopt(absl::GetFlag(FLAGS_convert_array_index_to_select));
  std::optional<int64_t> split_next_value_selects =
      NegativeIsNullopt(absl::GetFlag(FLAGS_split_next_value_selects));
  std::string ram_rewrites_pb = absl::GetFlag(FLAGS_ram_rewrites_pb);
  std::vector<RamRewrite> ram_rewrites_vec;
  if (!ram_rewrites_pb.empty()) {
    RamRewritesProto ram_rewrite_proto;
    XLS_RETURN_IF_ERROR(xls::ParseTextProtoFile(
        std::filesystem::path(ram_rewrites_pb), &ram_rewrite_proto));
    XLS_ASSIGN_OR_RETURN(ram_rewrites_vec,
                         RamRewritesFromProto(ram_rewrite_proto));
  }
  bool use_context_narrowing_analysis =
      absl::GetFlag(FLAGS_use_context_narrowing_analysis);
  bool optimize_for_best_case_throughput =
      absl::GetFlag(FLAGS_optimize_for_best_case_throughput);
  bool enable_resource_sharing = absl::GetFlag(FLAGS_enable_resource_sharing);
  bool force_resource_sharing = absl::GetFlag(FLAGS_force_resource_sharing);
  std::string area_model = absl::GetFlag(FLAGS_area_model);
  std::optional<std::string> pass_list = absl::GetFlag(FLAGS_passes);
  std::optional<int64_t> bisect_limit =
      absl::GetFlag(FLAGS_passes_bisect_limit);
  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));
  std::optional<std::string> pipeline_textproto =
      absl::GetFlag(FLAGS_passes_textproto);
  std::optional<std::string> pipeline_binproto =
      absl::GetFlag(FLAGS_passes_proto);
  std::variant<std::nullopt_t, std::string_view, PassPipelineProto>
      pass_pipeline = std::nullopt;
  if (absl::c_count_if(
          absl::Span<std::optional<std::string> const>{
              pass_list, pipeline_textproto, pipeline_binproto},
          [](const auto& v) -> bool { return v.has_value(); }) > 1) {
    return absl::InvalidArgumentError(
        "At most one of --pipeline_proto, --pipeline_textproto or --passes is "
        "allowed.");
  }
  if (pipeline_textproto) {
    XLS_ASSIGN_OR_RETURN(std::string data,
                         GetFileContents(*pipeline_textproto));
    PassPipelineProto res;
    XLS_RET_CHECK(google::protobuf::TextFormat::ParseFromString(data, &res));
    pass_pipeline = std::move(res);
  }
  if (pipeline_binproto) {
    XLS_ASSIGN_OR_RETURN(std::string data, GetFileContents(*pipeline_binproto));
    PassPipelineProto res;
    XLS_RET_CHECK(res.ParseFromString(data));
    pass_pipeline = std::move(res);
  }
  if (pass_list) {
    pass_pipeline = *pass_list;
  }

  bool debug_optimizations = absl::GetFlag(FLAGS_debug_optimizations);

  OptMetadata metadata;
  XLS_ASSIGN_OR_RETURN(
      std::string opt_ir,
      tools::OptimizeIrForTop(
          ir,
          OptOptions{
              .opt_level = opt_level,
              .top = top,
              .ir_dump_path = ir_dump_path,
              .skip_passes = std::move(skip_passes),
              .convert_array_index_to_select = convert_array_index_to_select,
              .split_next_value_selects = split_next_value_selects,
              .ram_rewrites = std::move(ram_rewrites_vec),
              .use_context_narrowing_analysis = use_context_narrowing_analysis,
              .optimize_for_best_case_throughput =
                  optimize_for_best_case_throughput,
              .enable_resource_sharing = enable_resource_sharing,
              .force_resource_sharing = force_resource_sharing,
              .area_model = area_model,
              .pass_pipeline = pass_pipeline,
              .bisect_limit = bisect_limit,
              .debug_optimizations = debug_optimizations,
          },
          &metadata));
  VLOG(2) << "Ran " << metadata.metrics.total_passes() << " passes";
  if (absl::GetFlag(FLAGS_pass_metrics_path)) {
    std::string tf;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(metadata.metrics, &tf));
    XLS_RETURN_IF_ERROR(
        SetFileContents(*absl::GetFlag(FLAGS_pass_metrics_path), tf));
  }

  if (output_path == "-") {
    std::cout << opt_ir;
  } else {
    XLS_RETURN_IF_ERROR(SetFileContents(output_path, opt_ir));
  }
  if (absl::GetFlag(FLAGS_passes_bisect_limit_is_error) && bisect_limit &&
      metadata.metrics.total_passes() >= *bisect_limit) {
    return absl::InternalError("passes bisect limit was reached.");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::tools

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (absl::GetFlag(FLAGS_list_passes)) {
    std::cout
        << xls::GetOptimizationPipelineGenerator().GetAvailablePassesStr();
    return 0;
  }
  if (positional_arguments.empty()) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                      argv[0]);
  }

  return xls::ExitStatus(xls::tools::RealMain(positional_arguments[0]));
}

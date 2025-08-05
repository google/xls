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

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/tool_timeout.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/tools/opt.h"
#include "xls/tools/opt_flags.h"
#include "xls/tools/opt_flags.pb.h"

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

ABSL_FLAG(bool, list_passes, false,
          "If passed list the names of all passes and exit.");

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

  XLS_ASSIGN_OR_RETURN(OptFlagsProto opt_flags, GetOptFlags(input_path));
  XLS_ASSIGN_OR_RETURN(OptOptions options, OptOptionsFromFlagsProto(opt_flags));

  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));

  OptMetadata metadata;
  XLS_ASSIGN_OR_RETURN(std::string optimized_ir,
                       OptimizeIrForTop(ir, options, &metadata));
  VLOG(2) << "Ran " << metadata.metrics.total_passes() << " passes";
  if (opt_flags.has_pass_metrics_path()) {
    std::string tf;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(metadata.metrics, &tf));
    XLS_RETURN_IF_ERROR(SetFileContents(opt_flags.pass_metrics_path(), tf));
  }

  if (output_path == "-") {
    std::cout << optimized_ir;
  } else {
    XLS_RETURN_IF_ERROR(SetFileContents(output_path, optimized_ir));
  }
  if (opt_flags.passes_bisect_limit_is_error() &&
      opt_flags.has_passes_bisect_limit() &&
      metadata.metrics.has_total_passes() &&
      metadata.metrics.total_passes() >= opt_flags.passes_bisect_limit()) {
    return absl::InternalError("passes bisect limit was reached.");
  }
  return absl::OkStatus();
}

absl::StatusOr<OptimizationPassPipelineGenerator> GetGeneratorForList(
    const OptFlagsProto& opt_flags) {
  OptimizationPassRegistry reg = GetOptimizationRegistry().OverridableClone();
  if (opt_flags.has_pipeline()) {
    XLS_RETURN_IF_ERROR(
        reg.RegisterPipelineProto(opt_flags.custom_registry(), "<pipeline>"));
  }
  return GetOptimizationPipelineGenerator(reg);
}
}  // namespace
}  // namespace xls::tools

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (absl::GetFlag(FLAGS_list_passes)) {
    absl::StatusOr<xls::OptFlagsProto> opt_flags =
        xls::GetOptFlags(std::nullopt);
    if (!opt_flags.ok()) {
      return xls::ExitStatus(opt_flags.status());
    }
    absl::StatusOr<xls::OptimizationPassPipelineGenerator> generator =
        xls::tools::GetGeneratorForList(*opt_flags);
    if (!generator.ok()) {
      return xls::ExitStatus(generator.status());
    }
    std::cout << generator->GetAvailablePassesStr();
    return 0;
  }
  if (positional_arguments.empty()) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                      argv[0]);
  }

  return xls::ExitStatus(xls::tools::RealMain(positional_arguments[0]));
}

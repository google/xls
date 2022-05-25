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

#include <numeric>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/block_metrics.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/ir_parser.h"
#include "xls/scheduling/pipeline_schedule.h"

const char kUsage[] = R"(
Dumps various codegen-related metrics about a block and corresponding Verilog
file. Designed to be used with run_benchmarks.py script.

Usage:
   benchmark_codegen_main --delay_model=DELAY_MODEL \
     OPT_IR_FILE BLOCK_IR_FILE VERILOG_FILE
)";

ABSL_FLAG(std::string, top, "",
          "Name of top block to use in lieu of the default.");
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from registry.");
ABSL_FLAG(int64_t, clock_period_ps, 0,
          "The number of picoseconds in a cycle to use when scheduling.");
ABSL_FLAG(int64_t, pipeline_stages, 0,
          "The number of stages in the pipeline when scheduling.");

namespace xls {
namespace {

absl::Status ScheduleAndPrintStats(Package* package,
                                   const DelayEstimator& delay_estimator,
                                   absl::optional<int64_t> clock_period_ps,
                                   absl::optional<int64_t> pipeline_stages) {
  SchedulingOptions options;
  if (clock_period_ps.has_value()) {
    options.clock_period_ps(*clock_period_ps);
  }
  if (pipeline_stages.has_value()) {
    options.pipeline_stages(*pipeline_stages);
  }

  absl::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity not set for package: %s.", package->name()));
  }
  absl::Time start = absl::Now();
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(top.value(), delay_estimator, options));
  absl::Duration total_time = absl::Now() - start;
  std::cout << absl::StreamFormat("Scheduling time: %dms\n",
                                  total_time / absl::Milliseconds(1));

  return absl::OkStatus();
}

absl::StatusOr<Block*> GetTopBlock(Package* package) {
  if (!absl::GetFlag(FLAGS_top).empty()) {
    return package->GetBlock(absl::GetFlag(FLAGS_top));
  }
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value()) {
    return absl::InvalidArgumentError("Package has no top defined");
  }

  if (!top.value()->IsBlock()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top entity of package is not a block: %s", top.value()->name()));
  }
  return top.value()->AsBlockOrDie();
}

absl::Status RealMain(absl::string_view opt_ir_path,
                      absl::string_view block_ir_path,
                      absl::string_view verilog_path,
                      absl::optional<const DelayEstimator*> delay_estimator) {
  XLS_VLOG(1) << "Reading optimized IR file: " << opt_ir_path;
  XLS_ASSIGN_OR_RETURN(std::string opt_ir_contents,
                       GetFileContents(opt_ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> opt_package,
                       Parser::ParsePackage(opt_ir_contents));

  XLS_VLOG(1) << "Reading block IR file: " << opt_ir_path;
  XLS_ASSIGN_OR_RETURN(std::string block_ir_contents,
                       GetFileContents(block_ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> block_package,
                       Parser::ParsePackage(block_ir_contents));

  XLS_VLOG(1) << "Reading Verilog file: " << verilog_path;
  XLS_ASSIGN_OR_RETURN(std::string verilog_contents,
                       GetFileContents(verilog_path));

  absl::optional<int64_t> clock_period_ps;
  if (absl::GetFlag(FLAGS_clock_period_ps) > 0) {
    clock_period_ps = absl::GetFlag(FLAGS_clock_period_ps);
  }
  absl::optional<int64_t> pipeline_stages;
  if (absl::GetFlag(FLAGS_pipeline_stages) > 0) {
    pipeline_stages = absl::GetFlag(FLAGS_pipeline_stages);
  }

  if (delay_estimator.has_value() &&
      (clock_period_ps.has_value() || pipeline_stages.has_value())) {
    // This may not be exactly how the scheduler is called because we're only
    // passing two of potentially numerous scheduling-related
    // arguments. However, since we're only printing the scheduling time this is
    // probably ok.
    // TODO(meheff): 2022/05/25 Fix this by maybe passing in proto of all the
    // scheduling options.
    XLS_RETURN_IF_ERROR(
        ScheduleAndPrintStats(opt_package.get(), *delay_estimator.value(),
                              clock_period_ps, pipeline_stages));
  }

  XLS_ASSIGN_OR_RETURN(Block * top, GetTopBlock(block_package.get()));
  XLS_ASSIGN_OR_RETURN(verilog::BlockMetricsProto metrics,
                       verilog::GenerateBlockMetrics(top, delay_estimator));
  std::cout << absl::StreamFormat("Flop count: %d\n", metrics.flop_count());
  std::cout << absl::StreamFormat(
      "Has feedthrough path: %s\n",
      metrics.feedthrough_path_exists() ? "true" : "false");
  if (metrics.has_max_reg_to_reg_delay_ps()) {
    std::cout << absl::StreamFormat("Max reg-to-reg delay: %dps\n",
                                    metrics.max_reg_to_reg_delay_ps());
  }
  if (metrics.has_max_input_to_reg_delay_ps()) {
    std::cout << absl::StreamFormat("Max input-to-reg delay: %dps\n",
                                    metrics.max_input_to_reg_delay_ps());
  }
  if (metrics.has_max_reg_to_output_delay_ps()) {
    std::cout << absl::StreamFormat("Max reg-to-output delay: %dps\n",
                                    metrics.max_reg_to_output_delay_ps());
  }
  if (metrics.has_max_feedthrough_path_delay_ps()) {
    std::cout << absl::StreamFormat("Max feedthrough path delay: %dps\n",
                                    metrics.max_feedthrough_path_delay_ps());
  }
  std::cout << absl::StreamFormat(
      "Lines of Verilog: %d\n",
      std::vector<std::string>(absl::StrSplit(verilog_contents, '\n')).size());

  return absl::OkStatus();
}

absl::StatusOr<absl::optional<const DelayEstimator*>> FetchDelayEstimator() {
  if (absl::GetFlag(FLAGS_delay_model).empty()) {
    return absl::nullopt;
  }
  XLS_ASSIGN_OR_RETURN(const DelayEstimator* delay_estimator,
                       GetDelayEstimator(absl::GetFlag(FLAGS_delay_model)));
  return delay_estimator;
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 3) {
    XLS_LOG(QFATAL) << absl::StreamFormat(
        "Expected invocation:\n  %s OPT_IR_FILE BLOCK_IR_FILE VERILOG_FILE",
        argv[0]);
  }

  absl::StatusOr<absl::optional<const xls::DelayEstimator*>>
      delay_estimator_or = xls::FetchDelayEstimator();
  XLS_QCHECK_OK(delay_estimator_or.status());

  XLS_QCHECK_OK(xls::RealMain(positional_arguments[0], positional_arguments[1],
                              positional_arguments[2],
                              delay_estimator_or.value()));
  return EXIT_SUCCESS;
}

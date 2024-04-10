// Copyright 2021 The XLS Authors
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
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/fdo/synthesized_delay_diff_utils.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

const char kUsage[] = R"(

Dumps delay information about an XLS function including per-node delay
information and critical-path. Example invocations:

Emit delay information about a function:
   delay_info_main --delay_model=unit --top=ENTRY IR_FILE

Emit delay information about a function including per-stage critical path
information:
   delay_info_main --delay_model=unit \
     --schedule_path=SCHEDULE_FILE \
     --top=ENTRY \
     IR_FILE
)";

ABSL_FLAG(
    std::string, top, "",
    "The name of the top entity. Currently, only functions are supported. "
    "Function to emit delay information about.");
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from registry.");
ABSL_FLAG(std::string, schedule_path, "",
          "Optional path to a pipeline schedule to use for emitting per-stage "
          "critical paths.");
ABSL_FLAG(bool, compare_to_synthesis, false,
          "Whether to compare the delay info from the XLS delay model to "
          "synthesizer output.");
ABSL_FLAG(
    std::string, yosys_path, "",
    "Path to the Yosys binary, required if using --compare_to_synthesis.");
ABSL_FLAG(std::string, sta_path, "",
          "Path to the STA binary, required if using --compare_to_synthesis.");
ABSL_FLAG(std::string, synthesis_libraries, "",
          "Path to the synthesis libraries, required if using "
          "--compare_to_synthesis.");
ABSL_FLAG(int, abs_delay_diff_min_ps, 0,
          "Return an error exit code if the absolute value of `synthesized "
          "delay - delay model prediction` is below this threshold. This "
          "enables use of delay_info_main as a helper for ir_minimizer_main, "
          "to find the minimal IR exhibiting a minimum difference. "
          "`compare_to_synthesis` must also be true.");

namespace xls::tools {
namespace {

absl::Status RealMain(std::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir, input_path));
  FunctionBase* top;
  if (absl::GetFlag(FLAGS_top).empty()) {
    if (!p->HasTop()) {
      return absl::InternalError(
          absl::StrFormat("Top entity not set for package: %s.", p->name()));
    }
    top = p->GetTop().value();
  } else {
    XLS_ASSIGN_OR_RETURN(top,
                         p->GetFunctionBaseByName(absl::GetFlag(FLAGS_top)));
  }

  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       GetDelayEstimator(absl::GetFlag(FLAGS_delay_model)));
  std::unique_ptr<synthesis::Synthesizer> synthesizer;
  if (absl::GetFlag(FLAGS_compare_to_synthesis)) {
    SchedulingOptions flags;
    flags.fdo_yosys_path(absl::GetFlag(FLAGS_yosys_path));
    flags.fdo_sta_path(absl::GetFlag(FLAGS_sta_path));
    flags.fdo_synthesis_libraries(absl::GetFlag(FLAGS_synthesis_libraries));
    XLS_ASSIGN_OR_RETURN(
        synthesizer,
        synthesis::GetSynthesizerManagerSingleton().MakeSynthesizer(
            flags.fdo_synthesizer_name(), flags));
  }
  std::optional<synthesis::SynthesizedDelayDiff> total_diff;
  if (absl::GetFlag(FLAGS_schedule_path).empty()) {
    XLS_ASSIGN_OR_RETURN(
        std::vector<CriticalPathEntry> critical_path,
        AnalyzeCriticalPath(top, /*clock_period_ps=*/std::nullopt,
                            *delay_estimator));
    std::cout << "# Critical path:\n";
    if (synthesizer) {
      XLS_ASSIGN_OR_RETURN(
          total_diff, SynthesizeAndGetDelayDiff(top, std::move(critical_path),
                                                synthesizer.get()));
      std::cout << SynthesizedDelayDiffToString(*total_diff);
    } else {
      std::cout << CriticalPathToString(critical_path);
    }
    std::cout << "\n";
  } else {
    XLS_ASSIGN_OR_RETURN(PackagePipelineSchedulesProto proto,
                         ParseTextProtoFile<PackagePipelineSchedulesProto>(
                             absl::GetFlag(FLAGS_schedule_path)));
    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::FromProto(top, proto));
    XLS_RETURN_IF_ERROR(schedule.Verify());
    XLS_ASSIGN_OR_RETURN(
        synthesis::SynthesizedDelayDiffByStage delay_diff,
        synthesis::CreateDelayDiffByStage(top, schedule, *delay_estimator,
                                          synthesizer.get()));
    total_diff = delay_diff.total_diff;
    for (int64_t i = 0; i < schedule.length(); ++i) {
      std::cout << absl::StrFormat("# Critical path for stage %d:\n", i);
      if (synthesizer) {
        std::cout << SynthesizedStageDelayDiffToString(
            delay_diff.stage_diffs[i], delay_diff.total_diff);
      } else {
        std::cout << CriticalPathToString(
            delay_diff.stage_diffs[i].critical_path);
      }
      std::cout << "\n";
    }
  }

  std::cout << "# Delay of all nodes:\n";
  for (Node* node : TopoSort(top)) {
    absl::StatusOr<int64_t> delay_status =
        delay_estimator->GetOperationDelayInPs(node);
    if (delay_status.ok()) {
      std::cout << absl::StreamFormat("%-15s : %5dps\n", node->GetName(),
                                      delay_status.value());
    } else {
      std::cout << absl::StreamFormat("%-15s : <unknown>\n", node->GetName());
    }
  }

  const int64_t abs_delay_diff_min_ps =
      absl::GetFlag(FLAGS_abs_delay_diff_min_ps);
  if (abs_delay_diff_min_ps != 0) {
    if (!total_diff.has_value() || !synthesizer) {
      return absl::InvalidArgumentError(
          "--abs_delay_diff_min_ps was specified without "
          "--compare_to_synthesis.");
    }
    if (std::abs(total_diff->synthesized_delay_ps - total_diff->xls_delay_ps) <
        abs_delay_diff_min_ps) {
      return absl::OutOfRangeError(
          "The yosys delay absolute diff was not in the specified range.");
    }
    std::cout << "The absolute delay diff is within the specified range.\n";
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::tools

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty()) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                      argv[0]);
  }

  return xls::ExitStatus(xls::tools::RealMain(positional_arguments[0]));
}

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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/extract_stage.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

const char kUsage[] = R"(

Dumps delay information about an XLS function including per-node delay
information and critical-path. Example invocations:

Emit delay information about a function:
   delay_info_main --delay_model=unit --entry=ENTRY IR_FILE

Emit delay information about a function including per-stage critical path
information:
   delay_info_main --delay_model=unit \
     --schedule_path=SCHEDULE_FILE \
     --entry=ENTRY \
     IR_FILE
)";

ABSL_FLAG(std::string, entry, "", "Function to emit delay information about.");
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from registry.");
ABSL_FLAG(std::string, schedule_path, "",
          "Optional path to a pipeline schedule to use for emitting per-stage "
          "critical paths.");

namespace xls::tools {
namespace {

absl::Status RealMain(absl::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir, input_path));
  Function* function;
  if (absl::GetFlag(FLAGS_entry).empty()) {
    XLS_ASSIGN_OR_RETURN(function, p->EntryFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(function, p->GetFunction(absl::GetFlag(FLAGS_entry)));
  }

  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       GetDelayEstimator(absl::GetFlag(FLAGS_delay_model)));

  if (absl::GetFlag(FLAGS_schedule_path).empty()) {
    XLS_ASSIGN_OR_RETURN(
        std::vector<CriticalPathEntry> critical_path,
        AnalyzeCriticalPath(function, /*clock_period_ps=*/absl::nullopt,
                            *delay_estimator));
    std::cout << "# Critical path:\n";
    std::cout << CriticalPathToString(critical_path);
    std::cout << "\n";
  } else {
    XLS_ASSIGN_OR_RETURN(PipelineScheduleProto proto,
                         ParseTextProtoFile<PipelineScheduleProto>(
                             absl::GetFlag(FLAGS_schedule_path)));
    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::FromProto(function, proto));
    XLS_RETURN_IF_ERROR(schedule.Verify());
    for (int64_t i = 0; i < schedule.length(); ++i) {
      XLS_ASSIGN_OR_RETURN(Function * stage_function,
                           ExtractStage(function, schedule, i));
      XLS_ASSIGN_OR_RETURN(
          std::vector<CriticalPathEntry> critical_path,
          AnalyzeCriticalPath(stage_function, /*clock_period_ps=*/absl::nullopt,
                              *delay_estimator));
      std::cout << absl::StrFormat("# Critical path for stage %d:\n", i);
      std::cout << CriticalPathToString(critical_path);
      std::cout << "\n";
    }
  }

  std::cout << "# Delay of all nodes:\n";
  for (Node* node : TopoSort(function)) {
    absl::StatusOr<int64_t> delay_status =
        delay_estimator->GetOperationDelayInPs(node);
    if (delay_status.ok()) {
      std::cout << absl::StreamFormat("%-15s : %5dps\n", node->GetName(),
                                      delay_status.value());
    } else {
      std::cout << absl::StreamFormat("%-15s : <unknown>\n", node->GetName());
    }
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::tools

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty()) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                          argv[0]);
  }

  XLS_QCHECK_OK(xls::tools::RealMain(positional_arguments[0]));
  return EXIT_SUCCESS;
}

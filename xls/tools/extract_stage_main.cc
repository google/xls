// Copyright 2020 Google LLC
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

// Simple driver for executing the ExtractStage() routine.
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/scheduling/extract_stage.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

ABSL_FLAG(std::string, ir_path, "", "Path to the IR file to load.");
ABSL_FLAG(std::string, function, "",
          "Function to extract from. "
          "If unspecified, a \"best guess\" will be selected.");
ABSL_FLAG(std::string, output_path, "", "Path to which to write output.");
ABSL_FLAG(std::string, schedule_path, "",
          "Path to the function's pipeline schedule.");
ABSL_FLAG(int, stage, -1, "Pipeline stage to extract.");

namespace xls {

absl::Status RealMain(const std::string& ir_path,
                      absl::optional<std::string> function_name,
                      const std::string& schedule_path, int stage,
                      const std::string& output_path) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  Function* function;
  if (function_name) {
    XLS_ASSIGN_OR_RETURN(function, package->GetFunction(function_name.value()));
  } else {
    XLS_ASSIGN_OR_RETURN(function, package->EntryFunction());
  }

  XLS_ASSIGN_OR_RETURN(
      PipelineScheduleProto proto,
      ParseTextProtoFile<PipelineScheduleProto>(schedule_path));
  XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                       PipelineSchedule::FromProto(function, proto));

  XLS_RETURN_IF_ERROR(ExtractStage(function, schedule, stage).status());
  XLS_RETURN_IF_ERROR(SetFileContents(output_path, package->DumpIr()));

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  XLS_QCHECK(!ir_path.empty()) << "--ir_path can't be empty!";

  absl::optional<std::string> function_name;
  if (!absl::GetFlag(FLAGS_function).empty()) {
    function_name = absl::GetFlag(FLAGS_function);
  }

  std::string schedule_path = absl::GetFlag(FLAGS_schedule_path);
  XLS_QCHECK(!schedule_path.empty()) << "--schedule_path can't be empty!";

  int stage = absl::GetFlag(FLAGS_stage);
  XLS_QCHECK(stage != -1) << "--stage must be specified!";

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  XLS_QCHECK(!output_path.empty()) << "--output path can't be empty!";
  XLS_QCHECK_OK(
      xls::RealMain(ir_path, function_name, schedule_path, stage, output_path));
  return 0;
}

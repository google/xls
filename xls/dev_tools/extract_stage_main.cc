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

// Simple driver for executing the ExtractStage() routine.
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
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
ABSL_FLAG(
    int, stage, -1,
    "Pipeline stage to extract, if not specified all stages are extracted.");

namespace xls {

static absl::Status RealMain(const std::string& ir_path,
                             std::optional<std::string> function_name,
                             const std::string& schedule_path, int stage,
                             const std::string& output_path) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  FunctionBase* function;
  if (function_name) {
    auto get_proc = package->GetFunction(function_name.value());
    if (get_proc.ok()) {
      function = get_proc.value();
    } else {
      XLS_ASSIGN_OR_RETURN(function, package->GetProc(function_name.value()));
    }
  } else {
    XLS_ASSIGN_OR_RETURN(function, package->GetTopAsFunction());
  }

  XLS_ASSIGN_OR_RETURN(PackageScheduleProto proto,
                       ParseTextProtoFile<PackageScheduleProto>(schedule_path));
  XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                       PipelineSchedule::FromProto(function, proto));
  std::vector<FunctionBase*> funcs = package->GetFunctionBases();

  if (stage == -1) {
    for (int i = 0; i < schedule.length(); ++i) {
      XLS_ASSIGN_OR_RETURN(Function * stage,
                           ExtractStage(function, schedule, i));
      XLS_RETURN_IF_ERROR(package->SetTop(stage));
    }
  } else {
    XLS_ASSIGN_OR_RETURN(Function * stage,
                         ExtractStage(function, schedule, stage));
    XLS_RETURN_IF_ERROR(package->SetTop(stage));
  }

  for (auto& f : funcs) {
    XLS_RETURN_IF_ERROR(package->RemoveFunctionBase(f));
  }
  while (!package->channels().empty()) {
    XLS_RETURN_IF_ERROR(package->RemoveChannel(package->channels().front()));
  }
  XLS_RETURN_IF_ERROR(SetFileContents(output_path, package->DumpIr()));

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  QCHECK(!ir_path.empty()) << "--ir_path can't be empty!";

  std::optional<std::string> function_name;
  if (!absl::GetFlag(FLAGS_function).empty()) {
    function_name = absl::GetFlag(FLAGS_function);
  }

  std::string schedule_path = absl::GetFlag(FLAGS_schedule_path);
  QCHECK(!schedule_path.empty()) << "--schedule_path can't be empty!";

  int stage = absl::GetFlag(FLAGS_stage);

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  QCHECK(!output_path.empty()) << "--output path can't be empty!";
  return xls::ExitStatus(
      xls::RealMain(ir_path, function_name, schedule_path, stage, output_path));
}

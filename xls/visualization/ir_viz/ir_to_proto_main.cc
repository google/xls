// Copyright 2023 The XLS Authors
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

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/visualization/ir_viz/ir_to_proto.h"
#include "xls/visualization/ir_viz/visualization.pb.h"

ABSL_FLAG(std::string, delay_model, "", "Delay model to use.");
ABSL_FLAG(std::optional<int64_t>, pipeline_stages, std::nullopt,
          "Pipeline stages to use when scheduling the function");
ABSL_FLAG(std::optional<std::string>, entry_name, std::nullopt, "Entry name");
ABSL_FLAG(bool, binary_format, false,
          "Whether to return the proto in binary serialized format (defaults "
          "to text)");
ABSL_FLAG(bool, token_dag, false,
          "Only output IR nodes associated with tokens");

constexpr std::string_view kUsage =
    R"(Expected: ir_to_json_main --delay_model=MODEL [--pipeline_stages=N] [--entry_name=ENTRY] /path/to/file.ir)";

namespace xls {
namespace {

// Returns the top entity to view in the visualizer. If the top is not set in
// the package, returns an arbitrary entity. If the package does not contain
// any entities, returns an error.
absl::StatusOr<FunctionBase*> GetFunctionBaseToView(Package* package) {
  std::optional<FunctionBase*> top = package->GetTop();
  if (top.has_value()) {
    return top.value();
  }
  if (!package->GetFunctionBases().empty()) {
    return package->GetFunctionBases().front();
  }
  return absl::NotFoundError(
      absl::StrFormat("No entities found in package: %s.", package->name()));
}

absl::Status RealMain(const std::filesystem::path& ir_path,
                      std::string_view delay_model_name,
                      std::optional<int64_t> pipeline_stages,
                      std::optional<std::string_view> entry_name,
                      bool token_dag) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));
  FunctionBase* func_base;
  if (entry_name.has_value()) {
    XLS_ASSIGN_OR_RETURN(func_base, package->GetFunction(entry_name.value()));
  } else {
    XLS_ASSIGN_OR_RETURN(func_base, GetFunctionBaseToView(package.get()));
  }
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       GetDelayEstimator(delay_model_name));
  XLS_ASSIGN_OR_RETURN(AreaEstimator * area_estimator,
                       GetAreaEstimator(delay_model_name));

  xls::viz::Package proto;
  if (pipeline_stages.has_value()) {
    // TODO(meheff): Support scheduled procs.
    XLS_RET_CHECK(func_base->IsFunction());
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func_base->AsFunctionOrDie(), *delay_estimator,
            SchedulingOptions().pipeline_stages(pipeline_stages.value())));
    XLS_ASSIGN_OR_RETURN(
        proto, IrToProto(package.get(), *delay_estimator, *area_estimator,
                         &schedule, func_base->name(), token_dag));
  } else {
    XLS_ASSIGN_OR_RETURN(
        proto, IrToProto(package.get(), *delay_estimator, *area_estimator,
                         /*schedule=*/nullptr, func_base->name(), token_dag));
  }
  google::protobuf::io::OstreamOutputStream cout(&std::cout);
  if (absl::GetFlag(FLAGS_binary_format)) {
    proto.SerializeToZeroCopyStream(&cout);
  } else {
    XLS_RET_CHECK(google::protobuf::TextFormat::Print(proto, &cout));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1 || positional_arguments[0].empty()) {
    LOG(QFATAL) << "Expected one position argument (IR path): " << argv[0]
                << " <ir_path>";
  }
  if (absl::GetFlag(FLAGS_delay_model).empty()) {
    LOG(QFATAL) << "--delay_model is required";
  }

  return xls::ExitStatus(xls::RealMain(
      positional_arguments[0], absl::GetFlag(FLAGS_delay_model),
      absl::GetFlag(FLAGS_pipeline_stages), absl::GetFlag(FLAGS_entry_name),
      absl::GetFlag(FLAGS_token_dag)));
}

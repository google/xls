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

#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/csv/csv_record.h"
#include "riegeli/csv/csv_writer.h"
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

constexpr std::string_view kUsage =
    R"(Expected: ir_to_csvs_main --delay_model=MODEL [--pipeline_stages=N] [--entry_name=ENTRY] /path/to/file.ir /path/to/node_output.csv /path/to/edge_output.csv)";

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

std::string ToFieldValue(bool v) { return v ? "true" : "false"; }

template <typename T>
std::string ToFieldValue(T v) {
  return absl::StrCat(v);
}

constexpr riegeli::CsvHeaderConstant kNodeHeader = {"name",
                                                    "label",
                                                    "ir",
                                                    "opcode",
                                                    "value",
                                                    "start",
                                                    "width",
                                                    "delay_ps",
                                                    "known_bits",
                                                    "on_critical_path",
                                                    "cycle",
                                                    "state_param_index",
                                                    "initial_value",
                                                    "area_um",
                                                    "file",
                                                    "line",
                                                    "range",
                                                    "all_locs"};
riegeli::CsvRecord NodeRecord(const viz::Node& node) {
  std::string locations =
      absl::StrJoin(node.loc(), "\n", [](std::string* out, const auto& loc) {
        absl::StrAppend(out, loc.file(), ":", loc.line());
      });
  // Trim off whitespace
  locations.erase(0, locations.find_first_not_of(" \n\r\t"));
  locations.erase(locations.find_last_not_of(" \n\r\t") + 1);
  CHECK(std::isnormal(node.attributes().start()));
  CHECK(std::isnormal(node.attributes().width()));
  CHECK(std::isnormal(node.attributes().cycle()));
  CHECK(std::isnormal(node.attributes().state_param_index()));
  CHECK(std::isnormal(node.attributes().area_um()));
  return riegeli::CsvRecord(
      *kNodeHeader,
      {
          node.id(),
          node.name(),
          node.ir(),
          node.opcode(),
          node.attributes().value(),
          node.attributes().has_start()
              ? ToFieldValue(static_cast<int64_t>(node.attributes().start()))
              : "",
          node.attributes().has_width()
              ? ToFieldValue(static_cast<int64_t>(node.attributes().width()))
              : "",
          node.attributes().has_delay_ps()
              ? ToFieldValue(node.attributes().delay_ps())
              : "",
          node.attributes().known_bits(),
          node.attributes().has_on_critical_path()
              ? ToFieldValue(node.attributes().on_critical_path())
              : "",
          node.attributes().has_cycle()
              ? ToFieldValue(static_cast<int64_t>(node.attributes().cycle()))
              : "",
          node.attributes().has_state_param_index()
              ? ToFieldValue(
                    static_cast<int64_t>(node.attributes().state_param_index()))
              : "",
          node.attributes().initial_value(),
          node.attributes().has_area_um()
              ? ToFieldValue(static_cast<int64_t>(node.attributes().area_um()))
              : "",
          node.loc_size() >= 1 && absl::c_all_of(node.loc(),
                                                 [&](auto loc) {
                                                   return loc.file() ==
                                                          node.loc()[0].file();
                                                 })
              ? node.loc()[0].file()
              : "",
          node.loc_size() == 1 ? ToFieldValue(node.loc()[0].line()) : "",
          node.attributes().has_ranges() ? node.attributes().ranges() : "",
          locations,
      });
}

constexpr riegeli::CsvHeaderConstant kEdgeHeader = {
    "node1", "node2", "id", "bit_width", "type", "on_critical_path"};
riegeli::CsvRecord EdgeRecord(const viz::Edge& edge) {
  CHECK(std::isnormal(edge.bit_width()));
  return riegeli::CsvRecord(
      *kEdgeHeader,
      {edge.source_id(), edge.target_id(), edge.id(),
       edge.has_bit_width()
           ? ToFieldValue(static_cast<int64_t>(edge.bit_width()))
           : "",
       edge.type(),
       edge.has_on_critical_path() ? ToFieldValue(edge.on_critical_path())
                                   : ""});
}

absl::Status RealMain(const std::filesystem::path& ir_path,
                      const std::filesystem::path& node_csv_path,
                      const std::filesystem::path& edge_csv_path,
                      std::string_view delay_model_name,
                      std::optional<int64_t> pipeline_stages,
                      std::optional<std::string_view> entry_name) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));
  FunctionBase* func_base;
  if (entry_name.has_value()) {
    XLS_ASSIGN_OR_RETURN(func_base,
                         package->GetFunctionBaseByName(entry_name.value()));
  } else {
    XLS_ASSIGN_OR_RETURN(func_base, GetFunctionBaseToView(package.get()));
  }
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       GetDelayEstimator(delay_model_name));
  XLS_ASSIGN_OR_RETURN(AreaEstimator * area_estimator,
                       GetAreaEstimator(delay_model_name));

  viz::Package package_proto;
  if (pipeline_stages.has_value()) {
    // TODO(meheff): Support scheduled procs.
    XLS_RET_CHECK(func_base->IsFunction());
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func_base->AsFunctionOrDie(), *delay_estimator,
            SchedulingOptions().pipeline_stages(pipeline_stages.value())));
    XLS_ASSIGN_OR_RETURN(package_proto,
                         IrToProto(package.get(), *delay_estimator, &schedule,
                                   func_base->name()));
  } else {
    XLS_ASSIGN_OR_RETURN(
        package_proto,
        IrToProto(package.get(), *delay_estimator, *area_estimator,
                  /*schedule=*/nullptr, func_base->name()));
  }

  const viz::FunctionBase* entry = nullptr;
  for (const auto& function_base : package_proto.function_bases()) {
    if (function_base.name() == func_base->name()) {
      entry = &function_base;
      break;
    }
  }
  if (entry == nullptr) {
    return absl::NotFoundError(
        absl::StrFormat("Entry %s missing from proto", func_base->name()));
  }

  riegeli::CsvWriterBase::Options node_options;
  node_options.set_header(*kNodeHeader);
  auto node_writer = riegeli::CsvWriter(
      riegeli::FdWriter(node_csv_path.string()), node_options);
  XLS_RETURN_IF_ERROR(node_writer.status());

  riegeli::CsvWriterBase::Options edge_options;
  edge_options.set_header(*kEdgeHeader);
  auto edge_writer = riegeli::CsvWriter(
      riegeli::FdWriter(edge_csv_path.string()), edge_options);
  XLS_RETURN_IF_ERROR(edge_writer.status());

  for (const auto& node : entry->nodes()) {
    node_writer.WriteRecord(NodeRecord(node));
  }
  for (const auto& edge : entry->edges()) {
    edge_writer.WriteRecord(EdgeRecord(edge));
  }

  absl::Status status;
  if (!edge_writer.Close()) {
    LOG(ERROR) << "Failed to close edge CSV writer";
    status.Update(edge_writer.status());
  }
  if (!node_writer.Close()) {
    LOG(ERROR) << "Failed to close node CSV writer";
    status.Update(node_writer.status());
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 3 || positional_arguments[0].empty()) {
    LOG(QFATAL) << "Expected three position arguments (IR path, node CSV "
                   "path, edge CSV path): "
                << argv[0] << " <ir_path> <node_csv_path> <edge_csv_path>";
  }
  if (absl::GetFlag(FLAGS_delay_model).empty()) {
    LOG(QFATAL) << "--delay_model is required";
  }

  return xls::ExitStatus(xls::RealMain(
      positional_arguments[0], positional_arguments[1], positional_arguments[2],
      absl::GetFlag(FLAGS_delay_model), absl::GetFlag(FLAGS_pipeline_stages),
      absl::GetFlag(FLAGS_entry_name)));
}

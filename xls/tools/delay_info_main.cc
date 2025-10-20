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
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_info.pb.h"
#include "xls/fdo/grpc_synthesizer.h"
#include "xls/fdo/synthesized_delay_diff_utils.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"
#include "xls/tools/scheduling_options_flags.h"

static constexpr std::string_view kUsage = R"(

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
ABSL_FLAG(std::string, schedule_path, "",
          "Optional path to a pipeline schedule to use for emitting per-stage "
          "critical paths.");
ABSL_FLAG(bool, schedule, false,
          "Run scheduling to generate a schedule for delay analysis, rather "
          "than reading a schedule via --schedule_path.");
ABSL_FLAG(bool, compare_to_synthesis, false,
          "Whether to compare the delay info from the XLS delay model to "
          "synthesizer output.");
ABSL_FLAG(std::string, synthesis_server, "ipv4:///0.0.0.0:10000",
          "The address, including port, of the gRPC server to use with "
          "--compare_to_synthesis.");
ABSL_FLAG(int, abs_delay_diff_min_ps, 0,
          "Return an error exit code if the absolute value of `synthesized "
          "delay - delay model prediction` is below this threshold. This "
          "enables use of delay_info_main as a helper for ir_minimizer_main, "
          "to find the minimal IR exhibiting a minimum difference. "
          "`compare_to_synthesis` must also be true.");
ABSL_FLAG(std::optional<int>, stage, std::nullopt,
          "Only analyze the specified, zero-based stage of the pipeline.");
ABSL_FLAG(std::optional<std::string>, proto_out, std::nullopt,
          "File to write a binary xls.DelayInfoProto to containing delay info "
          "of the input.");

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

  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      xls::GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       SetUpDelayEstimator(scheduling_options_flags_proto));
  std::unique_ptr<synthesis::Synthesizer> synthesizer;
  if (absl::GetFlag(FLAGS_compare_to_synthesis)) {
    synthesis::GrpcSynthesizerParameters parameters(
        absl::GetFlag(FLAGS_synthesis_server));
    XLS_ASSIGN_OR_RETURN(
        synthesizer,
        synthesis::GetSynthesizerManagerSingleton().MakeSynthesizer(
            parameters.name(), parameters));
  }
  std::optional<DelayInfoProto> delay_proto;
  if (absl::GetFlag(FLAGS_proto_out)) {
    delay_proto.emplace();
  }
  std::optional<synthesis::SynthesizedDelayDiff> total_diff;
  if (absl::GetFlag(FLAGS_schedule_path).empty() &&
      !absl::GetFlag(FLAGS_schedule)) {
    XLS_ASSIGN_OR_RETURN(
        std::vector<CriticalPathEntry> critical_path,
        AnalyzeCriticalPath(top, /*clock_period_ps=*/std::nullopt,
                            *delay_estimator));
    if (delay_proto) {
      *delay_proto->mutable_combinational_critical_path() =
          CriticalPathToProto(critical_path);
    }
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
    PackageScheduleProto proto;
    if (absl::GetFlag(FLAGS_schedule)) {
      if (!absl::GetFlag(FLAGS_schedule_path).empty()) {
        return absl::InvalidArgumentError(
            "Cannot specify both --schedule and --schedule_path.");
      }
      XLS_ASSIGN_OR_RETURN(
          SchedulingOptions scheduling_options,
          SetUpSchedulingOptions(scheduling_options_flags_proto, p.get()));
      XLS_ASSIGN_OR_RETURN(
          SchedulingResult result,
          Schedule(p.get(), scheduling_options, delay_estimator));
      proto = result.package_schedule;
    } else {
      XLS_ASSIGN_OR_RETURN(proto, ParseTextProtoFile<PackageScheduleProto>(
                                      absl::GetFlag(FLAGS_schedule_path)));
    }

    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::FromProto(top, proto));
    XLS_RETURN_IF_ERROR(schedule.Verify());
    std::optional<synthesis::SynthesizedDelayDiffByStage> diff_by_stage;
    std::optional<int> requested_stage = absl::GetFlag(FLAGS_stage);
    if (requested_stage.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          total_diff,
          CreateDelayDiffForStage(top, schedule, *delay_estimator,
                                  synthesizer.get(), *requested_stage));
    } else {
      XLS_ASSIGN_OR_RETURN(diff_by_stage, synthesis::CreateDelayDiffByStage(
                                              top, schedule, *delay_estimator,
                                              synthesizer.get()));
      total_diff = diff_by_stage->total_diff;
    }
    for (int64_t i = 0; i < schedule.length(); ++i) {
      if (requested_stage.has_value() && i != *requested_stage) {
        continue;
      }
      std::cout << absl::StrFormat("# Critical path for stage %d:\n", i);
      const std::vector<CriticalPathEntry>& real_critical_path =
          requested_stage.has_value()
              ? total_diff->critical_path
              : diff_by_stage->stage_diffs[i].critical_path;
      if (requested_stage.has_value()) {
        // If a particular stage was specified by the user, `total_diff` is for
        // that stage.
        std::cout << (synthesizer
                          ? SynthesizedDelayDiffToString(*total_diff)
                          : CriticalPathToString(total_diff->critical_path));
      } else if (synthesizer) {
        // `compare_to_synthesis` for all stages. Here we include the difference
        // in weighting of stages between the delay model and synthesis.
        std::cout << SynthesizedStageDelayDiffToString(
            diff_by_stage->stage_diffs[i],
            diff_by_stage->stage_percent_diffs[i]);
      } else {
        // Plain output for all stages without `compare_to_synthesis`.
        std::cout << CriticalPathToString(
            diff_by_stage->stage_diffs[i].critical_path);
      }
      if (delay_proto) {
        delay_proto->mutable_pipelined_critical_path()
            ->mutable_stage()
            ->emplace(i, CriticalPathToProto(real_critical_path));
      }
      std::cout << "\n";
    }
    if (top->IsProc()) {
      Proc* proc = top->AsProcOrDie();
      for (StateElement* state_element : proc->StateElements()) {
        std::cout << absl::StrFormat("# Critical path for state element %s:\n",
                                     state_element->name());
        XLS_ASSIGN_OR_RETURN(
            std::vector<CriticalPathEntry> state_critical_path,
            AnalyzeCriticalPath(
                top, /*clock_period_ps=*/std::nullopt, *delay_estimator,
                [&](Node* node) {
                  return node->Is<StateRead>() &&
                         node->As<StateRead>()->state_element() ==
                             state_element;
                },
                /*sink_filter=*/
                [&](Node* node) {
                  if (node->Is<StateRead>() &&
                      node->As<StateRead>()->state_element() == state_element) {
                    return true;
                  }
                  if (node->Is<Next>()) {
                    return node->As<Next>()->state_read() ==
                           proc->GetStateRead(state_element);
                  }
                  return false;
                }));
        std::cout << CriticalPathToString(state_critical_path) << "\n";
      }
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
    if (delay_proto) {
      DelayInfoNodeProto* node_proto = delay_proto->add_all_nodes();
      node_proto->set_id(node->id());
      node_proto->set_ir(node->ToStringWithOperandTypes());
      node_proto->set_op(ToOpProto(node->op()));
      if (delay_status.ok()) {
        node_proto->set_node_delay_ps(delay_status.value());
      }
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
  if (delay_proto) {
    XLS_RETURN_IF_ERROR(SetFileContents(*absl::GetFlag(FLAGS_proto_out),
                                        delay_proto->SerializeAsString()))
        << "Unable to write proto file";
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

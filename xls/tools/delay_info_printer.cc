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

#include "xls/tools/delay_info_printer.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
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
#include "xls/tools/delay_info_flags.pb.h"
#include "xls/tools/schedule.h"
#include "xls/tools/scheduling_options_flags.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls::tools {
namespace {

class DelayInfoPrinterImpl : public DelayInfoPrinter {
 public:
  absl::Status Init(DelayInfoFlagsProto flags) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p, LoadPackage(flags));
    XLS_ASSIGN_OR_RETURN(SchedulingOptionsFlagsProto scheduling_flags,
                         xls::GetSchedulingOptionsFlagsProto());
    XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                         SetUpDelayEstimator(scheduling_flags));

    std::unique_ptr<synthesis::Synthesizer> synthesizer;
    if (flags.compare_to_synthesis()) {
      synthesis::GrpcSynthesizerParameters parameters(flags.synthesis_server());
      XLS_ASSIGN_OR_RETURN(
          synthesizer,
          synthesis::GetSynthesizerManagerSingleton().MakeSynthesizer(
              parameters.name(), parameters));
    }
    return InitInternal(std::move(flags), std::move(p), delay_estimator,
                        std::move(synthesizer));
  }

  absl::Status Init(DelayInfoFlagsProto flags, DelayEstimator* delay_estimator,
                    std::unique_ptr<synthesis::Synthesizer> synthesizer) final {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p, LoadPackage(flags));
    return InitInternal(std::move(flags), std::move(p), delay_estimator,
                        std::move(synthesizer));
  }

  absl::StatusOr<std::unique_ptr<Package>> LoadPackage(
      const DelayInfoFlagsProto& flags) {
    std::string input_path = flags.input_path();
    if (input_path == "-") {
      input_path = "/dev/stdin";
    }
    XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));
    return Parser::ParsePackage(ir, input_path);
  }

  absl::Status InitInternal(
      DelayInfoFlagsProto flags, std::unique_ptr<Package> package,
      DelayEstimator* delay_estimator,
      std::unique_ptr<synthesis::Synthesizer> synthesizer) {
    flags_ = std::move(flags);
    package_ = std::move(package);
    delay_estimator_ = delay_estimator;
    synthesizer_ = std::move(synthesizer);

    if (!flags_.has_top()) {
      if (!package_->HasTop()) {
        return absl::InternalError(absl::StrFormat(
            "Top entity not set for package: %s.", package_->name()));
      }
      top_ = package_->GetTop().value();
    } else {
      XLS_ASSIGN_OR_RETURN(top_, package_->GetFunctionBaseByName(flags_.top()));
    }
    return absl::OkStatus();
  }

  absl::Status GenerateApplicableInfo() final {
    if (flags_.schedule_path().empty() && !flags_.schedule()) {
      XLS_RETURN_IF_ERROR(GenerateCombinationalCriticalPathInfo());
    } else {
      XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule, DetermineSchedule());
      XLS_RETURN_IF_ERROR(GenerateStageCriticalPathInfo(schedule));
      if (top_->IsProc()) {
        Proc* proc = top_->AsProcOrDie();
        XLS_RETURN_IF_ERROR(GenerateProcStateCriticalPathInfo(proc));
      }
    }

    XLS_RETURN_IF_ERROR(GenerateNodeDelays());
    XLS_RETURN_IF_ERROR(ValidateTotalDiff());

    if (flags_.has_proto_out()) {
      XLS_RETURN_IF_ERROR(
          SetFileContents(flags_.proto_out(), delay_proto_.SerializeAsString()))
          << "Unable to write proto file";
    }
    return absl::OkStatus();
  }

  absl::StatusOr<PipelineSchedule> DetermineSchedule() {
    PackageScheduleProto proto;
    if (flags_.schedule()) {
      if (!flags_.schedule_path().empty()) {
        return absl::InvalidArgumentError(
            "Cannot specify both --schedule and --schedule_path.");
      }
      XLS_ASSIGN_OR_RETURN(SchedulingOptionsFlagsProto scheduling_flags,
                           xls::GetSchedulingOptionsFlagsProto());
      XLS_ASSIGN_OR_RETURN(
          SchedulingOptions scheduling_options,
          SetUpSchedulingOptions(scheduling_flags, package_.get()));
      XLS_ASSIGN_OR_RETURN(
          SchedulingResult result,
          Schedule(package_.get(), scheduling_options, delay_estimator_));
      proto = result.package_schedule;
    } else {
      XLS_ASSIGN_OR_RETURN(proto, ParseTextProtoFile<PackageScheduleProto>(
                                      flags_.schedule_path()));
    }
    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::FromProto(top_, proto));
    XLS_RETURN_IF_ERROR(schedule.Verify());
    return schedule;
  }

  absl::Status GenerateCombinationalCriticalPathInfo() {
    XLS_ASSIGN_OR_RETURN(
        std::vector<CriticalPathEntry> critical_path,
        AnalyzeCriticalPath(top_, /*clock_period_ps=*/std::nullopt,
                            *delay_estimator_));
    *delay_proto_.mutable_combinational_critical_path() =
        CriticalPathToProto(critical_path);
    std::cout << "# Critical path:\n";
    if (flags_.compare_to_synthesis()) {
      XLS_ASSIGN_OR_RETURN(
          total_diff_, SynthesizeAndGetDelayDiff(top_, std::move(critical_path),
                                                 synthesizer_.get()));
      std::cout << SynthesizedDelayDiffToString(*total_diff_);
    } else {
      std::cout << CriticalPathToString(critical_path);
    }
    std::cout << "\n";
    return absl::OkStatus();
  }

  absl::Status GenerateStageCriticalPathInfo(const PipelineSchedule& schedule) {
    std::cout << "Generating stage info" << std::endl;
    std::optional<synthesis::SynthesizedDelayDiffByStage> diff_by_stage;
    std::optional<int> requested_stage;
    if (flags_.has_stage()) {
      requested_stage = flags_.stage();
    }
    if (requested_stage) {
      XLS_ASSIGN_OR_RETURN(
          total_diff_,
          CreateDelayDiffForStage(top_, schedule, *delay_estimator_,
                                  synthesizer_.get(), *requested_stage));
    } else {
      XLS_ASSIGN_OR_RETURN(diff_by_stage, synthesis::CreateDelayDiffByStage(
                                              top_, schedule, *delay_estimator_,
                                              synthesizer_.get()));
      total_diff_ = diff_by_stage->total_diff;
    }
    for (int64_t i = 0; i < schedule.length(); ++i) {
      if (requested_stage.has_value() && i != *requested_stage) {
        continue;
      }
      std::cout << absl::StrFormat("# Critical path for stage %d:\n", i);
      const std::vector<CriticalPathEntry>& real_critical_path =
          requested_stage.has_value()
              ? total_diff_->critical_path
              : diff_by_stage->stage_diffs[i].critical_path;
      if (requested_stage.has_value()) {
        // If a particular stage was specified by the user, `total_diff` is for
        // that stage.
        std::cout << (synthesizer_
                          ? SynthesizedDelayDiffToString(*total_diff_)
                          : CriticalPathToString(total_diff_->critical_path));
      } else if (synthesizer_) {
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

      delay_proto_.mutable_pipelined_critical_path()->mutable_stage()->emplace(
          i, CriticalPathToProto(real_critical_path));
      std::cout << "\n";
    }
    return absl::OkStatus();
  }

  absl::Status GenerateProcStateCriticalPathInfo(const Proc* proc) {
    for (StateElement* state_element : proc->StateElements()) {
      std::cout << absl::StrFormat("# Critical path for state element %s:\n",
                                   state_element->name());
      XLS_ASSIGN_OR_RETURN(
          std::vector<CriticalPathEntry> state_critical_path,
          AnalyzeCriticalPath(
              top_, /*clock_period_ps=*/std::nullopt, *delay_estimator_,
              [&](Node* node) {
                return node->Is<StateRead>() &&
                       node->As<StateRead>()->state_element() == state_element;
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
    return absl::OkStatus();
  }

  absl::Status GenerateNodeDelays() {
    std::cout << "# Delay of all nodes:\n";
    for (Node* node : TopoSort(top_)) {
      absl::StatusOr<int64_t> delay_status =
          delay_estimator_->GetOperationDelayInPs(node);
      if (delay_status.ok()) {
        std::cout << absl::StreamFormat("%-15s : %5dps\n", node->GetName(),
                                        delay_status.value());
      } else {
        std::cout << absl::StreamFormat("%-15s : <unknown>\n", node->GetName());
      }

      DelayInfoNodeProto* node_proto = delay_proto_.add_all_nodes();
      node_proto->set_id(node->id());
      node_proto->set_ir(node->ToStringWithOperandTypes());
      node_proto->set_op(ToOpProto(node->op()));
      if (delay_status.ok()) {
        node_proto->set_node_delay_ps(delay_status.value());
      }
    }
    return absl::OkStatus();
  }

  absl::Status ValidateTotalDiff() {
    const int64_t abs_delay_diff_min_ps = flags_.abs_delay_diff_min_ps();
    if (abs_delay_diff_min_ps != 0) {
      if (!total_diff_.has_value() || !synthesizer_) {
        return absl::InvalidArgumentError(
            "--abs_delay_diff_min_ps was specified without "
            "--compare_to_synthesis.");
      }
      if (std::abs(total_diff_->synthesized_delay_ps -
                   total_diff_->xls_delay_ps) < abs_delay_diff_min_ps) {
        return absl::OutOfRangeError(
            "The yosys delay absolute diff was not in the specified range.");
      }
      std::cout << "The absolute delay diff is within the specified range.\n";
    }
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<Package> package_;
  FunctionBase* top_;
  DelayInfoFlagsProto flags_;
  DelayEstimator* delay_estimator_;
  std::unique_ptr<synthesis::Synthesizer> synthesizer_;
  DelayInfoProto delay_proto_;
  std::optional<synthesis::SynthesizedDelayDiff> total_diff_;
};

}  // namespace

std::unique_ptr<DelayInfoPrinter> CreateDelayInfoPrinter() {
  return std::make_unique<DelayInfoPrinterImpl>();
}

}  // namespace xls::tools

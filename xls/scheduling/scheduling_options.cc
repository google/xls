// Copyright 2024 The XLS Authors
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

#include "xls/scheduling/scheduling_options.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/package.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

namespace {

absl::StatusOr<SchedulingOptions> OptionsFromFlagProto(
    Package* p, const SchedulingOptionsFlagsProto& proto) {
  // Some fields are pre-initialized with defaults
  SchedulingOptions scheduling_options;

  if (proto.has_opt_level()) {
    scheduling_options.opt_level(proto.opt_level());
  }
  if (proto.pipeline_stages() != 0) {
    scheduling_options.pipeline_stages(proto.pipeline_stages());
  }
  if (proto.clock_period_ps() != 0) {
    scheduling_options.clock_period_ps(proto.clock_period_ps());
  }
  if (proto.clock_margin_percent() != 0) {
    scheduling_options.clock_margin_percent(proto.clock_margin_percent());
  }
  if (proto.period_relaxation_percent() != 0) {
    scheduling_options.period_relaxation_percent(
        proto.period_relaxation_percent());
  }
  scheduling_options.minimize_clock_on_failure(
      proto.minimize_clock_on_failure());
  scheduling_options.recover_after_minimizing_clock(
      proto.recover_after_minimizing_clock());
  if (proto.worst_case_throughput() != 1) {
    scheduling_options.worst_case_throughput(proto.worst_case_throughput());
  }
  scheduling_options.minimize_worst_case_throughput(
      proto.minimize_worst_case_throughput());
  if (proto.has_dynamic_throughput_objective_weight()) {
    scheduling_options.dynamic_throughput_objective_weight(
        proto.dynamic_throughput_objective_weight());
  }
  if (proto.additional_input_delay_ps() != 0) {
    scheduling_options.additional_input_delay_ps(
        proto.additional_input_delay_ps());
  }
  if (proto.additional_output_delay_ps() != 0) {
    scheduling_options.additional_output_delay_ps(
        proto.additional_output_delay_ps());
  }
  if (!proto.additional_channel_delay_ps().empty()) {
    absl::flat_hash_map<std::string, int64_t> additional_channel_delay_ps(
        proto.additional_channel_delay_ps().begin(),
        proto.additional_channel_delay_ps().end());
    scheduling_options.additional_channel_delay_ps(
        std::move(additional_channel_delay_ps));
  }
  if (proto.ffi_fallback_delay_ps() != 0) {
    scheduling_options.ffi_fallback_delay_ps(proto.ffi_fallback_delay_ps());
  }

  for (const std::string& c : proto.io_constraints()) {
    std::vector<std::string> components = absl::StrSplit(c, ':');
    if (components.size() != 6) {
      return absl::InternalError(
          absl::StrFormat("Could not parse IO constraint: `%s`", c));
    }
    auto parse_dir =
        [&](const std::string& str) -> absl::StatusOr<IODirection> {
      if (str == "send") {
        return IODirection::kSend;
      }
      if (str == "recv") {
        return IODirection::kReceive;
      }
      return absl::InternalError(
          absl::StrFormat("Could not parse IO constraint: "
                          "invalid channel direction in `%s`",
                          c));
    };
    const std::string& source = components[0];
    XLS_ASSIGN_OR_RETURN(IODirection source_dir, parse_dir(components[1]));
    const std::string& target = components[2];
    XLS_ASSIGN_OR_RETURN(IODirection target_dir, parse_dir(components[3]));
    int64_t min_latency, max_latency;
    if (components[4] == "none") {
      min_latency = std::numeric_limits<int64_t>::min();
    } else if (!absl::SimpleAtoi(components[4], &min_latency)) {
      return absl::InternalError(
          absl::StrFormat("Could not parse IO constraint: "
                          "invalid minimum latency in `%s`",
                          c));
    }
    if (components[5] == "none") {
      max_latency = std::numeric_limits<int64_t>::max();
    } else if (!absl::SimpleAtoi(components[5], &max_latency)) {
      return absl::InternalError(
          absl::StrFormat("Could not parse IO constraint: "
                          "invalid maximum latency in `%s`",
                          c));
    }
    IOConstraint constraint(source, source_dir, target, target_dir, min_latency,
                            max_latency);
    scheduling_options.add_constraint(constraint);
  }
  if (proto.receives_first_sends_last()) {
    scheduling_options.add_constraint(RecvsFirstSendsLastConstraint());
  }
  if (proto.has_mutual_exclusion_z3_rlimit() &&
      proto.mutual_exclusion_z3_rlimit() >= 0) {
    scheduling_options.mutual_exclusion_z3_rlimit(
        proto.mutual_exclusion_z3_rlimit());
  }
  if (proto.has_default_next_value_z3_rlimit() &&
      proto.default_next_value_z3_rlimit() >= 0) {
    scheduling_options.default_next_value_z3_rlimit(
        proto.default_next_value_z3_rlimit());
  }

  if (p != nullptr) {
    for (const SchedulingConstraint& c : scheduling_options.constraints()) {
      if (std::holds_alternative<IOConstraint>(c)) {
        IOConstraint io_constr = std::get<IOConstraint>(c);
        if (!p->GetChannel(io_constr.SourceChannel()).ok()) {
          return absl::InternalError(absl::StrFormat(
              "Invalid channel name in IO constraint: %s; "
              "this name did not correspond to any channel in the package",
              io_constr.SourceChannel()));
        }
        if (!p->GetChannel(io_constr.TargetChannel()).ok()) {
          return absl::InternalError(absl::StrFormat(
              "Invalid channel name in IO constraint: %s; "
              "this name did not correspond to any channel in the package",
              io_constr.TargetChannel()));
        }
      }
    }
  }

  if (proto.has_failure_behavior()) {
    scheduling_options.failure_behavior(
        SchedulingFailureBehavior::FromProto(proto.failure_behavior()));
  }

  // The following fdo_* have valid default value init in scheduling_options.
  // Only copy proto->scheduling_options if the option is present in the proto.
  // If we copy obliviously, we could overwrite the valid default with
  // an invalid 0 or "" (the proto global defaults for unspecified fields).

  // When options are specified individually, unspecified FDO options
  //   will have their default value from absl flags copied to the proto.
  // When options are provided via proto, unspecified FDO options
  //   will get their default value from 'scheduling_options' initialization.

  // The value of FDO options should only be read from scheduling_options,
  // not the proto!

  if (proto.has_use_fdo()) {
    scheduling_options.use_fdo(proto.use_fdo());
  }

  if (proto.has_fdo_iteration_number()) {
    if (proto.fdo_iteration_number() < 2) {
      return absl::InternalError("fdo_iteration_number must be >= 2");
    }
    scheduling_options.fdo_iteration_number(proto.fdo_iteration_number());
  }

  if (proto.has_fdo_delay_driven_path_number()) {
    if (proto.fdo_delay_driven_path_number() < 0) {
      return absl::InternalError("delay_driven_path_number must be >= 0");
    }
    scheduling_options.fdo_delay_driven_path_number(
        proto.fdo_delay_driven_path_number());
  }

  if (proto.has_fdo_fanout_driven_path_number()) {
    if (proto.fdo_fanout_driven_path_number() < 0) {
      return absl::InternalError("fanout_driven_path_number must be >= 0");
    }
    scheduling_options.fdo_fanout_driven_path_number(
        proto.fdo_fanout_driven_path_number());
  }

  if (proto.has_fdo_refinement_stochastic_ratio()) {
    if (proto.fdo_refinement_stochastic_ratio() > 1.0 ||
        proto.fdo_refinement_stochastic_ratio() <= 0.0) {
      return absl::InternalError(
          "refinement_stochastic_ratio must be <= 1.0 and > 0.0");
    }
    scheduling_options.fdo_refinement_stochastic_ratio(
        proto.fdo_refinement_stochastic_ratio());
  }

  if (proto.has_fdo_path_evaluate_strategy()) {
    if (proto.fdo_path_evaluate_strategy() != "path" &&
        proto.fdo_path_evaluate_strategy() != "cone" &&
        proto.fdo_path_evaluate_strategy() != "window") {
      return absl::InternalError(
          "path_evaluate_strategy must be 'path', 'cone', or 'window'");
    }
    scheduling_options.fdo_path_evaluate_strategy(
        proto.fdo_path_evaluate_strategy());
  }

  if (proto.has_fdo_synthesizer_name()) {
    scheduling_options.fdo_synthesizer_name(proto.fdo_synthesizer_name());
  }

  // These have no default values
  scheduling_options.fdo_yosys_path(proto.fdo_yosys_path());
  scheduling_options.fdo_sta_path(proto.fdo_sta_path());
  scheduling_options.fdo_synthesis_libraries(proto.fdo_synthesis_libraries());
  scheduling_options.fdo_default_driver_cell(proto.fdo_default_driver_cell());
  scheduling_options.fdo_default_load(proto.fdo_default_load());

  scheduling_options.schedule_all_procs(proto.multi_proc());

  return scheduling_options;
}

}  // namespace

absl::StatusOr<DelayEstimator*> SetUpDelayEstimator(
    const SchedulingOptionsFlagsProto& flags) {
  return GetDelayEstimator(flags.delay_model());
}

absl::StatusOr<bool> IsDelayModelSpecifiedViaFlag(
    const SchedulingOptionsFlagsProto& flags) {
  return !flags.delay_model().empty();
}

absl::StatusOr<SchedulingOptions> SetUpSchedulingOptions(
    const SchedulingOptionsFlagsProto& flags, Package* p) {
  return OptionsFromFlagProto(p, flags);
}

}  // namespace xls

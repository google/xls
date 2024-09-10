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

#include "xls/tools/scheduling_options_flags.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/scheduling_options_flags.pb.h"

// LINT.IfChange
ABSL_FLAG(int64_t, opt_level, xls::kMaxOptLevel,
          absl::StrFormat("Optimization level. Ranges from 1 to %d.",
                          xls::kMaxOptLevel));
ABSL_FLAG(int64_t, clock_period_ps, 0,
          "Target clock period, in picoseconds. See "
          "https://google.github.io/xls/scheduling for details.");
ABSL_FLAG(int64_t, pipeline_stages, 0,
          "The number of stages in the generated pipeline. See "
          "https://google.github.io/xls/scheduling for details.");
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from registry.");
ABSL_FLAG(int64_t, clock_margin_percent, 0,
          "The percentage of clock period to set aside as a margin to ensure "
          "timing is met. Effectively, this lowers the clock period by this "
          "percentage amount for the purposes of scheduling. See "
          "https://google.github.io/xls/scheduling for details.");
ABSL_FLAG(int64_t, period_relaxation_percent, 0,
          "The percentage of clock period that will be relaxed when "
          "scheduling without an explicit --clock_period_ps. "
          "When set to 0, the minimum period that can satisfy scheduling "
          "constraints will be used. Increasing this will trade-off an "
          "increase in critical path delay in favor of decreased register "
          "count. See https://google.github.io/xls/scheduling for details.");
ABSL_FLAG(
    bool, minimize_clock_on_failure, true,
    "If true, when `--clock_period_ps` is given but is infeasible for "
    "scheduling, search for & report the shortest feasible clock period. "
    "Otherwise, just reports whether increasing the clock period can help.");
ABSL_FLAG(
    bool, recover_after_minimizing_clock, false,
    "If both this and `--minimize_clock_on_failure` are true and "
    "`--clock_period_ps` is given and infeasible for scheduling, search for & "
    "use the shortest feasible clock period - even if this does not meet the "
    "`--clock_period_ps` target - after printing a warning."
    "Otherwise, will stop with an error if `--clock_period_ps` is infeasible.");
ABSL_FLAG(bool, minimize_worst_case_throughput, false,
          "If true, when `--worst_case_throughput` is not given, search for & "
          "report the best possible worst-case throughput of the circuit "
          "(subject to all other constraints). If `--clock_period_ps` is not "
          "set, will first optimize for clock speed, and then find the best "
          "possible worst-case throughput within that constraint.");
ABSL_FLAG(std::optional<int64_t>, worst_case_throughput, std::nullopt,
          "Allow scheduling a pipeline with worst-case throughput no slower "
          "than once per N cycles. If unspecified and "
          "`--minimize_worst_case_throughput` is not set, enforces full "
          "throughput.\n"
          "Note: a higher value for --worst_case_throughput *decreases* the "
          "worst-case throughput, since this controls inverse throughput.\n"
          "\n"
          "If zero, no throughput bound will be enforced.\n"
          "If negative, XLS will find the fastest throughput achievable given "
          "all other constraints specified.");
ABSL_FLAG(int64_t, additional_input_delay_ps, 0,
          "The additional delay added to each receive node.");
ABSL_FLAG(int64_t, ffi_fallback_delay_ps, 0,
          "Delay of foreign function calls if not otherwise specified.");
ABSL_FLAG(std::vector<std::string>, io_constraints, {},
          "A comma-separated list of IO constraints, each of which is "
          "specified by a literal like `foo:send:bar:recv:3:5` which means "
          "that sends on channel `foo` must occur between 3 and 5 cycles "
          "(inclusive) before receives on channel `bar`. "
          "Note that for a constraint like `foo:send:foo:send:3:5`, no "
          "constraint will be applied between a node and itself; i.e.: this "
          "means all _different_ pairs of nodes sending on `foo` must be in "
          "cycles that differ by between 3 and 5. "
          "If the special minimum/maximum value `none` is used, then "
          "the minimum latency will be the lowest representable int64_t, "
          "and likewise for maximum latency.");
ABSL_FLAG(bool, receives_first_sends_last, false,
          "If true, this forces receives into the first cycle and sends into "
          "the last cycle.");
ABSL_FLAG(int64_t, mutual_exclusion_z3_rlimit, -1,
          "Resource limit for solver in mutual exclusion pass.");
ABSL_FLAG(int64_t, default_next_value_z3_rlimit, -1,
          "Resource limit for optimizer when attempting to prove a state param "
          "doesn't need a default next_value; if not specified, will not "
          "attempt this proof using Z3, but will still avoid adding a "
          "redundant default next_value in specific circumstances.");
ABSL_FLAG(std::string, scheduling_options_proto, "",
          "Path to a protobuf containing all scheduling options args.");
ABSL_FLAG(bool, explain_infeasibility, true,
          "If scheduling fails, re-run scheduling with extra slack variables "
          "in an attempt to explain why scheduling failed.");
ABSL_FLAG(
    std::optional<double>, infeasible_per_state_backedge_slack_pool,
    std::nullopt,
    "If specified, the specified value must be > 0. Setting this configures "
    "how the scheduling problem is reformulated in the case that it fails. If "
    "specified, this value will cause the reformulated problem to include "
    "per-state backedge slack variables, which increases the complexity. "
    "This value scales the objective such that adding slack to the per-state "
    "backedge is preferred up until total slack reaches the pool size, after "
    "which adding slack to the shared backedge slack variable is preferred. "
    "Increasing this value should give more specific information about how "
    "much slack each failing backedge needs at the cost of less actionable and "
    "harder to understand output.");
ABSL_FLAG(bool, use_fdo, false,
          "Use FDO (feedback-directed optimization) for pipeline scheduling. "
          "If 'false', then all 'fdo_*' options are ignored.");
ABSL_FLAG(int64_t, fdo_iteration_number, 5,
          "The number of FDO iterations during the pipeline scheduling. Must "
          "be an integer >= 2.");
ABSL_FLAG(int64_t, fdo_delay_driven_path_number, 1,
          "The number of delay-driven subgraphs in each FDO iteration. Must be "
          "a non-negative integer.");
ABSL_FLAG(int64_t, fdo_fanout_driven_path_number, 0,
          "The number of fanout-driven subgraphs in each FDO iteration. Must "
          "be a non-negative integer.");
ABSL_FLAG(
    float, fdo_refinement_stochastic_ratio, 1.0,
    "*path_number over refinement_stochastic_ratio paths are extracted and "
    "*path_number paths are randomly selected from them for synthesis in each "
    "FDO iteration. Must be a positive float <= 1.0.");
ABSL_FLAG(std::string, fdo_path_evaluate_strategy, "window",
          "Path evaluation strategy for FDO. Supports path, cone, and window.");
ABSL_FLAG(std::string, fdo_synthesizer_name, "yosys",
          "Name of synthesis backend for FDO. Only supports yosys.");
ABSL_FLAG(std::string, fdo_yosys_path, "", "Absolute path of yosys.");
ABSL_FLAG(std::string, fdo_sta_path, "", "Absolute path of OpenSTA.");
ABSL_FLAG(std::string, fdo_synthesis_libraries, "",
          "Synthesis and STA libraries.");
ABSL_FLAG(std::string, fdo_default_driver_cell, "",
          "Cell to assume is driving primary inputs");
ABSL_FLAG(std::string, fdo_default_load, "",
          "Cell to assume is being driven by primary outputs");
// TODO: google/xls#869 - Remove when proc-scoped channels supplant old-style
// procs.
ABSL_FLAG(bool, multi_proc, false,
          "If true, schedule all procs and codegen them all.");
// LINT.ThenChange(
//   //xls/build_rules/xls_providers.bzl,
//   //docs_src/codegen_options.md
// )
ABSL_FLAG(std::optional<std::string>, scheduling_options_used_textproto_file,
          std::nullopt,
          "If present, path to write a protobuf recording all schedule args "
          "used (including those set on the cmd line).");

namespace xls {

static absl::StatusOr<bool> SetOptionsFromFlags(
    SchedulingOptionsFlagsProto& proto) {
#define POPULATE_FLAG(__x)                                   \
  {                                                          \
    any_flags_set |= FLAGS_##__x.IsSpecifiedOnCommandLine(); \
    proto.set_##__x(absl::GetFlag(FLAGS_##__x));             \
  }
#define POPULATE_REPEATED_FLAG(__x)                                           \
  {                                                                           \
    any_flags_set |= FLAGS_##__x.IsSpecifiedOnCommandLine();                  \
    do {                                                                      \
      proto.mutable_##__x()->Clear();                                         \
      auto repeated_flag = absl::GetFlag(FLAGS_##__x);                        \
      proto.mutable_##__x()->Add(repeated_flag.begin(), repeated_flag.end()); \
    } while (0);                                                              \
  }
  bool any_flags_set = false;
  POPULATE_FLAG(opt_level);
  POPULATE_FLAG(clock_period_ps);
  POPULATE_FLAG(pipeline_stages);
  POPULATE_FLAG(delay_model);
  POPULATE_FLAG(clock_margin_percent);
  POPULATE_FLAG(period_relaxation_percent);
  POPULATE_FLAG(minimize_clock_on_failure);
  POPULATE_FLAG(recover_after_minimizing_clock);
  POPULATE_FLAG(minimize_worst_case_throughput);
  {
    any_flags_set |= FLAGS_worst_case_throughput.IsSpecifiedOnCommandLine();
    proto.set_worst_case_throughput(
        absl::GetFlag(FLAGS_worst_case_throughput)
            .value_or(proto.minimize_worst_case_throughput() ? 0 : 1));
  }
  POPULATE_FLAG(additional_input_delay_ps);
  POPULATE_FLAG(ffi_fallback_delay_ps);
  POPULATE_REPEATED_FLAG(io_constraints);
  POPULATE_FLAG(receives_first_sends_last);
  POPULATE_FLAG(mutual_exclusion_z3_rlimit);
  POPULATE_FLAG(default_next_value_z3_rlimit);
  POPULATE_FLAG(use_fdo);
  POPULATE_FLAG(fdo_iteration_number);
  POPULATE_FLAG(fdo_delay_driven_path_number);
  POPULATE_FLAG(fdo_fanout_driven_path_number);
  POPULATE_FLAG(fdo_refinement_stochastic_ratio);
  POPULATE_FLAG(fdo_path_evaluate_strategy);
  POPULATE_FLAG(fdo_synthesizer_name);
  POPULATE_FLAG(fdo_yosys_path);
  POPULATE_FLAG(fdo_sta_path);
  POPULATE_FLAG(fdo_synthesis_libraries);
  POPULATE_FLAG(fdo_default_driver_cell);
  POPULATE_FLAG(fdo_default_load);
  POPULATE_FLAG(multi_proc);
#undef POPULATE_FLAG
#undef POPULATE_REPEATED_FLAG

  // Failure behavior is a nested message, so handle directly instead of using
  // POPULATE_FLAG().
  any_flags_set |= FLAGS_explain_infeasibility.IsSpecifiedOnCommandLine();
  SchedulingFailureBehaviorProto* failure_behavior =
      proto.mutable_failure_behavior();
  failure_behavior->set_explain_infeasibility(
      absl::GetFlag(FLAGS_explain_infeasibility));
  std::optional<double> infeasible_per_state_backedge_slack_pool =
      absl::GetFlag(FLAGS_infeasible_per_state_backedge_slack_pool);
  if (infeasible_per_state_backedge_slack_pool.has_value()) {
    any_flags_set |= true;
    failure_behavior->set_infeasible_per_state_backedge_slack_pool(
        *infeasible_per_state_backedge_slack_pool);
  }

  return any_flags_set;
}

absl::StatusOr<SchedulingOptionsFlagsProto> GetSchedulingOptionsFlagsProto() {
  SchedulingOptionsFlagsProto proto;
  XLS_ASSIGN_OR_RETURN(bool any_individual_flags_set,
                       SetOptionsFromFlags(proto));
  if (any_individual_flags_set) {
    if (FLAGS_scheduling_options_proto.IsSpecifiedOnCommandLine()) {
      return absl::InvalidArgumentError(
          "Cannot combine 'scheduling_options_proto' and command line "
          "scheduling arguments");
    }
  } else if (FLAGS_scheduling_options_proto.IsSpecifiedOnCommandLine()) {
    XLS_RETURN_IF_ERROR(xls::ParseTextProtoFile(
        absl::GetFlag(FLAGS_scheduling_options_proto), &proto));
  }
  if (absl::GetFlag(FLAGS_scheduling_options_used_textproto_file)) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        *absl::GetFlag(FLAGS_scheduling_options_used_textproto_file), proto));
  }
  return proto;
}

static absl::StatusOr<SchedulingOptions> OptionsFromFlagProto(
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
  if (proto.additional_input_delay_ps() != 0) {
    scheduling_options.additional_input_delay_ps(
        proto.additional_input_delay_ps());
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
    std::string source = components[0];
    XLS_ASSIGN_OR_RETURN(IODirection source_dir, parse_dir(components[1]));
    std::string target = components[2];
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

absl::StatusOr<synthesis::Synthesizer*> SetUpSynthesizer(
    const SchedulingOptions& flags) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<synthesis::Synthesizer> synthesizer,
      synthesis::GetSynthesizerManagerSingleton().MakeSynthesizer(
          flags.fdo_synthesizer_name(), flags));
  return synthesizer.release();
}

}  // namespace xls

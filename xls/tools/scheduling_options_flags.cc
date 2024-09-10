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
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/tools/scheduling_options_flags.pb.h"

// LINT.IfChange
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

}  // namespace xls

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

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/scheduling/pipeline_schedule.h"

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
ABSL_FLAG(int64_t, additional_input_delay_ps, 0,
          "The additional delay added to each receive node.");
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
// LINT.ThenChange(
//   //xls/build_rules/xls_codegen_rules.bzl,
//   //docs_src/codegen_options.md
// )

namespace xls {

absl::StatusOr<SchedulingOptions> SetUpSchedulingOptions(Package* p) {
  SchedulingOptions scheduling_options;

  if (absl::GetFlag(FLAGS_pipeline_stages) != 0) {
    scheduling_options.pipeline_stages(absl::GetFlag(FLAGS_pipeline_stages));
  }
  if (absl::GetFlag(FLAGS_clock_period_ps) != 0) {
    scheduling_options.clock_period_ps(absl::GetFlag(FLAGS_clock_period_ps));
  }
  if (absl::GetFlag(FLAGS_clock_margin_percent) != 0) {
    scheduling_options.clock_margin_percent(
        absl::GetFlag(FLAGS_clock_margin_percent));
  }
  if (absl::GetFlag(FLAGS_period_relaxation_percent) != 0) {
    scheduling_options.period_relaxation_percent(
        absl::GetFlag(FLAGS_period_relaxation_percent));
  }
  if (absl::GetFlag(FLAGS_additional_input_delay_ps) != 0) {
    scheduling_options.additional_input_delay_ps(
        absl::GetFlag(FLAGS_additional_input_delay_ps));
  }
  for (const std::string& c : absl::GetFlag(FLAGS_io_constraints)) {
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
  if (absl::GetFlag(FLAGS_receives_first_sends_last)) {
    scheduling_options.add_constraint(RecvsFirstSendsLastConstraint());
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

  return scheduling_options;
}

absl::StatusOr<DelayEstimator*> SetUpDelayEstimator() {
  return GetDelayEstimator(absl::GetFlag(FLAGS_delay_model));
}

}  // namespace xls

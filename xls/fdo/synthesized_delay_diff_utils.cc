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

#include "xls/fdo/synthesized_delay_diff_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/extract_stage.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace synthesis {
namespace {

// Returns the percent of overall pipeline delay post-synthesis that is in the
// stage represented by `stage_diff`.
double GetSynthesizedPercentOfPipeline(
    const SynthesizedDelayDiff& stage_diff,
    const SynthesizedDelayDiff& overall_diff) {
  return overall_diff.synthesized_delay_ps == 0
             ? 0
             : static_cast<double>(stage_diff.synthesized_delay_ps) * 100.0 /
                   static_cast<double>(overall_diff.synthesized_delay_ps);
}

// Returns the percent of overall pipeline delay that is in the stage
// represented by `stage_diff` according to the XLS delay model.
double GetXlsPercentOfPipeline(const SynthesizedDelayDiff& stage_diff,
                               const SynthesizedDelayDiff& overall_diff) {
  return overall_diff.xls_delay_ps == 0
             ? 0
             : static_cast<double>(stage_diff.xls_delay_ps) * 100.0 /
                   static_cast<double>(overall_diff.xls_delay_ps);
}

// Returns the percent of the XLS delay model prediction that the actual
// synthesized delay represents. Most of the time, this is less than 100%, i.e.
// the delay model is pessimistic.
double GetSynthesizedPercentOfXlsDelay(const SynthesizedDelayDiff& diff) {
  return diff.xls_delay_ps == 0
             ? 0
             : static_cast<double>(diff.synthesized_delay_ps) * 100.0 /
                   static_cast<double>(diff.xls_delay_ps);
}

// Converts the given delay diff to a human-readable header suitable to prepend
// to the output of `CriticalPathToString`.
std::string SynthesizedDelayDiffToStringHeader(
    const SynthesizedDelayDiff& diff) {
  return absl::StrFormat(" %6dps as synthesized (%+dps)",
                         diff.synthesized_delay_ps,
                         diff.synthesized_delay_ps - diff.xls_delay_ps);
}

}  // namespace

absl::StatusOr<SynthesizedDelayDiff> SynthesizeAndGetDelayDiff(
    FunctionBase* f, std::vector<CriticalPathEntry> critical_path,
    synthesis::Synthesizer* synthesizer) {
  absl::flat_hash_set<Node*> nodes;
  nodes.insert(f->nodes().begin(), f->nodes().end());
  XLS_ASSIGN_OR_RETURN(int64_t delay,
                       synthesizer->SynthesizeNodesAndGetDelay(nodes));
  const int64_t xls_delay_ps =
      critical_path.empty() ? 0 : critical_path[0].path_delay_ps;
  return SynthesizedDelayDiff{
      .critical_path = std::move(critical_path),
      .xls_delay_ps = xls_delay_ps,
      .synthesized_delay_ps = delay,
  };
}

absl::StatusOr<SynthesizedDelayDiffByStage> CreateDelayDiffByStage(
    FunctionBase* f, const PipelineSchedule& schedule,
    const DelayEstimator& delay_estimator,
    synthesis::Synthesizer* synthesizer) {
  SynthesizedDelayDiffByStage result;
  result.stage_diffs.resize(schedule.length());
  result.stage_percent_diffs.resize(schedule.length());
  for (int i = 0; i < schedule.length(); ++i) {
    SynthesizedDelayDiff& stage_diff = result.stage_diffs[i];
    XLS_ASSIGN_OR_RETURN(Function * stage_function,
                         ExtractStage(f, schedule, i));
    XLS_ASSIGN_OR_RETURN(
        std::vector<CriticalPathEntry> critical_path,
        AnalyzeCriticalPath(stage_function, /*clock_period_ps=*/std::nullopt,
                            delay_estimator));
    if (synthesizer) {
      XLS_ASSIGN_OR_RETURN(
          stage_diff,
          SynthesizeAndGetDelayDiff(stage_function, std::move(critical_path),
                                    synthesizer));
    } else {
      stage_diff.xls_delay_ps =
          critical_path.empty() ? 0 : critical_path[0].path_delay_ps;
      stage_diff.critical_path = std::move(critical_path);
    }
    result.total_diff.synthesized_delay_ps += stage_diff.synthesized_delay_ps;
    result.total_diff.xls_delay_ps += stage_diff.xls_delay_ps;
  }
  for (int i = 0; i < schedule.length(); ++i) {
    const SynthesizedDelayDiff& stage_diff = result.stage_diffs[i];
    StagePercentDiff& stage_percent_diff = result.stage_percent_diffs[i];
    stage_percent_diff.synthesized_percent =
        GetSynthesizedPercentOfPipeline(stage_diff, result.total_diff);
    stage_percent_diff.xls_percent =
        GetXlsPercentOfPipeline(stage_diff, result.total_diff);
    double abs_diff = AbsPercentDiff(stage_percent_diff);
    result.total_stage_percent_diff_abs += abs_diff;
    result.max_stage_percent_diff_abs =
        std::max(result.max_stage_percent_diff_abs, abs_diff);
  }
  return result;
}

std::string SynthesizedDelayDiffToString(
    const SynthesizedDelayDiff& diff,
    std::optional<std::function<std::string(Node*)>> extra_info) {
  return SynthesizedDelayDiffToStringHeaderWithPercent(diff) + "\n" +
         CriticalPathToString(diff.critical_path, std::move(extra_info));
}

std::string SynthesizedStageDelayDiffToString(
    const SynthesizedDelayDiff& absolute_diff,
    const StagePercentDiff& percent_diff) {
  std::string result = SynthesizedDelayDiffToStringHeader(absolute_diff);
  absl::StrAppendFormat(
      &result, "; %.2f%% of synthesized pipeline vs. %.2f%% according to XLS.",
      percent_diff.synthesized_percent, percent_diff.xls_percent);
  return result + "\n" + CriticalPathToString(absolute_diff.critical_path);
}

std::string SynthesizedDelayDiffToStringHeaderWithPercent(
    const SynthesizedDelayDiff& diff) {
  return SynthesizedDelayDiffToStringHeader(diff) +
         absl::StrFormat("; %.2f%%", GetSynthesizedPercentOfXlsDelay(diff));
}

}  // namespace synthesis
}  // namespace xls

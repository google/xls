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

#ifndef XLS_FDO_SYNTHESIZED_DELAY_DIFF_UTILS_H_
#define XLS_FDO_SYNTHESIZED_DELAY_DIFF_UTILS_H_

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace synthesis {

// A difference in estimated delay between the XLS delay model and what a
// synthesis tool reports.
struct SynthesizedDelayDiff {
  std::vector<CriticalPathEntry> critical_path;
  int64_t xls_delay_ps = 0;
  int64_t synthesized_delay_ps = 0;
};

// A difference in the percent of a total pipeline delay that a particular stage
// accounts for, between the XLS delay model and what a synthesis tool reports.
struct StagePercentDiff {
  double xls_percent = 0;
  double synthesized_percent = 0;
};

// Returns the absolute value of the difference of XLS and synthesized percents
// of a pipeline accounted for by a given stage.
inline double AbsPercentDiff(const StagePercentDiff& diff) {
  return std::abs(diff.synthesized_percent - diff.xls_percent);
}

// A per-stage breakdown of the difference in estimated delay between the XLS
// delay model and what a synthesis tool reports.
struct SynthesizedDelayDiffByStage {
  SynthesizedDelayDiff total_diff;
  double max_stage_percent_diff_abs = 0;
  double total_stage_percent_diff_abs = 0;
  std::vector<SynthesizedDelayDiff> stage_diffs;
  std::vector<StagePercentDiff> stage_percent_diffs;
};

// Synthesizes `f` and returns the difference in reported delay, compared to
// what the XLS delay model predicts (according to `critical_path`). The
// `critical_path` can be obtained by calling `AnalyzeCriticalPath` first.
absl::StatusOr<SynthesizedDelayDiff> SynthesizeAndGetDelayDiff(
    FunctionBase* f, std::vector<CriticalPathEntry> critical_path,
    synthesis::Synthesizer* synthesizer);

// Creates a staged delay diff for all the stages in `f`, populating the XLS
// delay values and critical paths using `delay_estimator`. If `synthesizer` is
// non-null, this function also synthesizes each stage and populates the
// synthesized delays. This function does not currently populate the
// `critical_path` of the `total_diff`.
absl::StatusOr<SynthesizedDelayDiffByStage> CreateDelayDiffByStage(
    FunctionBase* f, const PipelineSchedule& schedule,
    const DelayEstimator& delay_estimator, synthesis::Synthesizer* synthesizer);

// Converts the given diff to a human-readable format that is an expanded form
// of what `CriticalPathToString` would return.
std::string SynthesizedDelayDiffToString(
    const SynthesizedDelayDiff& diff,
    std::optional<std::function<std::string(Node*)>> extra_info = std::nullopt);

// Like `SynthesizedDelayDiffToString(stage_diff)`, but includes the percent
// of the pipeline delay that is in the given stage, based on `overall_diff`.
std::string SynthesizedStageDelayDiffToString(
    const SynthesizedDelayDiff& absolute_diff,
    const StagePercentDiff& percent_diff);

// Returns just the synthesis-to-delay model overall comparison header that
// would be at the top of the `SynthesizedDelayDiffToString(diff)` output.
// Example: "1262ps as synthesized (-1035ps); 54.94%".
std::string SynthesizedDelayDiffToStringHeaderWithPercent(
    const SynthesizedDelayDiff& diff);

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_SYNTHESIZED_DELAY_DIFF_UTILS_H_

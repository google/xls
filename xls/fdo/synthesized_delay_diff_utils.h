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
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
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

// A per-stage breakdown of the difference in estimated delay between the XLS
// delay model and what a synthesis tool reports.
struct SynthesizedDelayDiffByStage {
  SynthesizedDelayDiff total_diff;
  std::vector<SynthesizedDelayDiff> stage_diffs;
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
std::string SynthesizedDelayDiffToString(const SynthesizedDelayDiff& diff);

// Like `SynthesizedDelayDiffToString(stage_diff)`, but includes the percent
// of the pipeline delay that is in the given stage, based on `overall_diff`.
std::string SynthesizedStageDelayDiffToString(
    const SynthesizedDelayDiff& stage_diff,
    const SynthesizedDelayDiff& overall_diff);

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_SYNTHESIZED_DELAY_DIFF_UTILS_H_

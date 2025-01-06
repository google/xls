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

#ifndef XLS_FDO_ITERATIVE_SDC_SCHEDULER_H_
#define XLS_FDO_ITERATIVE_SDC_SCHEDULER_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/fdo/delay_manager.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/sdc_scheduler.h"

namespace xls {

class IterativeSDCSchedulingModel : public SDCSchedulingModel {
  using DelayMap = absl::flat_hash_map<Node*, int64_t>;

 public:
  // Delay map is no longer needed as the delay calculation is completely
  // handled by the delay manager.
  IterativeSDCSchedulingModel(FunctionBase* func,
                              absl::flat_hash_set<Node*> dead_after_synthesis,
                              const DelayManager& delay_manager)
      : SDCSchedulingModel(func, std::move(dead_after_synthesis), DelayMap()),
        delay_manager_(delay_manager) {}

  // Overrides the original timing constraints builder. This method directly
  // call delay manager to extract the paths longer than the given clock period
  // instead of recalculating them.
  absl::Status AddTimingConstraints(int64_t clock_period_ps);

 private:
  const DelayManager& delay_manager_;
};

struct IterativeSDCSchedulingOptions {
  const synthesis::Synthesizer* synthesizer;
  int64_t iteration_number = 1;
  int64_t delay_driven_path_number = 0;
  int64_t fanout_driven_path_number = 0;
  float stochastic_ratio = 1.0;
  PathEvaluateStrategy path_evaluate_strategy = PathEvaluateStrategy::WINDOW;
};

// Runs iterative SDC scheduling. Compared to the original SDC, the iterative
// SDC scheduler will refine the delay estimations through low-level feedbacks,
// e.g., from OpenROAD, and improve the scheduling results iteratively.
absl::StatusOr<ScheduleCycleMap> ScheduleByIterativeSDC(
    FunctionBase* f, std::optional<int64_t> pipeline_stages,
    int64_t clock_period_ps, DelayManager& delay_manager,
    absl::Span<const SchedulingConstraint> constraints,
    const IterativeSDCSchedulingOptions& options,
    SchedulingFailureBehavior failure_behavior = SchedulingFailureBehavior{});

}  // namespace xls

#endif  // XLS_FDO_ITERATIVE_SDC_SCHEDULER_H_

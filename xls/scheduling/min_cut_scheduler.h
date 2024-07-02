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

#ifndef XLS_SCHEDULING_MIN_CUT_SCHEDULER_H_
#define XLS_SCHEDULING_MIN_CUT_SCHEDULER_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// Schedules the given function into a pipeline with the given clock
// period. Attempts to split nodes into stages such that the total number of
// flops in the pipeline stages is minimized without violating the target clock
// period.
absl::StatusOr<ScheduleCycleMap> MinCutScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints);

// Returns the list of ordering of cycles (pipeline stages) in which to compute
// min cut of the graph. Each min cut of the graph computes which XLS node
// values are in registers after a particular stage in the pipeline schedule. A
// min-cut must be computed for each stage in the schedule to determine the set
// of pipeline registers for the entire pipeline. The ordering of stages for
// which the min-cut is performed (e.g., stage 0 then 1, vs stage 1 then 0) can
// affect the total number of registers in the pipeline so multiple orderings
// are tried. This function returns this set of orderings.  Exposed for testing.
std::vector<std::vector<int64_t>> GetMinCutCycleOrders(int64_t length);

}  // namespace xls

#endif  // XLS_SCHEDULING_MIN_CUT_SCHEDULER_H_

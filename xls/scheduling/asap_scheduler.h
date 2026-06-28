// Copyright 2026 The XLS Authors
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

#ifndef XLS_SCHEDULING_ASAP_SCHEDULER_H_
#define XLS_SCHEDULING_ASAP_SCHEDULER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduler.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

class ASAPScheduler : public Scheduler {
 public:
  ASAPScheduler(const ScheduleGraph& graph, DelayEstimator& delay_estimator)
      : ASAPScheduler("ASAPScheduler", graph, delay_estimator) {}
  ~ASAPScheduler() override = default;

  absl::Status AddConstraints(
      absl::Span<const SchedulingConstraint> constraints) override {
    constraints_.insert(constraints_.end(), constraints.begin(),
                        constraints.end());
    return absl::OkStatus();
  }

  absl::StatusOr<ScheduleCycleMap> Schedule(
      std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
      SchedulingFailureBehavior failure_behavior,
      std::optional<int64_t> worst_case_throughput = std::nullopt) override;

  // An alternate interface that returns the full ASAP/ALAP bounds; mostly used
  // for other schedulers to build on top of this.
  absl::StatusOr<sched::ScheduleBounds> ComputeBounds(
      std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
      std::optional<int64_t> worst_case_throughput,
      bool get_helpful_error = true,
      int64_t max_upper_bound = sched::ScheduleBounds::kDefaultMaxUpperBound);

  const ScheduleGraph& graph() const { return graph_; }
  DelayEstimator& delay_estimator() const { return delay_estimator_; }
  absl::Span<const SchedulingConstraint> constraints() const {
    return constraints_;
  }

 protected:
  ASAPScheduler(std::string name, const ScheduleGraph& graph,
                DelayEstimator& delay_estimator)
      : Scheduler(std::move(name)),
        graph_(graph),
        delay_estimator_(delay_estimator) {}

  absl::Status GenerateHelpfulError(
      absl::Status&& orig_status, std::optional<int64_t> pipeline_stages,
      int64_t clock_period_ps, std::optional<int64_t> worst_case_throughput);

  // Helper to tighten bounds using the ASAP/ALAP bounds.
  // Exposed as `protected` so the random scheduler can build on top of this.
  static absl::Status TightenBounds(sched::ScheduleBounds& bounds,
                                    FunctionBase* f,
                                    std::optional<int64_t> schedule_length);
  const ScheduleGraph& graph_;
  DelayEstimator& delay_estimator_;
  std::vector<SchedulingConstraint> constraints_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_ASAP_SCHEDULER_H_

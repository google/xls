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

#ifndef XLS_SCHEDULING_RANDOM_SCHEDULER_H_
#define XLS_SCHEDULING_RANDOM_SCHEDULER_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/node.h"
#include "xls/scheduling/asap_scheduler.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduler.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

class RandomScheduler : public Scheduler {
 public:
  explicit RandomScheduler(const ScheduleGraph& graph,
                           DelayEstimator& delay_estimator, absl::BitGen bitgen)
      : Scheduler("RandomScheduler"),
        asap_(graph, delay_estimator),
        bitgen_(std::move(bitgen)) {}

  absl::Status AddConstraints(
      absl::Span<const SchedulingConstraint> constraints) override {
    return asap_.AddConstraints(constraints);
  }

  absl::StatusOr<ScheduleCycleMap> Schedule(
      std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
      SchedulingFailureBehavior failure_behavior,
      std::optional<int64_t> worst_case_throughput = std::nullopt) override;

 protected:
  // Helpers for testing (to allow one to suppress randomness)
  virtual absl::StatusOr<std::vector<Node*>> ShuffleNodes();
  virtual absl::StatusOr<int64_t> GetRandomCycle(Node* node, int64_t low,
                                                 int64_t high);

 private:
  // Helper to expose some internal ASAPScheduler methods for Random Scheduler
  // use.
  class ASAPSchedulerWrapper : public ASAPScheduler {
   public:
    ASAPSchedulerWrapper(const ScheduleGraph& graph,
                         DelayEstimator& delay_estimator)
        : ASAPScheduler("ASAPSchedulerWrapper", graph, delay_estimator) {}
    ~ASAPSchedulerWrapper() override = default;
    using ASAPScheduler::ComputeBounds;
    using ASAPScheduler::TightenBounds;
  };
  ASAPSchedulerWrapper asap_;
  absl::BitGen bitgen_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_RANDOM_SCHEDULER_H_

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

#ifndef XLS_SCHEDULING_SCHEDULER_H_
#define XLS_SCHEDULING_SCHEDULER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// An abstract base class for a scheduler.
class Scheduler {
 public:
  virtual ~Scheduler() = default;
  explicit Scheduler(std::string name) : name_(name) {}

  std::string_view name() const { return name_; }

  virtual absl::Status AddConstraints(
      absl::Span<const SchedulingConstraint> constraints) = 0;

  // Schedule the graph using the given constraints and this schedulers
  // strategy. Most schedulers try to either find or at least approximate the
  // solution to an optimization problem (eg minimize registers etc) but this
  // function doesn't actually require anything other than the schedule conforms
  // to the constraints passed both to it and the graph itself.
  //
  // If `pipeline_stages` is not specified, the solver will use the smallest
  // feasible value.
  //
  // If the problem is infeasible, `failure_behavior` configures what will be
  // done. If configured to do so, the scheduler will reformulate the problem
  // with slack variables and give actionable feedback on how to update the
  // design to be feasible to schedule.
  //
  // TODO(allight): Currently the tests for the scheduler expect that error
  // messages fit some pretty specific patterns which match what SDC does and
  // are not documented anywhere. We probably want to rationalize that all at
  // some point. For now we just implement some of the expected values for ASAP.
  // As all of these are failure paths even if a scheduler does not match the
  // messages everything will still mostly work.
  virtual absl::StatusOr<ScheduleCycleMap> Schedule(
      std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
      SchedulingFailureBehavior failure_behavior,
      std::optional<int64_t> worst_case_throughput = std::nullopt) = 0;

 private:
  std::string name_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULER_H_

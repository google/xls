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

#ifndef XLS_SCHEDULING_SCHEDULING_OPTIONS_H_
#define XLS_SCHEDULING_SCHEDULING_OPTIONS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/proc.h"

namespace xls {

// The strategy to use when scheduling pipelines.
enum class SchedulingStrategy {
  // Schedule all nodes a early as possible while satisfying dependency and
  // timing constraints.
  ASAP,

  // Approximately minimize the number of pipeline registers when scheduling
  // using a min-cut based algorithm.
  MIN_CUT,

  // Exactly minimize the number of pipeline registers when scheduling by
  // solving a system of difference constraints.
  SDC,

  // Create a random but sound schedule. This is useful for testing.
  RANDOM,
};

enum class IODirection { kReceive, kSend };

// This represents a constraint saying that interactions on the given
// `source_channel` of the type specified by the given `source_direction`
// must occur between `minimum_latency` and `maximum_latency` (inclusive) cycles
// before interactions on the given `target_channel` of the type specified by
// the given `target_direction`.
class IOConstraint {
 public:
  IOConstraint(std::string_view source_channel, IODirection source_direction,
               std::string_view target_channel, IODirection target_direction,
               int64_t minimum_latency, int64_t maximum_latency)
      : source_channel_(source_channel),
        source_direction_(source_direction),
        target_channel_(target_channel),
        target_direction_(target_direction),
        minimum_latency_(minimum_latency),
        maximum_latency_(maximum_latency) {}

  std::string SourceChannel() const { return source_channel_; }

  IODirection SourceDirection() const { return source_direction_; }

  std::string TargetChannel() const { return target_channel_; }

  IODirection TargetDirection() const { return target_direction_; }

  int64_t MinimumLatency() const { return minimum_latency_; }

  int64_t MaximumLatency() const { return maximum_latency_; }

 private:
  std::string source_channel_;
  IODirection source_direction_;
  std::string target_channel_;
  IODirection target_direction_;
  int64_t minimum_latency_;
  int64_t maximum_latency_;
};

// When this is present, receives will be scheduled in the first cycle and sends
// will be scheduled in the last cycle.
class RecvsFirstSendsLastConstraint {
 public:
  RecvsFirstSendsLastConstraint() {}
};

using SchedulingConstraint =
    std::variant<IOConstraint, RecvsFirstSendsLastConstraint>;

// Options to use when generating a pipeline schedule. At least a clock period
// or a pipeline length (or both) must be specified. See
// https://google.github.io/xls/scheduling/ for details on these options.
class SchedulingOptions {
 public:
  explicit SchedulingOptions(
      SchedulingStrategy strategy = SchedulingStrategy::SDC)
      : strategy_(strategy) {}

  // Returns the scheduling strategy.
  SchedulingStrategy strategy() const { return strategy_; }

  // Sets/gets the target clock period in picoseconds.
  SchedulingOptions& clock_period_ps(int64_t value) {
    clock_period_ps_ = value;
    return *this;
  }
  std::optional<int64_t> clock_period_ps() const { return clock_period_ps_; }

  // Sets/gets the target number of stages in the pipeline.
  SchedulingOptions& pipeline_stages(int64_t value) {
    pipeline_stages_ = value;
    return *this;
  }
  std::optional<int64_t> pipeline_stages() const { return pipeline_stages_; }

  // Sets/gets the percentage of clock period to set aside as a margin to ensure
  // timing is met. Effectively, this lowers the clock period by this percentage
  // amount for the purposes of scheduling.
  SchedulingOptions& clock_margin_percent(int64_t value) {
    clock_margin_percent_ = value;
    return *this;
  }
  std::optional<int64_t> clock_margin_percent() const {
    return clock_margin_percent_;
  }

  // Sets/gets the percentage of the estimated minimum period to relax so that
  // the scheduler may have more options to find an area-efficient
  // schedule without impacting timing.
  SchedulingOptions& period_relaxation_percent(int64_t value) {
    period_relaxation_percent_ = value;
    return *this;
  }
  std::optional<int64_t> period_relaxation_percent() const {
    return period_relaxation_percent_;
  }

  // Sets/gets the additional delay added to each receive node.
  //
  // TODO(tedhong): 2022-02-11, Update so that this sets/gets the
  // additional delay added to each input path.
  SchedulingOptions& additional_input_delay_ps(int64_t value) {
    additional_input_delay_ps_ = value;
    return *this;
  }
  std::optional<int64_t> additional_input_delay_ps() const {
    return additional_input_delay_ps_;
  }

  // Add a constraint to the set of scheduling constraints.
  SchedulingOptions& add_constraint(const SchedulingConstraint& constraint) {
    constraints_.push_back(constraint);
    return *this;
  }
  absl::Span<const SchedulingConstraint> constraints() const {
    return constraints_;
  }

  // The random seed, which is only used if the scheduler is `RANDOM`.
  SchedulingOptions& seed(int32_t value) {
    seed_ = value;
    return *this;
  }
  std::optional<int32_t> seed() const { return seed_; }

 private:
  SchedulingStrategy strategy_;
  std::optional<int64_t> clock_period_ps_;
  std::optional<int64_t> pipeline_stages_;
  std::optional<int64_t> clock_margin_percent_;
  std::optional<int64_t> period_relaxation_percent_;
  std::optional<int64_t> additional_input_delay_ps_;
  std::vector<SchedulingConstraint> constraints_;
  std::optional<int32_t> seed_;
};

// A map from node to cycle as a bare-bones representation of a schedule.
using ScheduleCycleMap = absl::flat_hash_map<Node*, int64_t>;

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_OPTIONS_H_

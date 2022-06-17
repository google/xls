// Copyright 2020 The XLS Authors
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

#ifndef XLS_SCHEDULING_PIPELINE_SCHEDULE_H_
#define XLS_SCHEDULING_PIPELINE_SCHEDULE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

namespace xls {

// The strategy to use when scheduling pipelines.
enum class SchedulingStrategy {
  // Schedule all nodes a early as possible while satisfying dependency and
  // timing constraints.
  ASAP,

  // Minimize the number of pipeline registers when scheduling.
  MINIMIZE_REGISTERS,

  // Minimize the number of pipeline registers when scheduling using SDC-method
  MINIMIZE_REGISTERS_SDC,
};

// Returns the list of ordering of cycles (pipeline stages) in which to compute
// min cut of the graph. Each min cut of the graph computes which XLS node
// values are in registers after a particular stage in the pipeline schedule. A
// min-cut must be computed for each stage in the schedule to determine the set
// of pipeline registers for the entire pipeline. The ordering of stages for
// which the min-cut is performed (e.g., stage 0 then 1, vs stage 1 then 0) can
// affect the total number of registers in the pipeline so multiple orderings
// are tried. This function returns this set of orderings.  Exposed for testing.
std::vector<std::vector<int64_t>> GetMinCutCycleOrders(int64_t length);

enum class IODirection { kReceive, kSend };

// This represents a constraint saying that interactions on the given
// `source_channel` of the type specified by the given `source_direction`
// must occur between `minimum_latency` and `maximum_latency` (inclusive) cycles
// before interactions on the given `target_channel` of the type specified by
// the given `target_direction`.
class SchedulingConstraint {
 public:
  SchedulingConstraint(absl::string_view source_channel,
                       IODirection source_direction,
                       absl::string_view target_channel,
                       IODirection target_direction, int64_t minimum_latency,
                       int64_t maximum_latency)
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

// Options to use when generating a pipeline schedule. At least a clock period
// or a pipeline length (or both) must be specified. See
// https://google.github.io/xls/scheduling/ for details on these options.
class SchedulingOptions {
 public:
  explicit SchedulingOptions(
      SchedulingStrategy strategy = SchedulingStrategy::MINIMIZE_REGISTERS_SDC)
      : strategy_(strategy) {}

  // Returns the scheduling strategy.
  SchedulingStrategy strategy() const { return strategy_; }

  // Sets/gets the target clock period in picoseconds.
  SchedulingOptions& clock_period_ps(int64_t value) {
    clock_period_ps_ = value;
    return *this;
  }
  absl::optional<int64_t> clock_period_ps() const { return clock_period_ps_; }

  // Sets/gets the target number of stages in the pipeline.
  SchedulingOptions& pipeline_stages(int64_t value) {
    pipeline_stages_ = value;
    return *this;
  }
  absl::optional<int64_t> pipeline_stages() const { return pipeline_stages_; }

  // Sets/gets the percentage of clock period to set aside as a margin to ensure
  // timing is met. Effectively, this lowers the clock period by this percentage
  // amount for the purposes of scheduling.
  SchedulingOptions& clock_margin_percent(int64_t value) {
    clock_margin_percent_ = value;
    return *this;
  }
  absl::optional<int64_t> clock_margin_percent() const {
    return clock_margin_percent_;
  }

  // Sets/gets the percentage of the estimated minimum period to relax so that
  // the scheduler may have more options to find an area-efficient
  // schedule without impacting timing.
  SchedulingOptions& period_relaxation_percent(int64_t value) {
    period_relaxation_percent_ = value;
    return *this;
  }
  absl::optional<int64_t> period_relaxation_percent() const {
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
  absl::optional<int64_t> additional_input_delay_ps() const {
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

 private:
  SchedulingStrategy strategy_;
  absl::optional<int64_t> clock_period_ps_;
  absl::optional<int64_t> pipeline_stages_;
  absl::optional<int64_t> clock_margin_percent_;
  absl::optional<int64_t> period_relaxation_percent_;
  absl::optional<int64_t> additional_input_delay_ps_;
  std::vector<SchedulingConstraint> constraints_;
};

// A map from node to cycle as a bare-bones representation of a schedule.
using ScheduleCycleMap = absl::flat_hash_map<Node*, int64_t>;

// Abstraction describing the binding of Nodes to cycles.
class PipelineSchedule {
 public:
  // Produces a feed-forward pipeline schedule using the given delay model and
  // scheduling options.
  static absl::StatusOr<PipelineSchedule> Run(
      FunctionBase* f, const DelayEstimator& delay_estimator,
      const SchedulingOptions& options);

  // Reconstructs a PipelineSchedule object from a proto representation.
  static absl::StatusOr<PipelineSchedule> FromProto(
      Function* function, const PipelineScheduleProto& proto);

  // Constructs a schedule for the given function with the given cycle map. If
  // length is not given, then the length equal to the largest cycle in cycle
  // map minus one.
  PipelineSchedule(FunctionBase* function_base, ScheduleCycleMap cycle_map,
                   absl::optional<int64_t> length = absl::nullopt);

  FunctionBase* function_base() const { return function_base_; }

  // Returns whether the given node is contained in this schedule.
  bool IsScheduled(Node* node) const { return cycle_map_.contains(node); }

  // Returns the cycle in which the node is placed. Dies if node has not
  // been placed in this schedule.
  int64_t cycle(const Node* node) const { return cycle_map_.at(node); }

  // Returns the nodes scheduled in the given cycle. The node order is
  // guaranteed to be topological.
  absl::Span<Node* const> nodes_in_cycle(int64_t cycle) const;

  std::string ToString() const;

  // Computes and returns the set of Nodes which are live out of the given
  // cycle. A node is live out of cycle N if it is scheduled at or before cycle
  // N and has users after cycle N.
  std::vector<Node*> GetLiveOutOfCycle(int64_t c) const;

  // Returns the number of stages in the pipeline. Use 'length' instead of
  // 'size' as 'size' is ambiguous in this context (number of resources? number
  // of nodes? number of cycles?). Note that codegen may add flops to the input
  // or output of the pipeline so this value may not be the same as the latency
  // of the pipeline.
  int64_t length() const { return cycle_to_nodes_.size(); }

  // Verifies various invariants of the schedule (each node scheduled exactly
  // once, node not scheduled before operands, etc.).
  absl::Status Verify() const;

  // Verifies that no path of nodes scheduled in the same cycle exceeds the
  // given clock period.
  absl::Status VerifyTiming(int64_t clock_period_ps,
                            const DelayEstimator& delay_estimator) const;

  // Returns a protobuf holding this object's scheduling info.
  PipelineScheduleProto ToProto() const;

  // Returns the number of internal registers in this schedule.
  int64_t CountFinalInteriorPipelineRegisters() const;

 private:
  FunctionBase* function_base_;

  // Map from node to the cycle in which it is scheduled.
  ScheduleCycleMap cycle_map_;

  // The nodes scheduled each cycle.
  std::vector<std::vector<Node*>> cycle_to_nodes_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_PIPELINE_SCHEDULE_H_

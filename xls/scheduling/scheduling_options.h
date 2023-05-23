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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"

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

// Force the given node into the given cycle. This is used for incremental
// scheduling in the scheduling pass pipeline, and is not currently exposed to
// users through codegen_main.
class NodeInCycleConstraint {
 public:
  NodeInCycleConstraint(Node* node, int64_t cycle)
      : node_(node), cycle_(cycle) {}

  Node* GetNode() const { return node_; }
  int64_t GetCycle() const { return cycle_; }

 private:
  Node* node_;
  int64_t cycle_;
};

// Force the given node to be less than or less than or equal to another node.
// The constraint will be `a - b ≤ max_difference`, so if you want to express
// `a ≤ b`, set max_difference to 0, and if you want to express `a < b`, set
// max_difference to -1.
class DifferenceConstraint {
 public:
  DifferenceConstraint(Node* a, Node* b, int64_t max_difference)
      : a_(a), b_(b), max_difference_(max_difference) {}

  Node* GetA() const { return a_; }
  Node* GetB() const { return b_; }
  int64_t GetMaxDifference() const { return max_difference_; }

 private:
  Node* a_;
  Node* b_;
  int64_t max_difference_;
};

// When this is present, receives will be scheduled in the first cycle and sends
// will be scheduled in the last cycle.
class RecvsFirstSendsLastConstraint {
 public:
  RecvsFirstSendsLastConstraint() = default;
};

// When this is present, state backedges will be forced to span over a single
// cycle. Not providing this is useful for implementing II > 1, but otherwise
// this should almost always be provided.
class BackedgeConstraint {
 public:
  BackedgeConstraint() = default;
};

using SchedulingConstraint =
    std::variant<IOConstraint, NodeInCycleConstraint, DifferenceConstraint,
                 RecvsFirstSendsLastConstraint, BackedgeConstraint>;

// When multiple of the same channel operations happen on the same channel,
// scheduling legalizes them through a combination of:
//  1. Requiring proven properties of the channel operations.
//  2. Runtime checks (assertions) that properties of the channel are true.
//  3. Arbitrary selection of priority between operations.
//
// Note that this does not apply to e.g. a send and receive on an internal
// SendReceive channel. This only applies when multiples of the same channel
// operation are being performed on the same channel.
enum class MultipleChannelOpsLegalizationStrictness {
  // Requires that channel operations be formally proven to be mutually
  // exclusive by Z3.
  kProvenMutuallyExclusive,
  // Requires that channel operations be mutually exclusive- enforced during
  // simulation via assertions.
  kRuntimeMutuallyExclusive,
  // For each proc, requires a total order on all operations on a channel. Note:
  // operations from different procs will not be ordered with respect to each
  // other.
  kTotalOrder,
  // Requires that a total order exists on every subset of channel operations
  // that fires at runtime. Adds assertions.
  kRuntimeOrdered,
  // For each proc, an arbitrary (respecting existing token relationships)
  // static priority is chosen for multiple channel operations. Operations
  // coming from different procs must be mutually exclusive (enforced via
  // assertions).
  kArbitraryStaticOrder,
};

inline bool AbslParseFlag(std::string_view text,
                          MultipleChannelOpsLegalizationStrictness* out,
                          std::string* error) {
  if (text == "proven_mutually_exclusive") {
    *out = MultipleChannelOpsLegalizationStrictness::kProvenMutuallyExclusive;
    return true;
  }
  if (text == "runtime_mutually_exclusive") {
    *out = MultipleChannelOpsLegalizationStrictness::kRuntimeMutuallyExclusive;
    return true;
  }
  if (text == "total_order") {
    *out = MultipleChannelOpsLegalizationStrictness::kTotalOrder;
    return true;
  }
  if (text == "runtime_ordered") {
    *out = MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered;
    return true;
  }
  if (text == "arbitrary_static_order") {
    *out = MultipleChannelOpsLegalizationStrictness::kArbitraryStaticOrder;
    return true;
  }
  *error = absl::StrFormat("Unrecognized strictness %s.", text);
  return false;
}
inline std::string AbslUnparseFlag(
    MultipleChannelOpsLegalizationStrictness in) {
  if (in ==
      MultipleChannelOpsLegalizationStrictness::kProvenMutuallyExclusive) {
    return "proven_mutually_exclusive";
  }
  if (in ==
      MultipleChannelOpsLegalizationStrictness::kRuntimeMutuallyExclusive) {
    return "runtime_mutually_exclusive";
  }
  if (in == MultipleChannelOpsLegalizationStrictness::kTotalOrder) {
    return "total_order";
  }
  if (in == MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered) {
    return "runtime_ordered";
  }
  if (in == MultipleChannelOpsLegalizationStrictness::kArbitraryStaticOrder) {
    return "arbitrary_static_order";
  }
  return "unknown";
}

// Options to use when generating a pipeline schedule. At least a clock period
// or a pipeline length (or both) must be specified. See
// https://google.github.io/xls/scheduling/ for details on these options.
class SchedulingOptions {
 public:
  explicit SchedulingOptions(
      SchedulingStrategy strategy = SchedulingStrategy::SDC)
      : strategy_(strategy),
        constraints_({BackedgeConstraint()}),
        multiple_channel_ops_legalization_strictness_(
            MultipleChannelOpsLegalizationStrictness::
                kProvenMutuallyExclusive) {}

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
  SchedulingOptions& clear_constraints() {
    constraints_.clear();
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

  // The rlimit used for mutual exclusion analysis.
  SchedulingOptions& mutual_exclusion_z3_rlimit(int64_t value) {
    mutual_exclusion_z3_rlimit_ = value;
    return *this;
  }
  std::optional<int64_t> mutual_exclusion_z3_rlimit() const {
    return mutual_exclusion_z3_rlimit_;
  }

  // The strictness setting for multiple channel ops legalization.
  SchedulingOptions& multiple_channel_ops_legalization_strictness(
      MultipleChannelOpsLegalizationStrictness value) {
    multiple_channel_ops_legalization_strictness_ = value;
    return *this;
  }
  MultipleChannelOpsLegalizationStrictness
  multiple_channel_ops_legalization_strictness() const {
    return multiple_channel_ops_legalization_strictness_;
  }

 private:
  SchedulingStrategy strategy_;
  std::optional<int64_t> clock_period_ps_;
  std::optional<int64_t> pipeline_stages_;
  std::optional<int64_t> clock_margin_percent_;
  std::optional<int64_t> period_relaxation_percent_;
  std::optional<int64_t> additional_input_delay_ps_;
  std::vector<SchedulingConstraint> constraints_;
  std::optional<int32_t> seed_;
  std::optional<int64_t> mutual_exclusion_z3_rlimit_;
  MultipleChannelOpsLegalizationStrictness
      multiple_channel_ops_legalization_strictness_;
};

// A map from node to cycle as a bare-bones representation of a schedule.
using ScheduleCycleMap = absl::flat_hash_map<Node*, int64_t>;

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_OPTIONS_H_

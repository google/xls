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
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/node.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

// The strategy to use when scheduling pipelines.
enum class SchedulingStrategy : int8_t {
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

enum class PathEvaluateStrategy : int8_t {
  PATH,
  CONE,
  WINDOW,
};

enum class IODirection : int8_t { kReceive, kSend };

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

  friend bool operator==(const IOConstraint& lhs, const IOConstraint& rhs) {
    return lhs.source_channel_ == rhs.source_channel_ &&
           lhs.source_direction_ == rhs.source_direction_ &&
           lhs.target_channel_ == rhs.target_channel_ &&
           lhs.target_direction_ == rhs.target_direction_ &&
           lhs.minimum_latency_ == rhs.minimum_latency_ &&
           lhs.maximum_latency_ == rhs.maximum_latency_;
  }
  friend bool operator!=(const IOConstraint& lhs, const IOConstraint& rhs) {
    return !(lhs == rhs);
  }

  template <typename H>
  friend H AbslHashValue(H h, const IOConstraint& s) {
    return H::combine(std::move(h), s.source_channel_, s.source_direction_,
                      s.target_channel_, s.target_direction_,
                      s.minimum_latency_, s.maximum_latency_);
  }

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

// When this is present, state backedges will be forced to span over at most II
// cycles.
class BackedgeConstraint {
 public:
  BackedgeConstraint() = default;
};

// When this is present, whenever we have a receive with a dependency on a send,
// the receive will always be scheduled at least `MinimumLatency()` cycles
// later. Since codegen currently blocks all execution within a stage if it
// contains a blocked receive, having this present with minimum latency 1 more
// accurately represents the user's expressed dependencies.
class SendThenRecvConstraint {
 public:
  explicit SendThenRecvConstraint(int64_t minimum_latency)
      : minimum_latency_(minimum_latency) {}

  int64_t MinimumLatency() const { return minimum_latency_; }

 private:
  int64_t minimum_latency_;
};

using SchedulingConstraint =
    std::variant<IOConstraint, NodeInCycleConstraint, DifferenceConstraint,
                 RecvsFirstSendsLastConstraint, BackedgeConstraint,
                 SendThenRecvConstraint>;

// Options for what the scheduler should do if scheduling fails.
struct SchedulingFailureBehavior {
  static SchedulingFailureBehavior FromProto(
      const SchedulingFailureBehaviorProto& proto) {
    SchedulingFailureBehavior failure_behavior;
    failure_behavior.explain_infeasibility = proto.explain_infeasibility();
    if (proto.has_infeasible_per_state_backedge_slack_pool()) {
      failure_behavior.infeasible_per_state_backedge_slack_pool =
          proto.infeasible_per_state_backedge_slack_pool();
    }
    return failure_behavior;
  }
  SchedulingFailureBehaviorProto ToProto() const {
    SchedulingFailureBehaviorProto proto;
    proto.set_explain_infeasibility(explain_infeasibility);
    if (infeasible_per_state_backedge_slack_pool.has_value()) {
      proto.set_infeasible_per_state_backedge_slack_pool(
          *infeasible_per_state_backedge_slack_pool);
    }
    return proto;
  }

  // If scheduling fails, re-run scheduling with extra slack variables in an
  // attempt to explain why scheduling failed.
  bool explain_infeasibility = true;

  // If specified, the specified value must be > 0. Setting this configures how
  // the scheduling problem is reformulated in the case that it fails. If
  // specified, this value will cause the reformulated problem to include
  // per-state backedge slack variables, which increases the complexity. This
  // value scales the objective such that adding slack to the per-state backedge
  // is preferred up until total slack reaches the pool size, after which adding
  // slack to the shared backedge slack variable is preferred. Increasing this
  // value should give more specific information about how much slack each
  // failing backedge needs at the cost of less actionable and harder to
  // understand output.
  std::optional<double> infeasible_per_state_backedge_slack_pool;
};

// Options to use when generating a pipeline schedule. At least a clock period
// or a pipeline length (or both) must be specified. See
// https://google.github.io/xls/scheduling/ for details on these options.
class SchedulingOptions {
 public:
  explicit SchedulingOptions(
      SchedulingStrategy strategy = SchedulingStrategy::SDC)
      : strategy_(strategy),
        minimize_clock_on_failure_(true),
        constraints_({
            BackedgeConstraint(),
            SendThenRecvConstraint(/*minimum_latency=*/1),
        }),
        use_fdo_(false),
        fdo_iteration_number_(5),
        fdo_delay_driven_path_number_(1),
        fdo_fanout_driven_path_number_(0),
        fdo_refinement_stochastic_ratio_(1.0),
        fdo_path_evaluate_strategy_(PathEvaluateStrategy::WINDOW),
        fdo_synthesizer_name_("yosys"),
        schedule_all_procs_(false) {}

  // Returns the scheduling strategy.
  SchedulingStrategy strategy() const { return strategy_; }

  // Sets/gets the target delay model
  SchedulingOptions& delay_model(std::string& value) {
    delay_model_ = value;
    return *this;
  }
  std::optional<std::string> delay_model() const { return delay_model_; }

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

  // Sets/gets whether to report the fastest feasible clock if scheduling is
  // infeasible at the user's specified clock.
  SchedulingOptions& minimize_clock_on_failure(bool value) {
    minimize_clock_on_failure_ = value;
    return *this;
  }
  std::optional<bool> minimize_clock_on_failure() const {
    return minimize_clock_on_failure_;
  }

  // Sets/gets the worst-case throughput bound to use when scheduling; for
  // procs, controls the length of state backedges allowed in scheduling.
  SchedulingOptions& worst_case_throughput(int64_t value) {
    worst_case_throughput_ = value;
    return *this;
  }
  std::optional<int64_t> worst_case_throughput() const {
    return worst_case_throughput_;
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

  // Set fallback estimation for ffi calls used in absence of more information.
  SchedulingOptions& ffi_fallback_delay_ps(int64_t value) {
    ffi_fallback_delay_ps_ = value;
    return *this;
  }
  std::optional<int64_t> ffi_fallback_delay_ps() const {
    return ffi_fallback_delay_ps_;
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

  // The rlimit used for default next-value omission optimization.
  SchedulingOptions& default_next_value_z3_rlimit(int64_t value) {
    default_next_value_z3_rlimit_ = value;
    return *this;
  }
  std::optional<int64_t> default_next_value_z3_rlimit() const {
    return default_next_value_z3_rlimit_;
  }

  // Struct that configures what should be done when scheduling fails. The
  // scheduling problem can be reformulated to give actionable feedback on how
  // to get a feasible schedule.
  SchedulingOptions& failure_behavior(SchedulingFailureBehavior value) {
    failure_behavior_ = value;
    return *this;
  }
  SchedulingFailureBehavior failure_behavior() const {
    return failure_behavior_;
  }

  // Enable FDO
  SchedulingOptions& use_fdo(bool value) {
    use_fdo_ = value;
    return *this;
  }
  bool use_fdo() const { return use_fdo_; }

  // The number of FDO iterations during the pipeline scheduling.
  SchedulingOptions& fdo_iteration_number(int64_t value) {
    fdo_iteration_number_ = value;
    return *this;
  }
  int64_t fdo_iteration_number() const { return fdo_iteration_number_; }

  // The number of delay-driven subgraphs in each FDO iteration.
  SchedulingOptions& fdo_delay_driven_path_number(int64_t value) {
    fdo_delay_driven_path_number_ = value;
    return *this;
  }
  int64_t fdo_delay_driven_path_number() const {
    return fdo_delay_driven_path_number_;
  }

  // The number of fanout-driven subgraphs in each FDO iteration.
  SchedulingOptions& fdo_fanout_driven_path_number(int64_t value) {
    fdo_fanout_driven_path_number_ = value;
    return *this;
  }
  int64_t fdo_fanout_driven_path_number() const {
    return fdo_fanout_driven_path_number_;
  }

  // *path_number over refinement_stochastic_ratio paths are extracted and
  // *path_number paths are randomly selected from them for synthesis in each
  // FDO iteration.
  SchedulingOptions& fdo_refinement_stochastic_ratio(float value) {
    fdo_refinement_stochastic_ratio_ = value;
    return *this;
  }
  float fdo_refinement_stochastic_ratio() const {
    return fdo_refinement_stochastic_ratio_;
  }

  // Support window, cone, and path for now.
  SchedulingOptions& fdo_path_evaluate_strategy(std::string_view value) {
    if (value == "path") {
      fdo_path_evaluate_strategy_ = PathEvaluateStrategy::PATH;
    } else if (value == "cone") {
      fdo_path_evaluate_strategy_ = PathEvaluateStrategy::CONE;
    } else {
      CHECK_EQ(value, "window") << "Unknown path evaluate strategy: " << value;
      fdo_path_evaluate_strategy_ = PathEvaluateStrategy::WINDOW;
    }
    return *this;
  }
  PathEvaluateStrategy fdo_path_evaluate_strategy() const {
    return fdo_path_evaluate_strategy_;
  }

  // Only support yosys for now.
  SchedulingOptions& fdo_synthesizer_name(std::string_view value) {
    fdo_synthesizer_name_ = value;
    return *this;
  }
  std::string fdo_synthesizer_name() const { return fdo_synthesizer_name_; }

  // Yosys path
  SchedulingOptions& fdo_yosys_path(std::string_view value) {
    fdo_yosys_path_ = value;
    return *this;
  }
  std::string fdo_yosys_path() const { return fdo_yosys_path_; }

  // STA path
  SchedulingOptions& fdo_sta_path(std::string_view value) {
    fdo_sta_path_ = value;
    return *this;
  }
  std::string fdo_sta_path() const { return fdo_sta_path_; }

  // Path to synth library (Liberty file)
  SchedulingOptions& fdo_synthesis_libraries(std::string_view value) {
    fdo_synthesis_libraries_ = value;
    return *this;
  }
  std::string fdo_synthesis_libraries() const {
    return fdo_synthesis_libraries_;
  }

  SchedulingOptions& schedule_all_procs(bool value) {
    schedule_all_procs_ = value;
    return *this;
  }
  bool schedule_all_procs() const { return schedule_all_procs_; }

 private:
  SchedulingStrategy strategy_;
  std::optional<int64_t> clock_period_ps_;
  std::optional<std::string> delay_model_;
  std::optional<int64_t> pipeline_stages_;
  std::optional<int64_t> clock_margin_percent_;
  std::optional<int64_t> period_relaxation_percent_;
  bool minimize_clock_on_failure_;
  std::optional<int64_t> worst_case_throughput_;
  std::optional<int64_t> additional_input_delay_ps_;
  std::optional<int64_t> ffi_fallback_delay_ps_;
  std::vector<SchedulingConstraint> constraints_;
  std::optional<int32_t> seed_;
  std::optional<int64_t> mutual_exclusion_z3_rlimit_;
  std::optional<int64_t> default_next_value_z3_rlimit_;
  SchedulingFailureBehavior failure_behavior_;
  bool use_fdo_;
  int64_t fdo_iteration_number_;
  int64_t fdo_delay_driven_path_number_;
  int64_t fdo_fanout_driven_path_number_;
  float fdo_refinement_stochastic_ratio_;
  PathEvaluateStrategy fdo_path_evaluate_strategy_;
  std::string fdo_synthesizer_name_;
  std::string fdo_yosys_path_;
  std::string fdo_sta_path_;
  std::string fdo_synthesis_libraries_;
  bool schedule_all_procs_;
};

// A map from node to cycle as a bare-bones representation of a schedule.
using ScheduleCycleMap = absl::flat_hash_map<Node*, int64_t>;

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_OPTIONS_H_

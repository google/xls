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

#ifndef XLS_SCHEDULING_SDC_SCHEDULER_H_
#define XLS_SCHEDULING_SDC_SCHEDULER_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"
#include "ortools/math_opt/cpp/math_opt.h"

namespace xls {

// A class used to build linear programming (LP) model for SDC scheduling. This
// class uses the LP solver from OR tools for problem solving. It provides
// methods to add scheduling constraints, set objectives, and extract solving
// results.
class ModelBuilder {
  using DelayMap = absl::flat_hash_map<Node*, int64_t>;

  static constexpr double kInfinity = std::numeric_limits<double>::infinity();

 public:
  ModelBuilder(FunctionBase* func, int64_t pipeline_length,
               int64_t clock_period_ps, const DelayMap& delay_map,
               std::string_view model_name = "");

  absl::Status AddDefUseConstraints(Node* node, std::optional<Node*> user);
  absl::Status AddCausalConstraint(Node* node, std::optional<Node*> user);
  absl::Status AddLifetimeConstraint(Node* node, std::optional<Node*> user);
  absl::Status AddBackedgeConstraints(const BackedgeConstraint& constraint);
  absl::Status AddTimingConstraints();
  absl::Status AddSchedulingConstraint(const SchedulingConstraint& constraint);
  absl::Status AddIOConstraint(const IOConstraint& constraint);
  absl::Status AddNodeInCycleConstraint(
      const NodeInCycleConstraint& constraint);
  absl::Status AddDifferenceConstraint(const DifferenceConstraint& constraint);
  absl::Status AddRFSLConstraint(
      const RecvsFirstSendsLastConstraint& constraint);
  absl::Status AddSendThenRecvConstraint(
      const SendThenRecvConstraint& constraint);

  absl::Status SetObjective();

  absl::Status SetPipelineLength(int64_t pipeline_length);

  void AddPipelineLengthSlack();
  absl::Status RemovePipelineLengthSlack();

  absl::StatusOr<int64_t> ExtractPipelineLength(
      const operations_research::math_opt::VariableMap<double>& variable_values)
      const;

  absl::Status AddSlackVariables();

  operations_research::math_opt::Model& Build() { return model_; }
  const operations_research::math_opt::Model& Build() const { return model_; }

  absl::StatusOr<ScheduleCycleMap> ExtractResult(
      const operations_research::math_opt::VariableMap<double>& variable_values)
      const;

  absl::Status ExtractError(
      const operations_research::math_opt::VariableMap<double>& variable_values)
      const;

  absl::flat_hash_map<Node*, operations_research::math_opt::Variable>
  GetCycleVars() const {
    return cycle_var_;
  }

  absl::flat_hash_map<Node*, operations_research::math_opt::Variable>
  GetLifetimeVars() const {
    return lifetime_var_;
  }

  operations_research::math_opt::LinearConstraint DiffAtMostConstraint(
      Node* x, Node* y, int64_t limit, std::string_view name);

  operations_research::math_opt::LinearConstraint DiffLessThanConstraint(
      Node* x, Node* y, int64_t limit, std::string_view name);

  operations_research::math_opt::LinearConstraint DiffAtLeastConstraint(
      Node* x, Node* y, int64_t limit, std::string_view name);

  operations_research::math_opt::LinearConstraint DiffGreaterThanConstraint(
      Node* x, Node* y, int64_t limit, std::string_view name);

  operations_research::math_opt::LinearConstraint DiffEqualsConstraint(
      Node* x, Node* y, int64_t diff, std::string_view name);

 private:
  operations_research::math_opt::Variable AddUpperBoundSlack(
      operations_research::math_opt::LinearConstraint c,
      std::optional<operations_research::math_opt::Variable> slack =
          std::nullopt);

  absl::Status RemoveUpperBoundSlack(
      operations_research::math_opt::Variable v,
      operations_research::math_opt::LinearConstraint upper_bound_with_slack,
      operations_research::math_opt::Variable slack);

  operations_research::math_opt::Variable AddLowerBoundSlack(
      operations_research::math_opt::LinearConstraint c,
      std::optional<operations_research::math_opt::Variable> slack =
          std::nullopt);

  std::pair<operations_research::math_opt::Variable,
            operations_research::math_opt::LinearConstraint>
  AddUpperBoundSlack(operations_research::math_opt::Variable v,
                     std::optional<operations_research::math_opt::Variable>
                         slack = std::nullopt);

  std::pair<operations_research::math_opt::Variable,
            operations_research::math_opt::LinearConstraint>
  AddLowerBoundSlack(operations_research::math_opt::Variable v,
                     std::optional<operations_research::math_opt::Variable>
                         slack = std::nullopt);

  FunctionBase* func_;
  operations_research::math_opt::Model model_;
  int64_t pipeline_length_;
  int64_t clock_period_ps_;
  const DelayMap& delay_map_;

  // Node's cycle after scheduling
  absl::flat_hash_map<Node*, operations_research::math_opt::Variable>
      cycle_var_;

  // Return-last constraint, if present
  std::optional<operations_research::math_opt::LinearConstraint>
      return_last_constraint_;

  // Node's lifetime, from when it finishes executing until it is consumed by
  // the last user.
  absl::flat_hash_map<Node*, operations_research::math_opt::Variable>
      lifetime_var_;

  // A placeholder node to represent an artificial sink node on the
  // data-dependence graph.
  operations_research::math_opt::Variable cycle_at_sinknode_;

  // A cache of the delay constraints.
  absl::flat_hash_map<Node*, std::vector<Node*>> delay_constraints_;

  absl::flat_hash_map<std::pair<Node*, Node*>,
                      operations_research::math_opt::LinearConstraint>
      backedge_constraint_;
  absl::flat_hash_map<Node*, operations_research::math_opt::LinearConstraint>
      send_last_constraint_;

  struct ConstraintPair {
    operations_research::math_opt::LinearConstraint lower;
    operations_research::math_opt::LinearConstraint upper;
  };
  absl::flat_hash_map<IOConstraint, std::vector<ConstraintPair>>
      io_constraints_;

  std::optional<operations_research::math_opt::Variable> pipeline_length_slack_;
  absl::flat_hash_map<Node*, operations_research::math_opt::LinearConstraint>
      cycle_upper_bound_with_slack_;

  std::optional<operations_research::math_opt::Variable> backedge_slack_;

  struct SlackPair {
    operations_research::math_opt::Variable min;
    operations_research::math_opt::Variable max;
  };
  absl::flat_hash_map<IOConstraint, SlackPair> io_slack_;
};

// Schedule to minimize the total pipeline registers using SDC scheduling
// the constraint matrix is totally unimodular, this ILP problem can be solved
// by LP.
//
// With `check_feasibility = true`, the objective function will be constant, and
// the LP solver will merely attempt to show that the generated set of
// constraints is feasible, rather than find an register-optimal schedule.
//
// If `pipeline_stages` is not specified, the solver will use the smallest
// feasible value.
//
// References:
//   - Cong, Jason, and Zhiru Zhang. "An efficient and versatile scheduling
//   algorithm based on SDC formulation." 2006 43rd ACM/IEEE Design Automation
//   Conference. IEEE, 2006.
//   - Zhang, Zhiru, and Bin Liu. "SDC-based modulo scheduling for pipeline
//   synthesis." 2013 IEEE/ACM International Conference on Computer-Aided Design
//   (ICCAD). IEEE, 2013.
absl::StatusOr<ScheduleCycleMap> SDCScheduler(
    FunctionBase* f, std::optional<int64_t> pipeline_stages,
    int64_t clock_period_ps, const DelayEstimator& delay_estimator,
    absl::Span<const SchedulingConstraint> constraints,
    bool check_feasibility = false, bool explain_infeasibility = true);

}  // namespace xls

#endif  // XLS_SCHEDULING_SDC_SCHEDULER_H_

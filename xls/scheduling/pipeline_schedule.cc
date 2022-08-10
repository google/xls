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

#include "xls/scheduling/pipeline_schedule.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/binary_search.h"
#include "xls/data_structures/graph_contraction.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/scheduling/function_partition.h"
#include "xls/scheduling/schedule_bounds.h"
#include "ortools/linear_solver/linear_solver.h"

namespace xls {
namespace {

// Returns the largest cycle value to which any node is mapped in the given
// ScheduleCycleMap. This is the maximum value of any value element in the map.
int64_t MaximumCycle(const ScheduleCycleMap& cycle_map) {
  int64_t max_cycle = 0;
  for (const auto& pair : cycle_map) {
    max_cycle = std::max(max_cycle, pair.second);
  }
  return max_cycle;
}

// Splits the nodes at the boundary between 'cycle' and 'cycle + 1' by
// performing a minimum cost cut and tightens the bounds accordingly. Upon
// return no node in the function will have a range which spans both 'cycle' and
// 'cycle + 1'.
absl::Status SplitAfterCycle(FunctionBase* f, int64_t cycle,
                             const DelayEstimator& delay_estimator,
                             sched::ScheduleBounds* bounds) {
  XLS_VLOG(3) << "Splitting after cycle " << cycle;

  // The nodes which need to be partitioned are those which can be scheduled in
  // either 'cycle' or 'cycle + 1'.
  std::vector<Node*> partitionable_nodes;
  for (Node* node : f->nodes()) {
    if (bounds->lb(node) <= cycle && bounds->ub(node) >= cycle + 1) {
      partitionable_nodes.push_back(node);
    }
  }

  std::pair<std::vector<Node*>, std::vector<Node*>> partitions =
      sched::MinCostFunctionPartition(f, partitionable_nodes);

  // Tighten bounds based on the cut.
  for (Node* node : partitions.first) {
    XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, cycle));
  }
  for (Node* node : partitions.second) {
    XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(node, cycle + 1));
  }

  return absl::OkStatus();
}

// Returns the number of pipeline registers (flops) on the interior of the
// pipeline not counting the input and output flops (if any).
absl::StatusOr<int64_t> CountInteriorPipelineRegisters(
    FunctionBase* f, const sched::ScheduleBounds& bounds) {
  int64_t registers = 0;
  for (Node* node : f->nodes()) {
    XLS_RET_CHECK_EQ(bounds.lb(node), bounds.ub(node)) << absl::StrFormat(
        "%s [%d, %d]", node->GetName(), bounds.lb(node), bounds.ub(node));
    int64_t latest_use = bounds.lb(node);
    for (Node* user : node->users()) {
      latest_use = std::max(latest_use, bounds.lb(user));
    }
    registers +=
        node->GetType()->GetFlatBitCount() * (latest_use - bounds.lb(node));
  }
  return registers;
}

// Schedules the given function into a pipeline with the given clock
// period. Attempts to split nodes into stages such that the total number of
// flops in the pipeline stages is minimized without violating the target clock
// period.
absl::StatusOr<ScheduleCycleMap> ScheduleToMinimizeRegisters(
    FunctionBase* f, int64_t pipeline_stages,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds) {
  XLS_VLOG(3) << "ScheduleToMinimizeRegisters()";
  XLS_VLOG(3) << "  pipeline stages = " << pipeline_stages;
  XLS_VLOG_LINES(4, f->DumpIr());

  XLS_VLOG(4) << "Initial bounds:";
  XLS_VLOG_LINES(4, bounds->ToString());

  // Try a number of different orderings of cycle boundary at which the min-cut
  // is performed and keep the best one.
  int64_t best_register_count = std::numeric_limits<int64_t>::max();
  std::optional<sched::ScheduleBounds> best_bounds;
  for (const std::vector<int64_t>& cut_order :
       GetMinCutCycleOrders(pipeline_stages - 1)) {
    XLS_VLOG(3) << absl::StreamFormat("Trying cycle order: {%s}",
                                      absl::StrJoin(cut_order, ", "));
    sched::ScheduleBounds trial_bounds = *bounds;
    // Partition the nodes at each cycle boundary. For each iteration, this
    // splits the nodes into those which must be scheduled at or before the
    // cycle and those which must be scheduled after. Upon loop completion each
    // node will have a range of exactly one cycle.
    for (int64_t cycle : cut_order) {
      XLS_RETURN_IF_ERROR(
          SplitAfterCycle(f, cycle, delay_estimator, &trial_bounds));
      XLS_RETURN_IF_ERROR(trial_bounds.PropagateLowerBounds());
      XLS_RETURN_IF_ERROR(trial_bounds.PropagateUpperBounds());
    }
    XLS_ASSIGN_OR_RETURN(int64_t trial_register_count,
                         CountInteriorPipelineRegisters(f, trial_bounds));
    if (!best_bounds.has_value() ||
        best_register_count > trial_register_count) {
      best_bounds = std::move(trial_bounds);
      best_register_count = trial_register_count;
    }
  }
  *bounds = std::move(*best_bounds);

  ScheduleCycleMap cycle_map;
  for (Node* node : f->nodes()) {
    XLS_RET_CHECK_EQ(bounds->lb(node), bounds->ub(node)) << node->GetName();
    cycle_map[node] = bounds->lb(node);
  }
  return cycle_map;
}

using DelayMap = absl::flat_hash_map<Node*, int64_t>;

// A helper function to compute each node's delay by calling the delay estimator
// The result is used by `ComputeCriticalCombPathsUntilNextCycle`,
// `ComputeSingleSourceCCPUntilNextCycle`,
// `ScheduleToMinimizeRegistersSDC`
absl::StatusOr<DelayMap> ComputeNodeDelays(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  DelayMap result;
  for (Node* node : f->nodes()) {
    XLS_ASSIGN_OR_RETURN(result[node],
                         delay_estimator.GetOperationDelayInPs(node));
  }
  return result;
}

// Returns the minimal set of schedule constraints which ensure that no
// combinational path in the schedule exceeds `clock_period_ps`. The returned
// map has a (potentially empty) vector entry for each node in `f`. The map
// value (vector of nodes) for node `x` is the set of nodes which must be
// scheduled at least one cycle later than `x`. That is, if `return_value[x]` is
// `S` then:
//
//   cycle(i) + 1 >= cycle(x) for i \in S
//
// The set of constraints is a minimal set which guarantees that no
// combinational path violates the clock period timing. Specifically, `(a, b)`
// is in the set of returned constraints (ie., `return_value[a]` contains `b`)
// iff critical-path distance from `a` to `b` including the delay of `a` and `b`
// is greater than `critical_path_period`, but the critical-path distance of the
// path *not* including the delay of `b` is *less than* `critical_path_period`.
absl::flat_hash_map<Node*, std::vector<Node*>>
ComputeCombinationalDelayConstraints(FunctionBase* f, int64_t clock_period_ps,
                                     const DelayMap& delay_map) {
  absl::flat_hash_map<Node*, std::vector<Node*>> result;
  result.reserve(f->node_count());

  // Compute all-pairs longest distance between all nodes in `f`. The distance
  // from node `a` to node `b` is defined as the length of the longest delay
  // path from `a` to `b` which includes the delay of the path endpoints `a` and
  // `b`. The all-pairs distance is stored in the map of vectors `node_to_index`
  // where `node_to_index[y]` holds the critical-path distances from each node
  // `x` to `y`.
  absl::flat_hash_map<Node*, std::vector<int64_t>> distances_to_node;
  distances_to_node.reserve(f->node_count());

  // Compute a map from Node* to the interval [0, node_count). These map values
  // serve as indices into a flat vector of distances.
  absl::flat_hash_map<Node*, int32_t> node_to_index;
  std::vector<Node*> index_to_node(f->node_count());
  node_to_index.reserve(f->node_count());
  int32_t index = 0;
  for (Node* node : f->nodes()) {
    node_to_index[node] = index;
    index_to_node[index] = node;
    index++;

    // Initialize the constraint map entry to an empty vector.
    result[node];
  }

  for (Node* node : TopoSort(f)) {
    int64_t node_index = node_to_index.at(node);
    int64_t node_delay = delay_map.at(node);
    std::vector<int64_t> distances = std::vector<int64_t>(f->node_count(), -1);

    // Compute the critical-path distance from `a` to `node` for all nodes `a`
    // from the distances of `a` to each operand of `node`.
    for (int64_t operand_i = 0; operand_i < node->operand_count();
         ++operand_i) {
      Node* operand = node->operand(operand_i);
      const std::vector<int64_t>& distances_to_operand =
          distances_to_node.at(operand);
      for (int64_t i = 0; i < f->node_count(); ++i) {
        int64_t operand_distance = distances_to_operand[i];
        if (operand_distance != -1) {
          if (distances[i] < operand_distance + node_delay) {
            distances[i] = operand_distance + node_delay;
            // Only add a constraint if the delay of `node` results in the
            // length of the critical-path crossing the `clock_period_ps`
            // boundary.
            if (operand_distance <= clock_period_ps &&
                operand_distance + node_delay > clock_period_ps) {
              result[index_to_node[i]].push_back(node);
            }
          }
        }
      }
    }

    distances[node_index] = node_delay;
    distances_to_node[node] = std::move(distances);
  }

  if (XLS_VLOG_IS_ON(4)) {
    XLS_VLOG(4) << "All-pairs critical-path distances:";
    for (Node* target : TopoSort(f)) {
      XLS_VLOG(4) << absl::StrFormat("  distances to %s:", target->GetName());
      for (int64_t i = 0; i < f->node_count(); ++i) {
        Node* source = index_to_node[i];
        XLS_VLOG(4) << absl::StrFormat(
            "    %s -> %s : %s", source->GetName(), target->GetName(),
            distances_to_node[target][i] == -1
                ? "(none)"
                : absl::StrCat(distances_to_node[target][i]));
      }
    }
    XLS_VLOG(4) << absl::StrFormat("Constraints (clock period: %dps):",
                                   clock_period_ps);
    for (Node* node : TopoSort(f)) {
      XLS_VLOG(4) << absl::StrFormat(
          "  %s: [%s]", node->GetName(),
          absl::StrJoin(result.at(node), ", ", NodeFormatter));
    }
  }
  return result;
}

// Schedule to minimize the total pipeline registers using SDC scheduling
// the constraint matrix is totally unimodular, this ILP problem can be solved
// by LP.
//
// References:
//   - Cong, Jason, and Zhiru Zhang. "An efficient and versatile scheduling
//   algorithm based on SDC formulation." 2006 43rd ACM/IEEE Design Automation
//   Conference. IEEE, 2006.
//   - Zhang, Zhiru, and Bin Liu. "SDC-based modulo scheduling for pipeline
//   synthesis." 2013 IEEE/ACM International Conference on Computer-Aided Design
//   (ICCAD). IEEE, 2013.
absl::StatusOr<ScheduleCycleMap> ScheduleToMinimizeRegistersSDC(
    FunctionBase* f, int64_t pipeline_stages,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    int64_t clock_period_ps,
    absl::Span<const SchedulingConstraint> constraints) {
  XLS_VLOG(3) << "ScheduleToMinimizeRegistersSDC()";
  XLS_VLOG(3) << "  pipeline stages = " << pipeline_stages;
  XLS_VLOG_LINES(4, f->DumpIr());

  XLS_VLOG(4) << "Initial bounds:";
  XLS_VLOG_LINES(4, bounds->ToString());

  namespace or_tools = ::operations_research;

  std::unique_ptr<or_tools::MPSolver> solver(
      or_tools::MPSolver::CreateSolver("GLOP"));
  if (!solver) {
    return absl::UnavailableError("GLOP solver unavailable.");
  }

  const double infinity = solver->infinity();

  // Node's cycle after scheduling
  absl::flat_hash_map<Node*, or_tools::MPVariable*> cycle_var;

  // Node's lifetime, from when it finishes executing until it is consumed by
  // the last user.
  absl::flat_hash_map<Node*, or_tools::MPVariable*> lifetime_var;
  for (Node* node : f->nodes()) {
    cycle_var[node] =
        solver->MakeNumVar(bounds->lb(node), bounds->ub(node), node->GetName());
    lifetime_var[node] = solver->MakeNumVar(
        0.0, infinity, absl::StrFormat("lifetime_%s", node->GetName()));
  }

  // Map from channel name to set of nodes that send/receive on that channel.
  absl::flat_hash_map<std::string, std::vector<Node*>> channel_to_nodes;
  for (Node* node : f->nodes()) {
    if (node->Is<Receive>() || node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      channel_to_nodes[channel->name()].push_back(node);
    }
  }

  // Scheduling constraints
  for (const SchedulingConstraint& constraint : constraints) {
    // We use `channel_to_nodes[...]` instead of `channel_to_nodes.at(...)`
    // below because we don't want to error out if a constraint is specified
    // that affects a channel with no associated send/receives in this proc.
    for (Node* source : channel_to_nodes[constraint.SourceChannel()]) {
      for (Node* target : channel_to_nodes[constraint.TargetChannel()]) {
        auto node_matches_direction = [](Node* node, IODirection dir) -> bool {
          return (node->Is<Send>() && dir == IODirection::kSend) ||
                 (node->Is<Receive>() && dir == IODirection::kReceive);
        };
        if (!node_matches_direction(source, constraint.SourceDirection())) {
          continue;
        }
        if (!node_matches_direction(target, constraint.TargetDirection())) {
          continue;
        }
        if (source == target) {
          continue;
        }

        or_tools::MPVariable* source_var = cycle_var.at(source);
        or_tools::MPVariable* target_var = cycle_var.at(target);

        // The desired constraint `cycle[target] - cycle[source] >= min_latency`
        // becomes `-(cycle[target] - cycle[source]) <= -min_latency`
        // becomes `cycle[source] - cycle[target] <= -min_latency`
        or_tools::MPConstraint* min_io_constraint = solver->MakeRowConstraint(
            -infinity, -constraint.MinimumLatency(),
            absl::StrFormat("min_io_%s_%s", source->GetName(),
                            target->GetName()));
        min_io_constraint->SetCoefficient(source_var, 1);
        min_io_constraint->SetCoefficient(target_var, -1);

        // Constraint: `cycle[target] - cycle[source] <= max_latency`
        or_tools::MPConstraint* max_io_constraint = solver->MakeRowConstraint(
            -infinity, constraint.MaximumLatency(),
            absl::StrFormat("max_io_%s_%s", source->GetName(),
                            target->GetName()));
        max_io_constraint->SetCoefficient(target_var, 1);
        max_io_constraint->SetCoefficient(source_var, -1);

        XLS_VLOG(2) << "Setting IO constraint: "
                    << absl::StrFormat("%d ≤ cycle[%s] - cycle[%s] ≤ %d",
                                       constraint.MinimumLatency(),
                                       target->GetName(), source->GetName(),
                                       constraint.MaximumLatency());
      }
    }
  }

  // A dummy node to represent an artificial sink node on the data-dependence
  // graph.
  or_tools::MPVariable* cycle_at_sinknode =
      solver->MakeNumVar(-infinity, infinity, "cycle_at_sinknode");

  for (Node* node : f->nodes()) {
    or_tools::MPVariable* lifetime_at_node = lifetime_var[node];
    or_tools::MPVariable* cycle_at_node = cycle_var[node];

    auto add_du_chains_related_constraints =
        [&](absl::string_view user_str, or_tools::MPVariable* cycle_at_user) {
          // Constraint: cycle[node] - cycle[node_user] <= 0
          or_tools::MPConstraint* causal = solver->MakeRowConstraint(
              -infinity, 0.0,
              absl::StrFormat("causal_%s_%s", node->GetName(), user_str));
          causal->SetCoefficient(cycle_at_node, 1);
          causal->SetCoefficient(cycle_at_user, -1);

          XLS_VLOG(2) << "Setting causal constraint: "
                      << absl::StrFormat("cycle[%s] - cycle[%s] ≥ 0", user_str,
                                         node->GetName());

          // Constraint: cycle[node_user] - cycle[node] - lifetime[node] <= 0
          or_tools::MPConstraint* lifetime = solver->MakeRowConstraint(
              -infinity, 0.0,
              absl::StrFormat("lifetime_%s_%s", node->GetName(), user_str));
          lifetime->SetCoefficient(cycle_at_user, 1);
          lifetime->SetCoefficient(cycle_at_node, -1);
          lifetime->SetCoefficient(lifetime_at_node, -1);

          XLS_VLOG(2) << "Setting lifetime constraint: "
                      << absl::StrFormat(
                             "lifetime[%s] + cycle[%s] - cycle[%s] ≥ 0",
                             node->GetName(), node->GetName(), user_str);
        };

    for (Node* user : node->users()) {
      add_du_chains_related_constraints(user->GetName(), cycle_var.at(user));
    }
    if (f->HasImplicitUse(node)) {
      add_du_chains_related_constraints("«sink»", cycle_at_sinknode);
    }
  }

  XLS_ASSIGN_OR_RETURN(auto delay_map, ComputeNodeDelays(f, delay_estimator));
  absl::flat_hash_map<Node*, std::vector<Node*>> delay_constraints =
      ComputeCombinationalDelayConstraints(f, clock_period_ps, delay_map);
  for (Node* source : f->nodes()) {
    for (Node* target : delay_constraints.at(source)) {
      or_tools::MPConstraint* timing = solver->MakeRowConstraint(
          1, infinity,
          absl::StrFormat("timing_%s_%s", source->GetName(),
                          target->GetName()));
      timing->SetCoefficient(cycle_var[target], 1);
      timing->SetCoefficient(cycle_var[source], -1);
      XLS_VLOG(2) << "Setting timing constraint: "
                  << absl::StrFormat("%s - %s ≥ 1", target->GetName(),
                                     source->GetName());
    }
  }

  or_tools::MPObjective* objective = solver->MutableObjective();
  for (Node* node : f->nodes()) {
    // This acts as a tie-breaker for underconstrained problems.
    objective->SetCoefficient(cycle_var[node], 1);
    // Minimize node lifetimes.
    // The scaling makes the tie-breaker small in comparison, and is a power
    // of two so that there's no imprecision (just add to exponent).
    objective->SetCoefficient(lifetime_var[node],
                              1024 * node->GetType()->GetFlatBitCount());
  }
  objective->SetMinimization();

  or_tools::MPSolver::ResultStatus status = solver->Solve();
  // Check that the problem has an optimal solution.
  if (status != or_tools::MPSolver::OPTIMAL) {
    XLS_VLOG(3) << "ScheduleToMinimizeRegistersSDC failed with " << status;
    return absl::InternalError(
        "The problem does not have an optimal solution!");
  }

  // Extract result from LP solver
  ScheduleCycleMap cycle_map;
  for (Node* node : f->nodes()) {
    double cycle = cycle_var[node]->solution_value();
    if (std::fabs(cycle - std::round(cycle)) > 0.001) {
      return absl::InternalError(
          "The scheduling result is expected to be integer");
    }
    cycle_map[node] = std::round(cycle);
  }
  return cycle_map;
}

// Returns the nodes of `f` which must be scheduled in the first stage of a
// pipeline. For functions this is parameters. For procs, this is receive nodes
// and next state nodes.
std::vector<Node*> FirstStageNodes(FunctionBase* f) {
  if (Function* function = dynamic_cast<Function*>(f)) {
    return std::vector<Node*>(function->params().begin(),
                              function->params().end());
  }
  if (Proc* proc = dynamic_cast<Proc*>(f)) {
    std::vector<Node*> nodes(proc->params().begin(), proc->params().end());
    for (Node* node : proc->nodes()) {
      // TODO(tedhong): 2021/10/14 Make this more flexible (ex. for ii>N),
      // where the next state node must be scheduled before a specific state
      // but not necessarily the 1st stage.
      if (std::find(proc->NextState().begin(), proc->NextState().end(), node) !=
          proc->NextState().end()) {
        nodes.push_back(node);
      }
    }
    return nodes;
  }

  return {};
}

// Returns the nodes of `f` which must be scheduled in the final stage of a
// pipeline. For functions this is the return value.
std::vector<Node*> FinalStageNodes(FunctionBase* f) {
  if (Function* function = dynamic_cast<Function*>(f)) {
    // If the return value is a parameter, then we do not force the return value
    // to be scheduled in the final stage because, as a parameter, the node must
    // be in the first stage.
    if (function->return_value()->Is<Param>()) {
      return {};
    }
    return {function->return_value()};
  }

  return {};
}

// Construct ScheduleBounds for the given function assuming the given
// clock period and delay estimator. `topo_sort` should be a topological sort of
// the nodes of `f`. If `schedule_length` is given then the upper bounds are
// set on the bounds object with the maximum upper bound set to
// `schedule_length` - 1. Otherwise, the maximum upper bound is set to the
// maximum lower bound.
absl::StatusOr<sched::ScheduleBounds> ConstructBounds(
    FunctionBase* f, int64_t clock_period_ps, std::vector<Node*> topo_sort,
    std::optional<int64_t> schedule_length,
    const DelayEstimator& delay_estimator) {
  sched::ScheduleBounds bounds(f, std::move(topo_sort), clock_period_ps,
                               delay_estimator);

  // Initially compute the lower bounds of all nodes.
  XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());

  int64_t upper_bound;
  if (schedule_length.has_value()) {
    if (schedule_length.value() <= bounds.max_lower_bound()) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Cannot be scheduled in %d stages. Computed lower bound is %d.",
          schedule_length.value(), bounds.max_lower_bound() + 1));
    }
    upper_bound = schedule_length.value() - 1;
  } else {
    upper_bound = bounds.max_lower_bound();
  }

  // Set the lower bound of nodes which must be in the final stage to
  // `upper_bound`
  bool rerun_lb_propagation = false;
  for (Node* node : FinalStageNodes(f)) {
    if (bounds.lb(node) != upper_bound) {
      XLS_RETURN_IF_ERROR(bounds.TightenNodeLb(node, upper_bound));
      if (!node->users().empty()) {
        rerun_lb_propagation = true;
      }
    }
  }
  // If fixing nodes in the final stage changed any lower bounds then
  // repropagate the lower bounds.
  if (rerun_lb_propagation) {
    XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());
  }

  if (bounds.max_lower_bound() > upper_bound) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Impossible to schedule Function/Proc %s; the following "
        "node(s) must be scheduled in the final cycle but that "
        "is impossible due to users of these node(s): %s",
        f->name(), absl::StrJoin(FinalStageNodes(f), ", ", NodeFormatter)));
  }

  // Set and propagate upper bounds.
  for (Node* node : f->nodes()) {
    XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, upper_bound));
  }
  for (Node* node : FirstStageNodes(f)) {
    if (bounds.lb(node) > 0) {
      return absl::ResourceExhaustedError(
          absl::StrFormat("Impossible to schedule Function/Proc %s; node `%s` "
                          "must be scheduled in the first cycle but that is "
                          "impossible due to the node's operand(s)",
                          f->name(), node->GetName()));
    }
    XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, 0));
  }
  XLS_RETURN_IF_ERROR(bounds.PropagateUpperBounds());

  return std::move(bounds);
}

// Returns the critical path through the given nodes (ordered topologically).
absl::StatusOr<int64_t> ComputeCriticalPath(
    absl::Span<Node* const> topo_sort, const DelayEstimator& delay_estimator) {
  int64_t function_cp = 0;
  absl::flat_hash_map<Node*, int64_t> node_cp;
  for (Node* node : topo_sort) {
    int64_t node_start = 0;
    for (Node* operand : node->operands()) {
      node_start = std::max(node_start, node_cp[operand]);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator.GetOperationDelayInPs(node));
    node_cp[node] = node_start + node_delay;
    function_cp = std::max(function_cp, node_cp[node]);
  }
  return function_cp;
}

// Returns the minimum clock period in picoseconds for which it is feasible to
// schedule the function into a pipeline with the given number of stages.
absl::StatusOr<int64_t> FindMinimumClockPeriod(
    FunctionBase* f, int64_t pipeline_stages,
    const DelayEstimator& delay_estimator) {
  XLS_VLOG(4) << "FindMinimumClockPeriod()";
  XLS_VLOG(4) << "  pipeline stages = " << pipeline_stages;
  auto topo_sort_it = TopoSort(f);
  std::vector<Node*> topo_sort(topo_sort_it.begin(), topo_sort_it.end());
  XLS_ASSIGN_OR_RETURN(int64_t function_cp,
                       ComputeCriticalPath(topo_sort, delay_estimator));
  // The lower bound of the search is the critical path delay evenly distributed
  // across all stages (rounded up), and the upper bound is simply the critical
  // path of the entire function. It's possible this upper bound is the best you
  // can do if there exists a single operation with delay equal to the
  // critical-path delay of the function.
  int64_t search_start = (function_cp + pipeline_stages - 1) / pipeline_stages;
  int64_t search_end = function_cp;
  XLS_VLOG(4) << absl::StreamFormat("Binary searching over interval [%d, %d]",
                                    search_start, search_end);
  XLS_ASSIGN_OR_RETURN(
      int64_t min_period,
      BinarySearchMinTrueWithStatus(
          search_start, search_end,
          [&](int64_t clk_period_ps) -> absl::StatusOr<bool> {
            absl::StatusOr<sched::ScheduleBounds> bounds_or = ConstructBounds(
                f, clk_period_ps, topo_sort,
                /*schedule_length=*/absl::nullopt, delay_estimator);
            if (!bounds_or.ok()) {
              return false;
            }
            return bounds_or.value().max_lower_bound() < pipeline_stages;
          }));
  XLS_VLOG(4) << "minimum clock period = " << min_period;

  return min_period;
}

// Returns a sequence of numbers from first to last where the zeroth element of
// the sequence is the middle element between first and last. Subsequent
// elements are selected recursively out of the two intervals before and after
// the middle element.
std::vector<int64_t> MiddleFirstOrder(int64_t first, int64_t last) {
  if (first == last) {
    return {first};
  }
  if (first == last - 1) {
    return {first, last};
  }

  int64_t middle = (first + last) / 2;
  std::vector<int64_t> head = MiddleFirstOrder(first, middle - 1);
  std::vector<int64_t> tail = MiddleFirstOrder(middle + 1, last);

  std::vector<int64_t> ret;
  ret.push_back(middle);
  ret.insert(ret.end(), head.begin(), head.end());
  ret.insert(ret.end(), tail.begin(), tail.end());
  return ret;
}

}  // namespace

std::vector<std::vector<int64_t>> GetMinCutCycleOrders(int64_t length) {
  if (length == 0) {
    return {{}};
  }
  if (length == 1) {
    return {{0}};
  }
  if (length == 2) {
    return {{0, 1}, {1, 0}};
  }
  // For lengths greater than 2, return forward, reverse and middle first
  // orderings.
  std::vector<std::vector<int64_t>> orders;
  std::vector<int64_t> forward(length);
  std::iota(forward.begin(), forward.end(), 0);
  orders.push_back(forward);

  std::vector<int64_t> reverse(length);
  std::iota(reverse.begin(), reverse.end(), 0);
  std::reverse(reverse.begin(), reverse.end());
  orders.push_back(reverse);

  orders.push_back(MiddleFirstOrder(0, length - 1));
  return orders;
}

PipelineSchedule::PipelineSchedule(FunctionBase* function_base,
                                   ScheduleCycleMap cycle_map,
                                   std::optional<int64_t> length)
    : function_base_(function_base), cycle_map_(std::move(cycle_map)) {
  // Build the mapping from cycle to the vector of nodes in that cycle.
  int64_t max_cycle = MaximumCycle(cycle_map_);
  if (length.has_value()) {
    XLS_CHECK_GT(*length, max_cycle);
    max_cycle = *length - 1;
  }
  // max_cycle is the latest cycle in which any node is scheduled so add one to
  // get the capacity because cycle numbers start at zero.
  cycle_to_nodes_.resize(max_cycle + 1);
  for (const auto& pair : cycle_map_) {
    Node* node = pair.first;
    int64_t cycle = pair.second;
    cycle_to_nodes_[cycle].push_back(node);
  }
  // The nodes in each cycle held in cycle_to_nodes_ must be in a topological
  // sort order.
  absl::flat_hash_map<Node*, int64_t> node_to_topo_index;
  int64_t i = 0;
  for (Node* node : TopoSort(function_base)) {
    node_to_topo_index[node] = i;
    ++i;
  }
  for (std::vector<Node*>& nodes_in_cycle : cycle_to_nodes_) {
    std::sort(nodes_in_cycle.begin(), nodes_in_cycle.end(),
              [&](Node* a, Node* b) {
                return node_to_topo_index[a] < node_to_topo_index[b];
              });
  }
}

absl::StatusOr<PipelineSchedule> PipelineSchedule::FromProto(
    Function* function, const PipelineScheduleProto& proto) {
  ScheduleCycleMap cycle_map;
  for (const auto& stage : proto.stages()) {
    for (const auto& node_name : stage.nodes()) {
      XLS_ASSIGN_OR_RETURN(Node * node, function->GetNode(node_name));
      cycle_map[node] = stage.stage();
    }
  }
  return PipelineSchedule(function, cycle_map);
}

absl::Span<Node* const> PipelineSchedule::nodes_in_cycle(int64_t cycle) const {
  if (cycle < cycle_to_nodes_.size()) {
    return cycle_to_nodes_[cycle];
  }
  return absl::Span<Node* const>();
}

std::vector<Node*> PipelineSchedule::GetLiveOutOfCycle(int64_t c) const {
  std::vector<Node*> live_out;
  for (int64_t i = 0; i <= c; ++i) {
    for (Node* node : nodes_in_cycle(i)) {
      if (node->function_base()->HasImplicitUse(node) ||
          absl::c_any_of(node->users(),
                         [&](Node* u) { return cycle(u) > c; })) {
        live_out.push_back(node);
      }
    }
  }
  return live_out;
}

namespace {
class DelayEstimatorWithInputDelay : public DelayEstimator {
 public:
  DelayEstimatorWithInputDelay(const DelayEstimator& base, int64_t input_delay)
      : DelayEstimator(absl::StrFormat("%s_with_input_delay", base.name())),
        base_delay_estimator_(&base),
        input_delay_(input_delay) {}

  virtual absl::StatusOr<int64_t> GetOperationDelayInPs(
      Node* node) const override {
    XLS_ASSIGN_OR_RETURN(int64_t base_delay,
                         base_delay_estimator_->GetOperationDelayInPs(node));

    return (node->op() == Op::kReceive) ? base_delay + input_delay_
                                        : base_delay;
  }

 private:
  const DelayEstimator* base_delay_estimator_;
  int64_t input_delay_;
};
}  // namespace

/*static*/ absl::StatusOr<PipelineSchedule> PipelineSchedule::Run(
    FunctionBase* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options) {
  int64_t input_delay = options.additional_input_delay_ps().has_value()
                            ? options.additional_input_delay_ps().value()
                            : 0;

  DelayEstimatorWithInputDelay delay_estimator_with_delay(delay_estimator,
                                                          input_delay);

  int64_t clock_period_ps;
  if (options.clock_period_ps().has_value()) {
    clock_period_ps = *options.clock_period_ps();

    if (options.clock_margin_percent().has_value()) {
      int64_t original_clock_period_ps = clock_period_ps;
      clock_period_ps -=
          (clock_period_ps * options.clock_margin_percent().value() + 50) / 100;
      if (clock_period_ps <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Clock period non-positive (%dps) after adjusting for margin. "
            "Original clock period: %dps, clock margin: %d%%",
            clock_period_ps, original_clock_period_ps,
            *options.clock_margin_percent()));
      }
    }
  } else {
    XLS_RET_CHECK(options.pipeline_stages().has_value());
    // A pipeline length is specified, but no target clock period. Determine
    // the minimum clock period for which the function can be scheduled in the
    // given pipeline length.
    XLS_ASSIGN_OR_RETURN(clock_period_ps,
                         FindMinimumClockPeriod(f, *options.pipeline_stages(),
                                                delay_estimator_with_delay));

    if (options.period_relaxation_percent().has_value()) {
      int64_t relaxation_percent = options.period_relaxation_percent().value();

      clock_period_ps += (clock_period_ps * relaxation_percent + 50) / 100;
    }
  }

  XLS_ASSIGN_OR_RETURN(
      sched::ScheduleBounds bounds,
      ConstructBounds(f, clock_period_ps, TopoSort(f).AsVector(),
                      options.pipeline_stages(), delay_estimator_with_delay));
  int64_t schedule_length = bounds.max_lower_bound() + 1;

  ScheduleCycleMap cycle_map;
  if (options.strategy() == SchedulingStrategy::MINIMIZE_REGISTERS) {
    XLS_ASSIGN_OR_RETURN(cycle_map, ScheduleToMinimizeRegisters(
                                        f, schedule_length,
                                        delay_estimator_with_delay, &bounds));
  } else if (options.strategy() == SchedulingStrategy::MINIMIZE_REGISTERS_SDC) {
    XLS_ASSIGN_OR_RETURN(
        cycle_map, ScheduleToMinimizeRegistersSDC(
                       f, schedule_length, delay_estimator_with_delay, &bounds,
                       clock_period_ps, options.constraints()));
  } else if (options.strategy() == SchedulingStrategy::RANDOM) {
    for (Node* node : TopoSort(f)) {
      int64_t lower_bound = bounds.lb(node);
      int64_t upper_bound = bounds.ub(node);
      std::mt19937 gen(options.seed().value_or(0));
      std::uniform_int_distribution<int64_t> distrib(lower_bound, upper_bound);
      int64_t cycle = distrib(gen);
      XLS_RETURN_IF_ERROR(bounds.TightenNodeLb(node, cycle));
      XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());
      XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, cycle));
      XLS_RETURN_IF_ERROR(bounds.PropagateUpperBounds());
      cycle_map[node] = cycle;
    }
  } else {
    XLS_RET_CHECK(options.strategy() == SchedulingStrategy::ASAP);
    XLS_RET_CHECK(!options.pipeline_stages().has_value());
    // Just schedule everything as soon as possible.
    for (Node* node : f->nodes()) {
      cycle_map[node] = bounds.lb(node);
    }
  }

  auto schedule = PipelineSchedule(f, cycle_map, options.pipeline_stages());
  XLS_RETURN_IF_ERROR(schedule.Verify());
  XLS_RETURN_IF_ERROR(
      schedule.VerifyTiming(clock_period_ps, delay_estimator_with_delay));

  // Verify that scheduling constraints are obeyed.
  {
    absl::flat_hash_map<std::string, std::vector<Node*>> channel_to_nodes;
    for (Node* node : f->nodes()) {
      if (node->Is<Receive>() || node->Is<Send>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
        channel_to_nodes[channel->name()].push_back(node);
      }
    }

    for (const SchedulingConstraint& constraint : options.constraints()) {
      // We use `channel_to_nodes[...]` instead of `channel_to_nodes.at(...)`
      // below because we don't want to error out if a constraint is specified
      // that affects a channel with no associated send/receives in this proc.
      for (Node* source : channel_to_nodes[constraint.SourceChannel()]) {
        for (Node* target : channel_to_nodes[constraint.TargetChannel()]) {
          if (source == target) {
            continue;
          }
          int64_t source_cycle = cycle_map.at(source);
          int64_t target_cycle = cycle_map.at(target);
          int64_t latency = target_cycle - source_cycle;
          if ((constraint.MinimumLatency() <= latency) &&
              (latency <= constraint.MaximumLatency())) {
            continue;
          }
          return absl::ResourceExhaustedError(absl::StrFormat(
              "Scheduling constraint violated: node %s was scheduled %d cycles "
              "before node %s which violates the constraint that ops on "
              "channel %s must be between %d and %d cycles (inclusive) before "
              "ops on channel %s.",
              source->ToString(), latency, target->ToString(),
              constraint.SourceChannel(), constraint.MinimumLatency(),
              constraint.MaximumLatency(), constraint.TargetChannel()));
        }
      }
    }
  }

  XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
  return schedule;
}

std::string PipelineSchedule::ToString() const {
  absl::flat_hash_map<const Node*, int64_t> topo_pos;
  int64_t pos = 0;
  for (Node* node : TopoSort(function_base_)) {
    topo_pos[node] = pos;
    pos++;
  }

  std::string result;
  for (int64_t cycle = 0; cycle <= length(); ++cycle) {
    absl::StrAppendFormat(&result, "Cycle %d:\n", cycle);
    // Emit nodes in topo-sort order for easier reading.
    std::vector<Node*> nodes(nodes_in_cycle(cycle).begin(),
                             nodes_in_cycle(cycle).end());
    std::sort(nodes.begin(), nodes.end(), [&](Node* a, Node* b) {
      return topo_pos.at(a) < topo_pos.at(b);
    });
    for (Node* node : nodes) {
      absl::StrAppendFormat(&result, "  %s\n", node->ToString());
    }
  }
  return result;
}

absl::Status PipelineSchedule::Verify() const {
  for (Node* node : function_base()->nodes()) {
    XLS_RET_CHECK(IsScheduled(node));
  }
  for (Node* node : function_base()->nodes()) {
    for (Node* operand : node->operands()) {
      XLS_RET_CHECK_LE(cycle(operand), cycle(node));
    }
  }

  // Verify initial nodes in cycle 0. Final nodes in final cycle.
  return absl::OkStatus();
}

absl::Status PipelineSchedule::VerifyTiming(
    int64_t clock_period_ps, const DelayEstimator& delay_estimator) const {
  // Critical path from start of the cycle that a node is scheduled through the
  // node itself. If the schedule meets timing, then this value should be less
  // than or equal to clock_period_ps for every node.
  absl::flat_hash_map<Node*, int64_t> node_cp;
  // The predecessor (operand) of the node through which the critical-path from
  // the start of the cycle extends.
  absl::flat_hash_map<Node*, Node*> cp_pred;
  // The node with the longest critical path from the start of the stage in the
  // entire schedule.
  Node* max_cp_node = nullptr;
  for (Node* node : TopoSort(function_base_)) {
    // The critical-path delay from the start of the stage to the start of the
    // node.
    int64_t cp_to_node_start = 0;
    cp_pred[node] = nullptr;
    for (Node* operand : node->operands()) {
      if (cycle(operand) == cycle(node)) {
        if (cp_to_node_start < node_cp.at(operand)) {
          cp_to_node_start = node_cp.at(operand);
          cp_pred[node] = operand;
        }
      }
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator.GetOperationDelayInPs(node));
    node_cp[node] = cp_to_node_start + node_delay;
    if (max_cp_node == nullptr || node_cp[node] > node_cp[max_cp_node]) {
      max_cp_node = node;
    }
  }

  if (node_cp[max_cp_node] > clock_period_ps) {
    std::vector<Node*> path;
    Node* node = max_cp_node;
    do {
      path.push_back(node);
      node = cp_pred[node];
    } while (node != nullptr);
    std::reverse(path.begin(), path.end());
    return absl::InternalError(absl::StrFormat(
        "Schedule does not meet timing (%dps). Longest failing path (%dps): %s",
        clock_period_ps, node_cp[max_cp_node],
        absl::StrJoin(path, " -> ", [&](std::string* out, Node* n) {
          absl::StrAppend(
              out, absl::StrFormat(
                       "%s (%dps)", n->GetName(),
                       delay_estimator.GetOperationDelayInPs(n).value()));
        })));
  }
  return absl::OkStatus();
}

PipelineScheduleProto PipelineSchedule::ToProto() const {
  PipelineScheduleProto proto;
  proto.set_function(function_base_->name());
  for (int i = 0; i < cycle_to_nodes_.size(); i++) {
    StageProto* stage = proto.add_stages();
    stage->set_stage(i);
    for (const Node* node : cycle_to_nodes_[i]) {
      stage->add_nodes(node->GetName());
    }
  }
  return proto;
}

int64_t PipelineSchedule::CountFinalInteriorPipelineRegisters() const {
  int64_t reg_count = 0;

  Function* as_func = dynamic_cast<Function*>(function_base_);
  for (int64_t stage = 0; stage < length(); ++stage) {
    for (Node* function_base_node : function_base_->nodes()) {
      if (cycle(function_base_node) > stage) {
        continue;
      }

      auto is_live_out_of_stage = [&](Node* n) {
        if (stage == length() - 1) {
          return false;
        }
        if (as_func && (n == as_func->return_value())) {
          return true;
        }
        for (Node* user : n->users()) {
          if (cycle(user) > stage) {
            return true;
          }
        }
        return false;
      };

      if (is_live_out_of_stage(function_base_node)) {
        reg_count += function_base_node->GetType()->GetFlatBitCount();
      }
    }
  }

  return reg_count;
}

}  // namespace xls

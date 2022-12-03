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

#include "xls/scheduling/sdc_scheduler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/strongly_connected_components.h"
#include "xls/data_structures/union_find.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/scheduling/schedule_bounds.h"
#include "ortools/linear_solver/linear_solver.h"

namespace or_tools = ::operations_research;

namespace xls {

namespace {

using DelayMap = absl::flat_hash_map<Node*, int64_t>;

// A helper function to compute each node's delay by calling the delay estimator
absl::StatusOr<DelayMap> ComputeNodeDelays(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  DelayMap result;
  for (Node* node : f->nodes()) {
    XLS_ASSIGN_OR_RETURN(result[node],
                         delay_estimator.GetOperationDelayInPs(node));
  }
  return result;
}

// All transitive children (operands, operands of operands, etc.) of the given
// node.
absl::btree_set<Node*, Node::NodeIdLessThan> Descendants(Node* root) {
  std::vector<Node*> stack;
  stack.push_back(root);
  absl::btree_set<Node*, Node::NodeIdLessThan> discovered;
  while (!stack.empty()) {
    Node* popped = stack.back();
    stack.pop_back();
    if (!discovered.contains(popped)) {
      discovered.insert(popped);
      for (Node* child : popped->operands()) {
        stack.push_back(child);
      }
    }
  }
  return discovered;
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

class ConstraintBuilder {
 public:
  ConstraintBuilder(FunctionBase* func, or_tools::MPSolver* solver,
                    int64_t pipeline_length, int64_t clock_period_ps,
                    const sched::ScheduleBounds& bounds,
                    const DelayMap& delay_map);

  absl::Status AddDefUseConstraints(Node* node, std::optional<Node*> user);
  absl::Status AddCausalConstraint(Node* node, std::optional<Node*> user);
  absl::Status AddLifetimeConstraint(Node* node, std::optional<Node*> user);
  absl::Status AddBackedgeConstraints();
  absl::Status AddTimingConstraints();
  absl::Status AddSchedulingConstraint(const SchedulingConstraint& constraint);
  absl::Status AddIOConstraint(const IOConstraint& constraint);
  absl::Status AddRFSLConstraint(
      const RecvsFirstSendsLastConstraint& constraint);

  absl::Status AddObjective();

  or_tools::MPSolver::ResultStatus Solve() { return solver_->Solve(); }

  absl::StatusOr<ScheduleCycleMap> ExtractResult() const;

  absl::flat_hash_map<Node*, or_tools::MPVariable*> GetCycleVars() const {
    return cycle_var_;
  }

  absl::flat_hash_map<Node*, or_tools::MPVariable*> GetLifetimeVars() const {
    return lifetime_var_;
  }

 private:
  or_tools::MPConstraint* DiffLessThanConstraint(Node* x, Node* y,
                                                 int64_t limit,
                                                 std::string_view name) {
    or_tools::MPConstraint* constraint = solver_->MakeRowConstraint(
        -infinity_, limit,
        absl::StrFormat("%s:%s-%s≤%d", name, x->GetName(), y->GetName(),
                        limit));
    constraint->SetCoefficient(cycle_var_.at(x), 1);
    constraint->SetCoefficient(cycle_var_.at(y), -1);
    return constraint;
  }

  or_tools::MPConstraint* DiffGreaterThanConstraint(Node* x, Node* y,
                                                    int64_t limit,
                                                    std::string_view name) {
    or_tools::MPConstraint* constraint = solver_->MakeRowConstraint(
        -infinity_, -limit,
        absl::StrFormat("%s:%s-%s≥%d", name, x->GetName(), y->GetName(),
                        limit));
    constraint->SetCoefficient(cycle_var_.at(x), -1);
    constraint->SetCoefficient(cycle_var_.at(y), 1);
    return constraint;
  }

  void DiffEqualsConstraint(Node* x, Node* y, int64_t diff,
                            std::string_view name) {
    if (x == y) {
      if (diff == 0) {
        return;
      }
      XLS_LOG(FATAL) << "DiffEqualsConstraint: " << x->GetName() << " - "
                     << y->GetName() << " = " << diff << " is unsatisfiable";
    }
    DiffLessThanConstraint(x, y, diff, name);
    DiffGreaterThanConstraint(x, y, diff, name);
  }

  FunctionBase* func_;
  or_tools::MPSolver* solver_;
  int64_t pipeline_length_;
  int64_t clock_period_ps_;
  const DelayMap& delay_map_;
  double infinity_;

  // Node's cycle after scheduling
  absl::flat_hash_map<Node*, or_tools::MPVariable*> cycle_var_;

  // Node's lifetime, from when it finishes executing until it is consumed by
  // the last user.
  absl::flat_hash_map<Node*, or_tools::MPVariable*> lifetime_var_;

  // A dummy node to represent an artificial sink node on the data-dependence
  // graph.
  or_tools::MPVariable* cycle_at_sinknode_;

  // A cache of the delay constraints.
  absl::flat_hash_map<Node*, std::vector<Node*>> delay_constraints_;
};

ConstraintBuilder::ConstraintBuilder(FunctionBase* func,
                                     or_tools::MPSolver* solver,
                                     int64_t pipeline_length,
                                     int64_t clock_period_ps,
                                     const sched::ScheduleBounds& bounds,
                                     const DelayMap& delay_map)
    : func_(func),
      solver_(solver),
      pipeline_length_(pipeline_length),
      clock_period_ps_(clock_period_ps),
      delay_map_(delay_map),
      infinity_(solver->infinity()) {
  for (Node* node : func_->nodes()) {
    cycle_var_[node] =
        solver_->MakeNumVar(bounds.lb(node), bounds.ub(node), node->GetName());
    lifetime_var_[node] = solver_->MakeNumVar(
        0.0, infinity_, absl::StrFormat("lifetime_%s", node->GetName()));
  }
  cycle_at_sinknode_ =
      solver->MakeNumVar(-infinity_, infinity_, "cycle_at_sinknode");
}

absl::Status ConstraintBuilder::AddDefUseConstraints(
    Node* node, std::optional<Node*> user) {
  XLS_RETURN_IF_ERROR(AddCausalConstraint(node, user));
  XLS_RETURN_IF_ERROR(AddLifetimeConstraint(node, user));
  return absl::OkStatus();
}

absl::Status ConstraintBuilder::AddCausalConstraint(Node* node,
                                                    std::optional<Node*> user) {
  or_tools::MPVariable* cycle_at_node = cycle_var_.at(node);
  or_tools::MPVariable* cycle_at_user =
      user.has_value() ? cycle_var_.at(user.value()) : cycle_at_sinknode_;

  std::string user_str = user.has_value() ? user.value()->GetName() : "«sink»";

  // Constraint: cycle[node] - cycle[node_user] <= 0
  or_tools::MPConstraint* causal = solver_->MakeRowConstraint(
      -infinity_, 0.0,
      absl::StrFormat("causal_%s_%s", node->GetName(), user_str));
  causal->SetCoefficient(cycle_at_node, 1);
  causal->SetCoefficient(cycle_at_user, -1);

  XLS_VLOG(2) << "Setting causal constraint: "
              << absl::StrFormat("cycle[%s] - cycle[%s] ≥ 0", user_str,
                                 node->GetName());

  return absl::OkStatus();
}

absl::Status ConstraintBuilder::AddLifetimeConstraint(
    Node* node, std::optional<Node*> user) {
  or_tools::MPVariable* cycle_at_node = cycle_var_.at(node);
  or_tools::MPVariable* lifetime_at_node = lifetime_var_.at(node);
  or_tools::MPVariable* cycle_at_user =
      user.has_value() ? cycle_var_.at(user.value()) : cycle_at_sinknode_;

  std::string user_str = user.has_value() ? user.value()->GetName() : "«sink»";

  // Constraint: cycle[node_user] - cycle[node] - lifetime[node] <= 0
  or_tools::MPConstraint* lifetime = solver_->MakeRowConstraint(
      -infinity_, 0.0,
      absl::StrFormat("lifetime_%s_%s", node->GetName(), user_str));
  lifetime->SetCoefficient(cycle_at_user, 1);
  lifetime->SetCoefficient(cycle_at_node, -1);
  lifetime->SetCoefficient(lifetime_at_node, -1);

  XLS_VLOG(2) << "Setting lifetime constraint: "
              << absl::StrFormat("lifetime[%s] + cycle[%s] - cycle[%s] ≥ 0",
                                 node->GetName(), node->GetName(), user_str);

  return absl::OkStatus();
}

// This ensures that state backedges don't span more than one cycle, which is
// necessary while II = 1.
absl::Status ConstraintBuilder::AddBackedgeConstraints() {
  Proc* proc = dynamic_cast<Proc*>(func_);
  if (proc == nullptr) {
    return absl::OkStatus();
  }

  using StateIndex = int64_t;

  absl::flat_hash_set<Node*> params(proc->StateParams().begin(),
                                    proc->StateParams().end());

  // This constructs the state dependence graph, which is defined in the
  // following way:
  //
  // First, construct a bipartite graph where nodes are either state params or
  // next state nodes, and there is an edge from a state param to a next state
  // node iff there is a path between them in the DFG. For example, this might
  // look like `{(p1, n1), (p2, n1), (p2, n2), (p3, n3)}`.
  //
  // Then, identify each param node with its corresponding next node.
  // Using the previous example, this would look like
  // `{(v1, v1), (v2, v1), (v2, v2), (v3, v3)}`.
  absl::btree_map<StateIndex, absl::btree_set<StateIndex>> graph;
  for (StateIndex target = 0; target < proc->GetStateElementCount(); ++target) {
    for (Node* param : Descendants(proc->GetNextStateElement(target))) {
      if (!params.contains(param)) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(StateIndex source, proc->GetStateParamIndex(
                                                  dynamic_cast<Param*>(param)));
      graph[source].insert(target);
    }
  }

  // Note that the Strongly Connected Components (SCC) algorithm represents a
  // graph as a sparse adjacency matrix. If a node has no edges, then the node
  // will not be in the graph. As a result, the following ensures that the next
  // state is always in the same cycle as its corresponding param, which is
  // currently required by codegen.
  for (StateIndex i = 0; i < proc->GetStateElementCount(); ++i) {
    graph[i].insert(i);
  }

  // A strongly connected component in the state dependence graph is a set of
  // state params / next state nodes that must be scheduled in the same cycle.
  std::vector<absl::btree_set<StateIndex>> sccs =
      StronglyConnectedComponents<StateIndex>(graph);
  for (const absl::btree_set<StateIndex>& scc : sccs) {
    absl::btree_set<Node*, Node::NodeIdLessThan> nodes;
    for (const StateIndex& index : scc) {
      nodes.insert(proc->GetStateParam(index));
      nodes.insert(proc->GetNextStateElement(index));
    }
    Node* rep = *(nodes.begin());
    for (Node* other : nodes) {
      if (rep == other) {
        continue;
      }
      DiffEqualsConstraint(rep, other, 0, "backedge");
      XLS_VLOG(2) << "Setting backedge constraint: "
                  << absl::StrFormat("cycle[%s] = cycle[%s]", rep->GetName(),
                                     other->GetName());
    }
  }

  return absl::OkStatus();
}

absl::Status ConstraintBuilder::AddTimingConstraints() {
  if (delay_constraints_.empty()) {
    delay_constraints_ = ComputeCombinationalDelayConstraints(
        func_, clock_period_ps_, delay_map_);
  }

  for (Node* source : func_->nodes()) {
    for (Node* target : delay_constraints_.at(source)) {
      DiffGreaterThanConstraint(target, source, 1, "timing");
      XLS_VLOG(2) << "Setting timing constraint: "
                  << absl::StrFormat("1 ≤ %s - %s", target->GetName(),
                                     source->GetName());
    }
  }

  return absl::OkStatus();
}

absl::Status ConstraintBuilder::AddSchedulingConstraint(
    const SchedulingConstraint& constraint) {
  if (std::holds_alternative<IOConstraint>(constraint)) {
    return AddIOConstraint(std::get<IOConstraint>(constraint));
  }
  if (std::holds_alternative<RecvsFirstSendsLastConstraint>(constraint)) {
    return AddRFSLConstraint(
        std::get<RecvsFirstSendsLastConstraint>(constraint));
  }
  return absl::InternalError("Unhandled scheduling constraint type");
}

absl::Status ConstraintBuilder::AddIOConstraint(
    const IOConstraint& constraint) {
  // Map from channel name to set of nodes that send/receive on that channel.
  absl::flat_hash_map<std::string, std::vector<Node*>> channel_to_nodes;
  for (Node* node : func_->nodes()) {
    if (node->Is<Receive>() || node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      channel_to_nodes[channel->name()].push_back(node);
    }
  }

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

      DiffGreaterThanConstraint(target, source, constraint.MinimumLatency(),
                                "io");
      DiffLessThanConstraint(target, source, constraint.MaximumLatency(), "io");

      XLS_VLOG(2) << "Setting IO constraint: "
                  << absl::StrFormat("%d ≤ cycle[%s] - cycle[%s] ≤ %d",
                                     constraint.MinimumLatency(),
                                     target->GetName(), source->GetName(),
                                     constraint.MaximumLatency());
    }
  }

  return absl::OkStatus();
}

absl::Status ConstraintBuilder::AddRFSLConstraint(
    const RecvsFirstSendsLastConstraint& constraint) {
  for (Node* node : func_->nodes()) {
    if (node->Is<Receive>()) {
      or_tools::MPConstraint* recv_constraint = solver_->MakeRowConstraint(
          -infinity_, 0, absl::StrFormat("recv_%s", node->GetName()));
      recv_constraint->SetCoefficient(cycle_var_.at(node), 1);

      XLS_VLOG(2) << "Setting receive-in-first-cycle constraint: "
                  << absl::StrFormat("cycle[%s] ≤ 0", node->GetName());
    }
    if (node->Is<Send>()) {
      or_tools::MPConstraint* send_constraint = solver_->MakeRowConstraint(
          -infinity_, -(pipeline_length_ - 1),
          absl::StrFormat("send_%s", node->GetName()));
      send_constraint->SetCoefficient(cycle_var_.at(node), -1);

      XLS_VLOG(2) << "Setting send-in-last-cycle constraint: "
                  << absl::StrFormat("%d ≤ cycle[%s]", pipeline_length_ - 1,
                                     node->GetName());
    }
  }

  return absl::OkStatus();
}

absl::Status ConstraintBuilder::AddObjective() {
  or_tools::MPObjective* objective = solver_->MutableObjective();
  for (Node* node : func_->nodes()) {
    // This acts as a tie-breaker for underconstrained problems.
    objective->SetCoefficient(cycle_var_.at(node), 1);
    // Minimize node lifetimes.
    // The scaling makes the tie-breaker small in comparison, and is a power
    // of two so that there's no imprecision (just add to exponent).
    objective->SetCoefficient(lifetime_var_.at(node),
                              1024 * node->GetType()->GetFlatBitCount());
  }
  objective->SetMinimization();
  return absl::OkStatus();
}

absl::StatusOr<ScheduleCycleMap> ConstraintBuilder::ExtractResult() const {
  ScheduleCycleMap cycle_map;
  for (Node* node : func_->nodes()) {
    double cycle = cycle_var_.at(node)->solution_value();
    if (std::fabs(cycle - std::round(cycle)) > 0.001) {
      return absl::InternalError(
          "The scheduling result is expected to be integer");
    }
    cycle_map[node] = std::round(cycle);
  }
  return cycle_map;
}

}  // namespace

absl::StatusOr<ScheduleCycleMap> SDCScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints,
    bool check_feasibility) {
  XLS_VLOG(3) << "SDCScheduler()";
  XLS_VLOG(3) << "  pipeline stages = " << pipeline_stages;
  XLS_VLOG_LINES(4, f->DumpIr());

  XLS_VLOG(4) << "Initial bounds:";
  XLS_VLOG_LINES(4, bounds->ToString());

  std::unique_ptr<or_tools::MPSolver> solver(
      or_tools::MPSolver::CreateSolver("GLOP"));
  if (!solver) {
    return absl::UnavailableError("GLOP solver unavailable.");
  }

  XLS_ASSIGN_OR_RETURN(DelayMap delay_map,
                       ComputeNodeDelays(f, delay_estimator));

  ConstraintBuilder builder(f, solver.get(), pipeline_stages, clock_period_ps,
                            *bounds, delay_map);

  for (const SchedulingConstraint& constraint : constraints) {
    XLS_RETURN_IF_ERROR(builder.AddSchedulingConstraint(constraint));
  }

  for (Node* node : f->nodes()) {
    for (Node* user : node->users()) {
      XLS_RETURN_IF_ERROR(builder.AddDefUseConstraints(node, user));
    }
    if (f->IsFunction() && f->HasImplicitUse(node)) {
      XLS_RETURN_IF_ERROR(builder.AddDefUseConstraints(node, std::nullopt));
    }
  }

  XLS_RETURN_IF_ERROR(builder.AddTimingConstraints());
  XLS_RETURN_IF_ERROR(builder.AddBackedgeConstraints());

  if (!check_feasibility) {
    XLS_RETURN_IF_ERROR(builder.AddObjective());
  }

  or_tools::MPSolver::ResultStatus status = builder.Solve();

  if (status != or_tools::MPSolver::OPTIMAL) {
    XLS_VLOG(1) << "SDCScheduler failed with " << status;
    return absl::InternalError("The problem does not have an optimal solution");
  }

  return builder.ExtractResult();
}

}  // namespace xls

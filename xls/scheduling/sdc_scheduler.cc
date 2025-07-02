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
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/schedule_util.h"
#include "xls/scheduling/scheduling_options.h"
#include "ortools/math_opt/cpp/math_opt.h"

namespace xls {

namespace {

using DelayMap = absl::flat_hash_map<Node*, int64_t>;
namespace math_opt = ::operations_research::math_opt;

constexpr double kObjectiveScaling = 1024;

// A helper function to compute each node's delay by calling the delay
// estimator; treats all dead-after-synthesis nodes as having a delay of 0.
absl::StatusOr<DelayMap> ComputeNodeDelays(
    const ScheduleGraph& graph, const DelayEstimator& delay_estimator) {
  DelayMap result;
  for (const ScheduleNode& schedule_node : graph.nodes()) {
    Node* node = schedule_node.node;
    if (schedule_node.is_dead_after_synthesis) {
      result[node] = 0;
    } else {
      XLS_ASSIGN_OR_RETURN(result[node],
                           delay_estimator.GetOperationDelayInPs(node));
    }
  }
  return result;
}

// Compute all-pairs longest distance between all nodes in `f`. The distance
// from node `a` to node `b` is defined as the length of the longest delay path
// from `a`'s start to `b`'s end, which includes the delay of the path endpoints
// `a` and `b`. The all-pairs distance is stored in the map of maps
// `distances_to_node` where `distances_to_node[y][x]` (if present) is the
// critical-path distance from `x` to `y`.
absl::flat_hash_map<Node*, absl::flat_hash_map<Node*, int64_t>>
ComputeDistancesToNodes(const ScheduleGraph& graph, const DelayMap& delay_map) {
  absl::flat_hash_map<Node*, absl::flat_hash_map<Node*, int64_t>>
      distances_to_node;
  distances_to_node.reserve(graph.nodes().size());
  for (const ScheduleNode& schedule_node : graph.nodes()) {
    // Initialize the distance map entry to an empty map.
    distances_to_node[schedule_node.node];
  }

  for (const ScheduleNode& schedule_node : graph.nodes()) {
    absl::flat_hash_map<Node*, int64_t>& distances =
        distances_to_node.at(schedule_node.node);

    // The critical path from `node` to `node` is always `node_delay` long.
    int64_t node_delay = delay_map.at(schedule_node.node);
    distances_to_node.at(schedule_node.node)[schedule_node.node] = node_delay;

    // Compute the critical-path distance from `a` to `node` for all descendants
    // `a` of each operand, extending the critical path from `a` to each operand
    // of `node` by `node_delay`.
    for (Node* operand : schedule_node.predecessors) {
      for (auto [a, operand_distance] : distances_to_node.at(operand)) {
        auto [it, newly_reachable] =
            distances.try_emplace(a, operand_distance + node_delay);
        if (!newly_reachable) {
          if (it->second >= operand_distance + node_delay) {
            continue;
          }
          it->second = operand_distance + node_delay;
        }
      }
    }
  }

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "All-pairs critical-path distances:";
    for (const ScheduleNode& schedule_target : graph.nodes()) {
      Node* target = schedule_target.node;
      VLOG(4) << absl::StrFormat("  distances to %s:", target->GetName());
      for (const ScheduleNode& schedule_source : graph.nodes()) {
        Node* source = schedule_source.node;
        VLOG(4) << absl::StrFormat(
            "    %s -> %s : %s", source->GetName(), target->GetName(),
            distances_to_node.at(target).contains(source)
                ? absl::StrCat(distances_to_node.at(target).at(source))
                : "(none)");
      }
    }
  }

  return distances_to_node;
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
ComputeCombinationalDelayConstraints(
    const ScheduleGraph& graph, int64_t clock_period_ps,
    const absl::flat_hash_map<Node*, absl::flat_hash_map<Node*, int64_t>>&
        distances_to_node,
    const DelayMap& delay_map) {
  absl::flat_hash_map<Node*, std::vector<Node*>> result;
  result.reserve(graph.nodes().size());
  for (const ScheduleNode& a : graph.nodes()) {
    result[a.node];
  }

  for (const ScheduleNode& schedule_node : graph.nodes()) {
    Node* node = schedule_node.node;
    const int64_t node_delay = delay_map.at(node);

    // For each ancestor `a`, check whether the critical-path length from `a`'s
    // start to `node`'s end crosses a `clock_period_ps` boundary due to
    // `node`'s delay. If so, we need a constraint to ensure that `node` is in a
    // later stage than `a`.
    //
    // NOTE: The order in which we iterate over the ancestors `a` here does not
    // matter. As long as our iteration over `node` is deterministic, we will
    // push the same sequence of `node`s into each `result[a]` every time.
    for (auto [a, distance] : distances_to_node.at(node)) {
      if (distance > clock_period_ps &&
          distance - node_delay <= clock_period_ps) {
        result.at(a).push_back(node);
      }
    }
  }

  if (VLOG_IS_ON(4)) {
    VLOG(4) << absl::StrFormat("Constraints (clock period: %dps):",
                               clock_period_ps);
    for (const ScheduleNode& schedule_node : graph.nodes()) {
      Node* node = schedule_node.node;
      VLOG(4) << absl::StrFormat("  %s: [%s]", node->GetName(),
                                 absl::StrJoin(result.at(node), ", "));
    }
  }
  return result;
}

}  // namespace

SDCSchedulingModel::SDCSchedulingModel(
    ScheduleGraph graph, const DelayMap& delay_map,
    std::optional<int64_t> initiation_interval)
    : graph_(std::move(graph)),
      model_(absl::StrCat("sdc_model:", graph_.name())),
      delay_map_(delay_map),
      initiation_interval_(initiation_interval),
      last_stage_(model_.AddContinuousVariable(0.0, kMaxStages, "last_stage")),
      cycle_at_sinknode_(model_.AddContinuousVariable(-kInfinity, kInfinity,
                                                      "cycle_at_sinknode")) {
  // when subclassed for Iterative SDC, delay_map_ and distances_to_node_
  // are not used.
  if (!delay_map_.empty()) {
    distances_to_node_ = ComputeDistancesToNodes(graph_, delay_map_);
  }

  auto get_model_node_name = [this](Node* node) {
    return graph_.IsFunctionBaseScoped()
               ? node->GetName()
               : absl::StrCat(node->function_base()->name(),
                              "::", node->GetName());
  };

  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    if (IsUntimed(node)) {
      continue;
    }
    std::string model_node_name = get_model_node_name(node);
    cycle_var_.emplace(
        node, model_.AddContinuousVariable(0.0, kMaxStages, model_node_name));
    model_.AddLinearConstraint(
        cycle_var_.at(node) <= last_stage_,
        absl::StrFormat("pipeline_length:%s", model_node_name));
    lifetime_var_.emplace(
        node,
        model_.AddContinuousVariable(
            0.0, kInfinity, absl::StrFormat("lifetime_%s", model_node_name)));
    if (node->Is<Next>()) {
      unwanted_inverse_throughput_var_.emplace(
          node, model_.AddContinuousVariable(
                    0.0, kInfinity,
                    absl::StrFormat("unwanted_inverse_throughput_%s",
                                    model_node_name)));
    }

    if (schedule_node.schedule_in_first_stage) {
      model_.AddLinearConstraint(
          cycle_var_.at(node) <= 0,
          absl::StrCat("in_first_stage:", model_node_name));
    }
    if (schedule_node.schedule_in_last_stage) {
      CHECK(!schedule_node.schedule_in_first_stage);
      model_.AddLinearConstraint(
          cycle_var_.at(node) >= last_stage_,
          absl::StrCat("in_last_stage:", model_node_name));
    }
  }
}

absl::Status SDCSchedulingModel::AddAllDefUseConstraints() {
  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    if (IsUntimed(node)) {
      continue;
    }
    for (Node* successor : schedule_node.successors) {
      XLS_RETURN_IF_ERROR(AddDefUseConstraints(node, successor));
    }
    if (schedule_node.is_live_out) {
      XLS_RETURN_IF_ERROR(AddDefUseConstraints(node, std::nullopt));
    }
  }
  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddDefUseConstraints(
    Node* node, std::optional<Node*> user) {
  // Nodes must be scheduled no later than their users.
  XLS_RETURN_IF_ERROR(AddCausalConstraint(node, user));

  if (node->Is<StateRead>() && user.has_value() && user.value()->Is<Next>()) {
    Next* next = user.value()->As<Next>();
    if (next->state_read() == node) {
      XLS_RETURN_IF_ERROR(AddThroughputConstraint(node->As<StateRead>(), next));
    }
    if (next->value() != node && next->predicate() != node) {
      XLS_RET_CHECK_EQ(next->state_read(), node);
      // We don't need to keep the param's value alive to this user, so no need
      // for a lifetime constraint.
      return absl::OkStatus();
    }
  }

  // If the user is dead after synthesis, we don't count its contribution to the
  // lifetime, assuming the synthesis tool will be able to strip any pipeline
  // registers used to persist the value.
  if (user.has_value() &&
      graph_.GetScheduleNode(*user).is_dead_after_synthesis) {
    return absl::OkStatus();
  }
  return AddLifetimeConstraint(node, user);
}

absl::Status SDCSchedulingModel::AddCausalConstraint(
    Node* node, std::optional<Node*> user) {
  math_opt::Variable cycle_at_node = cycle_var_.at(node);
  math_opt::Variable cycle_at_user =
      user.has_value() ? cycle_var_.at(user.value()) : cycle_at_sinknode_;

  std::string_view user_name = "<cycle_at_sinknode>";
  if (user.has_value()) {
    user_name = (*user)->GetNameView();
  }
  XLS_RET_CHECK_EQ(cycle_at_node.storage(), cycle_at_user.storage())
      << *node << " " << user_name;
  if (cycle_at_user == cycle_at_node) {
    return absl::OkStatus();
  }

  // Explicit delay nodes must lag their inputs by a certain number of cycles.
  int64_t min_delay = 0;
  if (user.has_value() && user.value()->Is<MinDelay>()) {
    min_delay = user.value()->As<MinDelay>()->delay();
  }

  std::string user_str = user.has_value() ? user.value()->GetName() : "«sink»";

  model_.AddLinearConstraint(
      cycle_at_user - cycle_at_node >= static_cast<double>(min_delay),
      absl::StrFormat("causal_%s_%s", node->GetName(), user_str));
  VLOG(2) << "Setting causal constraint: "
          << absl::StrFormat("cycle[%s] - cycle[%s] ≥ %d", user_str,
                             node->GetName(), min_delay);

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddLifetimeConstraint(
    Node* node, std::optional<Node*> user) {
  if (IsUntimed(node)) {
    return absl::OkStatus();
  }
  math_opt::Variable cycle_at_node = cycle_var_.at(node);
  math_opt::Variable lifetime_at_node = lifetime_var_.at(node);
  math_opt::Variable cycle_at_user =
      user.has_value() ? cycle_var_.at(user.value()) : cycle_at_sinknode_;

  if (cycle_at_user == cycle_at_node) {
    return absl::OkStatus();
  }

  std::string user_str = user.has_value() ? user.value()->GetName() : "«sink»";

  model_.AddLinearConstraint(
      lifetime_at_node + cycle_at_node - cycle_at_user >= 0,
      absl::StrFormat("lifetime_%s_%s", node->GetName(), user_str));
  VLOG(2) << "Setting lifetime constraint: "
          << absl::StrFormat("lifetime[%s] + cycle[%s] - cycle[%s] ≥ 0",
                             node->GetName(), node->GetName(), user_str);

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddThroughputConstraint(StateRead* state_read,
                                                         Next* next_value) {
  XLS_RET_CHECK(graph_.IncludesProc());

  StateElement* state_element = state_read->state_element();

  math_opt::Variable cycle_at_param = cycle_var_.at(state_read);
  math_opt::Variable cycle_at_next = cycle_var_.at(next_value);
  math_opt::Variable unwanted_inverse_throughput_at_next =
      unwanted_inverse_throughput_var_.at(next_value);

  model_.AddLinearConstraint(
      // TODO: https://github.com/google/xls/issues/2071 - incorporate target
      // throughput in the RHS of this constraint.
      unwanted_inverse_throughput_at_next + cycle_at_param - cycle_at_next >= 0,
      absl::StrFormat("throughput_%s_%s", state_read->GetName(),
                      next_value->GetName()));
  VLOG(2) << "Setting throughput constraint: "
          << absl::StreamFormat(
                 "unwanted_inverse_throughput[%s] + cycle[%s] - cycle[%s] ≥ 0",
                 state_element->name(), state_read->GetName(),
                 next_value->GetName());

  return absl::OkStatus();
}

// This ensures that state backedges don't span more than II cycles, which is
// necessary while enforcing a target II.
absl::Status SDCSchedulingModel::AddBackedgeConstraints(
    const BackedgeConstraint& constraint) {
  const int64_t II = initiation_interval_.value_or(1);

  for (const ScheduleBackedge& backedge : graph_.backedges()) {
    if (!backedge.distance.has_value() ||
        !std::holds_alternative<LessThanInitiationInterval>(
            *backedge.distance)) {
      return absl::UnimplementedError(
          "Unsupported backedge type in SDC schedule");
    }
    if (II <= 0) {
      // Distance of backedge is constrained by the II, but the worst-case
      // throughput constraint is not set.
      continue;
    }

    VLOG(2) << "Setting backedge constraint (II): "
            << absl::StrFormat("cycle[%s] - cycle[%s] < %d",
                               backedge.source->GetName(),
                               backedge.destination->GetName(), II);
    backedge_constraint_.emplace(
        std::make_pair(backedge.destination, backedge.source),
        DiffLessThanConstraint(backedge.source, backedge.destination, II,
                               "backedge"));
  }

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddSchedulingConstraint(
    const SchedulingConstraint& constraint) {
  if (std::holds_alternative<BackedgeConstraint>(constraint)) {
    return AddBackedgeConstraints(std::get<BackedgeConstraint>(constraint));
  }
  if (std::holds_alternative<IOConstraint>(constraint)) {
    return AddIOConstraint(std::get<IOConstraint>(constraint));
  }
  if (std::holds_alternative<NodeInCycleConstraint>(constraint)) {
    return AddNodeInCycleConstraint(
        std::get<NodeInCycleConstraint>(constraint));
  }
  if (std::holds_alternative<DifferenceConstraint>(constraint)) {
    return AddDifferenceConstraint(std::get<DifferenceConstraint>(constraint));
  }
  if (std::holds_alternative<RecvsFirstSendsLastConstraint>(constraint)) {
    return AddRFSLConstraint(
        std::get<RecvsFirstSendsLastConstraint>(constraint));
  }
  if (std::holds_alternative<SendThenRecvConstraint>(constraint)) {
    return AddSendThenRecvConstraint(
        std::get<SendThenRecvConstraint>(constraint));
  }
  if (std::holds_alternative<SameChannelConstraint>(constraint)) {
    return AddSameChannelConstraint(
        std::get<SameChannelConstraint>(constraint));
  }
  return absl::InternalError("Unhandled scheduling constraint type");
}

absl::Status SDCSchedulingModel::AddIOConstraint(
    const IOConstraint& constraint) {
  // Map from channel name to set of nodes that send/receive on that channel.
  absl::flat_hash_map<std::string, std::vector<Node*>> channel_to_nodes;
  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    if (node->Is<Receive>() || node->Is<Send>()) {
      channel_to_nodes[node->As<ChannelNode>()->channel_name()].push_back(node);
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

      VLOG(2) << "Setting IO constraint: "
              << absl::StrFormat("%d ≤ cycle[%s] - cycle[%s] ≤ %d",
                                 constraint.MinimumLatency(), target->GetName(),
                                 source->GetName(),
                                 constraint.MaximumLatency());
      io_constraints_[constraint].push_back({
          .lower = DiffAtLeastConstraint(target, source,
                                         constraint.MinimumLatency(), "io"),
          .upper = DiffAtMostConstraint(target, source,
                                        constraint.MaximumLatency(), "io"),
      });
    }
  }

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddNodeInCycleConstraint(
    const NodeInCycleConstraint& constraint) {
  Node* node = constraint.GetNode();
  int64_t cycle = constraint.GetCycle();

  model_.AddLinearConstraint(cycle_var_.at(node) == static_cast<double>(cycle),
                             absl::StrFormat("nic_%s", node->GetName()));
  VLOG(2) << "Setting node-in-cycle constraint: "
          << absl::StrFormat("cycle[%s] = %d", node->GetName(), cycle);

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddDifferenceConstraint(
    const DifferenceConstraint& constraint) {
  Node* a = constraint.GetA();
  Node* b = constraint.GetB();
  int64_t max_difference = constraint.GetMaxDifference();
  DiffAtMostConstraint(a, b, max_difference, "diff");

  VLOG(2) << "Setting difference constraint: "
          << absl::StrFormat("cycle[%s] - cycle[%s] ≤ %d", a->GetName(),
                             b->GetName(), max_difference);

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddRFSLConstraint(
    const RecvsFirstSendsLastConstraint& constraint) {
  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    if (node->Is<Receive>()) {
      VLOG(2) << "Setting receive-in-first-cycle constraint: "
              << absl::StrFormat("cycle[%s] ≤ 0", node->GetName());
      model_.AddLinearConstraint(cycle_var_.at(node) <= 0,
                                 absl::StrFormat("recv_%s", node->GetName()));
    } else if (node->Is<Send>()) {
      VLOG(2) << "Setting send-in-last-cycle constraint: "
              << absl::StrFormat("%s ≤ cycle[%s]", last_stage_.name(),
                                 node->GetName());
      model_.AddLinearConstraint(cycle_var_.at(node) >= last_stage_,
                                 absl::StrFormat("send_%s", node->GetName()));
    }
  }

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddSendThenRecvConstraint(
    const SendThenRecvConstraint& constraint) {
  CHECK_GE(constraint.MinimumLatency(), 0);
  if (constraint.MinimumLatency() == 0) {
    return absl::OkStatus();
  }

  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* recv = schedule_node.node;
    if (!recv->Is<Receive>()) {
      continue;
    }

    // Look for a Send dependency.
    //
    // Technically, we probably don't need to trace back through the predicate
    // operand; the only operation we have today that takes a token and returns
    // data is a Receive (and technically tuple construction, but that just gets
    // weird), so we'd end up terminating our search before reaching a Send
    // anyway. But - just in case we ever add some other operation, we'll trace
    // up both paths to be sure.
    std::vector<Node*> stack(recv->operands().begin(), recv->operands().end());
    absl::flat_hash_set<Node*> seen;
    while (!stack.empty()) {
      Node* node = stack.back();
      stack.pop_back();
      if (seen.contains(node)) {
        continue;
      }
      seen.insert(node);

      if (node->Is<Send>()) {
        // Ensure that this send happens before the receive that depends on it.
        DiffAtLeastConstraint(recv, node, constraint.MinimumLatency(),
                              "send_then_recv");
        // We don't need to trace any further back on this line, since any
        // earlier sends are transitively handled.
        continue;
      }
      if (node->Is<Receive>()) {
        // No need to trace any further back on this line; this node will also
        // be the root of a search, and will get a similar appropriate delay.
        continue;
      }
      stack.insert(stack.end(), node->operands().begin(),
                   node->operands().end());
    }
  }
  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::AddSameChannelConstraint(
    const SameChannelConstraint& constraint) {
  CHECK_GE(constraint.MinimumLatency(), 0);
  if (constraint.MinimumLatency() == 0) {
    return absl::OkStatus();
  }

  // Collects all operations on each channel in topological order; used to order
  // arbitrary-static-order channels.
  absl::flat_hash_map<ChannelRef, std::vector<Node*>> ops_on_channel;
  // Used for all other channels.
  absl::flat_hash_map<
      Node*, absl::flat_hash_map<ChannelRef,
                                 absl::btree_set<Node*, Node::NodeIdLessThan>>>
      node_io_dependencies;
  node_io_dependencies.reserve(graph_.nodes().size());
  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    absl::flat_hash_map<ChannelRef,
                        absl::btree_set<Node*, Node::NodeIdLessThan>>&
        io_dependencies = node_io_dependencies[node];
    for (Node* operand : schedule_node.predecessors) {
      auto it = node_io_dependencies.find(operand);
      if (it == node_io_dependencies.end()) {
        continue;
      }
      for (const auto& [channel, nodes] : it->second) {
        io_dependencies[channel].insert(nodes.begin(), nodes.end());
      }
    }

    if (!node->Is<ChannelNode>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(ChannelRef channel,
                         node->As<ChannelNode>()->GetChannelRef());
    std::optional<ChannelStrictness> strictness = ChannelRefStrictness(channel);

    if (strictness == ChannelStrictness::kArbitraryStaticOrder) {
      // Find the most recent operation (of the same type) on this channel.
      auto it = std::find_if(ops_on_channel[channel].rbegin(),
                             ops_on_channel[channel].rend(),
                             [op = node->op()](Node* predecessor) {
                               return predecessor->op() == op;
                             });
      if (it != ops_on_channel[channel].rend()) {
        DiffAtLeastConstraint(node, *it, constraint.MinimumLatency(),
                              "same_channel_arbitrary_order");
      }
      ops_on_channel[channel].push_back(node);
    } else {
      ops_on_channel[channel].push_back(node);

      absl::btree_set<Node*, Node::NodeIdLessThan>& dependencies =
          io_dependencies[channel];
      if (dependencies.empty()) {
        dependencies.insert(schedule_node.node);
        continue;
      }

      // This operation has dependencies on other operations on the same
      // channel. Ensure that this operation is scheduled at least
      // MinimumLatency cycles after any dependencies of the same type, at which
      // point we have accounted for the delay with respect to them; further
      // operations in the chain only need to delay with respect to this node.
      absl::btree_set<Node*, Node::NodeIdLessThan> new_dependencies;
      for (Node* dependency : dependencies) {
        if (dependency->op() == schedule_node.node->op()) {
          DiffAtLeastConstraint(schedule_node.node, dependency,
                                constraint.MinimumLatency(),
                                "same_channel_dependency");
        } else {
          new_dependencies.insert(dependency);
        }
      }
      new_dependencies.insert(schedule_node.node);
      dependencies = std::move(new_dependencies);
    }
  }
  return absl::OkStatus();
}

void SDCSchedulingModel::SetObjective(std::optional<double> throughput_weight) {
  math_opt::LinearExpression objective;
  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    if (IsUntimed(node)) {
      continue;
    }
    // Maximize throughput at the user-specified weight.
    if (auto it = unwanted_inverse_throughput_var_.find(node);
        it != unwanted_inverse_throughput_var_.end() &&
        throughput_weight.value_or(0.0) != 0.0) {
      CHECK(node->Is<Next>());
      int64_t num_paths =
          node->function_base()
              ->next_values(node->As<Next>()->state_read()->As<StateRead>())
              .size();
      objective += (kObjectiveScaling * *throughput_weight / num_paths) *
                   unwanted_inverse_throughput_var_.at(node);
    }
    // This acts as a tie-breaker for under-constrained problems, favoring ASAP
    // schedules.
    objective += cycle_var_.at(node);
    // Minimize node lifetimes.
    // The scaling makes the tie-breaker small in comparison, and is a power
    // of two so that there's no imprecision (just add to exponent).
    objective += kObjectiveScaling *
                 static_cast<double>(node->GetType()->GetFlatBitCount()) *
                 lifetime_var_.at(node);
  }
  model_.Minimize(objective);
}

void SDCSchedulingModel::RemoveObjective() { model_.Minimize(0.0); }

absl::StatusOr<ScheduleCycleMap> SDCSchedulingModel::ExtractResult(
    const math_opt::VariableMap<double>& variable_values) const {
  ScheduleCycleMap cycle_map;
  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* node = schedule_node.node;
    if (IsUntimed(node)) {
      continue;
    }
    double cycle = variable_values.at(cycle_var_.at(node));
    if (std::fabs(cycle - std::round(cycle)) > 0.001) {
      return absl::InternalError(
          "The scheduling result is expected to be integer");
    }
    cycle_map[node] = std::round(cycle);
  }
  return cycle_map;
}

void SDCSchedulingModel::SetClockPeriod(int64_t clock_period_ps) {
  absl::flat_hash_map<Node*, std::vector<Node*>> prev_delay_constraints =
      std::move(delay_constraints_);
  delay_constraints_ = ComputeCombinationalDelayConstraints(
      graph_, clock_period_ps, distances_to_node_, delay_map_);

  for (const ScheduleNode& schedule_node : graph_.nodes()) {
    Node* source = schedule_node.node;
    if (IsUntimed(source)) {
      continue;
    }
    if (!prev_delay_constraints.empty()) {
      // Check over all the prior constraints, dropping any that are obsolete.
      absl::flat_hash_set<Node*> new_targets(
          delay_constraints_.at(source).begin(),
          delay_constraints_.at(source).end());
      for (Node* target : prev_delay_constraints.at(source)) {
        if (new_targets.contains(target)) {
          continue;
        }

        // No longer related; remove constraint.
        auto it = timing_constraint_.find(std::make_pair(source, target));
        model_.DeleteLinearConstraint(it->second);
        timing_constraint_.erase(it);
      }
    }

    // Add all new constraints, avoiding duplicates for any that already exist.
    for (Node* target : delay_constraints_.at(source)) {
      auto key = std::make_pair(source, target);
      if (timing_constraint_.contains(key)) {
        continue;
      }

      // Newly related; add constraint.
      VLOG(2) << "Setting timing constraint: "
              << absl::StrFormat("1 ≤ %s - %s", target->GetName(),
                                 source->GetName());
      timing_constraint_.emplace(
          key, DiffAtLeastConstraint(target, source, 1, "timing"));
    }
  }
}

absl::Status SDCSchedulingModel::SetWorstCaseThroughput(
    int64_t worst_case_throughput) {
  if (!graph_.IsSingleProc()) {
    return absl::UnimplementedError(
        "SetWorstCaseThroughput only supports procs, since it controls state "
        "backedges");
  }
  if (initiation_interval_.value_or(1) == worst_case_throughput) {
    return absl::OkStatus();
  }
  initiation_interval_ = worst_case_throughput;
  for (auto& [nodes, constraint] : backedge_constraint_) {
    model_.DeleteLinearConstraint(constraint);
  }
  backedge_constraint_.clear();
  return AddBackedgeConstraints(BackedgeConstraint());
}

void SDCSchedulingModel::SetPipelineLength(
    std::optional<int64_t> pipeline_length) {
  if (pipeline_length.has_value()) {
    model_.set_lower_bound(last_stage_,
                           static_cast<double>(*pipeline_length - 1));
    model_.set_upper_bound(last_stage_,
                           static_cast<double>(*pipeline_length - 1));
  } else {
    model_.set_lower_bound(last_stage_, 0.0);
    model_.set_upper_bound(last_stage_, kInfinity);
  }
}

void SDCSchedulingModel::MinimizePipelineLength() {
  model_.Minimize(last_stage_);
}

absl::StatusOr<int64_t> SDCSchedulingModel::ExtractPipelineLength(
    const operations_research::math_opt::VariableMap<double>& variable_values)
    const {
  double last_stage = variable_values.at(last_stage_);
  if (std::fabs(last_stage - std::round(last_stage)) > 0.001) {
    return absl::InternalError(absl::StrCat(
        "The optimum pipeline length is expected to be an integer, was: ",
        last_stage + 1.0));
  }
  return static_cast<int64_t>(std::round(last_stage)) + 1;
}

absl::Status SDCSchedulingModel::AddSlackVariables(
    std::optional<double> infeasible_per_state_backedge_slack_pool) {
  if (infeasible_per_state_backedge_slack_pool.has_value()) {
    XLS_RET_CHECK_GT(*infeasible_per_state_backedge_slack_pool, 0)
        << "infeasible_per_state_backedge_slack_pool must be positive";
  }
  // Add slack variables to all relevant constraints.

  // Remove any pre-existing objective, and declare that we'll be minimizing our
  // new objective.
  model_.Minimize(0);

  // First, try to minimize the depth of the pipeline. We assume users are most
  // willing to relax this; i.e., they care about throughput more than latency.
  if (last_stage_.upper_bound() < kInfinity) {
    auto [last_stage_slack, last_stage_ub] = AddUpperBoundSlack(last_stage_);
    model_.AddToObjective(last_stage_slack);
    last_stage_slack_ = last_stage_slack;
  }

  // Next, relax the state back-edge length restriction (if present). We assume
  // users are reasonably willing to relax this; i.e., they care about
  // throughput, but they care more about the I/O constraints they've specified.
  if (!backedge_constraint_.empty()) {
    double backedge_slack_objective_scale = static_cast<double>(1 << 10);
    shared_backedge_slack_ = model_.AddVariable(
        0.0, kInfinity, /*is_integer=*/false, "backedge_slack");
    model_.AddToObjective(
        (backedge_slack_objective_scale * shared_backedge_slack_.value()));
    double node_to_node_slack_objective_scale = 0.0;
    if (infeasible_per_state_backedge_slack_pool.has_value()) {
      node_to_node_slack_objective_scale =
          (backedge_slack_objective_scale /
           infeasible_per_state_backedge_slack_pool.value()) *
          // Make slightly larger to break ties with shared backedge slack.
          (1 + 1e-6);
    }
    for (auto& [nodes, constraint] : backedge_constraint_) {
      AddUpperBoundSlack(constraint, shared_backedge_slack_);
      if (infeasible_per_state_backedge_slack_pool.has_value()) {
        auto [itr, inserted] = node_backedge_slack_.try_emplace(
            nodes,
            model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                               absl::StrFormat("%v_to_%v_backedge_slack",
                                               *nodes.first, *nodes.second)));
        XLS_RET_CHECK(inserted);
        operations_research::math_opt::Variable& node_to_node_slack =
            itr->second;
        model_.AddToObjective(node_to_node_slack_objective_scale *
                              node_to_node_slack);
        AddUpperBoundSlack(constraint, node_to_node_slack);
      }
    }
  }

  // Finally, relax the I/O constraints, if nothing else works.
  for (auto& [io_constraint, constraints] : io_constraints_) {
    math_opt::Variable min_slack = model_.AddVariable(
        0, kInfinity, /*is_integer=*/false,
        absl::StrCat("io_min_", io_constraint.SourceChannel(), "→",
                     io_constraint.TargetChannel(), "_slack"));
    math_opt::Variable max_slack = model_.AddVariable(
        0, kInfinity, /*is_integer=*/false,
        absl::StrCat("io_max_", io_constraint.SourceChannel(), "→",
                     io_constraint.TargetChannel(), "_slack"));
    model_.AddToObjective((1 << 20) * min_slack);
    model_.AddToObjective((1 << 20) * max_slack);
    io_slack_.emplace(io_constraint, SlackPair{
                                         .min = min_slack,
                                         .max = max_slack,
                                     });

    for (auto& [min_constraint, max_constraint] : constraints) {
      AddLowerBoundSlack(min_constraint, min_slack);
      AddUpperBoundSlack(max_constraint, max_slack);
    }
  }

  return absl::OkStatus();
}

absl::Status SDCSchedulingModel::ExtractError(
    const math_opt::VariableMap<double>& variable_values) const {
  std::vector<std::string> problems;
  std::vector<std::string> suggestions;
  if (last_stage_slack_.has_value()) {
    double last_stage_slack = variable_values.at(*last_stage_slack_);
    if (last_stage_slack > 0.001) {
      int64_t new_pipeline_length =
          static_cast<int64_t>(std::round(variable_values.at(last_stage_))) + 1;
      problems.push_back("the specified pipeline length");
      suggestions.push_back(
          absl::StrCat("`--pipeline_stages=", new_pipeline_length, "`"));
    }
  }
  if (shared_backedge_slack_.has_value()) {
    double backedge_slack = variable_values.at(*shared_backedge_slack_);
    if (backedge_slack > 0.001) {
      int64_t new_backedge_length =
          initiation_interval_.value_or(1) +
          static_cast<int64_t>(std::round(backedge_slack));
      if (initiation_interval_.value_or(1) == 1) {
        problems.push_back("full throughput");
      } else {
        problems.push_back("the specified throughput");
      }
      suggestions.push_back(
          absl::StrCat("`--worst_case_throughput=", new_backedge_length, "`"));
    }
    for (const auto& [nodes, node_backedge_var] : node_backedge_slack_) {
      double node_backedge = variable_values.at(node_backedge_var);
      if (node_backedge > 0.001) {
        if (problems.back() != "full throughput") {
          problems.push_back("full throughput");
        }
        suggestions.push_back(absl::StrFormat(
            "looking at paths between %v and %v (needs %d additional slack)",
            *nodes.first, *nodes.second,
            static_cast<int64_t>(std::round(node_backedge))));
      }
    }
  }
  if (!problems.empty()) {
    if (problems.size() == 1 || problems.size() == 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          graph_.name(), ": cannot achieve ", absl::StrJoin(problems, " or "),
          ". Try ", absl::StrJoin(suggestions, " and ")));
    }
    return absl::InvalidArgumentError(absl::StrCat(
        graph_.name(), ": cannot achieve ",
        absl::StrJoin(
            absl::MakeConstSpan(problems).subspan(0, problems.size() - 1),
            ", "),
        " or ", problems[problems.size() - 1], ". Try ",
        absl::StrJoin(
            absl::MakeConstSpan(suggestions).subspan(0, suggestions.size() - 1),
            ", "),
        " and ", suggestions[suggestions.size() - 1]));
  }

  std::vector<std::string> io_problems;
  for (auto& [io_constraint, slacks] : io_slack_) {
    double min_slack = variable_values.at(slacks.min);
    double max_slack = variable_values.at(slacks.max);

    std::vector<std::string> latency_suggestions;
    if (min_slack > 0.001) {
      int64_t new_min_latency = io_constraint.MinimumLatency() -
                                static_cast<int64_t>(std::round(min_slack));
      latency_suggestions.push_back(
          absl::StrCat("minimum latency ≤ ", new_min_latency));
    }
    if (max_slack > 0.001) {
      int64_t new_max_latency = io_constraint.MaximumLatency() +
                                static_cast<int64_t>(std::round(max_slack));
      latency_suggestions.push_back(
          absl::StrCat("maximum latency ≥ ", new_max_latency));
    }

    if (latency_suggestions.empty()) {
      continue;
    }
    io_problems.push_back(absl::StrCat(
        io_constraint.SourceChannel(), "→", io_constraint.TargetChannel(),
        " with ", absl::StrJoin(latency_suggestions, ", ")));
  }
  if (!io_problems.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        graph_.name(),
        ": cannot satisfy the given I/O constraints. Would succeed with: ",
        absl::StrJoin(io_problems, ", ",
                      [](std::string* out, const std::string& entry) {
                        absl::StrAppend(out, "{", entry, "}");
                      })));
  }

  return absl::UnknownError(
      absl::StrCat("reason unknown for ", graph_.name(), "."));
}

math_opt::LinearConstraint SDCSchedulingModel::DiffAtMostConstraint(
    Node* x, Node* y, int64_t limit, std::string_view name) {
  return model_.AddLinearConstraint(
      cycle_var_.at(x) - cycle_var_.at(y) <= static_cast<double>(limit),
      absl::StrFormat("%s:%s-%s≤%d", name, x->GetName(), y->GetName(), limit));
}

math_opt::LinearConstraint SDCSchedulingModel::DiffLessThanConstraint(
    Node* x, Node* y, int64_t limit, std::string_view name) {
  return model_.AddLinearConstraint(
      cycle_var_.at(x) - cycle_var_.at(y) <= static_cast<double>(limit - 1),
      absl::StrFormat("%s:%s-%s<%d", name, x->GetName(), y->GetName(), limit));
}

math_opt::LinearConstraint SDCSchedulingModel::DiffAtLeastConstraint(
    Node* x, Node* y, int64_t limit, std::string_view name) {
  CHECK(!IsUntimed(y));
  return model_.AddLinearConstraint(
      cycle_var_.at(x) - cycle_var_.at(y) >= static_cast<double>(limit),
      absl::StrFormat("%s:%s-%s≥%d", name, x->GetName(), y->GetName(), limit));
}

math_opt::LinearConstraint SDCSchedulingModel::DiffGreaterThanConstraint(
    Node* x, Node* y, int64_t limit, std::string_view name) {
  return model_.AddLinearConstraint(
      cycle_var_.at(x) - cycle_var_.at(y) >= static_cast<double>(limit + 1),
      absl::StrFormat("%s:%s-%s≥%d", name, x->GetName(), y->GetName(), limit));
}

math_opt::LinearConstraint SDCSchedulingModel::DiffEqualsConstraint(
    Node* x, Node* y, int64_t diff, std::string_view name) {
  if (x == y) {
    LOG(FATAL) << "DiffEqualsConstraint: " << x->GetName() << " - "
               << y->GetName() << " = " << diff << " is unsatisfiable";
  }
  return model_.AddLinearConstraint(
      cycle_var_.at(x) - cycle_var_.at(y) == static_cast<double>(diff),
      absl::StrFormat("%s:%s-%s=%d", name, x->GetName(), y->GetName(), diff));
}

math_opt::Variable SDCSchedulingModel::AddUpperBoundSlack(
    math_opt::LinearConstraint c, std::optional<math_opt::Variable> slack) {
  CHECK_LT(c.upper_bound(), kInfinity)
      << "The constraint " << c.name() << " has no upper bound.";
  if (slack.has_value()) {
    CHECK_EQ(c.coefficient(*slack), 0.0)
        << "The slack variable " << slack->name()
        << " is already referenced in the constraint " << c.name() << ".";
  } else {
    slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                               absl::StrCat(c.name(), "_ub_slack"));
  }
  model_.set_coefficient(c, *slack, -1.0);
  return *slack;
}

absl::Status SDCSchedulingModel::RemoveUpperBoundSlack(
    math_opt::Variable v, math_opt::LinearConstraint upper_bound_with_slack,
    math_opt::Variable slack) {
  XLS_RET_CHECK_EQ(upper_bound_with_slack.coefficient(v), 1.0);
  XLS_RET_CHECK_EQ(upper_bound_with_slack.coefficient(slack), -1.0);
  model_.set_upper_bound(v, upper_bound_with_slack.upper_bound());
  model_.DeleteLinearConstraint(upper_bound_with_slack);
  return absl::OkStatus();
}

math_opt::Variable SDCSchedulingModel::AddLowerBoundSlack(
    math_opt::LinearConstraint c, std::optional<math_opt::Variable> slack) {
  CHECK_GT(c.lower_bound(), -kInfinity)
      << "The constraint " << c.name() << " has no lower bound.";
  if (slack.has_value()) {
    CHECK_EQ(c.coefficient(*slack), 0.0)
        << "The slack variable " << slack->name()
        << " is already referenced in the constraint " << c.name() << ".";
  } else {
    slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                               absl::StrCat(c.name(), "_lb_slack"));
  }
  model_.set_coefficient(c, *slack, 1.0);
  return *slack;
}

std::pair<math_opt::Variable, math_opt::LinearConstraint>
SDCSchedulingModel::AddUpperBoundSlack(
    math_opt::Variable v, std::optional<math_opt::Variable> slack) {
  CHECK_LT(v.upper_bound(), kInfinity)
      << "The variable " << v.name() << " has no fixed upper bound.";
  if (!slack.has_value()) {
    slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                               absl::StrCat(v.name(), "_ub_slack"));
  }
  math_opt::LinearConstraint upper_bound = model_.AddLinearConstraint(
      v - *slack <= v.upper_bound(), absl::StrCat(v.name(), "_ub"));
  model_.set_upper_bound(v, kInfinity);
  return {*slack, upper_bound};
}

std::pair<math_opt::Variable, math_opt::LinearConstraint>
SDCSchedulingModel::AddLowerBoundSlack(
    math_opt::Variable v, std::optional<math_opt::Variable> slack) {
  CHECK_GT(v.lower_bound(), -kInfinity)
      << "The variable " << v.name() << " has no fixed lower bound.";
  if (!slack.has_value()) {
    slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                               absl::StrCat(v.name(), "_lb_slack"));
  }
  math_opt::LinearConstraint lower_bound = model_.AddLinearConstraint(
      v + *slack >= v.lower_bound(), absl::StrCat(v.name(), "_lb"));
  model_.set_lower_bound(v, -kInfinity);
  return {*slack, lower_bound};
}

absl::StatusOr<std::unique_ptr<SDCScheduler>> SDCScheduler::Create(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  absl::flat_hash_set<Node*> dead_after_synthesis =
      GetDeadAfterSynthesisNodes(f);
  ScheduleGraph graph = ScheduleGraph::Create(f, dead_after_synthesis);
  XLS_ASSIGN_OR_RETURN(DelayMap delay_map,
                       ComputeNodeDelays(graph, delay_estimator));
  std::optional<int64_t> initiation_interval =
      f->IsProc() ? std::optional<int64_t>(
                        f->AsProcOrDie()->GetInitiationInterval().value_or(1))
                  : std::nullopt;
  std::unique_ptr<SDCScheduler> scheduler(new SDCScheduler(
      std::move(graph), initiation_interval, std::move(delay_map)));
  XLS_RETURN_IF_ERROR(scheduler->Initialize());
  return std::move(scheduler);
}

absl::StatusOr<std::unique_ptr<SDCScheduler>> SDCScheduler::Create(
    ScheduleGraph graph, const DelayEstimator& delay_estimator) {
  XLS_ASSIGN_OR_RETURN(DelayMap delay_map,
                       ComputeNodeDelays(graph, delay_estimator));
  std::unique_ptr<SDCScheduler> scheduler(
      new SDCScheduler(std::move(graph), std::nullopt, std::move(delay_map)));
  XLS_RETURN_IF_ERROR(scheduler->Initialize());
  return std::move(scheduler);
}

SDCScheduler::SDCScheduler(ScheduleGraph graph,
                           std::optional<int64_t> initiation_interval,
                           DelayMap&& delay_map)
    : delay_map_(std::move(delay_map)),
      model_(std::move(graph), delay_map_, initiation_interval) {}

absl::Status SDCScheduler::Initialize() {
  XLS_ASSIGN_OR_RETURN(
      solver_, math_opt::NewIncrementalSolver(&model_.UnderlyingModel(),
                                              math_opt::SolverType::kGlop));
  XLS_RETURN_IF_ERROR(model_.AddAllDefUseConstraints());
  return absl::OkStatus();
}

absl::Status SDCScheduler::AddConstraints(
    absl::Span<const SchedulingConstraint> constraints) {
  for (const SchedulingConstraint& constraint : constraints) {
    XLS_RETURN_IF_ERROR(model_.AddSchedulingConstraint(constraint));
  }
  return absl::OkStatus();
}

absl::Status SDCScheduler::BuildError(
    const math_opt::SolveResult& result,
    SchedulingFailureBehavior failure_behavior) {
  CHECK_NE(result.termination.reason, math_opt::TerminationReason::kOptimal);

  if (failure_behavior.explain_infeasibility &&
      (result.termination.reason == math_opt::TerminationReason::kInfeasible ||
       result.termination.reason ==
           math_opt::TerminationReason::kInfeasibleOrUnbounded)) {
    XLS_RETURN_IF_ERROR(model_.AddSlackVariables(
        failure_behavior.infeasible_per_state_backedge_slack_pool));
    XLS_ASSIGN_OR_RETURN(math_opt::SolveResult result_with_slack,
                         solver_->Solve());
    if (result_with_slack.termination.reason ==
            math_opt::TerminationReason::kOptimal ||
        result_with_slack.termination.reason ==
            math_opt::TerminationReason::kFeasible) {
      XLS_RETURN_IF_ERROR(
          model_.ExtractError(result_with_slack.variable_values()));
    }
  }

  // We don't know why the solver failed to find an optimal solution to our LP
  // problem; it could be an infeasibility issue (which needs more work to
  // analyze), a timeout, a precision error, or more. For now, just return a
  // simple error hinting at the problem.
  return absl::InternalError(
      absl::StrCat("The problem does not have an optimal solution; solver "
                   "terminated with ",
                   math_opt::EnumToString(result.termination.reason)));
}

absl::StatusOr<ScheduleCycleMap> SDCScheduler::Schedule(
    std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
    SchedulingFailureBehavior failure_behavior, bool check_feasibility,
    std::optional<int64_t> worst_case_throughput,
    std::optional<double> dynamic_throughput_objective_weight) {
  model_.SetClockPeriod(clock_period_ps);
  if (worst_case_throughput.has_value()) {
    if (model_.initiation_interval().value_or(1) != *worst_case_throughput) {
      XLS_RETURN_IF_ERROR(
          model_.SetWorstCaseThroughput(*worst_case_throughput));
    }
  }

  model_.SetPipelineLength(pipeline_stages);
  if (!pipeline_stages.has_value() && !check_feasibility) {
    // Find the minimum feasible pipeline length.
    model_.MinimizePipelineLength();
    XLS_ASSIGN_OR_RETURN(
        const math_opt::SolveResult result_with_minimized_pipeline_length,
        solver_->Solve());
    if (result_with_minimized_pipeline_length.termination.reason !=
        math_opt::TerminationReason::kOptimal) {
      return BuildError(result_with_minimized_pipeline_length,
                        failure_behavior);
    }
    XLS_ASSIGN_OR_RETURN(
        const int64_t min_pipeline_length,
        model_.ExtractPipelineLength(
            result_with_minimized_pipeline_length.variable_values()));
    model_.SetPipelineLength(min_pipeline_length);
  }

  if (check_feasibility) {
    model_.RemoveObjective();
  } else {
    model_.SetObjective(dynamic_throughput_objective_weight);
  }

  XLS_ASSIGN_OR_RETURN(math_opt::SolveResult result, solver_->Solve());
  if (result.termination.reason == math_opt::TerminationReason::kOptimal ||
      (check_feasibility &&
       result.termination.reason == math_opt::TerminationReason::kFeasible)) {
    return model_.ExtractResult(result.variable_values());
  }
  return BuildError(result, failure_behavior);
}

}  // namespace xls

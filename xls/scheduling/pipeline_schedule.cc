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
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/fdo/delay_manager.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

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

}  // namespace

PipelineSchedule::PipelineSchedule(FunctionBase* function_base,
                                   ScheduleCycleMap cycle_map,
                                   std::optional<int64_t> length)
    : function_base_(function_base), cycle_map_(std::move(cycle_map)) {
  // Build the mapping from cycle to the vector of nodes in that cycle.
  int64_t max_cycle = MaximumCycle(cycle_map_);
  if (length.has_value()) {
    CHECK_GT(*length, max_cycle);
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

void PipelineSchedule::RemoveNode(Node* node) {
  CHECK(cycle_map_.contains(node))
      << "Tried to remove a node from a schedule that it doesn't contain";
  int64_t old_cycle = cycle_map_.at(node);
  std::vector<Node*>& ref = cycle_to_nodes_.at(old_cycle);
  ref.erase(std::remove(ref.begin(), ref.end(), node), std::end(ref));
  cycle_map_.erase(node);
}

absl::StatusOr<PipelineSchedule> PipelineSchedule::FromProto(
    FunctionBase* function, const PipelineScheduleProto& proto) {
  ScheduleCycleMap cycle_map;
  for (const auto& stage : proto.stages()) {
    for (const auto& timed_node : stage.timed_nodes()) {
      // NOTE: we handle timing with our estimator, so ignore timings from proto
      // but it might be useful in the future to e.g. detect regressions.
      XLS_ASSIGN_OR_RETURN(Node * node, function->GetNode(timed_node.node()));
      cycle_map[node] = stage.stage();
    }
  }
  return PipelineSchedule(function, cycle_map);
}

absl::StatusOr<PipelineSchedule> PipelineSchedule::SingleStage(
    FunctionBase* function) {
  ScheduleCycleMap cycle_map;
  for (Node* node : function->nodes()) {
    cycle_map.emplace(node, 0);
  }
  return PipelineSchedule(function, cycle_map);
}

absl::Span<Node* const> PipelineSchedule::nodes_in_cycle(int64_t cycle) const {
  if (cycle < cycle_to_nodes_.size()) {
    return cycle_to_nodes_[cycle];
  }
  return absl::Span<Node* const>();
}

bool PipelineSchedule::IsLiveOutOfCycle(Node* node, int64_t c) const {
  Function* as_func = dynamic_cast<Function*>(function_base_);

  if (cycle(node) > c) {
    return false;
  }

  if (c >= length() - 1) {
    return false;
  }

  if ((as_func != nullptr) && (node == as_func->return_value())) {
    return true;
  }

  for (Node* user : node->users()) {
    if (cycle(user) <= c) {
      continue;
    }
    if (user->Is<Next>()) {
      Next* user_next = user->As<Next>();
      if (user_next->predicate() != node && user_next->value() != node) {
        CHECK_EQ(user_next->param(), node);
        // This Next node only uses this Param node to target the state register
        // it needs to write to; it doesn't actually need the value read out of
        // the Param node, so we don't need to keep the value in pipeline
        // registers for its sake.
        continue;
      }
    }
    return true;
  }

  return false;
}

std::vector<Node*> PipelineSchedule::GetLiveOutOfCycle(int64_t c) const {
  std::vector<Node*> live_out;

  for (int64_t i = 0; i <= c; ++i) {
    for (Node* node : nodes_in_cycle(i)) {
      if (IsLiveOutOfCycle(node, c)) {
        live_out.push_back(node);
      }
    }
  }

  return live_out;
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

      if (node->Is<MinDelay>()) {
        XLS_RET_CHECK_LE(cycle(operand),
                         cycle(node) - node->As<MinDelay>()->delay());
      }
    }
  }
  if (function_base()->IsProc()) {
    Proc* proc = function_base()->AsProcOrDie();
    for (int64_t index = 0; index < proc->GetStateElementCount(); ++index) {
      Node* param = proc->GetStateParam(index);
      Node* next_state = proc->GetNextStateElement(index);
      // Verify that we determine the new state within II cycles of accessing
      // the current param.
      XLS_RET_CHECK_LT(
          cycle(next_state),
          cycle(param) + proc->GetInitiationInterval().value_or(1));
    }
    for (Next* next : proc->next_values()) {
      Node* param = next->param();
      // Verify that no write happens before the corresponding read.
      XLS_RET_CHECK_LE(cycle(param), cycle(next));
      // Verify that we determine the new state within II cycles of accessing
      // the current param.
      XLS_RET_CHECK_LT(
          cycle(next),
          cycle(param) + proc->GetInitiationInterval().value_or(1));
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

absl::Status PipelineSchedule::VerifyTiming(
    int64_t clock_period_ps, const DelayManager& delay_manager) const {
  PathExtractOptions options;
  options.cycle_map = &cycle_map_;
  XLS_ASSIGN_OR_RETURN(PathInfo critical_path,
                       delay_manager.GetLongestPath(options));

  auto [delay, source, target] = critical_path;
  if (delay > clock_period_ps) {
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> path,
                         delay_manager.GetFullCriticalPath(source, target));
    return absl::InternalError(absl::StrFormat(
        "Schedule does not meet timing (%dps). Longest failing path (%dps): %s",
        clock_period_ps, delay,
        absl::StrJoin(path, " -> ", [&](std::string* out, Node* n) {
          absl::StrAppend(
              out, absl::StrFormat("%s (%dps)", n->GetName(),
                                   delay_manager.GetNodeDelay(n).value()));
        })));
  }
  return absl::OkStatus();
}

absl::Status PipelineSchedule::VerifyConstraints(
    absl::Span<const SchedulingConstraint> constraints,
    std::optional<int64_t> worst_case_throughput) const {
  absl::flat_hash_map<std::string, std::vector<Node*>> channel_to_nodes;
  absl::flat_hash_map<Node*, absl::btree_set<Node*>> send_predecessors;
  int64_t last_cycle = 0;
  for (Node* node : TopoSort(function_base_)) {
    if (node->Is<Receive>() || node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      channel_to_nodes[channel->name()].push_back(node);
    }
    last_cycle = std::max(last_cycle, cycle_map_.at(node));

    for (Node* operand : node->operands()) {
      if (operand->Is<Send>()) {
        send_predecessors[node].insert(operand);
      }
      send_predecessors[node].insert(send_predecessors[operand].begin(),
                                     send_predecessors[operand].end());
    }
  }

  constexpr auto plural_s = [](int64_t count) {
    if (count == 1) {
      return "";
    }
    return "s";
  };

  auto matches_direction = [](IODirection direction, Node* node) -> bool {
    switch (direction) {
      case IODirection::kReceive:
        return node->Is<Receive>();
      case IODirection::kSend:
        return node->Is<Send>();
    }
  };

  for (const SchedulingConstraint& constraint : constraints) {
    if (std::holds_alternative<IOConstraint>(constraint)) {
      IOConstraint io_constr = std::get<IOConstraint>(constraint);
      // We use `channel_to_nodes[...]` instead of `channel_to_nodes.at(...)`
      // below because we don't want to error out if a constraint is specified
      // that affects a channel with no associated send/receives in this proc.
      for (Node* source : channel_to_nodes[io_constr.SourceChannel()]) {
        for (Node* target : channel_to_nodes[io_constr.TargetChannel()]) {
          if (source == target) {
            continue;
          }
          // Check that the source and target nodes are the correct kind of
          // channel operations. If this is a "loopback" channel, a mixture of
          // sends and receives might be in channel_to_nodes and you only want
          // to perform the check in the correct direction.
          if (!matches_direction(io_constr.SourceDirection(), source) ||
              !matches_direction(io_constr.TargetDirection(), target)) {
            continue;
          }
          int64_t source_cycle = cycle_map_.at(source);
          int64_t target_cycle = cycle_map_.at(target);
          int64_t latency = target_cycle - source_cycle;
          if ((io_constr.MinimumLatency() <= latency) &&
              (latency <= io_constr.MaximumLatency())) {
            continue;
          }
          return absl::ResourceExhaustedError(absl::StrFormat(
              "Scheduling constraint violated: node %s was scheduled %d "
              "cycles before node %s which violates the constraint that ops "
              "on channel %s must be between %d and %d cycles (inclusive) "
              "before ops on channel %s.",
              source->ToString(), latency, target->ToString(),
              io_constr.SourceChannel(), io_constr.MinimumLatency(),
              io_constr.MaximumLatency(), io_constr.TargetChannel()));
        }
      }
    } else if (std::holds_alternative<NodeInCycleConstraint>(constraint)) {
      NodeInCycleConstraint nic_constr =
          std::get<NodeInCycleConstraint>(constraint);
      const int64_t cycle = cycle_map_.at(nic_constr.GetNode());
      if (cycle != nic_constr.GetCycle()) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Scheduling constraint violated: node %s was scheduled in cycle "
            "%d which violates the constraint that this node must be in "
            "cycle %d.",
            nic_constr.GetNode()->ToString(), cycle, nic_constr.GetCycle()));
      }
    } else if (std::holds_alternative<DifferenceConstraint>(constraint)) {
      DifferenceConstraint diff_constr =
          std::get<DifferenceConstraint>(constraint);
      // a - b <= max_difference
      const int64_t cycle_a = cycle_map_.at(diff_constr.GetA());
      const int64_t cycle_b = cycle_map_.at(diff_constr.GetB());
      if (cycle_a - cycle_b > diff_constr.GetMaxDifference()) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Scheduling constraint violated: node %s was scheduled %d cycle%s "
            "before node %s which violates the constraint that node %s must "
            "be no more than %d cycle%s before node %s.",
            diff_constr.GetA()->ToString(), cycle_a - cycle_b,
            plural_s(cycle_a - cycle_b), diff_constr.GetB()->ToString(),
            diff_constr.GetA()->ToString(), diff_constr.GetMaxDifference(),
            plural_s(diff_constr.GetMaxDifference()),
            diff_constr.GetB()->ToString()));
      }
    } else if (std::holds_alternative<RecvsFirstSendsLastConstraint>(
                   constraint)) {
      for (Node* node : function_base_->nodes()) {
        if (node->Is<Receive>() && cycle_map_.at(node) != 0) {
          return absl::ResourceExhaustedError(absl::StrFormat(
              "Scheduling constraint violated: node %s was scheduled in "
              "cycle %d which violates the constraint that all receives must "
              "be in cycle 0.",
              node->ToString(), cycle_map_.at(node)));
        }
        if (node->Is<Send>() && cycle_map_.at(node) != last_cycle) {
          return absl::ResourceExhaustedError(absl::StrFormat(
              "Scheduling constraint violated: node %s was scheduled in "
              "cycle %d which violates the constraint that all sends must "
              "be in the last cycle (cycle %d).",
              node->ToString(), cycle_map_.at(node), last_cycle));
        }
      }
    } else if (std::holds_alternative<BackedgeConstraint>(constraint)) {
      if (!function_base_->IsProc()) {
        continue;
      }
      const int64_t max_backedge = worst_case_throughput.value_or(1) - 1;
      Proc* proc = function_base_->AsProcOrDie();
      for (int64_t index = 0; index < proc->GetStateElementCount(); ++index) {
        Param* param = proc->GetStateParam(index);
        Node* next_state = proc->GetNextStateElement(index);
        int64_t backedge_length =
            cycle_map_.at(next_state) - cycle_map_.at(param);
        if (backedge_length > max_backedge) {
          return absl::ResourceExhaustedError(absl::StrFormat(
              "Scheduling constraint violated: param %s was scheduled for "
              "access %d cycle%s before node %s, its next value, which "
              "violates the constraint that we can achieve a worst-case "
              "throughput of one iteration per %d cycle%s without external "
              "stalls.",
              param->name(), backedge_length, plural_s(backedge_length),
              next_state->ToString(), worst_case_throughput.value_or(1),
              plural_s(worst_case_throughput.value_or(1))));
        }
      }
      for (Next* next : proc->next_values()) {
        Param* param = next->param()->As<Param>();
        int64_t backedge_length = cycle_map_.at(next) - cycle_map_.at(param);
        if (backedge_length > max_backedge) {
          return absl::ResourceExhaustedError(absl::StrFormat(
              "Scheduling constraint violated: param %s was scheduled for "
              "access %d cycle%s before node %s, its next value, which "
              "violates the constraint that we can achieve a worst-case "
              "throughput of one iteration per %d cycle%s without external "
              "stalls.",
              param->name(), backedge_length, plural_s(backedge_length),
              next->ToString(), worst_case_throughput.value_or(1),
              plural_s(worst_case_throughput.value_or(1))));
        }
      }
    } else if (std::holds_alternative<SendThenRecvConstraint>(constraint)) {
      const SendThenRecvConstraint str_const =
          std::get<SendThenRecvConstraint>(constraint);
      for (Node* recv : function_base_->nodes()) {
        if (!recv->Is<Receive>()) {
          continue;
        }
        for (Node* send : send_predecessors.at(recv)) {
          int64_t send_then_recv_latency =
              cycle_map_.at(recv) - cycle_map_.at(send);
          if (send_then_recv_latency < str_const.MinimumLatency()) {
            return absl::ResourceExhaustedError(absl::StrFormat(
                "Scheduling constraint violated: node %s was scheduled for %d "
                "cycle%s before node %s, which violates the constraint that "
                "all receives must be scheduled at least %d cycle%s after "
                "sends on which they depend.",
                send->ToString(), send_then_recv_latency,
                plural_s(send_then_recv_latency), recv->ToString(),
                str_const.MinimumLatency(),
                plural_s(str_const.MinimumLatency())));
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

PipelineScheduleProto PipelineSchedule::ToProto(
    const DelayEstimator& delay_estimator) const {
  // Compute nodes and paths delays.
  absl::flat_hash_map<Node*, int64_t> node_delays;
  absl::flat_hash_map<Node*, int64_t> node_path_delays;
  for (Node* node : TopoSort(function_base_)) {
    int64_t delay_to_node_start = 0;
    for (Node* operand : node->operands()) {
      if (cycle(operand) == cycle(node)) {
        if (delay_to_node_start < node_path_delays.at(operand)) {
          delay_to_node_start = node_path_delays.at(operand);
        }
      }
    }
    int64_t node_delay = delay_estimator.GetOperationDelayInPs(node).value();
    int64_t path_delay = delay_to_node_start + node_delay;
    node_delays[node] = node_delay;
    node_path_delays[node] = path_delay;
  }

  PipelineScheduleProto proto;
  proto.set_function(function_base_->name());
  for (int i = 0; i < cycle_to_nodes_.size(); i++) {
    StageProto* stage = proto.add_stages();
    stage->set_stage(i);
    for (Node* node : cycle_to_nodes_[i]) {
      TimedNodeProto* timed_node = stage->add_timed_nodes();
      timed_node->set_node(node->GetName());
      timed_node->set_node_delay_ps(node_delays[node]);
      timed_node->set_path_delay_ps(node_path_delays[node]);
    }
  }
  return proto;
}

int64_t PipelineSchedule::CountFinalInteriorPipelineRegisters() const {
  int64_t reg_count = 0;

  for (int64_t stage = 0; stage < length(); ++stage) {
    for (Node* function_base_node : function_base_->nodes()) {
      if (cycle(function_base_node) > stage) {
        continue;
      }

      if (IsLiveOutOfCycle(function_base_node, stage)) {
        reg_count += function_base_node->GetType()->GetFlatBitCount();
      }
    }
  }

  return reg_count;
}

/* static */ absl::StatusOr<PackagePipelineSchedules>
PackagePipelineSchedulesFromProto(Package* p,
                                  const PackagePipelineSchedulesProto& proto) {
  PackagePipelineSchedules schedules;
  for (const auto& [fb_name, proto_schedule] : proto.schedules()) {
    XLS_VLOG(3) << absl::StreamFormat(
        "Converting proto for Functionbase with name %s", fb_name);
    XLS_ASSIGN_OR_RETURN(FunctionBase * fb, p->GetFunctionBaseByName(fb_name));
    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::FromProto(fb, proto_schedule));
    schedules.insert({fb, std::move(schedule)});
  }
  return schedules;
}

PackagePipelineSchedulesProto PackagePipelineSchedulesToProto(
    const PackagePipelineSchedules& schedules,
    const DelayEstimator& delay_estimator) {
  PackagePipelineSchedulesProto proto;
  for (const auto& [fb, schedule] : schedules) {
    proto.mutable_schedules()->insert(
        {fb->name(), schedule.ToProto(delay_estimator)});
  }
  return proto;
}

}  // namespace xls

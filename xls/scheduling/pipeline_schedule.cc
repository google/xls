// Copyright 2020 Google LLC
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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/binary_search.h"
#include "xls/ir/node_iterator.h"
#include "xls/scheduling/function_partition.h"
#include "xls/scheduling/schedule_bounds.h"

namespace xls {
namespace {

// Returns the largest cycle value to which any node is mapped in the given
// ScheduleCycleMap. This is the maximum value of any value element in the map.
int64 MaximumCycle(const ScheduleCycleMap& cycle_map) {
  int64 max_cycle = 0;
  for (const auto& pair : cycle_map) {
    max_cycle = std::max(max_cycle, pair.second);
  }
  return max_cycle;
}

// Splits the nodes at the boundary between 'cycle' and 'cycle + 1' by
// performing a minimum cost cut and tightens the bounds accordingly. Upon
// return no node in the function will have a range which spans both 'cycle' and
// 'cycle + 1'.
absl::Status SplitAfterCycle(Function* f, int64 cycle,
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
xabsl::StatusOr<int64> CountInteriorPipelineRegisters(
    Function* f, const sched::ScheduleBounds& bounds) {
  int64 registers = 0;
  for (Node* node : f->nodes()) {
    XLS_RET_CHECK_EQ(bounds.lb(node), bounds.ub(node));
    int64 latest_use = bounds.lb(node);
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
xabsl::StatusOr<ScheduleCycleMap> ScheduleToMinimizeRegisters(
    Function* f, int64 pipeline_stages, const DelayEstimator& delay_estimator,
    sched::ScheduleBounds* bounds) {
  XLS_VLOG(3) << "ScheduleToMinimizeRegisters()";
  XLS_VLOG(3) << "  pipeline stages = " << pipeline_stages;
  XLS_VLOG_LINES(4, f->DumpIr());

  XLS_VLOG(4) << "Initial bounds:";
  XLS_VLOG_LINES(4, bounds->ToString());

  // Try a number of different orderings of cycle boundary at which the min-cut
  // is performed and keep the best one.
  int64 best_register_count;
  absl::optional<sched::ScheduleBounds> best_bounds;
  for (const std::vector<int64>& cut_order :
       GetMinCutCycleOrders(pipeline_stages - 1)) {
    XLS_VLOG(3) << absl::StreamFormat("Trying cycle order: {%s}",
                                      absl::StrJoin(cut_order, ", "));
    sched::ScheduleBounds trial_bounds = *bounds;
    // Partition the nodes at each cycle boundary. For each iteration, this
    // splits the nodes into those which must be scheduled at or before the
    // cycle and those which must be scheduled after. Upon loop completion each
    // node will have a range of exactly one cycle.
    for (int64 cycle : cut_order) {
      XLS_RETURN_IF_ERROR(
          SplitAfterCycle(f, cycle, delay_estimator, &trial_bounds));
      XLS_RETURN_IF_ERROR(trial_bounds.PropagateLowerBounds());
      XLS_RETURN_IF_ERROR(trial_bounds.PropagateUpperBounds());
    }
    XLS_ASSIGN_OR_RETURN(int64 trial_register_count,
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

// Returns the critical path of the function given a topological sort of its
// nodes.
xabsl::StatusOr<int64> FunctionCriticalPath(
    absl::Span<Node* const> topo_sort, const DelayEstimator& delay_estimator) {
  int64 function_cp = 0;
  absl::flat_hash_map<Node*, int64> node_cp;
  for (Node* node : topo_sort) {
    int64 node_start = 0;
    for (Node* operand : node->operands()) {
      node_start = std::max(node_start, node_cp[operand]);
    }
    XLS_ASSIGN_OR_RETURN(int64 node_delay,
                         delay_estimator.GetOperationDelayInPs(node));
    node_cp[node] = node_start + node_delay;
    function_cp = std::max(function_cp, node_cp[node]);
  }
  return function_cp;
}

// Returns the minimum clock period in picoseconds for which it is feasible to
// schedule the function into a pipeline with the given number of stages.
xabsl::StatusOr<int64> FindMinimumClockPeriod(
    Function* f, int64 pipeline_stages, const DelayEstimator& delay_estimator) {
  XLS_VLOG(4) << "FindMinimumClockPeriod()";
  XLS_VLOG(4) << "  pipeline stages = " << pipeline_stages;
  auto topo_sort_it = TopoSort(f);
  std::vector<Node*> topo_sort(topo_sort_it.begin(), topo_sort_it.end());
  XLS_ASSIGN_OR_RETURN(int64 function_cp,
                       FunctionCriticalPath(topo_sort, delay_estimator));
  // The lower bound of the search is the critical path delay evenly distributed
  // across all stages (rounded up), and the upper bound is simply the critical
  // path of the entire function. It's possible this upper bound is the best you
  // can do if there exists a single operation with delay equal to the
  // critical-path delay of the function.
  int64 search_start = (function_cp + pipeline_stages - 1) / pipeline_stages;
  int64 search_end = function_cp;
  XLS_VLOG(4) << absl::StreamFormat("Binary searching over interval [%d, %d]",
                                    search_start, search_end);
  XLS_ASSIGN_OR_RETURN(int64 min_period,
                       BinarySearchMinTrueWithStatus(
                           search_start, search_end,
                           [&](int64 clk_period_ps) -> xabsl::StatusOr<bool> {
                             // If any node does not fit in the clock period,
                             // fail outright.
                             for (Node* node : f->nodes()) {
                               XLS_ASSIGN_OR_RETURN(
                                   int64 node_delay,
                                   delay_estimator.GetOperationDelayInPs(node));
                               if (node_delay > clk_period_ps) {
                                 return false;
                               }
                             }
                             sched::ScheduleBounds bounds(
                                 f, topo_sort, clk_period_ps, delay_estimator);
                             XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());
                             return bounds.max_lower_bound() < pipeline_stages;
                           }));
  XLS_VLOG(4) << "minimum clock period = " << min_period;
  return min_period;
}

// Returns a sequence of numbers from first to last where the zeroth element of
// the sequence is the middle element between first and last. Subsequent
// elements are selected recursively out of the two intervals before and after
// the middle element.
std::vector<int64> MiddleFirstOrder(int64 first, int64 last) {
  if (first == last) {
    return {first};
  }
  if (first == last - 1) {
    return {first, last};
  }

  int64 middle = (first + last) / 2;
  std::vector<int64> head = MiddleFirstOrder(first, middle - 1);
  std::vector<int64> tail = MiddleFirstOrder(middle + 1, last);

  std::vector<int64> ret;
  ret.push_back(middle);
  ret.insert(ret.end(), head.begin(), head.end());
  ret.insert(ret.end(), tail.begin(), tail.end());
  return ret;
}

}  // namespace

std::vector<std::vector<int64>> GetMinCutCycleOrders(int64 length) {
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
  std::vector<std::vector<int64>> orders;
  std::vector<int64> forward(length);
  std::iota(forward.begin(), forward.end(), 0);
  orders.push_back(forward);

  std::vector<int64> reverse(length);
  std::iota(reverse.begin(), reverse.end(), 0);
  std::reverse(reverse.begin(), reverse.end());
  orders.push_back(reverse);

  orders.push_back(MiddleFirstOrder(0, length - 1));
  return orders;
}

PipelineSchedule::PipelineSchedule(Function* function,
                                   ScheduleCycleMap cycle_map,
                                   absl::optional<int64> length)
    : function_(function), cycle_map_(std::move(cycle_map)) {
  // Build the mapping from cycle to the vector of nodes in that cycle.
  int64 max_cycle = MaximumCycle(cycle_map_);
  if (length.has_value()) {
    XLS_CHECK_GT(*length, max_cycle);
    max_cycle = *length - 1;
  }
  // max_cycle is the latest cycle in which any node is scheduled so add one to
  // get the capacity because cycle numbers start at zero.
  cycle_to_nodes_.resize(max_cycle + 1);
  for (const auto& pair : cycle_map_) {
    Node* node = pair.first;
    int64 cycle = pair.second;
    cycle_to_nodes_[cycle].push_back(node);
  }
  // The nodes in each cycle held in cycle_to_nodes_ must be in a topological
  // sort order.
  absl::flat_hash_map<Node*, int64> node_to_topo_index;
  int64 i = 0;
  for (Node* node : TopoSort(function)) {
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

xabsl::StatusOr<PipelineSchedule> PipelineSchedule::FromProto(
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

absl::Span<Node* const> PipelineSchedule::nodes_in_cycle(int64 cycle) const {
  if (cycle < cycle_to_nodes_.size()) {
    return cycle_to_nodes_[cycle];
  }
  return absl::Span<Node* const>();
}

std::vector<Node*> PipelineSchedule::GetLiveOutOfCycle(int64 c) const {
  std::vector<Node*> live_out;
  for (int64 i = 0; i <= c; ++i) {
    for (Node* node : nodes_in_cycle(i)) {
      if (node == node->function()->return_value() ||
          absl::c_any_of(node->users(),
                         [&](Node* u) { return cycle(u) > c; })) {
        live_out.push_back(node);
      }
    }
  }
  return live_out;
}

/*static*/ xabsl::StatusOr<PipelineSchedule> PipelineSchedule::Run(
    Function* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options) {
  int64 clock_period_ps;
  if (options.clock_period_ps().has_value()) {
    clock_period_ps = *options.clock_period_ps();

    if (options.clock_margin_percent().has_value()) {
      int64 original_clock_period_ps = clock_period_ps;
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
    XLS_ASSIGN_OR_RETURN(
        clock_period_ps,
        FindMinimumClockPeriod(f, *options.pipeline_stages(), delay_estimator));
  }

  sched::ScheduleBounds bounds(f, clock_period_ps, delay_estimator);
  XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());

  int64 max_ub;
  if (options.pipeline_stages().has_value()) {
    XLS_RET_CHECK_GE(*options.pipeline_stages(), bounds.max_lower_bound());
    max_ub = *options.pipeline_stages() - 1;
  } else {
    max_ub = bounds.max_lower_bound();
  }

  for (Node* node : f->nodes()) {
    XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, max_ub));
  }
  XLS_RETURN_IF_ERROR(bounds.PropagateUpperBounds());
  ScheduleCycleMap cycle_map;
  if (options.strategy() == SchedulingStrategy::MINIMIZE_REGISTERS) {
    XLS_ASSIGN_OR_RETURN(
        cycle_map,
        ScheduleToMinimizeRegisters(f, max_ub + 1, delay_estimator, &bounds));
  } else {
    XLS_RET_CHECK(options.strategy() == SchedulingStrategy::ASAP);
    XLS_RET_CHECK(!options.pipeline_stages().has_value());
    // Just schedule everything as soon as possible.
    for (Node* node : f->nodes()) {
      cycle_map[node] = bounds.lb(node);
    }
  }
  auto schedule = PipelineSchedule(f, cycle_map, options.pipeline_stages());
  XLS_RETURN_IF_ERROR(schedule.VerifyTiming(clock_period_ps, delay_estimator));
  XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
  return schedule;
}

std::string PipelineSchedule::ToString() const {
  absl::flat_hash_map<const Node*, int64> topo_pos;
  int64 pos = 0;
  for (Node* node : TopoSort(function_)) {
    topo_pos[node] = pos;
    pos++;
  }

  std::string result;
  for (int64 cycle = 0; cycle <= length(); ++cycle) {
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
  for (Node* node : function()->nodes()) {
    XLS_RET_CHECK(IsScheduled(node));
  }
  for (Node* node : function()->nodes()) {
    for (Node* operand : node->operands()) {
      XLS_RET_CHECK_LE(cycle(operand), cycle(node));
    }
  }
  return absl::OkStatus();
}

absl::Status PipelineSchedule::VerifyTiming(
    int64 clock_period_ps, const DelayEstimator& delay_estimator) const {
  // Critical path from start of the cycle that a node is scheduled through the
  // node itself. If the schedule meets timing, then this value should be less
  // than or equal to clock_period_ps for every node.
  absl::flat_hash_map<Node*, int64> node_cp;
  // The predecessor (operand) of the node through which the critical-path from
  // the start of the cycle extends.
  absl::flat_hash_map<Node*, Node*> cp_pred;
  // The node with the longest critical path from the start of the stage in the
  // entire schedule.
  Node* max_cp_node = nullptr;
  for (Node* node : TopoSort(function_)) {
    // The critical-path delay from the start of the stage to the start of the
    // node.
    int64 cp_to_node_start = 0;
    cp_pred[node] = nullptr;
    for (Node* operand : node->operands()) {
      if (cycle(operand) == cycle(node)) {
        if (cp_to_node_start < node_cp.at(operand)) {
          cp_to_node_start = node_cp.at(operand);
          cp_pred[node] = operand;
        }
      }
    }
    XLS_ASSIGN_OR_RETURN(int64 node_delay,
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

PipelineScheduleProto PipelineSchedule::ToProto() {
  PipelineScheduleProto proto;
  proto.set_function(function_->name());
  for (int i = 0; i < cycle_to_nodes_.size(); i++) {
    StageProto* stage = proto.add_stages();
    stage->set_stage(i);
    for (const Node* node : cycle_to_nodes_[i]) {
      stage->add_nodes(node->GetName());
    }
  }
  return proto;
}

}  // namespace xls

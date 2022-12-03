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
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/scheduling/min_cut_scheduler.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/sdc_scheduler.h"

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

// Returns the nodes of `f` which must be scheduled in the first stage of a
// pipeline. For functions this is parameters.
std::vector<Node*> FirstStageNodes(FunctionBase* f) {
  if (Function* function = dynamic_cast<Function*>(f)) {
    return std::vector<Node*>(function->params().begin(),
                              function->params().end());
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
    const DelayEstimator& delay_estimator,
    absl::Span<const SchedulingConstraint> constraints) {
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
                f, clk_period_ps, topo_sort, pipeline_stages, delay_estimator);
            if (!bounds_or.ok()) {
              return false;
            }
            sched::ScheduleBounds bounds = bounds_or.value();
            absl::StatusOr<ScheduleCycleMap> scm =
                SDCScheduler(f, pipeline_stages, clk_period_ps, delay_estimator,
                             &bounds, constraints, /*check_feasibility=*/true);
            return scm.ok();
          }));
  XLS_VLOG(4) << "minimum clock period = " << min_period;

  return min_period;
}

}  // namespace

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
    FunctionBase* function, const PipelineScheduleProto& proto) {
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

bool PipelineSchedule::IsLiveOutOfCycle(Node* node, int64_t c) const {
  Function* as_func = dynamic_cast<Function*>(function_base_);

  if (c == length() - 1) {
    return false;
  }

  if ((as_func != nullptr) && (node == as_func->return_value())) {
    return true;
  }

  for (Node* user : node->users()) {
    if (cycle(user) > c) {
      return true;
    }
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

/*static*/ absl::StatusOr<PipelineSchedule> PipelineSchedule::Run(
    FunctionBase* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options) {
  int64_t input_delay = options.additional_input_delay_ps().has_value()
                            ? options.additional_input_delay_ps().value()
                            : 0;

  DecoratingDelayEstimator input_delay_added(
      "input_delay_added", delay_estimator,
      [input_delay](Node* node, int64_t base_delay) {
        return node->op() == Op::kReceive ? base_delay + input_delay
                                          : base_delay;
      });

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
    XLS_ASSIGN_OR_RETURN(
        clock_period_ps,
        FindMinimumClockPeriod(f, *options.pipeline_stages(), input_delay_added,
                               options.constraints()));

    if (options.period_relaxation_percent().has_value()) {
      int64_t relaxation_percent = options.period_relaxation_percent().value();

      clock_period_ps += (clock_period_ps * relaxation_percent + 50) / 100;
    }
  }

  XLS_ASSIGN_OR_RETURN(
      sched::ScheduleBounds bounds,
      ConstructBounds(f, clock_period_ps, TopoSort(f).AsVector(),
                      options.pipeline_stages(), input_delay_added));
  int64_t schedule_length = bounds.max_lower_bound() + 1;
  if (options.pipeline_stages().has_value()) {
    schedule_length = options.pipeline_stages().value();
  }

  ScheduleCycleMap cycle_map;
  if (options.strategy() == SchedulingStrategy::MIN_CUT) {
    XLS_ASSIGN_OR_RETURN(
        cycle_map,
        MinCutScheduler(f, schedule_length, clock_period_ps, input_delay_added,
                        &bounds, options.constraints()));
  } else if (options.strategy() == SchedulingStrategy::SDC) {
    XLS_ASSIGN_OR_RETURN(
        cycle_map,
        SDCScheduler(f, schedule_length, clock_period_ps, input_delay_added,
                     &bounds, options.constraints()));
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
      schedule.VerifyTiming(clock_period_ps, input_delay_added));

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
            int64_t source_cycle = cycle_map.at(source);
            int64_t target_cycle = cycle_map.at(target);
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
  if (function_base()->IsProc()) {
    Proc* proc = function_base()->AsProcOrDie();
    for (int64_t index = 0; index < proc->GetStateElementCount(); ++index) {
      Node* param = proc->GetStateParam(index);
      Node* next_state = proc->GetNextStateElement(index);
      XLS_RET_CHECK_EQ(cycle(param), cycle(next_state));
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

}  // namespace xls

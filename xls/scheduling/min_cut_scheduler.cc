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

#include "xls/scheduling/min_cut_scheduler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/scheduling/function_partition.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

namespace {

// Splits the nodes at the boundary between 'cycle' and 'cycle + 1' by
// performing a minimum cost cut and tightens the bounds accordingly. Upon
// return no node in the function will have a range which spans both 'cycle' and
// 'cycle + 1'.
absl::Status SplitAfterCycle(FunctionBase* f, int64_t cycle,
                             const DelayEstimator& delay_estimator,
                             sched::ScheduleBounds* bounds) {
  VLOG(3) << "Splitting after cycle " << cycle;

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

absl::StatusOr<ScheduleCycleMap> MinCutScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints) {
  VLOG(3) << "MinCutScheduler()";
  VLOG(3) << "  pipeline stages = " << pipeline_stages;
  XLS_VLOG_LINES(4, f->DumpIr());

  VLOG(4) << "Initial bounds:";
  XLS_VLOG_LINES(4, bounds->ToString());

  for (const SchedulingConstraint& constraint : constraints) {
    if (std::holds_alternative<RecvsFirstSendsLastConstraint>(constraint)) {
      for (Node* node : f->nodes()) {
        if (node->Is<Receive>()) {
          XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, 0));
          XLS_RETURN_IF_ERROR(bounds->PropagateUpperBounds());
        }
        if (node->Is<Send>()) {
          XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(node, pipeline_stages - 1));
          XLS_RETURN_IF_ERROR(bounds->PropagateLowerBounds());
        }
      }
    } else {
      return absl::InternalError(
          "MinCutScheduler doesn't support constraints "
          "other than receives-first-sends-last.");
    }
  }

  for (Node* node : f->nodes()) {
    if (node->Is<MinDelay>()) {
      return absl::InternalError(
          "MinCutScheduler doesn't support min_delay nodes.");
    }
  }

  // The state backedge must be in the first cycle.
  if (Proc* proc = dynamic_cast<Proc*>(f)) {
    for (Node* node : proc->params()) {
      XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, 0));
      XLS_RETURN_IF_ERROR(bounds->PropagateUpperBounds());
    }
    for (Node* node : proc->NextState()) {
      XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, 0));
      XLS_RETURN_IF_ERROR(bounds->PropagateUpperBounds());
    }
  }

  // Try a number of different orderings of cycle boundary at which the min-cut
  // is performed and keep the best one.
  int64_t best_register_count = std::numeric_limits<int64_t>::max();
  std::optional<sched::ScheduleBounds> best_bounds;
  for (const std::vector<int64_t>& cut_order :
       GetMinCutCycleOrders(pipeline_stages - 1)) {
    VLOG(3) << absl::StreamFormat("Trying cycle order: {%s}",
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

}  // namespace xls

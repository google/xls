// Copyright 2020 The XLS Authors
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

#include "xls/scheduling/schedule_bounds.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"

namespace xls {
namespace sched {

ScheduleBounds::ScheduleBounds(FunctionBase* f, int64_t clock_period_ps,
                               const DelayEstimator& delay_estimator)
    : clock_period_ps_(clock_period_ps), delay_estimator_(&delay_estimator) {
  auto topo_sort_it = TopoSort(f);
  topo_sort_ = std::vector<Node*>(topo_sort_it.begin(), topo_sort_it.end());
  Reset();
}

ScheduleBounds::ScheduleBounds(FunctionBase* f, std::vector<Node*> topo_sort,
                               int64_t clock_period_ps,
                               const DelayEstimator& delay_estimator)
    : topo_sort_(std::move(topo_sort)),
      clock_period_ps_(clock_period_ps),
      delay_estimator_(&delay_estimator) {
  Reset();
}

void ScheduleBounds::Reset() {
  max_lower_bound_ = 0;
  min_upper_bound_ = 0;
  for (Node* node : topo_sort_) {
    bounds_[node] = {0, std::numeric_limits<int64_t>::max()};
    max_lower_bound_ = 0;
    min_upper_bound_ = std::numeric_limits<int64_t>::max();
  }
}

std::string ScheduleBounds::ToString() const {
  std::string out = "Bounds:\n";
  if (!bounds_.empty()) {
    for (Node* node : TopoSort(bounds_.begin()->first->function_base())) {
      if (bounds_.contains(node)) {
        absl::StrAppendFormat(&out, "  %s : [%d, %d]\n", node->GetName(),
                              lb(node), ub(node));
      }
    }
  }
  return out;
}

absl::Status ScheduleBounds::PropagateLowerBounds() {
  VLOG(4) << "PropagateLowerBounds()";
  // The delay in picoseconds from the beginning of a cycle to the start of the
  // node.
  absl::flat_hash_map<Node*, int64_t> in_cycle_delay;

  // Compute the lower bound of each node based on the lower bounds of the
  // operands of the node.
  for (Node* node : topo_sort_) {
    int64_t& node_in_cycle_delay = in_cycle_delay[node];
    VLOG(4) << absl::StreamFormat("  %s : original lb=%d", node->GetName(),
                                  lb(node));
    for (Node* operand : node->operands()) {
      int64_t operand_lb = lb(operand);
      if (operand_lb < lb(node)) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(int64_t operand_delay,
                           delay_estimator_->GetOperationDelayInPs(operand));
      if (operand_lb > lb(node)) {
        VLOG(4) << absl::StreamFormat(
            "    tightened lb to %d because of operand %s", operand_lb,
            operand->GetName());
        XLS_RETURN_IF_ERROR(TightenNodeLb(node, operand_lb));
        node_in_cycle_delay = in_cycle_delay.at(operand) + operand_delay;
        continue;
      }
      node_in_cycle_delay = std::max(
          node_in_cycle_delay, in_cycle_delay.at(operand) + operand_delay);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator_->GetOperationDelayInPs(node));
    if (node_delay > clock_period_ps_) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Node %s has a greater delay (%dps) than the clock period (%dps)",
          node->GetName(), node_delay, clock_period_ps_));
    }
    if (node_in_cycle_delay + node_delay > clock_period_ps_) {
      // Node does not fit in this cycle. Move to next cycle.
      VLOG(4) << "    overflows clock period, tightened lb to " << lb(node) + 1;
      XLS_RETURN_IF_ERROR(TightenNodeLb(node, lb(node) + 1));
      node_in_cycle_delay = 0;
    }
  }
  return absl::OkStatus();
}

absl::Status ScheduleBounds::PropagateUpperBounds() {
  VLOG(4) << "PropagateUpperBounds()";
  // The delay in picoseconds from the end of a cycle to the end of the node.
  absl::flat_hash_map<Node*, int64_t> in_cycle_delay;

  // Compute the upper bound of each node based on the upper bounds of the
  // users of the node.
  for (auto it = topo_sort_.rbegin(); it != topo_sort_.rend(); ++it) {
    Node* node = *it;
    int64_t& node_in_cycle_delay = in_cycle_delay[node];
    VLOG(4) << absl::StreamFormat("  %s : original ub=%d", node->GetName(),
                                  ub(node));
    for (Node* user : node->users()) {
      int64_t user_ub = ub(user);
      if (user_ub == std::numeric_limits<int64_t>::max() ||
          user_ub > ub(node)) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(int64_t user_delay,
                           delay_estimator_->GetOperationDelayInPs(user));
      if (user_ub < ub(node)) {
        VLOG(4) << absl::StreamFormat(
            "    tightened ub to %d because of user %s", user_ub,
            user->GetName());
        XLS_RETURN_IF_ERROR(TightenNodeUb(node, user_ub));
        node_in_cycle_delay = in_cycle_delay.at(user) + user_delay;
        continue;
      }
      node_in_cycle_delay =
          std::max(node_in_cycle_delay, in_cycle_delay.at(user) + user_delay);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator_->GetOperationDelayInPs(node));
    if (node_delay > clock_period_ps_) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Node %s has a greater delay (%dps) than the clock period (%dps)",
          node->GetName(), node_delay, clock_period_ps_));
    }
    if (node_in_cycle_delay + node_delay > clock_period_ps_) {
      // Node does not fit in this cycle. Move to next cycle.
      VLOG(4) << "    overflows clock period, tightened ub to " << ub(node) - 1;
      XLS_RETURN_IF_ERROR(TightenNodeUb(node, ub(node) - 1));
      node_in_cycle_delay = 0;
    }
  }
  return absl::OkStatus();
}

/* static */ absl::StatusOr<ScheduleBounds>
ScheduleBounds::ComputeAsapAndAlapBounds(
    FunctionBase* f, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator) {
  VLOG(4) << "ComputeAsapAndAlapBounds()";
  ScheduleBounds bounds(f, clock_period_ps, delay_estimator);
  XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());
  VLOG(4) << "Setting all upper bounds to max-lower-bound "
          << bounds.max_lower_bound();
  for (Node* node : f->nodes()) {
    XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, bounds.max_lower_bound()));
  }
  XLS_RETURN_IF_ERROR(bounds.PropagateUpperBounds());
  return bounds;
}

}  // namespace sched
}  // namespace xls

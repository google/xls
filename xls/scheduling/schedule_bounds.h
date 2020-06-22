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

#ifndef XLS_SCHEDULING_SCHEDULE_BOUNDS_H_
#define XLS_SCHEDULING_SCHEDULE_BOUNDS_H_

#include <limits>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"

namespace xls {
namespace sched {

// An abstraction holding lower and upper bounds for each node in a
// function. The bounds are constraints on cycles in which a node may be
// scheduled.
class ScheduleBounds {
 public:
  // Returns a object with the lower bounds of each node set to the earliest
  // possible cycle which satisfies dependency and clock period
  // constraints. Similarly, upper bounds are set to the latest possible cycle
  // The upper bounds of nodes with no uses (leaf nodes) are set to the maximum
  // lower bound of any node.
  static xabsl::StatusOr<ScheduleBounds> ComputeAsapAndAlapBounds(
      Function* f, int64 clock_period_ps,
      const DelayEstimator& delay_estimator);

  // Upon construction all parameters have lower and upper bounds of 0. All
  // other nodes have a lower bound of 1 and an upper bound of INT64_MAX.
  ScheduleBounds(Function* f, int64 clock_period_ps,
                 const DelayEstimator& delay_estimator);

  // Constructor which uses an existing topological sort to avoid having to
  // recompute it.
  ScheduleBounds(Function* f, std::vector<Node*> topo_sort,
                 int64 clock_period_ps, const DelayEstimator& delay_estimator);

  ScheduleBounds(const ScheduleBounds& other) = default;
  ScheduleBounds(ScheduleBounds&& other) = default;
  ScheduleBounds& operator=(const ScheduleBounds& other) = default;
  ScheduleBounds& operator=(ScheduleBounds&& other) = default;

  // Resets node bounds to their initial unconstrained values.
  void Reset();

  // Return the lower/upper bound of the given node.
  int64 lb(Node* node) const { return bounds_.at(node).first; }
  int64 ub(Node* node) const { return bounds_.at(node).second; }

  // Return the lower and upper bound as a pair (lower bound is first element).
  const std::pair<int64, int64>& bounds(Node* node) const {
    return bounds_.at(node);
  }

  // Sets the lower bound of the given node to the maximum of its existing value
  // and the given value. Raises an error if the new value results in infeasible
  // bounds (lower bound is greater than upper bound).
  absl::Status TightenNodeLb(Node* node, int64 value) {
    XLS_RET_CHECK_GE(ub(node), value) << node;
    bounds_.at(node).first = std::max(bounds_.at(node).first, value);
    max_lower_bound_ = std::max(max_lower_bound_, value);
    return absl::OkStatus();
  }

  // Sets the upper bound of the given node to the minimum of its existing value
  // and the given value. Raises an error if the new value results in infeasible
  // bounds (lower bound is greater than upper bound).
  absl::Status TightenNodeUb(Node* node, int64 value) {
    XLS_CHECK_LE(lb(node), value);
    XLS_RET_CHECK_LE(lb(node), value) << node;
    bounds_.at(node).second = std::min(bounds_.at(node).second, value);
    min_upper_bound_ = std::min(min_upper_bound_, value);
    return absl::OkStatus();
  }

  // Returns the maximum lower (upper) bound of any node in the function.
  int64 max_lower_bound() const { return max_lower_bound_; }
  int64 min_upper_bound() const { return min_upper_bound_; }

  std::string ToString() const;

  // Updates the lower (upper) bounds of each node such that dependency and
  // clock period constraints are met for every node. Should be called after
  // calling TightenNodeLb (TightenNodeUb) to propagate the tightened bound
  // throughout the graph. This method only tightens bounds (increases lower
  // bounds and decreases upper bounds). Returns an error if propagation results
  // in infeasible bounds (lower bound is greater than upper bound for a node).
  absl::Status PropagateLowerBounds();
  absl::Status PropagateUpperBounds();

 private:
  // A topological sort of the nodes in the function.
  std::vector<Node*> topo_sort_;

  int64 clock_period_ps_;
  const DelayEstimator* delay_estimator_;

  // The bounds of each node stored as a {lower, upper} pair.
  absl::flat_hash_map<Node*, std::pair<int64, int64>> bounds_;

  int64 max_lower_bound_;
  int64 min_upper_bound_;
};

}  // namespace sched
}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULE_BOUNDS_H_

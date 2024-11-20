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

#include "xls/scheduling/scheduling_wrapper_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

absl::StatusOr<bool> SchedulingWrapperPass::RunInternal(
    SchedulingUnit* unit, const SchedulingPassOptions& options,
    PassResults* results) const {
  // Keep a set of nodeids because Node* can be invalidated by the wrapped pass,
  // e.g. when DCE removes a node.
  absl::btree_set<int64_t> nodeids_before;
  // FunctionBase* can also be invalidated by wrapped pass, so be careful not to
  // dereference elements from it after running the wrapped pass unless you've
  // checked that it's still there.
  absl::flat_hash_map<int64_t, FunctionBase*> id_to_function;
  // Only populate this with nodes from before running the wrapped pass, so
  // elements could be invalid. We build this to keep track of nodes to remove
  // from the schedule.
  absl::flat_hash_map<int64_t, Node*> id_to_node;
  XLS_ASSIGN_OR_RETURN(std::vector<FunctionBase*> before_schedulable_functions,
                       unit->GetSchedulableFunctions());
  for (FunctionBase* f : before_schedulable_functions) {
    // No need to save nodes_before for unscheduled FunctionBases.
    if (!unit->schedules().contains(f)) {
      continue;
    }
    for (Node* node : f->nodes()) {
      nodeids_before.insert(node->id());
      id_to_function[node->id()] = node->function_base();
      id_to_node[node->id()] = node;
    }
  }

  XLS_ASSIGN_OR_RETURN(
      bool changed,
      wrapped_pass_->Run(
          unit->GetPackage(),
          OptimizationPassOptions(options).WithOptLevel(opt_level_), results));
  if (!changed) {
    return false;
  }

  XLS_ASSIGN_OR_RETURN(std::vector<FunctionBase*> after_schedulable_functions,
                       unit->GetSchedulableFunctions());

  // Check for FunctionBases that have been removed by the wrapped pass and
  // remove any schedules they might have.
  {
    absl::flat_hash_set<FunctionBase*> after_schedulable_functions_set(
        after_schedulable_functions.begin(), after_schedulable_functions.end());
    absl::erase_if(
        unit->schedules(),
        [&after_schedulable_functions_set](
            const std::pair<FunctionBase* const, PipelineSchedule>& itr) {
          return !after_schedulable_functions_set.contains(itr.first);
        });
  }

  absl::btree_set<int64_t> nodeids_after;
  for (FunctionBase* f : after_schedulable_functions) {
    // No need to save nodes_after for unscheduled FunctionBases, we only care
    // about removing nodes that were removed and exist only before the wrapped
    // pass is run.
    if (!unit->schedules().contains(f)) {
      continue;
    }
    for (Node* node : f->nodes()) {
      nodeids_after.insert(node->id());
      id_to_function[node->id()] = node->function_base();
    }
  }

  std::vector<int64_t> symmetric_difference;
  symmetric_difference.reserve(
      std::max(nodeids_before.size(), nodeids_after.size()));
  std::set_symmetric_difference(nodeids_before.begin(), nodeids_before.end(),
                                nodeids_after.begin(), nodeids_after.end(),
                                std::back_inserter(symmetric_difference));

  for (int64_t nodeid : symmetric_difference) {
    if (nodeids_before.contains(nodeid)) {
      // This means the wrapped pass has removed this node. It's possible that
      // the schedule has already been removed because a new node was added, so
      // we check if the schedule is present first.
      XLS_RET_CHECK(!nodeids_after.contains(nodeid));
      auto itr = unit->schedules().find(id_to_function.at(nodeid));
      if (itr != unit->schedules().end()) {
        itr->second.RemoveNode(id_to_node.at(nodeid));
      }
      continue;
    }

    // This means there's a new node that has been added by the wrapped pass.
    if (reschedule_new_nodes_) {
      // need to reschedule, remove the current schedule if it hasn't been
      // removed already.
      unit->schedules().erase(id_to_function.at(nodeid));
    } else {
      return absl::InternalError(
          absl::StrFormat("SchedulingWrapperPass(%s) can't create new nodes "
                          "when reschedule_new_nodes_ is false.",
                          wrapped_pass_->short_name()));
    }
  }

  return true;
}

}  // namespace xls

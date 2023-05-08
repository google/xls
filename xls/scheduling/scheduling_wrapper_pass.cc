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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"

namespace xls {

absl::StatusOr<bool> SchedulingWrapperPass::RunInternal(
    SchedulingUnit<>* unit, const SchedulingPassOptions& options,
    PassResults* results) const {
  if (!unit->schedule.has_value()) {
    return wrapped_pass_->Run(unit->ir, PassOptions(), results);
  }

  absl::flat_hash_map<int64_t, Node*> nodes_before;
  for (FunctionBase* f : unit->ir->GetFunctionBases()) {
    for (Node* node : f->nodes()) {
      nodes_before[node->id()] = node;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool changed,
                       wrapped_pass_->Run(unit->ir, PassOptions(), results));
  if (!changed) {
    return false;
  }

  absl::flat_hash_map<int64_t, Node*> nodes_after;
  for (FunctionBase* f : unit->ir->GetFunctionBases()) {
    for (Node* node : f->nodes()) {
      nodes_after[node->id()] = node;
    }
  }

  auto itr = std::find_if(nodes_after.begin(), nodes_after.end(),
                          [&nodes_before](const std::pair<int64_t, Node*>& kv) {
                            return !nodes_before.contains(kv.first);
                          });
  if (itr != nodes_after.end()) {
    if (reschedule_new_nodes_) {
      // need to reschedule, delete the current schedule.
      unit->schedule = std::nullopt;
    } else {
      return absl::InternalError(
          absl::StrFormat("SchedulingWrapperPass(%s) can't create new nodes "
                          "when reschedule_new_nodes_ is false.",
                          wrapped_pass_->short_name()));
    }
  }

  if (unit->schedule.has_value()) {
    for (const auto& [before, before_node] : nodes_before) {
      if (!nodes_after.contains(before)) {
        unit->schedule.value().RemoveNode(before_node);
      }
    }
  }

  return true;
}

}  // namespace xls

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

  for (const auto& [after, after_node] : nodes_after) {
    if (!nodes_before.contains(after)) {
      return absl::InternalError(
          absl::StrFormat("SchedulingWrapperPass can't handle passes that "
                          "create new nodes: wrapped over %s",
                          wrapped_pass_->short_name()));
    }
  }

  for (const auto& [before, before_node] : nodes_before) {
    if (!nodes_after.contains(before) && unit->schedule.has_value()) {
      unit->schedule.value().RemoveNode(before_node);
    }
  }

  return true;
}

}  // namespace xls

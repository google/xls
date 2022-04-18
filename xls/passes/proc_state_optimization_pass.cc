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

#include "xls/passes/proc_state_optimization_pass.h"

#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

absl::StatusOr<bool> RemoveZeroWidthStateElements(Proc* proc) {
  std::vector<int64_t> to_remove;
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (proc->GetStateElementType(i)->GetFlatBitCount() == 0) {
      to_remove.push_back(i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }
  for (int64_t i : to_remove) {
    XLS_RETURN_IF_ERROR(proc->GetStateParam(i)
                            ->ReplaceUsesWithNew<Literal>(
                                ZeroOfType(proc->GetStateElementType(i)))
                            .status());
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

// TODO(meheff): 4/7/2022 Remove elements whose only use is in the computation
// of their own `next` value.
// TODO(meheff): 4/7/2022 Remove sets of elements whose only uses are to compute
// the collective `next` values of the set.
absl::StatusOr<bool> RemoveDeadStateElements(Proc* proc) {
  std::vector<int64_t> to_remove;
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (proc->GetStateParam(i)->users().empty()) {
      to_remove.push_back(i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }
  for (int64_t i : to_remove) {
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> ProcStateOptimizationPass::RunOnProcInternal(
    Proc* proc, const PassOptions& options, PassResults* results) const {
  bool changed = false;
  XLS_ASSIGN_OR_RETURN(bool zero_width_changed,
                       RemoveZeroWidthStateElements(proc));
  changed |= zero_width_changed;

  XLS_ASSIGN_OR_RETURN(bool dead_changed, RemoveDeadStateElements(proc));
  changed |= dead_changed;

  // TODO(meheff): 4/7/2022 Remove elements which are static (i.e, never change
  // from their initial value).

  return changed;
}

}  // namespace xls

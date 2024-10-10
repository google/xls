// Copyright 2024 The XLS Authors
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

#include "xls/passes/next_node_modernize_pass.h"

#include <cstdint>
#include <optional>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

absl::StatusOr<bool> NextNodeModernizePass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  if (!proc->next_values().empty()) {
    VLOG(1) << "NextNodeModernizePass triggered on an IR that already "
               "has next-nodes present: "
            << proc->DumpIr();
    return false;
  }

  for (int64_t index = 0; index < proc->GetStateElementCount(); ++index) {
    Param* param = proc->GetStateParam(index);
    Node* next_value = proc->GetNextStateElement(index);
    XLS_RETURN_IF_ERROR(
        proc->MakeNodeWithName<Next>(param->loc(), /*param=*/param,
                                     /*value=*/next_value,
                                     /*predicate=*/std::nullopt,
                                     absl::StrCat(param->name(), "_next"))
            .status());

    if (next_value != static_cast<Node*>(param)) {
      // Nontrivial next-state element; remove it so we pass verification.
      XLS_RETURN_IF_ERROR(proc->SetNextStateElement(index, param));
    }
  }

  return proc->GetStateElementCount() > 0;
}

REGISTER_OPT_PASS(NextNodeModernizePass);

}  // namespace xls

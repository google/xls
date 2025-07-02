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

#include "xls/passes/dce_pass.h"

#include <cstdint>
#include <deque>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

absl::StatusOr<bool> DeadCodeEliminationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  auto is_deletable = [](Node* n) {
    // Don't remove invokes, they will be removed by inlining. The invoked
    // functions could have side effects, so DCE shouldn't remove them.
    //
    // TODO: google/xls#1806 -  consider making invokes side-effecting if we can
    // deal with FFI well.
    return !n->function_base()->HasImplicitUse(n) && !n->Is<Invoke>() &&
           (!OpIsSideEffecting(n->op()) || n->Is<Gate>());
  };

  std::deque<Node*> worklist;
  for (Node* n : f->nodes()) {
    if (n->users().empty() && is_deletable(n)) {
      worklist.push_back(n);
    }
  }
  int64_t removed_count = 0;
  absl::flat_hash_set<Node*> unique_operands;
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop_front();

    // A node may appear more than once as an operand of 'node'. Keep track of
    // which operands have been handled in a set.
    unique_operands.clear();
    for (Node* operand : node->operands()) {
      if (unique_operands.insert(operand).second) {
        if (HasSingleUse(operand) && is_deletable(operand)) {
          worklist.push_back(operand);
        }
      }
    }
    VLOG(3) << "DCE removing " << node->ToString();
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
    removed_count++;
  }

  VLOG(2) << "Removed " << removed_count << " dead nodes";
  return removed_count > 0;
}

}  // namespace xls

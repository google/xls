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

#include "xls/passes/dfe_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/block.h"
#include "xls/ir/proc.h"

namespace xls {
namespace {

void MarkReachedFunctions(FunctionBase* func,
                          absl::flat_hash_set<FunctionBase*>* reached) {
  if (reached->contains(func)) {
    return;
  }
  reached->insert(func);
  // Iterate over statements and find invocations or references.
  for (Node* node : func->nodes()) {
    switch (node->op()) {
      case Op::kCountedFor:
        MarkReachedFunctions(node->As<CountedFor>()->body(), reached);
        break;
      case Op::kInvoke:
        MarkReachedFunctions(node->As<Invoke>()->to_apply(), reached);
        break;
      case Op::kMap:
        MarkReachedFunctions(node->As<Map>()->to_apply(), reached);
        break;
      default:
        break;
    }
  }
}

}  // namespace

// Starting from the return_value(s), DFS over all nodes. Unvisited
// nodes, or parameters, are dead.
absl::StatusOr<bool> DeadFunctionEliminationPass::RunInternal(
    Package* p, const PassOptions& options, PassResults* results) const {
  absl::flat_hash_set<FunctionBase*> reached;
  // TODO(meheff): Package:EntryFunction check fails if there is not a function
  // named "main". Ideally as an invariant a Package should always have an entry
  // function, but for now look for it and bail if it does not exist.
  absl::optional<FunctionBase*> top = p->GetTop();
  if (!top.has_value() || !top.value()->IsFunction()) {
    return false;
  }
  Function* func = top.value()->AsFunctionOrDie();

  MarkReachedFunctions(func, &reached);

  // Blocks and procs are not deleted from the package so any references from
  // these constructs must remain.
  // TODO(https://github.com/google/xls/issues/531): 2021/12/6 Eliminate dead
  // procs/blocks when the default entry function heuristics are eliminated.
  for (const std::unique_ptr<Proc>& proc : p->procs()) {
    MarkReachedFunctions(proc.get(), &reached);
  }
  for (const std::unique_ptr<Block>& block : p->blocks()) {
    MarkReachedFunctions(block.get(), &reached);
  }

  // Accumulate a list of nodes to unlink.
  std::vector<Function*> to_unlink;
  for (std::unique_ptr<Function>& f : p->functions()) {
    if (!reached.contains(f)) {
      XLS_VLOG(2) << "Dead Function Elimination: " << f->name();
      to_unlink.push_back(f.get());
    }
  }
  for (Function* function : to_unlink) {
    XLS_RETURN_IF_ERROR(p->RemoveFunction(function));
  }
  return !to_unlink.empty();
}

}  // namespace xls

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

namespace xls {
namespace {

void MarkReachedFunctions(Function* func,
                          absl::flat_hash_set<Function*>* reached) {
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
  absl::flat_hash_set<Function*> reached;
  // TODO(meheff): Package:EntryFunction check fails if there is not a function
  // named "main". Ideally as an invariant a Package should always have an entry
  // function, but for now look for it and bail if it does not exist.
  absl::StatusOr<Function*> func = p->EntryFunction();
  if (!func.ok()) {
    return false;
  }

  MarkReachedFunctions(*func, &reached);

  // Accumulate a list of nodes to unlink.
  std::vector<Function*> to_unlink;
  for (std::unique_ptr<Function>& f : p->functions()) {
    if (!reached.contains(f)) {
      XLS_VLOG(2) << "Dead Function Elimination: " << f->name();
      to_unlink.push_back(f.get());
    }
  }
  if (!to_unlink.empty()) {
    p->DeleteDeadFunctions(to_unlink);
  }
  return !to_unlink.empty();
}

}  // namespace xls

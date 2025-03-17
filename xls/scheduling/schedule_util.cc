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

#include "xls/scheduling/schedule_util.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"

namespace xls {

absl::flat_hash_set<Node*> GetDeadAfterSynthesisNodes(FunctionBase* f) {
  absl::flat_hash_set<Node*> dead_after_synthesis;
  for (Node* node : ReverseTopoSort(f)) {
    // Does this node have any visible effects of its own (not counting
    // non-synthesized effects, like asserts or traces)? If so, it's live.
    if (f->HasImplicitUse(node)) {
      continue;
    }
    if (OpIsSideEffecting(node->op()) &&
        !node->OpIn({Op::kAssert, Op::kCover, Op::kTrace})) {
      continue;
    }

    // Otherwise, if all of its users are dead after synthesis, then this node
    // is too.
    if (absl::c_all_of(node->users(), [&](Node* user) {
          return dead_after_synthesis.contains(user);
        })) {
      dead_after_synthesis.insert(node);
    }
  }
  return dead_after_synthesis;
}

}  // namespace xls

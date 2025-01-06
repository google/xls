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

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"

namespace xls {

absl::flat_hash_set<Node*> GetDeadAfterSynthesisNodes(FunctionBase* f) {
  absl::flat_hash_set<Node*> dead_after_synthesis;
  std::vector<Node*> to_visit;
  auto mark_dead = [&](Node* node) {
    auto [_, inserted] = dead_after_synthesis.insert(node);
    if (inserted) {
      to_visit.insert(to_visit.end(), node->operands().begin(),
                      node->operands().end());
    }
  };
  for (Node* node : f->nodes()) {
    if (node->OpIn({Op::kAssert, Op::kCover, Op::kTrace})) {
      mark_dead(node);
    }
  }
  while (!to_visit.empty()) {
    Node* node = to_visit.back();
    to_visit.pop_back();
    if (dead_after_synthesis.contains(node)) {
      continue;
    }

    // Does this node have any visible effects of its own? If so, it's live.
    if (f->HasImplicitUse(node)) {
      continue;
    }
    if (OpIsSideEffecting(node->op())) {
      continue;
    }

    // Otherwise, if all of its users are dead after synthesis, then this node
    // is too.
    if (absl::c_all_of(node->users(), [&](Node* user) {
          return dead_after_synthesis.contains(user);
        })) {
      mark_dead(node);
    }
  }
  return dead_after_synthesis;
}

}  // namespace xls

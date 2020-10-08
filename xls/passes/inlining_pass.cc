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

#include "xls/passes/inlining_pass.h"

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace {

bool ShouldInline(Invoke* invoke) {
  // Inline all the things!
  //
  // TODO(xls-team): 2019-04-12 May want a more sophisticated policy than this
  // one day.
  return true;
}

// Finds an "effectively used" (has users or is return value) invoke in the
// function f, or returns nullptr if none is found.
Invoke* FindInvoke(Function* f) {
  for (Node* node : TopoSort(f)) {
    if (node->Is<Invoke>() &&
        (node == f->return_value() || !node->users().empty()) &&
        ShouldInline(node->As<Invoke>())) {
      return node->As<Invoke>();
    }
  }
  return nullptr;
}

// Unrolls the node "loop" by replacing it with a sequence of dependent
// invocations.
absl::Status InlineInvoke(Invoke* invoke, Function* f) {
  Function* invoked = invoke->to_apply();
  absl::flat_hash_map<Node*, Node*> invoked_node_to_replacement;
  for (int64 i = 0; i < invoked->params().size(); ++i) {
    Node* param = invoked->params()[i];
    invoked_node_to_replacement[param] = invoke->operand(i);
  }
  for (Node* node : TopoSort(invoked)) {
    if (invoked_node_to_replacement.find(node) !=
        invoked_node_to_replacement.end()) {
      // Already taken care of (e.g. parameters above).
      continue;
    }
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(invoked_node_to_replacement.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(Node * new_node, node->Clone(new_operands, f));
    invoked_node_to_replacement[node] = new_node;
  }

  XLS_RETURN_IF_ERROR(invoke
                          ->ReplaceUsesWith(invoked_node_to_replacement.at(
                              invoked->return_value()))
                          .status());
  return f->RemoveNode(invoke);
}

}  // namespace

absl::StatusOr<bool> InliningPass::RunOnFunction(Function* f,
                                                 const PassOptions& options,
                                                 PassResults* results) const {
  bool changed = false;
  while (true) {
    Invoke* invoke = FindInvoke(f);
    if (invoke == nullptr) {
      break;
    }
    XLS_RETURN_IF_ERROR(InlineInvoke(invoke, f));
    changed = true;
  }
  return changed;
}

}  // namespace xls

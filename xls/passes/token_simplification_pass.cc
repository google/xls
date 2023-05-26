// Copyright 2023 The XLS Authors
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

#include "xls/passes/token_simplification_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"

namespace xls {

namespace {

// Find all token-containing nodes that the given node transitively depends on.
absl::flat_hash_set<Node*> DependenciesOf(Node* root) {
  std::vector<Node*> stack;
  stack.push_back(root);
  absl::flat_hash_set<Node*> discovered;
  while (!stack.empty()) {
    Node* popped = stack.back();
    stack.pop_back();
    if (!TypeHasToken(popped->GetType())) {
      continue;
    }
    if (!discovered.contains(popped)) {
      discovered.insert(popped);
      for (Node* child : popped->operands()) {
        stack.push_back(child);
      }
    }
  }
  return discovered;
}

absl::StatusOr<bool> SimplifyTrivialAfterAll(AfterAll* node) {
  // AfterAll([x]) = x

  FunctionBase* f = node->function_base();
  if (node->operand_count() == 1) {
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(node->operand(0)));
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
    return true;
  }
  return false;
}

absl::StatusOr<bool> CollapseAfterAll(AfterAll* node) {
  // If all users of an AfterAll are in AfterAlls, then we can collapse them
  // together, e.g.:
  // AfterAll([…, AfterAll([…]), …]) = AfterAll([…, …, …])

  FunctionBase* f = node->function_base();

  if (f->HasImplicitUse(node)) {
    return false;
  }

  for (Node* user : node->users()) {
    if (!user->Is<AfterAll>()) {
      return false;
    }
  }

  bool changed = false;

  // Prevents iterator invalidation
  std::vector<Node*> users(node->users().begin(), node->users().end());

  for (Node* user_node : users) {
    AfterAll* user = user_node->As<AfterAll>();
    std::vector<Node*> new_operands;
    new_operands.reserve(node->operand_count() + user->operand_count() - 1);
    for (Node* operand : user->operands()) {
      if (operand != node) {
        new_operands.push_back(operand);
      }
    }
    for (Node* operand : node->operands()) {
      new_operands.push_back(operand);
    }

    XLS_ASSIGN_OR_RETURN(Node * replacement,
                         f->MakeNode<AfterAll>(SourceInfo(), new_operands));
    XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(replacement));
    XLS_RETURN_IF_ERROR(f->RemoveNode(user));
    changed = true;
  }

  XLS_RETURN_IF_ERROR(f->RemoveNode(node));

  return changed;
}

absl::StatusOr<bool> ReplaceOverlappingAfterAll(AfterAll* node) {
  // If x depends on y, AfterAll([…, x, …, y, …]) = AfterAll([…, x, …, x, …])

  bool changed = false;
  for (int64_t i = 0; i < node->operand_count(); ++i) {
    Node* operand = node->operand(i);
    absl::flat_hash_set<Node*> deps = DependenciesOf(operand);
    for (int64_t j = 0; j < node->operand_count(); ++j) {
      Node* other_operand = node->operand(j);
      if (deps.contains(other_operand) && operand != other_operand) {
        XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(j, operand));
        changed = true;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> RemoveDuplicateAfterAll(AfterAll* node) {
  // AfterAll([…, x, …, x, …]) = AfterAll([…, x, …])

  FunctionBase* f = node->function_base();
  absl::flat_hash_set<Node*> operands(node->operands().begin(),
                                      node->operands().end());
  if (operands.size() != node->operand_count()) {
    std::vector<Node*> sorted_operands(operands.begin(), operands.end());
    std::sort(sorted_operands.begin(), sorted_operands.end(),
              Node::NodeIdLessThan());
    XLS_ASSIGN_OR_RETURN(Node * replacement,
                         f->MakeNode<AfterAll>(SourceInfo(), sorted_operands));
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(replacement));
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
    return true;
  }

  return false;
}

int64_t NumberOfTokensInType(Type* type) {
  if (type->IsToken()) {
    return 1;
  }
  if (type->IsArray()) {
    return NumberOfTokensInType(type->AsArrayOrDie()->element_type()) *
           type->AsArrayOrDie()->size();
  }
  if (type->IsTuple()) {
    int64_t result = 0;
    for (Type* element_type : type->AsTupleOrDie()->element_types()) {
      result += NumberOfTokensInType(element_type);
    }
    return result;
  }
  return 0;
}

}  // namespace

absl::StatusOr<bool> TokenSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  for (Node* node : f->nodes()) {
    if (NumberOfTokensInType(node->GetType()) > 1) {
      return false;
    }
  }

  bool changed = false;

  for (Node* node : ReverseTopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         CollapseAfterAll(node->As<AfterAll>()));
    changed |= subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         ReplaceOverlappingAfterAll(node->As<AfterAll>()));
    changed |= subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         RemoveDuplicateAfterAll(node->As<AfterAll>()));
    changed |= subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         SimplifyTrivialAfterAll(node->As<AfterAll>()));
    changed |= subpass_changed;
  }

  return changed;
}

}  // namespace xls

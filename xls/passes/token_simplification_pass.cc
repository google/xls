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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

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

      if (popped->Is<Invoke>()) {
        // Disregard any dependencies that pass through Invoke nodes; we can't
        // tell whether the function invocation necessarily establishes a
        // dependency chain from all of its inputs to all of its outputs.
        continue;
      }
      for (Node* child : popped->operands()) {
        stack.push_back(child);
      }
    }
  }
  return discovered;
}

absl::StatusOr<bool> SimplifyTrivialMinDelay(MinDelay* node) {
  // MinDelay(x, delay=0) = x
  // and
  // MinDelay(AfterAll(), delay=…) = AfterAll()
  FunctionBase* f = node->function_base();

  if (node->delay() == 0) {
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(node->operand(0)));
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
    return true;
  }

  Node* operand_node = node->operand(0);
  if (!operand_node->Is<AfterAll>()) {
    return false;
  }
  AfterAll* operand = operand_node->As<AfterAll>();
  if (operand->operand_count() == 0) {
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(operand));
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
    return true;
  }
  return false;
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

absl::StatusOr<bool> CollapseMinDelay(MinDelay* node) {
  // If all users of a MinDelay are MinDelays, then we can collapse them
  // together, e.g.:
  // MinDelay(MinDelay(…, delay=1), delay=2) = MinDelay(…, delay=3)

  if (!node->operand(0)->Is<MinDelay>()) {
    return false;
  }

  FunctionBase* f = node->function_base();
  XLS_RETURN_IF_ERROR(
      node->ReplaceUsesWithNew<MinDelay>(
              node->operand(0)->operand(0),
              node->delay() + node->operand(0)->As<MinDelay>()->delay())
          .status());
  XLS_RETURN_IF_ERROR(f->RemoveNode(node));
  return true;
}

absl::StatusOr<bool> PushDownMinDelay(AfterAll* node) {
  // If all operands of an AfterAll are MinDelays, then we can push the smallest
  // delay through the AfterAll, canonicalizing our delay graph & simplifying
  // later passes. (This also reduces the number of MinDelays if more than one
  // has the minimum delay specified.)
  // e,g.:
  //
  // AfterAll(MinDelay(…, delay=1), MinDelay(…, delay=1), MinDelay(…, delay=3))
  //   =
  // MinDelay(AfterAll(…, …, MinDelay(…, delay=2)), delay=1)

  FunctionBase* f = node->function_base();

  if (node->operand_count() == 0) {
    return false;
  }

  int64_t least_delay = std::numeric_limits<int64_t>::max();
  for (Node* operand : node->operands()) {
    if (!operand->Is<MinDelay>()) {
      return false;
    }
    least_delay = std::min(least_delay, operand->As<MinDelay>()->delay());
  }
  if (least_delay <= 0) {
    return false;
  }

  // Retain these so we can clean them up if we're the only user.
  const std::vector<Node*> operands(node->operands().begin(),
                                    node->operands().end());
  // Prevents iterator invalidation
  const std::vector<Node*> users(node->users().begin(), node->users().end());

  std::vector<Node*> new_operands;
  new_operands.reserve(operands.size());
  for (Node* input_node : operands) {
    MinDelay* input = input_node->As<MinDelay>();
    int64_t new_delay = input->delay() - least_delay;

    Node* new_input = nullptr;
    if (new_delay > 0) {
      XLS_ASSIGN_OR_RETURN(
          new_input,
          f->MakeNode<MinDelay>(SourceInfo(), input->operand(0), new_delay));
    } else {
      new_input = input->operand(0);
    }
    new_operands.push_back(new_input);
  }
  XLS_ASSIGN_OR_RETURN(Node * new_node,
                       f->MakeNode<AfterAll>(SourceInfo(), new_operands));
  XLS_RETURN_IF_ERROR(
      node->ReplaceUsesWithNew<MinDelay>(new_node, least_delay).status());
  XLS_RETURN_IF_ERROR(f->RemoveNode(node));
  return true;
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

    // NOTE: `DependenciesOf` does not find transitive dependencies through
    // Invoke nodes, since it can't tell which outputs have dependency chains to
    // which inputs. By skipping those, this is a more conservative
    // optimization, and is safe to run even before function inlining; it just
    // may miss some optimization opportunities. Rerunning the pass after
    // function inlining will clean up any of these opportunities that remain.
    //
    // In particular, if we assumed Invoke nodes created dependencies from all
    // of their inputs to their output, we would accidentally treat the Invoke
    // node's output as overlapping with all of its input tokens, which could
    // cause us to remove the only path from a side-effecting op's token to the
    // sink... but IR verification couldn't catch the mistake until after
    // function inlining, since the verification logic also treats functions as
    // opaque. (See the CL that introduced this comment for how this could
    // cause important problems.)
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
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext* context) const {
  for (Node* node : f->nodes()) {
    if (NumberOfTokensInType(node->GetType()) > 1) {
      return false;
    }
  }

  bool changed = false;

  for (Node* node : TopoSort(f)) {
    if (!node->Is<MinDelay>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         SimplifyTrivialMinDelay(node->As<MinDelay>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<MinDelay>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         CollapseMinDelay(node->As<MinDelay>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         PushDownMinDelay(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : ReverseTopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         CollapseAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         ReplaceOverlappingAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         RemoveDuplicateAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         SimplifyTrivialAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  if (changed) {
    for (Node* node : TopoSort(f)) {
      if (!node->Is<MinDelay>()) {
        continue;
      }
      XLS_RETURN_IF_ERROR(CollapseMinDelay(node->As<MinDelay>()).status());
    }
  }

  return changed;
}

REGISTER_OPT_PASS(TokenSimplificationPass);

}  // namespace xls

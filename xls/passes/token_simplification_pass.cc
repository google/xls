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
#include <iterator>
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

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

// If x depends on y, AfterAll([…, x, …, y, …]) = AfterAll([…, x, …, x, …])
//
// This function is invoked on x to transform the AfterAll() nodes for which it
// is an operand.
//
// NOTE: We do not find transitive dependencies through Invoke nodes, since it
// can't tell which outputs have dependency chains to which inputs. By skipping
// those, this is a more conservative optimization, and is safe to run even
// before function inlining; it just may miss some optimization opportunities.
// Rerunning the pass after function inlining will clean up any of these
// opportunities that remain.
//
// In particular, if we assumed Invoke nodes created dependencies from all of
// their inputs to their output, we would accidentally treat the Invoke node's
// output as overlapping with all of its input tokens, which could cause us to
// remove the only path from a side-effecting op's token to the sink... but IR
// verification couldn't catch the mistake until after function inlining, since
// the verification logic also treats functions as opaque. (See the CL that
// introduced this comment for how this could cause important problems.)
//
// Perhaps surprisingly, we've observed this transformation as being a
// significant contributor to runtime for large designs. As a performance
// optimization, we take a mapping of nodes to their index in a topo sort. This
// gives us a cheap test to determine if the node we're considering could
// possibly be an intermediate node to a relevant after_all. In practice, this
// seems to be more performant than e.g. computing the transitive closure on
// token-typed nodes' dependencies. If you see this function in a profile, file
// an issue as it might be worth revisiting this in the future.
absl::StatusOr<bool> ReplaceOverlappingAfterAll(
    Node* node, const absl::flat_hash_map<Node*, int32_t>& topo_sort_index) {
  bool changed = false;

  auto can_analyze = [](Node* n) {
    return !n->Is<Invoke>() && TypeHasToken(n->GetType());
  };

  // These should only contain nodes for which can_analyze is true.
  std::vector<Node*> worklist;
  absl::flat_hash_set<Node*> visited;

  bool saw_after_all = false;
  // We only consider updating users of node that are after_all nodes.
  // We record the last topo sort index of node's after_all users so that we can
  // avoid looking at other transitive users later in the topo sort.
  int32_t last_topo_sort_index = -1;
  for (Node* child : node->users()) {
    if (child->Is<AfterAll>()) {
      saw_after_all = true;
      last_topo_sort_index =
          std::max(last_topo_sort_index, topo_sort_index.at(child));
    }
    if (can_analyze(child)) {
      absl::c_copy_if(child->users(), std::back_inserter(worklist),
                      can_analyze);
    }
  }
  if (!saw_after_all) {
    return false;
  }

  while (!worklist.empty()) {
    Node* user = worklist.back();
    worklist.pop_back();
    DCHECK(can_analyze(user));
    if (topo_sort_index.at(user) > last_topo_sort_index) {
      // user can't possibly be a transitive operand of one of node's after_all
      // users, so we can skip it.
      continue;
    }
    auto [_, inserted] = visited.insert(user);
    if (!inserted) {
      continue;
    }
    if (user->Is<AfterAll>() && node->HasUser(user)) {
      absl::Span<Node* const> operands = user->operands();
      // It doesn't matter which operand we replace node with, so just pick one.
      auto iter = absl::c_find_if(
          operands, [node](Node* const operand) { return operand != node; });
      XLS_RET_CHECK(iter != operands.end());
      VLOG(5) << "Replacing " << node->ToString() << " with "
              << (*iter)->ToString() << " in " << user->ToString();
      XLS_RET_CHECK(user->ReplaceOperand(node, *iter));
      changed = true;
    } else {
      absl::c_copy_if(user->users(), std::back_inserter(worklist), can_analyze);
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
    PassResults* results, OptimizationContext& context) const {
  for (Node* node : f->nodes()) {
    if (NumberOfTokensInType(node->GetType()) > 1) {
      return false;
    }
  }

  bool changed = false;

  for (Node* node : context.TopoSort(f)) {
    if (!node->Is<MinDelay>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         SimplifyTrivialMinDelay(node->As<MinDelay>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : context.TopoSort(f)) {
    if (!node->Is<MinDelay>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         CollapseMinDelay(node->As<MinDelay>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : context.TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         PushDownMinDelay(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : context.ReverseTopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         CollapseAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  {
    std::vector<Node*> topo_sort = context.TopoSort(f);
    absl::flat_hash_map<Node*, int32_t> topo_sort_index;
    topo_sort_index.reserve(topo_sort.size());
    for (int32_t i = 0; i < topo_sort.size(); ++i) {
      topo_sort_index[topo_sort[i]] = i;
    }
    for (Node* node : topo_sort) {
      if (node->Is<Invoke>() || !TypeHasToken(node->GetType())) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                           ReplaceOverlappingAfterAll(node, topo_sort_index));
      changed = changed || subpass_changed;
    }
  }

  for (Node* node : context.TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         RemoveDuplicateAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  for (Node* node : context.TopoSort(f)) {
    if (!node->Is<AfterAll>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool subpass_changed,
                         SimplifyTrivialAfterAll(node->As<AfterAll>()));
    changed = changed || subpass_changed;
  }

  if (changed) {
    for (Node* node : context.TopoSort(f)) {
      if (!node->Is<MinDelay>()) {
        continue;
      }
      XLS_RETURN_IF_ERROR(CollapseMinDelay(node->As<MinDelay>()).status());
    }
  }

  return changed;
}

}  // namespace xls

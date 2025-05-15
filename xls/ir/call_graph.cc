// Copyright 2022 The XLS Authors
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

#include "xls/ir/call_graph.h"

#include <iterator>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls {

namespace {
// Returns the function called directly by the given node. Nodes which call
// functions include: map, invoke, etc. If the node does not call a function
// std::nullopt is returned.
std::optional<Function*> CalledFunction(Node* node) {
  switch (node->op()) {
    case Op::kCountedFor:
      return node->As<CountedFor>()->body();
    case Op::kDynamicCountedFor:
      return node->As<DynamicCountedFor>()->body();
    case Op::kMap:
      return node->As<Map>()->to_apply();
    case Op::kInvoke:
      return node->As<Invoke>()->to_apply();
    default:
      return std::nullopt;
  }
}

// Returns the functions called directly by the nodes of the given FunctionBase.
std::vector<FunctionBase*> CalledFunctions(FunctionBase* function_base) {
  absl::flat_hash_set<FunctionBase*> called_set;
  std::vector<FunctionBase*> called;
  for (Node* node : function_base->nodes()) {
    if (std::optional<Function*> callee = CalledFunction(node)) {
      auto [_, inserted] = called_set.insert(callee.value());
      if (inserted) {
        called.push_back(callee.value());
      }
    }
  }
  if (function_base->IsBlock()) {
    // Add instantiations.
    Block* blk = function_base->AsBlockOrDie();
    for (Instantiation* i : blk->GetInstantiations()) {
      if (i->kind() == InstantiationKind::kBlock) {
        Block* instantiated =
            i->AsBlockInstantiation().value()->instantiated_block();
        auto [_, inserted] = called_set.insert(instantiated);
        if (inserted) {
          called.push_back(instantiated);
        }
      }
    }
  }
  return called;
}
}  // namespace

absl::StatusOr<CallGraph> CallGraph::Create(Package* package) {
  CallGraph call_graph_analysis(package);
  // NB We could dedup but its useful to know how many edges there are.
  for (FunctionBase* f : package->GetFunctionBases()) {
    for (Node* node : f->nodes()) {
      std::optional<Function*> callee = CalledFunction(node);
      if (callee.has_value()) {
        call_graph_analysis.callee_functions_[f].push_back(node);
        call_graph_analysis.caller_functions_[*callee].push_back(f);
      }
    }
  }
  return call_graph_analysis;
}

// Recursive DFS visitor of the call graph induced by invoke
// instructions. Builds a post order of functions in the post_order vector.
static void DfsVisit(FunctionBase* f,
                     absl::flat_hash_set<FunctionBase*>* visited,
                     std::vector<FunctionBase*>* post_order) {
  visited->insert(f);
  for (FunctionBase* callee : CalledFunctions(f)) {
    if (!visited->contains(callee)) {
      DfsVisit(callee, visited, post_order);
    }
  }
  post_order->push_back(f);
}

std::vector<FunctionBase*> GetDependentFunctions(FunctionBase* function_base) {
  absl::flat_hash_set<FunctionBase*> visited;
  std::vector<FunctionBase*> post_order;
  DfsVisit(function_base, &visited, &post_order);

  if (VLOG_IS_ON(2)) {
    VLOG(2) << absl::StreamFormat("DependentFunctions(%s):",
                                  function_base->name());
    for (FunctionBase* f : post_order) {
      VLOG(2) << "  " << f->name();
    }
  }
  return post_order;
}

// Returns the functions which are roots in the call graph, that is, the
// functions which are not called by any other functions.
static std::vector<FunctionBase*> GetRootFunctions(const Package* p) {
  absl::flat_hash_set<FunctionBase*> called_functions;
  for (FunctionBase* f : p->GetFunctionBases()) {
    for (FunctionBase* callee : CalledFunctions(f)) {
      called_functions.insert(callee);
    }
  }
  std::vector<FunctionBase*> roots;
  for (FunctionBase* f : p->GetFunctionBases()) {
    if (!called_functions.contains(f)) {
      roots.push_back(f);
    }
  }
  return roots;
}

std::vector<FunctionBase*> FunctionsInPostOrder(const Package* p) {
  absl::flat_hash_set<FunctionBase*> visited;
  std::vector<FunctionBase*> post_order;
  for (FunctionBase* f : GetRootFunctions(p)) {
    DfsVisit(f, &visited, &post_order);
  }
  return post_order;
}

absl::StatusOr<Function*> CloneFunctionAndItsDependencies(
    Function* to_clone, std::string_view new_name, Package* target_package,
    absl::flat_hash_map<const Function*, Function*> call_remapping) {
  std::vector<FunctionBase*> dependent_functions =
      GetDependentFunctions(to_clone);
  for (const FunctionBase* dependent_function : dependent_functions) {
    if (dependent_function == dependent_functions.back()) {
      continue;
    }
    XLS_RET_CHECK(dependent_function->IsFunction());
    auto function = static_cast<const Function*>(dependent_function);
    XLS_ASSIGN_OR_RETURN(
        Function * dependent_function_clone,
        function->Clone(function->name(), target_package, call_remapping));
    call_remapping.insert({function, dependent_function_clone});
  }
  return to_clone->Clone(new_name, target_package, call_remapping);
}

std::vector<Node*> GetNodesWhichCall(Function* f) {
  std::vector<Node*> nodes;
  for (FunctionBase* other : f->package()->GetFunctionBases()) {
    if (other == f) {
      continue;
    }
    absl::c_copy_if(other->nodes(), std::back_inserter(nodes),
                    [&](Node* n) { return f == CalledFunction(n); });
  }
  return nodes;
}

}  // namespace xls

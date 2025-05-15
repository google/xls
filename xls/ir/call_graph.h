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

#ifndef XLS_IR_CALL_GRAPH_H_
#define XLS_IR_CALL_GRAPH_H_

#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {

class CallGraph {
 public:
  static absl::StatusOr<CallGraph> Create(Package* package);

  // All invoke/map/etc that 'fb' directly contains.
  absl::Span<Node* const> FunctionsCalledBy(FunctionBase* fb) const {
    if (!callee_functions_.contains(fb)) {
      return absl::Span<Node* const>();
    }
    return callee_functions_.at(fb);
  }

  // All functions which call 'fb'. NB A function called multiple times will be
  // included more than once in this list.
  absl::Span<FunctionBase* const> FunctionsWhichCall(FunctionBase* fb) const {
    if (!caller_functions_.contains(fb)) {
      return absl::Span<FunctionBase* const>();
    }
    return caller_functions_.at(fb);
  }

 private:
  explicit CallGraph(Package* package) : package_(package) {}
  Package* package_;
  absl::flat_hash_map<FunctionBase*, std::vector<Node*>> callee_functions_;
  absl::flat_hash_map<FunctionBase*, std::vector<FunctionBase*>>
      caller_functions_;
};

// Returns the functions called and blocks instantiated transitively by the
// given FunctionBase. Called functions/instantiated blocks are returned before
// callee/instantiator FunctionBases in the returned order. The final element in
// the returned vector is `function_base`.
std::vector<FunctionBase*> GetDependentFunctions(FunctionBase* function_base);

// Clones transitively the given function and its dependencies.
absl::StatusOr<Function*> CloneFunctionAndItsDependencies(
    Function* to_clone, std::string_view new_name,
    Package* target_package = nullptr,
    absl::flat_hash_map<const Function*, Function*> call_remapping = {});

// Returns the functions in package 'p' in a DFS post order traversal of the
// call graph induced by function-invoking nodes. Called FunctionBases are
// returned before callee FunctionBases in the returned order.
std::vector<FunctionBase*> FunctionsInPostOrder(const Package* p);

// Returns a list of all the nodes which reference the given function as a
// target.
std::vector<Node*> GetNodesWhichCall(Function* f);

// Returns true if node 'n' targets function 'f'.
bool FunctionIsTargetedBy(Function* f, Node* n);

}  // namespace xls

#endif  // XLS_IR_CALL_GRAPH_H_

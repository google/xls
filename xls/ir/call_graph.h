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

#include <vector>

#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

// Returns the function called directly by the given node. Nodes which call
// functions include: map, invoke, etc. If the node does not call a function
// std::nullopt is returned.
std::optional<Function*> CalledFunction(Node* node);

// Returns the functions called directly by the nodes of the given FunctionBase.
std::vector<Function*> CalledFunctions(FunctionBase* f);

// Returns the functions called transitively by the given FunctionBase. Called
// functions are returned before callee FunctionBases in the returned order. The
// final element in the returned vector is `function_base`.
std::vector<FunctionBase*> GetDependentFunctions(FunctionBase* function_base);

// Clones transitively the given function and its dependencies.
absl::StatusOr<Function*> CloneFunctionAndItsDependencies(
    Function* to_clone, std::string_view new_name,
    Package* target_package = nullptr,
    absl::flat_hash_map<const Function*, Function*> call_remapping = {});

// Returns the functions in package 'p' in a DFS post order traversal of the
// call graph induced by function-invoking nodes. Called FunctionBases are
// returned before callee FunctionBases in the returned order.
std::vector<FunctionBase*> FunctionsInPostOrder(Package* p);

}  // namespace xls

#endif  // XLS_IR_CALL_GRAPH_H_

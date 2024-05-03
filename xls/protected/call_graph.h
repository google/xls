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

#ifndef XLS_PROTECTED_CALL_GRAPH_H_
#define XLS_PROTECTED_CALL_GRAPH_H_

// This file reproduces a subset of call_graph.h. It is being used externally
// but isn't the most generically useful function, so we put it in the
// "protected" directory.
//
// This function is subject to change.

#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"

namespace xls {

// Clones transitively the given function and its dependencies.
absl::StatusOr<Function*> CloneFunctionAndItsDependencies(
    Function* to_clone, std::string_view new_name,
    Package* target_package = nullptr,
    absl::flat_hash_map<const Function*, Function*> call_remapping = {});

}  // namespace xls

#endif  // XLS_PROTECTED_CALL_GRAPH_H_

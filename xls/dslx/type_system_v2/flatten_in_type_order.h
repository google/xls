// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_FLATTEN_IN_TYPE_ORDER_
#define XLS_DSLX_TYPE_SYSTEM_V2_FLATTEN_IN_TYPE_ORDER_

#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/import_data.h"

namespace xls::dslx {

// Flattens an AST subtree to a vector in a topological order of type
// information flow. This means any node y whose type depends on either the type
// or constant value of a node x will appear after x in the returned vector. If
// `include_parametric_entities` is false, then the flattening ignores any
// encountered parametric functions/procs and does not include even their roots.
absl::StatusOr<std::vector<const AstNode*>> FlattenInTypeOrder(
    const ImportData& import_data, const AstNode* root,
    bool include_parametric_entities);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_FLATTEN_IN_TYPE_ORDER_

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
#ifndef XLS_DSLX_FRONTEND_AST_CLONER_H_
#define XLS_DSLX_FRONTEND_AST_CLONER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

// Creates a deep copy of the given AST node (inside the same module). All nodes
// in the tree are duplicated.
absl::StatusOr<AstNode*> CloneAst(AstNode* root);

// Variant of CloneAst that won't clone type definitions -- this is useful for
// e.g. cloning return types without recursing into cloned definitions which
// would change nominal types.
absl::StatusOr<AstNode*> CloneAstSansTypeDefinitions(AstNode* root);

// Helper wrapper for the above that downcasts the result to the given
// apparent type (derived type of AstNode).
template <typename T>
inline absl::StatusOr<T*> CloneNodeSansTypeDefinitions(T* root) {
  XLS_ASSIGN_OR_RETURN(AstNode * cloned, CloneAstSansTypeDefinitions(root));
  return down_cast<T*>(cloned);
}

absl::StatusOr<std::unique_ptr<Module>> CloneModule(Module* module);

// Verifies that the AST node tree rooted at `new_root` does not contain any of
// the AST nodes in the tree rooted at `old_root`. In practice, this will verify
// that a clone doesn't contain any 'old' AST nodes.
absl::Status VerifyClone(const AstNode* old_root, const AstNode* new_root);

// Helper for CloneAst that uses the apparent (derived) type given by the
// parameter as the return type. (This helps encapsulate casts to be safer.)
template <typename T>
inline absl::StatusOr<T*> CloneNode(T* node) {
  XLS_ASSIGN_OR_RETURN(AstNode * cloned, CloneAst(node));
  return down_cast<T*>(cloned);
}

// Helper that vectorizes the CloneNode routine.
template <typename T>
inline absl::StatusOr<std::vector<T*>> CloneNodes(absl::Span<T* const> nodes) {
  std::vector<T*> results;
  results.reserve(nodes.size());
  for (T* n : nodes) {
    XLS_ASSIGN_OR_RETURN(T * cloned, CloneNode<T>(n));
    results.push_back(cloned);
  }
  return results;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_CLONER_H_

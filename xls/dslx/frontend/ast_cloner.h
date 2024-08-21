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
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

// A function that can be used to override the cloning behavior for certain
// nodes during a `CloneAst` operations. A replacer can be used to replace
// targeted nodes with something else entirely, or it can just "clone" those
// nodes differently than the default logic.
using CloneReplacer =
    absl::AnyInvocable<absl::StatusOr<std::optional<AstNode*>>(const AstNode*)>;

// This function is directly usable as the `replacer` argument for `CloneAst`
// when a direct clone with no replacements is desired.
inline std::optional<AstNode*> NoopCloneReplacer(const AstNode* original_node) {
  return std::nullopt;
}

// A replacer function that performs shallow clones of `TypeRef` nodes, pointing
// the clone to the original `TypeDefinition` object. This is useful for e.g.
// cloning return types without recursing into cloned definitions which would
// change nominal types.
std::optional<AstNode*> PreserveTypeDefinitionsReplacer(
    const AstNode* original_node);

// Creates a `CloneReplacer` that replaces references to the given `def` with
// the given `replacement`.
CloneReplacer NameRefReplacer(const NameDef* def, Expr* replacement);

// Creates a deep copy of the given AST node (inside the same module), generally
// duplicating all nodes. The given `replacer` may override whether and how a
// given node gets duplicated. The `replacer` is invoked for each original node
// about to be cloned. If it returns `nullopt` (which is the default), then
// cloning proceeds as normal. If it returns an `AstNode*`, then that pointer is
// used as a wholesale replacement subtree, and cloning does not delve into the
// children of the original node.
absl::StatusOr<AstNode*> CloneAst(AstNode* root,
                                  CloneReplacer replacer = &NoopCloneReplacer);

absl::StatusOr<std::unique_ptr<Module>> CloneModule(
    Module* module, CloneReplacer replacer = &NoopCloneReplacer);

// Verifies that the AST node tree rooted at `new_root` does not contain any of
// the AST nodes in the tree rooted at `old_root`. In practice, this will verify
// that a clone doesn't contain any 'old' AST nodes.
absl::Status VerifyClone(const AstNode* old_root, const AstNode* new_root);

// Helper for CloneAst that uses the apparent (derived) type given by the
// parameter as the return type. (This helps encapsulate casts to be safer.)
template <typename T>
inline absl::StatusOr<T*> CloneNode(
    T* node, CloneReplacer replacer = &NoopCloneReplacer) {
  XLS_ASSIGN_OR_RETURN(AstNode * cloned, CloneAst(node, std::move(replacer)));
  return down_cast<T*>(cloned);
}

// Helper that vectorizes the CloneNode routine.
template <typename T>
inline absl::StatusOr<std::vector<T*>> CloneNodes(
    absl::Span<T* const> nodes, CloneReplacer replacer = &NoopCloneReplacer) {
  std::vector<T*> results;
  results.reserve(nodes.size());
  for (T* n : nodes) {
    XLS_ASSIGN_OR_RETURN(T * cloned, CloneNode<T>(n, replacer));
    results.push_back(cloned);
  }
  return results;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_CLONER_H_

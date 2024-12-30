// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_ZIP_AST_H_
#define XLS_DSLX_FRONTEND_ZIP_AST_H_

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

// Traverses `lhs` and `rhs`, invoking `lhs_visitor` and then `rhs_visitor` for
// each corresponding node pair.
//
// The expectation is that `lhs` and `rhs` are structurally equivalent, meaning
// each corresponding node is of the same class and has the same number of
// children.
//
// If a structural mismatch is encountered, then the `accept_mismatch_callback`
// is invoked, and if it errors, the error is propagated out of `ZipAst`. On
// success of `accept_mismatch_callback`, the mismatching subtree is ignored,
// and `ZipAst` proceeds.
absl::Status ZipAst(
    const AstNode* lhs, const AstNode* rhs, AstNodeVisitor* lhs_visitor,
    AstNodeVisitor* rhs_visitor,
    absl::AnyInvocable<absl::Status(const AstNode*, const AstNode*)>
        accept_mismatch_callback);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_V2_ZIP_AST_H_

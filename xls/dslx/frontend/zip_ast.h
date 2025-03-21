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
#include "absl/strings/substitute.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

// Options for the behavior of `ZipAst`.
struct ZipAstOptions {
  // Whether to consider it a mismatch if an LHS `NameRef` refers to a different
  // def than the RHS one.
  bool check_defs_for_name_refs = false;

  // If true, then two `NameRef`s to the same `ParametricBinding` of the same
  // function or struct are considered different. This means the caller is
  // considering the two ASTs being zipped to be in two different parametric
  // contexts, so that, for example `uN[N]` must have a different instantiation
  // of `N` on one side than the other, even though it's the same `N` at the AST
  // level. Note that references to *different* parametric bindings that look
  // the same (e.g. the `N` in two different functions) fall under the purview
  // of the `check_defs_for_name_refs` flag.
  bool refs_to_same_parametric_are_different = false;

  // The callback for handling mismatches. By default, this generates an error.
  absl::AnyInvocable<absl::Status(const AstNode*, const AstNode*)>
      accept_mismatch_callback =
          [](const AstNode* x, const AstNode* y) -> absl::Status {
    return absl::InvalidArgumentError(
        absl::Substitute("Mismatch: $0 vs. $1", x->ToString(), y->ToString()));
  };
};

// Traverses `lhs` and `rhs`, invoking `lhs_visitor` and then `rhs_visitor` for
// each corresponding node pair.
//
// The expectation is that `lhs` and `rhs` are structurally equivalent, meaning
// each corresponding node is of the same class and has the same number of
// children.
//
// If a structural mismatch is encountered, then
// `options.accept_mismatch_callback` is invoked, and if it errors, the error is
// propagated out of `ZipAst`. On success of `options.accept_mismatch_callback`,
// the mismatching subtree is ignored, and `ZipAst` proceeds.
absl::Status ZipAst(const AstNode* lhs, const AstNode* rhs,
                    AstNodeVisitor* lhs_visitor, AstNodeVisitor* rhs_visitor,
                    ZipAstOptions options = ZipAstOptions{});

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_V2_ZIP_AST_H_

// Copyright 2020 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_DEDUCE_H_
#define XLS_DSLX_TYPE_SYSTEM_DEDUCE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Returns the deduced type for "node", or an error status.
//
// On success, adds node to DeduceCtx::type_info memoization map as a side
// effect.
//
// Note: due to transitivity, as a side effect the type_info mapping is filled
// with all the deductions that were necessary to determine (deduce) the
// resulting type of `node`.
absl::StatusOr<std::unique_ptr<Type>> Deduce(const AstNode* node,
                                             DeduceCtx* ctx);

// Resolves "type_" via provided symbolic bindings.
//
// Uses the symbolic bindings of the function we're currently inside of to
// resolve parametric types.
//
// Args:
//  type: Type to resolve any contained dims for.
//  ctx: Deduction context to use in resolving the dims.
//
// Returns:
//  "type" with dimensions resolved according to bindings in "ctx".
absl::StatusOr<std::unique_ptr<Type>> Resolve(const Type& type, DeduceCtx* ctx);

// Helpers that sequences Deduce, then Resolve.
absl::StatusOr<std::unique_ptr<Type>> DeduceAndResolve(const AstNode* node,
                                                       DeduceCtx* ctx);

// Converts an AST expression in "dimension position" (e.g. in an array type
// annotation's size) and converts it into a `TypeDim` value that can be used
// in, e.g., an `ArrayType`. The result is either a constexpr-evaluated value or
// a `ParametricExpression` (for a parametric binding that has not yet been
// defined).
//
// Note: this is not capable of evaluating more complex ASTs; it assumes
// something is either fully constexpr-evaluatable, or symbolic.
absl::StatusOr<TypeDim> DimToConcreteUsize(const Expr* dim_expr,
                                           DeduceCtx* ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_H_

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

// These are self-contained Expr (AST node) deduction rules. They are broken out
// into their own translation unit as they can be cleanly separated from the
// infrastructure they use.

#ifndef XLS_DSLX_TYPE_SYSTEM_DEDUCE_EXPR_H_
#define XLS_DSLX_TYPE_SYSTEM_DEDUCE_EXPR_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<Type>> DeduceArray(const Array* node,
                                                  DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceConstantArray(
    const ConstantArray* node, DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceNumber(const Number* node,
                                                   DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceString(const String* string,
                                                   DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceConditional(const Conditional* node,
                                                        DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceUnop(const Unop* node,
                                                 DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceBinop(const Binop* node,
                                                  DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceTupleIndex(const TupleIndex* node,
                                                       DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceXlsTuple(const XlsTuple* node,
                                                     DeduceCtx* ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_EXPR_H_

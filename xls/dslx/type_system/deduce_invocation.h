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

#ifndef XLS_DSLX_TYPE_SYSTEM_DEDUCE_INVOCATION_H_
#define XLS_DSLX_TYPE_SYSTEM_DEDUCE_INVOCATION_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/ast_env.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<Type>> DeduceInvocation(const Invocation* node,
                                                       DeduceCtx* ctx);

// Helper that deduces the concrete types of the arguments and appends them to
// "instantiate_args" -- getting them in this form is often a pre-step in
// instantiating a parametric function or proc.
absl::Status AppendArgsForInstantiation(
    const Instantiation* inst, const Expr* callee, absl::Span<Expr* const> args,
    DeduceCtx* ctx, std::vector<InstantiateArg>* instantiate_args);

// Generic function to do the heavy lifting of deducing the type of an
// Invocation or Spawn's constituent functions.
absl::StatusOr<TypeAndParametricEnv> DeduceInstantiation(
    DeduceCtx* ctx, const Invocation* invocation,
    const std::function<absl::StatusOr<Function*>(const Instantiation*,
                                                  DeduceCtx*)>& resolve_fn,
    const AstEnv& constexpr_env);

absl::StatusOr<std::unique_ptr<Type>> DeduceFormatMacro(const FormatMacro* node,
                                                        DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceZeroMacro(const ZeroMacro* node,
                                                      DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceAllOnesMacro(
    const AllOnesMacro* node, DeduceCtx* ctx);
}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_INVOCATION_H_

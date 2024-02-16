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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPECHECK_INVOCATION_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPECHECK_INVOCATION_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

absl::StatusOr<TypeAndParametricEnv> TypecheckInvocation(
    DeduceCtx* ctx, const Invocation* invocation,
    const absl::flat_hash_map<std::variant<const Param*, const ProcMember*>,
                              InterpValue>& constexpr_env);

// Typechecks the function's parametrics' and parameters' types.
//
// Returns the deduced types for all of the parameters of "f".
absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>>
TypecheckFunctionParams(Function& f, DeduceCtx* ctx);

// Interprets an expression we're forced to evaluate at typechecking-time.
absl::StatusOr<InterpValue> InterpretExpr(
    ImportData* import_data, TypeInfo* type_info, Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPECHECK_INVOCATION_H_

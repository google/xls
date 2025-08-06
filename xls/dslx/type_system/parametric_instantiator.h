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

#ifndef XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_H_
#define XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_H_

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"

namespace xls::dslx {

// Instantiates a function invocation using the bindings derived from args'
// types.
//
// Args:
//  span: Invocation span causing the instantiation to occur.
//  callee_fn: The function being instantiated.
//  function_type: Type (possibly parametric) of the function being
//    instantiated.
//  args: Arguments driving the instantiation of the function signature.
//  ctx: Type deduction context, e.g. used in constexpr evaluation.
//  typed_parametrics: Contains expressions being given as parametrics that
//    must be evaluated along with their inferred types.
//  explicit_bindings: Environment to use for evaluating the
//    typed_parametrics expressions; e.g. for the example above if the
//    caller invoked `const M: u32 = 42; f<M>(x)`, this environment would
//    be `{N: u32:42}` (since M is passed as the N value for the callee).
absl::StatusOr<TypeAndParametricEnv> InstantiateFunction(
    Span span, Function& callee_fn, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricWithType> typed_parametrics,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_bindings,
    absl::Span<const ParametricBinding* absl_nonnull const>
        parametric_bindings);

// Instantiates a struct using the bindings derived from args' types.
//
// See InstantiateFunction() above.
absl::StatusOr<TypeAndParametricEnv> InstantiateStruct(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<Type> const> member_types, DeduceCtx* ctx,
    absl::Span<const ParametricWithType> typed_parametrics,
    absl::Span<const ParametricBinding* absl_nonnull const>
        parametric_bindings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_H_

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

#ifndef XLS_DSLX_TYPE_SYSTEM_CPP_PARAMETRIC_INSTANTIATOR_H_
#define XLS_DSLX_TYPE_SYSTEM_CPP_PARAMETRIC_INSTANTIATOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_constraint.h"
#include "xls/dslx/type_system/type_and_bindings.h"

namespace xls::dslx {

// Instantiates a function invocation using the bindings derived from args'
// types.
//
// Args:
//  span: Invocation span causing the instantiation to occur.
//  function_type: Type (possibly parametric) of the function being
//    instantiated.
//  args: Arguments driving the instantiation of the function signature.
//  ctx: Type deduction context, e.g. used in constexpr evaluation.
//  parametric_constraints: Contains expressions being given as parametrics that
//    must be evaluated. They are called "constraints" because they may be
//    in conflict as a result of deductive inference; e.g. for
//    `f<N: u32, R: u32 = N+N>(x: bits[N]) -> bits[R] { x }` we'll find the
//    "constraint" on R of being `N+N` is incorrect/infeasible (when N != 0).
//  explicit_bindings: Environment to use for evaluating the
//    parametric_constraints expressions; e.g. for the example above if the
//    caller invoked `const M: u32 = 42; f<M>(x)`, this environment would
//    be `{N: u32:42}` (since M is passed as the N value for the callee).
absl::StatusOr<TypeAndBindings> InstantiateFunction(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_bindings);

// Instantiates a struct using the bindings derived from args' types.
//
// See InstantiateFunction() above.
absl::StatusOr<TypeAndBindings> InstantiateStruct(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<ConcreteType> const> member_types,
    DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_CPP_PARAMETRIC_INSTANTIATOR_H_

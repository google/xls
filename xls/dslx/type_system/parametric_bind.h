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

#ifndef XLS_DSLX_TYPE_SYSTEM_SYMBOLIC_BIND_H_
#define XLS_DSLX_TYPE_SYSTEM_SYMBOLIC_BIND_H_

#include <memory>
#include <string>

#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Attrs:
//  span: The span where this argument is being passed in the invocation, used
//    in returned positional error messages.
//  parametric_binding_types: Mapping from parametric symbol to its type; e.g.
//    in the signature for `p` below, this would map "p" to BitsType u32.
//  parametric_default_exprs: Mapping from parametric symbol to its "default
//    expression"; e.g. in `p<X: u32, Y: u32 = X+X>` the expression for X would
//    be nullptr and for Y would be `X+X`.
//  parametric_env: The environment we're populating with parametric bindings
//    (symbols to interpreter values).
//  deduce_ctx: Made available for error reporting.
struct ParametricBindContext {
  const Span& span;
  const absl::flat_hash_map<std::string, std::unique_ptr<Type>>&
      parametric_binding_types;
  const absl::flat_hash_map<std::string, Expr*>& parametric_default_exprs;
  absl::flat_hash_map<std::string, InterpValue>& parametric_env;
  DeduceCtx& deduce_ctx;
};

// Binds a formal argument's parametric symbol via an actual argument, if
// applicable.
//
// Returns an appropriate error if the binding has a type error or is in
// conflict with a previous binding.
//
// This is key to the process that populates parametrics automatically from
// actual arguments; e.g.
//
//    fn p<X: u32>(x: uN[X])
//
// Implicitly gets X=8 bound via this invocation using an actual argument in
// the caller where we know the concrete type:
//
//    p(u8:42)
//
// Args:
//  param_type: The type of the (formal) parameter.
//  param_dim: The dimension held by param_type, this may be parametric.
//  arg_type: The type of the *actual* argument being passed to the formal
//    parameter.
//  arg_dim: The dimension held by arg_type.
//  ctx: The parametric binding context.
//  deduce_ctx: Used for reporting errors.
absl::Status ParametricBindTypeDim(const Type& param_type,
                                   const TypeDim& param_dim,
                                   const Type& arg_type, const TypeDim& arg_dim,
                                   ParametricBindContext& ctx);

// As described above in ParametricBindTypeDim, but handles arbitrary
// parameter types / argument type bindings.
absl::Status ParametricBind(const Type& param_type, const Type& arg_type,
                            ParametricBindContext& ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_SYMBOLIC_BIND_H_

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

// Describes (parametric) builtin functions and how to typecheck those builtins.

#ifndef XLS_DSLX_DSLX_BUILTINS_SIGNATURES_H_
#define XLS_DSLX_DSLX_BUILTINS_SIGNATURES_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_constraint.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"

namespace xls::dslx {

// Data provided to the lambdas below (of type SignatureFn) that help us do
// deduction and handling for parametric builtin function instantiation.
struct SignatureData {
  // Argument types for the parametric builtin.
  const std::vector<const Type*>& arg_types;
  const std::vector<Span>& arg_spans;
  const std::vector<ExprOrType>& arg_explicit_parametrics;
  // Name of the builtin.
  std::string_view name;
  // Span that we're invoking the builtin from (span for the entire invocation).
  const Span& span;
  // Any "higher order" parametric bindings e.g. for the callee in the case of
  // map.
  std::optional<std::vector<ParametricConstraint>> parametric_bindings;
  // Callback that can be used to perform constexpr evaluation on one of the
  // function arguments; which is requested is given by argno.
  const std::function<absl::StatusOr<InterpValue>(int64_t argno)>&
      constexpr_eval;

  // Argument expressions -- note that these should generally not be needed by
  // type inferencing routines for builtins, except in special cases like map()
  // where we know an argument must be a literal function reference that we may
  // want to use in instantiation machinery. (Even in that case ideally we'd
  // only need the types, but we use the function in some cases for soundness
  // checking.)
  absl::Span<Expr* const> args;
};

// Deduction rule that determines the FunctionType and any associated symbolic
// bindings for a parametric builtin function.
using SignatureFn = std::function<absl::StatusOr<TypeAndParametricEnv>(
    const SignatureData&, DeduceCtx*)>;

// Returns a function that produces the type of builtin_name.
//
// Many builtins are parametric, and so the function type is determined (or type
// errors are raised) based on the types that are presented to it as arguments.
//
// The returned function is then invoked (conceptually) as:
//
//     fsignature = get_fsignature(builtin_name)
//     fn_type, parametric_env = fsignature(arg_types, builtin_name,
//                                             invocation_span)
//
// Where the second line provides the argument types presented to the builtin.
//
// This is similar conceptually to type deduction, just the builtin functions
// have no definitions in the source code, and sometimes we do fancier rules
// than we currently have support for in the DSL. As parametric support grows,
// however, one day these may all be a special "builtin" module.
absl::StatusOr<SignatureFn> GetParametricBuiltinSignature(
    std::string_view builtin_name);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DSLX_BUILTINS_SIGNATURES_H_

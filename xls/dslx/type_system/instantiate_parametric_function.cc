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

#include "xls/dslx/type_system/instantiate_parametric_function.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_instantiator.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Walks through all of the explicit parametric values that the
// invocation provided; e.g.
//
//    foo<A, B, C>(...)
//       ^~~~~~~~^------- these!
static absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
EvaluateExplicitParametrics(
    DeduceCtx* ctx, DeduceCtx* parent_ctx,
    absl::Span<ExprOrType const> explicit_parametrics,
    absl::Span<const ParametricBinding* const> parametric_bindings,
    const Span& callee_span) {
  absl::flat_hash_map<std::string, InterpValue> callee_parametric_env;

  for (int64_t i = 0; i < explicit_parametrics.size(); ++i) {
    const ParametricBinding* binding = parametric_bindings[i];
    ExprOrType eot = explicit_parametrics[i];

    // We cannot currently provide types to user-defined parametric functions.
    if (!std::holds_alternative<Expr*>(eot)) {
      auto* type_annotation = std::get<TypeAnnotation*>(eot);
      return TypeInferenceErrorStatus(
          type_annotation->span(), nullptr,
          absl::StrFormat("Parametric function invocation cannot take "
                          "type `%s` -- parametric must be an expression",
                          type_annotation->ToString()));
    }

    auto* parametric_expr = std::get<Expr*>(eot);

    VLOG(5) << "Populating callee parametric `" << binding->ToString()
            << "` via invocation expression: " << parametric_expr->ToString();

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> binding_type,
                         ParametricBindingToType(*binding, ctx));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_expr_type,
                         parent_ctx->Deduce(parametric_expr));

    if (*binding_type != *parametric_expr_type) {
      return ctx->TypeMismatchError(callee_span, nullptr, *binding_type,
                                    parametric_expr, *parametric_expr_type,
                                    "Explicit parametric type mismatch.");
    }

    // We have to be at least one fn deep to be instantiating a parametric, so
    // referencing fn_stack::back is safe.
    XLS_RET_CHECK(!parent_ctx->fn_stack().empty());

    // Evaluate the explicit parametric expression -- note that we have to do
    // this in the /caller's/ parametric env, since that's where the expression
    // is.
    XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
        parent_ctx->import_data(), parent_ctx->type_info(),
        parent_ctx->warnings(),
        /*bindings=*/parent_ctx->GetCurrentParametricEnv(),
        /*expr=*/parametric_expr,
        /*type=*/parametric_expr_type.get()));

    // The value we're instantiating the function with must be constexpr -- we
    // can't instantiate with values determined at runtime, of course.
    if (!parent_ctx->type_info()->IsKnownConstExpr(parametric_expr)) {
      return TypeInferenceErrorStatus(
          parametric_expr->span(), parametric_expr_type.get(),
          absl::StrFormat("Parametric expression `%s` was not constexpr -- "
                          "parametric values must be compile-time constants",
                          parametric_expr->ToString()));
    }

    XLS_ASSIGN_OR_RETURN(
        InterpValue const_expr,
        parent_ctx->type_info()->GetConstExpr(parametric_expr));
    callee_parametric_env.insert(
        {binding->identifier(), std::move(const_expr)});
  }

  return callee_parametric_env;
}

absl::StatusOr<TypeAndParametricEnv> InstantiateParametricFunction(
    DeduceCtx* ctx, DeduceCtx* parent_ctx, const Invocation* invocation,
    Function& callee_fn, const FunctionType& fn_type,
    const std::vector<InstantiateArg>& instantiate_args) {
  VLOG(5) << "InstantiateParametricFunction; callee_fn: "
          << callee_fn.identifier() << " invocation: `"
          << invocation->ToString() << "` @ " << invocation->span();

  {
    // As a special case, flag recursion (that otherwise shows up as unresolved
    // parametrics), so the cause is more clear to the user (vs the symptom).
    Function* const current = &callee_fn;
    CHECK(current != nullptr);
    for (int64_t i = 0; i < ctx->fn_stack().size(); ++i) {
      Function* previous = (ctx->fn_stack().rbegin() + i)->f();
      if (current == previous) {
        return TypeInferenceErrorStatus(
            invocation->span(), nullptr,
            absl::StrFormat("Recursive call to `%s` detected during "
                            "type-checking -- recursive calls are unsupported.",
                            callee_fn.identifier()));
      }
    }
  }

  const std::vector<ParametricBinding*>& parametric_bindings =
      callee_fn.parametric_bindings();
  if (invocation->explicit_parametrics().size() > parametric_bindings.size()) {
    return ArgCountMismatchErrorStatus(
        invocation->span(),
        absl::StrFormat(
            "Too many parametric values supplied; limit: %d given: %d",
            parametric_bindings.size(),
            invocation->explicit_parametrics().size()));
  }

  using ParametricEnvMap = absl::flat_hash_map<std::string, InterpValue>;
  XLS_ASSIGN_OR_RETURN(ParametricEnvMap callee_parametric_env,
                       EvaluateExplicitParametrics(
                           ctx, parent_ctx, invocation->explicit_parametrics(),
                           parametric_bindings, invocation->callee()->span()));

  // The bindings that were not explicitly filled by the caller are taken from
  // the callee directly; e.g. if caller invokes as `parametric()` it supplies
  // 0 parmetrics directly, but callee may have:
  //
  //    `fn parametric<N: u32 = 5>() { ... }`
  //
  // and thus needs the `N: u32 = 5` to be filled here.
  absl::Span<ParametricBinding* const> non_explicit =
      absl::MakeSpan(parametric_bindings)
          .subspan(invocation->explicit_parametrics().size());
  std::vector<ParametricWithType> typed_parametrics;
  for (ParametricBinding* remaining_binding : non_explicit) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> binding_type,
                         ParametricBindingToType(*remaining_binding, ctx));
    typed_parametrics.push_back(
        ParametricWithType(*remaining_binding, std::move(binding_type)));
  }

  return InstantiateFunction(invocation->span(), callee_fn, fn_type,
                             instantiate_args, ctx, typed_parametrics,
                             callee_parametric_env, parametric_bindings);
}

}  // namespace xls::dslx

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
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_constraint.h"
#include "xls/dslx/type_system/parametric_instantiator.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<ConcreteType>> ParametricBindingToType(
    ParametricBinding* binding, DeduceCtx* ctx) {
  Module* binding_module = binding->owner();
  ImportData* import_data = ctx->import_data();
  XLS_ASSIGN_OR_RETURN(TypeInfo * binding_type_info,
                       import_data->GetRootTypeInfo(binding_module));
  auto binding_ctx = ctx->MakeCtx(binding_type_info, binding_module);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> metatype,
                       binding_ctx->Deduce(binding->type_annotation()));
  return UnwrapMetaType(std::move(metatype), binding->type_annotation()->span(),
                        "parametric binding type");
}

absl::StatusOr<TypeAndParametricEnv> InstantiateParametricFunction(
    DeduceCtx* ctx, DeduceCtx* parent_ctx, const Invocation* invocation,
    Function& callee_fn, const FunctionType& fn_type,
    const std::vector<InstantiateArg>& instantiate_args) {
  XLS_VLOG(5) << "InstantiateParametricFunction; callee_fn: "
              << callee_fn.identifier();

  {
    // As a special case, flag recursion (that otherwise shows up as unresolved
    // parametrics), so the cause is more clear to the user (vs the symptom).
    Function* const current = &callee_fn;
    XLS_CHECK(current != nullptr);
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
  absl::flat_hash_map<std::string, InterpValue> explicit_bindings;
  std::vector<ParametricConstraint> parametric_constraints;
  parametric_constraints.reserve(callee_fn.parametric_bindings().size());
  if (invocation->explicit_parametrics().size() > parametric_bindings.size()) {
    return ArgCountMismatchErrorStatus(
        invocation->span(),
        absl::StrFormat(
            "Too many parametric values supplied; limit: %d given: %d",
            parametric_bindings.size(),
            invocation->explicit_parametrics().size()));
  }

  for (int64_t i = 0; i < invocation->explicit_parametrics().size(); ++i) {
    ParametricBinding* binding = parametric_bindings[i];
    ExprOrType eot = invocation->explicit_parametrics()[i];

    // We cannot currently provide types to user-defined parametric functions.
    if (!std::holds_alternative<Expr*>(eot)) {
      auto* type_annotation = std::get<TypeAnnotation*>(eot);
      return TypeInferenceErrorStatus(
          type_annotation->span(), nullptr,
          absl::StrFormat("Parametric function invocation `%s` cannot take "
                          "type `%s` -- parametric must be an expression",
                          invocation->ToString(), type_annotation->ToString()));
    }

    auto* value = std::get<Expr*>(eot);

    XLS_VLOG(5) << "Populating callee parametric `" << binding->ToString()
                << "` via invocation expression: " << value->ToString();

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> binding_type,
                         ParametricBindingToType(binding, ctx));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> value_type,
                         parent_ctx->Deduce(value));

    if (*binding_type != *value_type) {
      return ctx->TypeMismatchError(invocation->callee()->span(), nullptr,
                                    *binding_type, value, *value_type,
                                    "Explicit parametric type mismatch.");
    }

    // We have to be at least one fn deep to be instantiating a parametric, so
    // referencing fn_stack::back is safe.
    XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
        parent_ctx->import_data(), parent_ctx->type_info(),
        parent_ctx->warnings(), parent_ctx->fn_stack().back().parametric_env(),
        value, value_type.get()));

    // The value we're instantiating the function with must be constexpr -- we
    // can't instantiate with values determined at runtime, of course.
    if (!parent_ctx->type_info()->IsKnownConstExpr(value)) {
      return TypeInferenceErrorStatus(
          value->span(), value_type.get(),
          absl::StrFormat("Parametric expression `%s` was not constexpr -- "
                          "parametric values must be compile-time constants",
                          value->ToString()));
    }

    explicit_bindings.insert(
        {binding->identifier(),
         parent_ctx->type_info()->GetConstExpr(value).value()});
  }

  // The bindings that were not explicitly filled by the caller are taken from
  // the callee directly; e.g. if caller invokes as `parametric()` it supplies
  // 0 parmetrics directly, but callee may have:
  //
  //    `fn parametric<N: u32 = 5>() { ... }`
  //
  // and thus needs the `N: u32 = 5` to be filled here.
  for (ParametricBinding* remaining_binding :
       absl::MakeSpan(parametric_bindings)
           .subspan(invocation->explicit_parametrics().size())) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> binding_type,
                         ParametricBindingToType(remaining_binding, ctx));
    parametric_constraints.push_back(
        ParametricConstraint(*remaining_binding, std::move(binding_type)));
  }

  return InstantiateFunction(invocation->span(), fn_type, instantiate_args, ctx,
                             parametric_constraints, explicit_bindings);
}

}  // namespace xls::dslx

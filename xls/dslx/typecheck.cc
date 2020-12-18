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

#include "xls/dslx/typecheck.h"

#include "xls/dslx/deduce.h"
#include "xls/dslx/dslx_builtins.h"

namespace xls::dslx {

// Checks the function's parametrics' and arguments' types.
static absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>>
CheckFunctionParams(Function* f, DeduceCtx* ctx) {
  for (ParametricBinding* parametric : f->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type()));
    if (parametric->expr() != nullptr) {
      // TODO(leary): 2020-07-06 Fully document the behavior of parametric
      // function calls in parametric expressions.
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                           ctx->Deduce(parametric->expr()));
      if (*expr_type != *parametric_binding_type) {
        return XlsTypeErrorStatus(parametric->span(), *parametric_binding_type,
                                  *expr_type,
                                  "Annotated type of derived parametric value "
                                  "did not match inferred type.");
      }
    }
    ctx->type_info()->SetItem(parametric->name_def(), *parametric_binding_type);
  }

  std::vector<std::unique_ptr<ConcreteType>> param_types;
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> param_type,
                         ctx->Deduce(param));
    ctx->type_info()->SetItem(param->name_def(), *param_type);
    param_types.push_back(std::move(param_type));
  }

  return param_types;
}

absl::Status CheckFunction(Function* f, DeduceCtx* ctx) {
  const std::string& fn_name = ctx->fn_stack().back().name;

  std::vector<std::unique_ptr<ConcreteType>> param_types;
  std::unique_ptr<ConcreteType> annotated_return_type;

  // First, get the types of the function's parametrics, args, and return type.
  if (f->IsParametric() && f->identifier() == fn_name) {
    // Parametric functions are evaluated per invocation. If we're currently
    // inside of this function, it just means that we already have the type
    // signature, and now we just need to evaluate the body.
    absl::optional<ConcreteType*> f_type = ctx->type_info()->GetItem(f);
    XLS_RET_CHECK(f_type.has_value());
    auto* function_type = dynamic_cast<FunctionType*>(f_type.value());
    XLS_RET_CHECK(function_type != nullptr);
    annotated_return_type = function_type->return_type().CloneToUnique();
    param_types = CloneToUnique(absl::MakeSpan(function_type->params()));
  } else {
    XLS_VLOG(1) << "Type-checking signature for function: " << f->identifier();
    XLS_ASSIGN_OR_RETURN(param_types, CheckFunctionParams(f, ctx));
    if (f->IsParametric()) {
      // We just needed the type signature so that we can instantiate this
      // invocation. Let's return this for now, and typecheck the body once we
      // have symbolic bindings.
      if (f->return_type() == nullptr) {
        annotated_return_type = TupleType::MakeNil();
      } else {
        XLS_ASSIGN_OR_RETURN(annotated_return_type,
                             ctx->Deduce(f->return_type()));
      }
      FunctionType function_type(std::move(param_types),
                                 std::move(annotated_return_type));
      ctx->type_info()->SetItem(f->name_def(), function_type);
      ctx->type_info()->SetItem(f, function_type);
      return absl::OkStatus();
    }
  }

  XLS_VLOG(1) << "Type-checking body for function: " << f->identifier();

  // Second, typecheck the return type of the function.
  // Note: if there is no annotated return type, we assume nil.
  std::unique_ptr<ConcreteType> return_type;
  if (f->return_type() == nullptr) {
    return_type = TupleType::MakeNil();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, DeduceAndResolve(f->return_type(), ctx));
  }

  // Third, typecheck the body of the function.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_return_type,
                       DeduceAndResolve(f->body(), ctx));

  // Finally, assert type consistency between body and annotated return type.
  XLS_VLOG(3) << absl::StrFormat(
      "Resolved return type: %s => %s resolved body type: %s => %s",
      annotated_return_type->ToString(), return_type->ToString(),
      body_return_type->ToString(), body_return_type->ToString());
  if (*return_type != *body_return_type) {
    return XlsTypeErrorStatus(
        f->body()->span(), *body_return_type, *return_type,
        absl::StrFormat("Return type of function body for '%s' did not match "
                        "the annotated return type.",
                        f->identifier()));
  }

  FunctionType function_type(std::move(param_types),
                             std::move(body_return_type));
  ctx->type_info()->SetItem(f->name_def(), function_type);
  ctx->type_info()->SetItem(f, function_type);
  return absl::OkStatus();
}

absl::Status CheckTest(Test* t, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_return_type,
                       ctx->Deduce(t->body()));
  if (body_return_type->IsNil()) {
    return absl::OkStatus();
  }
  return XlsTypeErrorStatus(
      t->body()->span(), *body_return_type, *ConcreteType::MakeNil(),
      absl::StrFormat("Return type of test body for '%s' did not match the "
                      "expected test return type `()`.",
                      t->identifier()));
}

absl::StatusOr<NameDef*> InstantiateBuiltinParametric(
    BuiltinNameDef* builtin_name, Invocation* invocation, DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> arg_types;
  for (Expr* arg : invocation->args()) {
    absl::optional<ConcreteType*> arg_type = ctx->type_info()->GetItem(arg);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                         Resolve(**arg_type, ctx));
    arg_types.push_back(std::move(resolved));
  }

  absl::optional<std::string> map_fn_name;
  absl::optional<Function*> map_fn;
  absl::optional<std::vector<ParametricBinding*>>
      higher_order_parametric_bindings;
  if (builtin_name->identifier() == "map") {
    Expr* map_fn_ref = invocation->args()[1];
    if (auto* colon_ref = dynamic_cast<ColonRef*>(map_fn_ref)) {
      map_fn_name = colon_ref->attr();
      Import* import_node = colon_ref->ResolveImportSubject().value();
      absl::optional<const ImportedInfo*> imported =
          ctx->type_info()->GetImported(import_node);
      XLS_RET_CHECK(imported.has_value());
      XLS_ASSIGN_OR_RETURN(map_fn,
                           (*imported)->module->GetFunction(*map_fn_name));
      higher_order_parametric_bindings = (*map_fn)->parametric_bindings();
    } else {
      auto* name_ref = dynamic_cast<NameRef*>(map_fn_ref);
      map_fn_name = name_ref->identifier();
      if (!GetParametricBuiltins().contains(*map_fn_name)) {
        XLS_ASSIGN_OR_RETURN(map_fn, ctx->module()->GetFunction(*map_fn_name));
        higher_order_parametric_bindings = (*map_fn)->parametric_bindings();
      }
    }
  }

  std::vector<const ConcreteType*> arg_type_ptrs;
  arg_type_ptrs.reserve(arg_types.size());
  for (auto& arg_type : arg_types) {
    arg_type_ptrs.push_back(arg_type.get());
  }

  XLS_ASSIGN_OR_RETURN(SignatureFn fsignature, GetParametricBuiltinSignature(
                                                   builtin_name->identifier()));
  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      fsignature(arg_type_ptrs, builtin_name->identifier(), invocation->span(),
                 ctx, higher_order_parametric_bindings));
  FunctionType* fn_type = dynamic_cast<FunctionType*>(tab.type.get());
  XLS_RET_CHECK(fn_type != nullptr) << tab.type->ToString();

  const SymbolicBindings& fn_symbolic_bindings =
      ctx->fn_stack().back().symbolic_bindings;
  ctx->type_info()->AddInvocationSymbolicBindings(
      invocation, fn_symbolic_bindings, tab.symbolic_bindings);
  ctx->type_info()->SetItem(invocation->callee(), *fn_type);
  ctx->type_info()->SetItem(invocation, fn_type->return_type());

  if (builtin_name->identifier() == "map") {
    Expr* map_fn_ref = invocation->args()[1];
    if (GetParametricBuiltins().contains(*map_fn_name) ||
        !(*map_fn)->IsParametric()) {
      // A builtin higher-order parametric fn would've been typechecked when we
      // were going through the arguments of this invocation. If the function
      // wasn't parametric, then we're good to go.
      return nullptr;
    }

    if (ctx->type_info()->HasInstantiation(invocation, tab.symbolic_bindings)) {
      // We've already typechecked this parametric function using these
      // bindings.
      return nullptr;
    }

    // If the higher order function is parametric, we need to typecheck its body
    // with the symbolic bindings we just computed.
    if (auto* colon_ref = dynamic_cast<ColonRef*>(map_fn_ref)) {
      absl::optional<Import*> import = colon_ref->ResolveImportSubject();
      XLS_RET_CHECK(import.has_value());
      absl::optional<const ImportedInfo*> import_info =
          ctx->type_info()->GetImported(*import);
      XLS_RET_CHECK(import_info.has_value());

      auto invocation_imported_type_info = std::make_shared<TypeInfo>(
          (*import_info)->module, /*parent=*/(*import_info)->type_info);
      std::shared_ptr<DeduceCtx> imported_ctx =
          ctx->MakeCtx(invocation_imported_type_info, (*import_info)->module);
      imported_ctx->fn_stack().push_back(
          FnStackEntry{*map_fn_name, tab.symbolic_bindings});
      // We need to typecheck this imported function with respect to its module.
      XLS_RETURN_IF_ERROR(
          ctx->typecheck_function()(map_fn.value(), imported_ctx.get()));
      ctx->type_info()->AddInstantiation(invocation, tab.symbolic_bindings,
                                         invocation_imported_type_info);
    } else {
      // If the higher-order parametric fn is in this module, let's try to push
      // it onto the typechecking stack.
      ctx->fn_stack().push_back(
          FnStackEntry{*map_fn_name, tab.symbolic_bindings});

      // Create a "derived" type info (a child type info with the current type
      // info as a parent), and note that it exists for an instantiation (in the
      // parent type info).
      std::shared_ptr<TypeInfo> parent_type_info = ctx->type_info();
      ctx->AddDerivedTypeInfo();
      parent_type_info->AddInstantiation(invocation, tab.symbolic_bindings,
                                         ctx->type_info());
      auto* name_ref = dynamic_cast<NameRef*>(map_fn_ref);
      return absl::get<NameDef*>(name_ref->name_def());
    }
  }

  return nullptr;
}

}  // namespace xls::dslx

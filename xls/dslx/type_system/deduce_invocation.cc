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

#include "xls/dslx/type_system/deduce_invocation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/ast_env.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_zero_value.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {

// Creates a function invocation on the first element of the given array.
//
// We need to create a fake invocation to deduce the type of a function
// in the case where map is called with a builtin as the map function. Normally,
// map functions (including parametric ones) have their types deduced when their
// `Function` nodes are encountered (where a similar fake `Invocation` node is
// created).
//
// Builtins don't have `Function` nodes (since they're not userspace functions),
// so that inference can't occur, so we essentially perform that synthesis and
// deduction here.
//
// Args:
//   module: AST node owner.
//   span_: The location in the code where analysis is occurring.
//   callee: The function to be invoked.
//   arg_array: The array of arguments (at least one) to the function.
//   explicit_parametrics: Any parametrics specified with the function name,
//       e.g. `X` and `Y` in a call like `map(arr, f<X, Y>)`.
//
// Returns:
//   An invocation node for the given function when called with an element in
//   the argument array.
static Invocation* CreateElementInvocation(
    Module* module, const Span& span, Expr* callee, Expr* arg_array,
    AstNode* parent, std::vector<ExprOrType> explicit_parametrics) {
  BuiltinNameDef* name =
      module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32);
  BuiltinTypeAnnotation* annotation =
      module->Make<BuiltinTypeAnnotation>(span, BuiltinType::kU32, name);
  Number* index_number =
      module->Make<Number>(span, "0", NumberKind::kOther, annotation);
  Index* index = module->Make<Index>(span, arg_array, index_number);
  Invocation* result = module->Make<Invocation>(
      span, callee, std::vector<Expr*>{index}, std::move(explicit_parametrics));
  result->SetParentNonLexical(parent);
  return result;
}

template <typename Resolution>
static absl::StatusOr<std::unique_ptr<Type>> DeduceMapInvocation(
    const Invocation* node, DeduceCtx* ctx, Resolution resolve_fn) {
  const absl::Span<Expr* const>& args = node->args();
  if (args.size() != 2) {
    return ArgCountMismatchErrorStatus(
        node->span(),
        absl::StrFormat(
            "Expected 2 arguments to `map` builtin but got %d argument(s).",
            args.size()),
        ctx->file_table());
  }

  // First, get the input element type.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> arg0_type,
                       ctx->DeduceAndResolve(args[0]));

  Expr* callee = args[1];
  std::vector<ExprOrType> explicit_parametrics;
  auto* callee_ref = dynamic_cast<FunctionRef*>(callee);
  if (callee_ref != nullptr) {
    // Currently there is only a `callee_ref` if explicit parametrics are used.
    explicit_parametrics = callee_ref->explicit_parametrics();
    callee = callee_ref->callee();
  }
  // If the callee is an imported function, we need to check that it is public.
  if (auto* colon_ref = dynamic_cast<ColonRef*>(callee); colon_ref != nullptr) {
    XLS_ASSIGN_OR_RETURN(Function * callee_fn,
                         ResolveFunction(colon_ref, ctx->type_info()));
    if (!callee_fn->is_public()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::StrFormat("Attempted to refer to module member %s that "
                          "is not public.",
                          callee->ToString()),
          ctx->file_table());
    }
  }

  // Then get the type and bindings for the mapping fn.
  Invocation* element_invocation =
      CreateElementInvocation(ctx->module(), node->span(), /*callee=*/callee,
                              /*arg_array=*/args[0], /*parent=*/node->parent(),
                              std::move(explicit_parametrics));
  XLS_ASSIGN_OR_RETURN(TypeAndParametricEnv tab,
                       ctx->typecheck_invocation()(ctx, element_invocation,
                                                   /*constexpr_env=*/{}));
  const FnStackEntry& caller_fn_entry = ctx->fn_stack().back();
  const ParametricEnv& caller_bindings = caller_fn_entry.parametric_env();
  Function* caller = caller_fn_entry.f();

  // We can't blindly resolve the function or else we might fail due to look up
  // a parametric builtin.
  bool callee_needs_implicit_token = false;
  Function* callee_fn = nullptr;
  if (IsBuiltinFn(callee)) {
    callee_needs_implicit_token = GetBuiltinFnRequiresImplicitToken(callee);
  } else {
    XLS_ASSIGN_OR_RETURN(Function * fn, resolve_fn(callee, ctx, node->span()));
    XLS_RET_CHECK(fn != nullptr);

    XLS_ASSIGN_OR_RETURN(TypeInfo * root_type_info,
                         ctx->import_data()->GetRootTypeInfoForNode(fn));
    std::optional<bool> callee_opt =
        root_type_info->GetRequiresImplicitToken(*fn);
    XLS_RET_CHECK(callee_opt.has_value())
        << "user-defined function should have an annotation for whether it "
           "requires an implicit token: "
        << fn->identifier();
    callee_needs_implicit_token = callee_opt.value();
    callee_fn = fn;
  }

  std::optional<TypeInfo*> dti = ctx->type_info()->GetInvocationTypeInfo(
      element_invocation, caller_bindings);
  XLS_RETURN_IF_ERROR(ctx->type_info()->AddInvocationTypeInfo(
      *node, callee_fn, caller, caller_bindings, tab.parametric_env,
      dti.has_value() ? *dti : nullptr));

  // If the callee function needs an implicit token type (e.g. because it has
  // a fail!() or cover!() operation transitively) then so do we.
  if (callee_needs_implicit_token) {
    UseImplicitToken(ctx);
  }

  ArrayType* arg0_array_type = dynamic_cast<ArrayType*>(arg0_type.get());
  return std::make_unique<ArrayType>(tab.type->CloneToUnique(),
                                     arg0_array_type->size());
}

static absl::StatusOr<std::optional<Function*>> ResolveStructColonRefToFn(
    ColonRef* ref, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(ColonRefSubjectT subject,
                       ResolveColonRefSubjectForTypeChecking(
                           ctx->import_data(), ctx->type_info(), ref));
  std::optional<StructDef*> struct_def;
  if (std::holds_alternative<StructDef*>(subject)) {
    struct_def = std::get<StructDef*>(subject);
  } else if (std::holds_alternative<TypeRefTypeAnnotation*>(subject)) {
    TypeRefTypeAnnotation* type_ref = std::get<TypeRefTypeAnnotation*>(subject);
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                         ctx->import_data()->GetRootTypeInfoForNode(type_ref));
    XLS_ASSIGN_OR_RETURN(
        struct_def,
        DerefToStruct(ref->span(), type_ref->ToString(), *type_ref, ti));
  }

  if (struct_def.has_value()) {
    if (!(*struct_def)->impl().has_value()) {
      return TypeInferenceErrorStatus(
          ref->span(), nullptr,
          absl::StrFormat("Struct '%s' has no impl defining '%s'",
                          (*struct_def)->identifier(), ref->attr()),
          ctx->file_table());
    }
    Impl* impl = *(*struct_def)->impl();
    std::optional<Function*> resolved = impl->GetFunction(ref->attr());
    if (!resolved.has_value()) {
      return TypeInferenceErrorStatus(
          ref->span(), nullptr,
          absl::StrFormat("Function with name '%s' is not defined by the impl "
                          "for struct '%s'.",
                          ref->attr(), (*struct_def)->identifier()),
          ctx->file_table());
    }
    return *resolved;
  }
  return std::nullopt;
}

// Resolves "ref" to an AST function.
static absl::StatusOr<Function*> ResolveColonRefToFnForInvocation(
    ColonRef* ref, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::optional<Function*> struct_fn,
                       ResolveStructColonRefToFn(ref, ctx));
  if (struct_fn.has_value()) {
    return *struct_fn;
  }

  std::optional<ImportSubject> import = ref->ResolveImportSubject();
  if (!import.has_value()) {
    return TypeInferenceErrorStatus(
        ref->span(), nullptr,
        absl::StrFormat("Colon-reference subject `%s` did not refer to a "
                        "module or struct, so `%s` cannot be invoked.",
                        ToAstNode(ref->subject())->ToString(), ref->ToString()),
        ctx->file_table());
  }
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << ref->ToString();
  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported_info,
                       ctx->type_info()->GetImportedOrError(import.value()));
  Module* module = imported_info->module;

  XLS_ASSIGN_OR_RETURN(Function * resolved,
                       GetMemberOrTypeInferenceError<Function>(
                           module, ref->attr(), ref->span()));
  if (!resolved->is_public()) {
    return TypeInferenceErrorStatus(
        ref->span(), nullptr,
        absl::StrFormat("Attempted to resolve a module member that was not "
                        "public; `%s` defined in module `%s` @ %s",
                        resolved->identifier(), module->name(),
                        resolved->span().ToString(ctx->file_table())),
        ctx->file_table());
  }
  return resolved;
}

// Resolves a `UseTreeEntry` to a `Function` (for purposes of doing an
// invocation).
static absl::StatusOr<Function*> ResolveUsedFunctionForInvocation(
    UseTreeEntry* use_tree_entry, TypeInfo* type_info) {
  XLS_ASSIGN_OR_RETURN(ImportedInfo * imported_info,
                       type_info->GetImportedOrError(use_tree_entry));
  const auto& payload = use_tree_entry->payload();
  XLS_RET_CHECK(std::holds_alternative<NameDef*>(payload));
  const NameDef* name_def = std::get<NameDef*>(payload);
  XLS_ASSIGN_OR_RETURN(
      Function * fn,
      GetMemberOrTypeInferenceError<Function>(
          imported_info->module, name_def->identifier(), name_def->span()));
  return fn;
}

absl::StatusOr<TypeAndParametricEnv> DeduceInstantiation(
    DeduceCtx* ctx, const Invocation* invocation,
    const std::function<absl::StatusOr<Function*>(const Instantiation*,
                                                  DeduceCtx*)>& resolve_fn,
    const AstEnv& constexpr_env) {
  bool is_parametric_fn = false;
  // We can't resolve builtins as AST Functions, hence this check.
  if (!IsBuiltinFn(invocation->callee())) {
    XLS_ASSIGN_OR_RETURN(Function * f, resolve_fn(invocation, ctx));
    is_parametric_fn = f->IsParametric() || f->proc().has_value();
  }

  // If this is a parametric function invocation, then we need to typecheck
  // the resulting [function] instantiation before we can deduce its return
  // type (or else we won't find it in our TypeInfo).
  if (IsBuiltinFn(invocation->callee()) || is_parametric_fn) {
    return ctx->typecheck_invocation()(ctx, invocation, constexpr_env);
  }

  // If it's non-parametric, then we assume it's been already checked at module
  // top.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> callee_type,
                       ctx->Deduce(invocation->callee()));
  FunctionType* callee_fn_type = dynamic_cast<FunctionType*>(callee_type.get());

  // Callee must be a function.
  if (callee_fn_type == nullptr) {
    return TypeInferenceErrorStatus(
        invocation->callee()->span(), callee_type.get(),
        absl::StrFormat("Invocation callee `%s` is not a function",
                        invocation->callee()->ToString()),
        ctx->file_table());
  }

  return TypeAndParametricEnv{callee_fn_type->return_type().CloneToUnique(),
                              {}};
}

// If the callee is a struct instance, and the function is a method, then the
// first arg is that struct instance.
absl::Status SeedSelfArg(const Attr* callee, DeduceCtx* ctx,
                         std::vector<InstantiateArg>* args) {
  // Attempt to deduce the type first since that's required to collect the impl
  // function, if it exists. If deduction fails, it could be because the
  // reference is not to a struct with an impl. Ignore the error at this stage,
  // it will bubble up later if the attr is not an appropriate callee or the
  // number of parameters isn't correct.
  absl::StatusOr<std::unique_ptr<Type>> type = ctx->Deduce(callee->lhs());
  if (!type.ok()) {
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(std::optional<Function*> fn,
                       ImplFnFromCallee(callee, ctx->type_info()));
  if (!fn.has_value() || !(*fn)->IsMethod()) {
    return absl::OkStatus();
  }
  args->push_back(
      InstantiateArg{std::move(type.value()), (callee->lhs())->span()});
  return absl::OkStatus();
}

absl::Status AppendArgsForInstantiation(
    const Instantiation* inst, const Expr* callee, absl::Span<Expr* const> args,
    DeduceCtx* ctx, std::vector<InstantiateArg>* instantiate_args) {
  if (const Attr* attr = dynamic_cast<const Attr*>(callee)) {
    XLS_RETURN_IF_ERROR(SeedSelfArg(attr, ctx, instantiate_args));
  }

  for (Expr* arg : args) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         ctx->DeduceAndResolve(arg));
    VLOG(5) << absl::StreamFormat(
        "AppendArgsForInstantiation; arg: `%s` deduced: `%s` @ %s",
        arg->ToString(), type->ToString(),
        arg->span().ToString(ctx->file_table()));
    if (type->IsMeta()) {
      return TypeInferenceErrorStatus(
          arg->span(), type.get(), "Cannot pass a type as a function argument.",
          ctx->file_table());
    }
    instantiate_args->push_back(InstantiateArg{std::move(type), arg->span()});
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Type>> DeduceInvocation(const Invocation* node,
                                                       DeduceCtx* ctx) {
  VLOG(5) << "Deducing type for invocation: " << node->ToString();

  // Detect direct recursion. Indirect recursion is currently not syntactically
  // possible (as of 2023-08-22) since you cannot refer to a name that has not
  // yet been defined in the grammar.
  const FnStackEntry& entry = ctx->fn_stack().back();
  if (entry.f() != nullptr &&
      IsNameRefTo(node->callee(), entry.f()->name_def())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Recursion of function `%s` detected -- recursion is "
                        "currently unsupported.",
                        node->callee()->ToString()),
        ctx->file_table());
  }

  auto resolve_callee = [](Expr* callee, DeduceCtx* ctx,
                           const Span& base_span) -> absl::StatusOr<Function*> {
    // Check for attr associated with an impl function.
    if (auto* attr = dynamic_cast<Attr*>(callee); attr != nullptr) {
      XLS_ASSIGN_OR_RETURN(std::optional<Function*> impl_fn,
                           ImplFnFromCallee(attr, ctx->type_info()));
      if (impl_fn.has_value()) {
        return *impl_fn;
      }
    }

    Function* fn;
    if (auto* colon_ref = dynamic_cast<ColonRef*>(callee);
        colon_ref != nullptr) {
      XLS_ASSIGN_OR_RETURN(fn,
                           ResolveColonRefToFnForInvocation(colon_ref, ctx));
    } else if (auto* name_ref = dynamic_cast<NameRef*>(callee);
               name_ref != nullptr) {
      AstNode* definer = name_ref->GetDefiner();
      if (auto* use_tree_entry = dynamic_cast<UseTreeEntry*>(definer);
          use_tree_entry != nullptr) {
        // We'll go fetch the function from the other module.
        XLS_ASSIGN_OR_RETURN(fn, ResolveUsedFunctionForInvocation(
                                     use_tree_entry, ctx->type_info()));
      } else {
        fn = dynamic_cast<Function*>(definer);
        if (fn == nullptr) {
          return TypeInferenceErrorStatus(
              callee->span(), nullptr,
              absl::StrFormat("Invocation callee `%s` is not a function",
                              callee->ToString()),
              ctx->file_table());
        }
      }
    } else {
      return TypeInferenceErrorStatus(
          base_span, nullptr,
          "An invocation callee must be a function, with a possible scope "
          "indicated using `::` or `.` in the case of an instance method",
          ctx->file_table());
    }
    return fn;
  };
  // Find the callee as a DSLX Function from the expression.
  auto resolve_fn = [&](const Instantiation* node,
                        DeduceCtx* ctx) -> absl::StatusOr<Function*> {
    Expr* callee = node->callee();
    return resolve_callee(callee, ctx, node->span());
  };
  // Map is special.
  if (IsBuiltinFn(node->callee(), "map")) {
    return DeduceMapInvocation(node, ctx, resolve_callee);
  }

  // Gather up the type of all the (actual) arguments.
  std::vector<InstantiateArg> args;
  XLS_RETURN_IF_ERROR(AppendArgsForInstantiation(node, node->callee(),
                                                 node->args(), ctx, &args));

  XLS_ASSIGN_OR_RETURN(
      TypeAndParametricEnv tab,
      DeduceInstantiation(ctx, node, resolve_fn, /*constexpr_env=*/{}));

  Type* ct = ctx->type_info()->GetItem(node->callee()).value();
  FunctionType* ft = dynamic_cast<FunctionType*>(ct);
  if (args.size() != ft->params().size()) {
    return ArgCountMismatchErrorStatus(
        node->span(),
        absl::StrFormat("Expected %d parameter(s) but got %d arguments.",
                        ft->params().size(), node->args().size()),
        ctx->file_table());
  }

  for (int i = 0; i < args.size(); i++) {
    if (*args[i].type() != *ft->params()[i]) {
      return ctx->TypeMismatchError(
          args[i].span(), nullptr, *ft->params()[i], nullptr, *args[i].type(),
          "Mismatch between parameter and argument types.");
    }
  }

  // We can't blindly resolve the function or else we might fail due to look up
  // a parametric builtin.
  bool callee_needs_implicit_token = false;
  if (IsBuiltinFn(node->callee())) {
    callee_needs_implicit_token =
        GetBuiltinFnRequiresImplicitToken(node->callee());
  } else {
    XLS_ASSIGN_OR_RETURN(Function * fn, resolve_fn(node, ctx));
    XLS_RET_CHECK(fn != nullptr);

    XLS_ASSIGN_OR_RETURN(TypeInfo * root_type_info,
                         ctx->import_data()->GetRootTypeInfoForNode(fn));
    std::optional<bool> callee_opt =
        root_type_info->GetRequiresImplicitToken(*fn);
    XLS_RET_CHECK(callee_opt.has_value())
        << "user-defined function should have an annotation for whether it "
           "requires an implicit token: "
        << fn->identifier();
    callee_needs_implicit_token = callee_opt.value();
    // Parametric and proc functions will be added separately.
    if (!fn->IsParametric() && !fn->IsInProc()) {
      XLS_RETURN_IF_ERROR(
          ctx->type_info()->AddInvocation(*node, fn, entry.f()));
    }
  }

  // If the callee function needs an implicit token type (e.g. because it has
  // a fail!() or cover!() operation transitively) then so do we.
  if (callee_needs_implicit_token) {
    UseImplicitToken(ctx);
  }

  return std::move(tab.type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceFormatMacro(const FormatMacro* node,
                                                        DeduceCtx* ctx) {
  if (node->verbosity().has_value()) {
    XLS_RETURN_IF_ERROR(ctx->Deduce(*node->verbosity()).status());
  }
  int64_t arg_count = OperandsExpectedByFormat(node->format());

  if (arg_count != node->args().size()) {
    return ArgCountMismatchErrorStatus(
        node->span(),
        absl::StrFormat("%s macro expects %d argument(s) from format but has "
                        "%d argument(s)",
                        node->macro(), arg_count, node->args().size()),
        ctx->file_table());
  }

  if (node->macro() == "assert_fmt!") {
    // assert_fmt!-like macro.
    VLOG(5) << "DeduceFormatMacro (assert_fmt!-like): " << node->ToString();
    UseImplicitToken(ctx);

    Expr* condition = node->condition().value();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> condition_type,
                         ctx->Deduce(condition));
    std::unique_ptr<Type> u1 = BitsType::MakeU1();
    if (*condition_type != *u1) {
      return ctx->TypeMismatchError(
          condition->span(), nullptr, *condition_type, nullptr, *u1,
          "assert_fmt! condition must be a boolean value.");
    }

    for (Expr* arg : node->args()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                           ctx->DeduceAndResolve(arg));
      if (!ctx->type_info()->IsKnownConstExpr(arg)) {
        return NotConstantErrorStatus(arg->span(), arg, ctx->file_table());
      }
      XLS_RETURN_IF_ERROR(
          ValidateFormatMacroArgument(*type, arg->span(), ctx->file_table()));
    }

    if (VLOG_IS_ON(5)) {
      XLS_ASSIGN_OR_RETURN(
          std::string formatted_msg,
          EvaluateFormatString(ctx->import_data(), ctx->type_info(),
                               ctx->warnings(), ctx->GetCurrentParametricEnv(),
                               std::vector<FormatStep>(node->format().begin(),
                                                       node->format().end()),
                               node->args()));
      VLOG(5) << "assert_fmt! formatted message: " << formatted_msg;
    }

    return Type::MakeUnit();
  }

  // trace_fmt!-like macro.
  for (Expr* arg : node->args()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         ctx->DeduceAndResolve(arg));
    XLS_RETURN_IF_ERROR(
        ValidateFormatMacroArgument(*type, arg->span(), ctx->file_table()));
  }

  // trace_fmt! (and any future friends) require threading implicit tokens for
  // control just like cover! and fail! do.
  UseImplicitToken(ctx);

  return std::make_unique<TokenType>();
}

absl::StatusOr<std::unique_ptr<Type>> DeduceZeroMacro(const ZeroMacro* node,
                                                      DeduceCtx* ctx) {
  VLOG(5) << "DeduceZeroMacro; node: `" << node->ToString() << "`";
  // Note: since it's a macro the parser checks arg count and parametric count.
  //
  // This says the type of the parametric type arg is the type of the result.
  // However, we have to check that all of the transitive type within the
  // parametric argument type are "zero capable".
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_type,
                       ctx->DeduceAndResolve(ToAstNode(node->type())));
  XLS_ASSIGN_OR_RETURN(parametric_type,
                       UnwrapMetaType(std::move(parametric_type), node->span(),
                                      "zero! macro type", ctx->file_table()));
  XLS_RET_CHECK(!parametric_type->IsMeta());

  XLS_ASSIGN_OR_RETURN(
      InterpValue value,
      MakeZeroValue(*parametric_type, *ctx->import_data(), node->span()));
  ctx->type_info()->NoteConstExpr(node, value);
  return parametric_type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceAllOnesMacro(
    const AllOnesMacro* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceAllOnesMacro; node: `" << node->ToString() << "`";
  // Note: since it's a macro the parser checks arg count and parametric count.
  //
  // This says the type of the parametric type arg is the type of the result.
  // However, we have to check that all of the transitive type within the
  // parametric argument type are "all-ones capable".
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_type,
                       ctx->DeduceAndResolve(ToAstNode(node->type())));
  XLS_ASSIGN_OR_RETURN(
      parametric_type,
      UnwrapMetaType(std::move(parametric_type), node->span(),
                     "all_ones! macro type", ctx->file_table()));
  XLS_RET_CHECK(!parametric_type->IsMeta());

  XLS_ASSIGN_OR_RETURN(
      InterpValue value,
      MakeAllOnesValue(*parametric_type, *ctx->import_data(), node->span()));
  ctx->type_info()->NoteConstExpr(node, value);
  return parametric_type;
}

}  // namespace xls::dslx

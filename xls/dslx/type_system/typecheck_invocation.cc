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

#include "xls/dslx/type_system/typecheck_invocation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/dslx_builtins_signatures.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/ast_env.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/instantiate_parametric_function.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/scoped_fn_stack_entry.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "re2/re2.h"

namespace xls::dslx {

static absl::StatusOr<std::unique_ptr<DeduceCtx>> GetImportedDeduceCtx(
    DeduceCtx* ctx, const Invocation* invocation,
    const ParametricEnv& caller_bindings,
    std::variant<UseTreeEntry*, Import*> import_key) {
  auto it = ctx->type_info()->GetRootImports().find(import_key);
  XLS_RET_CHECK(it != ctx->type_info()->GetRootImports().end())
      << "Could not find import for key: " << ToAstNode(import_key)->ToString();
  const ImportedInfo* imported = &it->second;

  XLS_ASSIGN_OR_RETURN(
      TypeInfo * imported_type_info,
      ctx->type_info_owner().New(imported->module, imported->type_info));
  std::unique_ptr<DeduceCtx> imported_ctx =
      ctx->MakeCtx(imported_type_info, imported->module);
  imported_ctx->AddFnStackEntry(FnStackEntry::MakeTop(imported->module));

  return imported_ctx;
}

static absl::Status TypecheckIsAcceptableWideningCast(DeduceCtx* ctx,
                                                      const Invocation* node) {
  // Use type_info rather than ctx->Deduce as this Invocation's nodes have
  // already been deduced and placed into type_info.
  TypeInfo* type_info = ctx->type_info();
  const Expr* from_expr = node->args().at(0);

  std::optional<Type*> maybe_from_type = type_info->GetItem(from_expr);
  std::optional<Type*> maybe_to_type = type_info->GetItem(node);

  XLS_RET_CHECK(maybe_from_type.has_value());
  XLS_RET_CHECK(maybe_to_type.has_value());

  std::optional<BitsLikeProperties> from_bits_like =
      GetBitsLike(*maybe_from_type.value());
  std::optional<BitsLikeProperties> to_bits_like =
      GetBitsLike(*maybe_to_type.value());

  if (!from_bits_like.has_value() || !to_bits_like.has_value()) {
    return ctx->TypeMismatchError(
        node->span(), from_expr, *maybe_from_type.value(), node,
        *maybe_to_type.value(),
        absl::StrFormat("widening_cast must cast bits to bits, not %s to %s.",
                        maybe_from_type.value()->ToErrorString(),
                        maybe_to_type.value()->ToErrorString()));
  }

  XLS_ASSIGN_OR_RETURN(bool signed_input,
                       from_bits_like->is_signed.GetAsBool());
  XLS_ASSIGN_OR_RETURN(bool signed_output, to_bits_like->is_signed.GetAsBool());

  XLS_ASSIGN_OR_RETURN(int64_t old_bit_count,
                       from_bits_like->size.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(int64_t new_bit_count, to_bits_like->size.GetAsInt64());

  bool can_cast =
      ((signed_input == signed_output) && (new_bit_count >= old_bit_count)) ||
      (!signed_input && signed_output && (new_bit_count > old_bit_count));

  if (!can_cast) {
    return ctx->TypeMismatchError(
        node->span(), from_expr, *maybe_from_type.value(), node,
        *maybe_to_type.value(),
        absl::StrFormat("Can not cast from type %s (%d bits) to"
                        " %s (%d bits) with widening_cast",
                        ToTypeString(from_bits_like.value()), old_bit_count,
                        ToTypeString(to_bits_like.value()), new_bit_count));
  }

  return absl::OkStatus();
}

static absl::Status ValidateWithinProc(std::string_view builtin_name,
                                       const Span& span, DeduceCtx* ctx) {
  if (!ctx->WithinProc()) {
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat("Cannot %s() outside of a proc", builtin_name),
        ctx->file_table());
  }
  return absl::OkStatus();
}

static absl::Status TypecheckCoverBuiltinInvocation(
    DeduceCtx* ctx, const Invocation* invocation) {
  // Make sure that the coverpoint's identifier is valid in both Verilog
  // and DSLX - notably, we don't support Verilog escaped strings.
  // TODO(rspringer): 2021-05-26: Ensure only one instance of an identifier
  // in a design.
  String* identifier_node = dynamic_cast<String*>(invocation->args()[0]);
  XLS_RET_CHECK(identifier_node != nullptr);
  if (identifier_node->text().empty()) {
    return InvalidIdentifierErrorStatus(
        invocation->span(), "An identifier must be specified with a cover! op.",
        ctx->file_table());
  }

  std::string identifier = identifier_node->text();
  if (identifier[0] == '\\') {
    return InvalidIdentifierErrorStatus(
        invocation->span(), "Verilog escaped strings are not supported.",
        ctx->file_table());
  }

  // We don't support Verilog "escaped strings", so we only have to worry
  // about regular identifier matching.
  if (!RE2::FullMatch(identifier, "[a-zA-Z_][a-zA-Z0-9$_]*")) {
    return InvalidIdentifierErrorStatus(
        invocation->span(),
        "A coverpoint identifier must start with a letter or underscore, "
        "and otherwise consist of letters, digits, underscores, and/or "
        "dollar signs.",
        ctx->file_table());
  }

  return absl::OkStatus();
}

// Note: caller may be nullptr when it is the top level (e.g. module-level
// constant) performing the invocation.
static absl::StatusOr<TypeAndParametricEnv>
TypecheckParametricBuiltinInvocation(DeduceCtx* ctx,
                                     const Invocation* invocation,
                                     Function* caller) {
  Expr* callee = invocation->callee();
  NameRef* callee_nameref = dynamic_cast<NameRef*>(callee);

  // Note: we already know it's a builtin invocation, so comparisons to name
  // strings are ok here.
  std::string callee_name = callee_nameref->identifier();

  static const absl::flat_hash_set<std::string> kShouldBeInProc = {
      "join",
      "recv",
      "recv_if",
      "send",
      "send_if",
      "recv_non_blocking",
      "recv_if_non_blocking",
  };

  if (kShouldBeInProc.contains(callee_name)) {
    XLS_RETURN_IF_ERROR(
        ValidateWithinProc(callee_name, invocation->span(), ctx));
  }

  std::vector<std::unique_ptr<Type>> arg_types;
  std::vector<Span> arg_spans;
  for (Expr* arg : invocation->args()) {
    XLS_ASSIGN_OR_RETURN(auto arg_type, ctx->Deduce(arg));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved,
                         ctx->Resolve(*arg_type));
    arg_types.push_back(std::move(resolved));
    arg_spans.push_back(arg->span());
  }

  if ((callee_name == "fail!" || callee_name == "assert!" ||
       callee_name == "cover!") &&
      caller != nullptr) {
    ctx->type_info()->NoteRequiresImplicitToken(*caller, true);
  }

  VLOG(5) << "Instantiating builtin parametric: "
          << callee_nameref->identifier();
  XLS_ASSIGN_OR_RETURN(
      SignatureFn fsignature,
      GetParametricBuiltinSignature(callee_nameref->identifier()));

  // Callback that signatures can use to request constexpr evaluation of their
  // arguments -- this is a special builtin superpower used by e.g. range().
  auto constexpr_eval = [&](int64_t argno) -> absl::StatusOr<InterpValue> {
    Expr* arg = invocation->args()[argno];

    XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
        ctx->import_data(), ctx->type_info(), ctx->warnings(),
        ctx->GetCurrentParametricEnv(), arg, nullptr))
        << "while evaluating: " << arg->ToString();

    if (!ctx->type_info()->IsKnownConstExpr(arg)) {
      return TypeInferenceErrorStatus(
          invocation->args().at(argno)->span(), nullptr,
          absl::StrFormat(
              "Argument to built-in function `%s` must be constexpr; "
              "could not evaluate argument to constexpr: `%s`",
              callee_name, arg->ToString()),
          ctx->file_table());
    }

    return ctx->type_info()->GetConstExpr(arg).value();
  };

  std::vector<const Type*> arg_type_ptrs;
  arg_type_ptrs.reserve(arg_types.size());
  for (int64_t i = 0; i < arg_types.size(); ++i) {
    const std::unique_ptr<Type>& arg_type = arg_types.at(i);
    if (arg_type->IsMeta()) {
      return TypeInferenceErrorStatus(
          invocation->args().at(i)->span(), arg_type.get(),
          "Argument to built-in function must be a value, not a type.",
          ctx->file_table());
    }
    arg_type_ptrs.push_back(arg_type.get());
  }

  XLS_ASSIGN_OR_RETURN(
      TypeAndParametricEnv tab,
      fsignature(
          SignatureData{
              .arg_types = arg_type_ptrs,
              .arg_spans = arg_spans,
              .arg_explicit_parametrics = invocation->explicit_parametrics(),
              .name = callee_nameref->identifier(),
              .span = invocation->span(),
              .constexpr_eval = constexpr_eval,
              .args = invocation->args(),
          },
          ctx));

  FunctionType* fn_type = dynamic_cast<FunctionType*>(tab.type.get());
  XLS_RET_CHECK(fn_type != nullptr) << tab.type->ToString();

  const ParametricEnv& fn_parametric_env = ctx->GetCurrentParametricEnv();

  VLOG(5) << "TypeInfo::AddInvocationCallBindings; type_info: "
          << ctx->type_info() << "; node: `" << invocation->ToString()
          << "`; caller_env: " << fn_parametric_env
          << "; callee_env: " << tab.parametric_env;

  // Note that, since this is not a user-defined function, there is no derived
  // type information for it.
  XLS_RETURN_IF_ERROR(
      ctx->type_info()->AddInvocationTypeInfo(*invocation, caller,
                                              /*caller_env=*/fn_parametric_env,
                                              /*callee_env=*/tab.parametric_env,
                                              /*derived_type_info=*/nullptr));

  ctx->type_info()->SetItem(invocation->callee(), *fn_type);
  // We don't want to store a type on a BuiltinNameDef.
  if (std::holds_alternative<const NameDef*>(callee_nameref->name_def())) {
    ctx->type_info()->SetItem(ToAstNode(callee_nameref->name_def()), *fn_type);
  }

  ctx->type_info()->SetItem(invocation, fn_type->return_type());

  // Special check for additional builtin type constraints.
  if (callee_nameref->identifier() == "widening_cast") {
    XLS_RETURN_IF_ERROR(TypecheckIsAcceptableWideningCast(ctx, invocation));
  }

  // array_size is always a constexpr result since it just needs the type
  // information
  if (callee_nameref->identifier() == "array_size") {
    auto* array_type = down_cast<const ArrayType*>(fn_type->params()[0].get());
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    ctx->type_info()->NoteConstExpr(
        invocation, InterpValue::MakeU32(static_cast<int32_t>(array_size)));
  }

  // bit_count and element_count are similar to array_size, but uses the
  // parametric argument rather than a value.
  if (callee_nameref->identifier() == "bit_count" ||
      callee_nameref->identifier() == "element_count") {
    auto* annotation =
        std::get<TypeAnnotation*>(invocation->explicit_parametrics()[0]);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         ctx->DeduceAndResolve(annotation));
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         callee_nameref->identifier() == "element_count"
                             ? GetElementCountAsInterpValue(type.get())
                             : GetBitCountAsInterpValue(type.get()));
    ctx->type_info()->NoteConstExpr(invocation, value);
  }

  if (callee_nameref->identifier() == "cover!") {
    XLS_RETURN_IF_ERROR(TypecheckCoverBuiltinInvocation(ctx, invocation));
  }

  // fsignature returns a tab w/a fn type, not the fn return type (which is
  // what we actually want). We hack up `tab` to make this consistent with
  // InstantiateFunction.
  tab.type = fn_type->return_type().CloneToUnique();
  return tab;
}

// Inspects to see whether the given `ref` is a reference to an externally
// defined entity -- if so, returns the "import key" we can use to resolve the
// module information for it.
static std::optional<std::variant<UseTreeEntry*, Import*>> IsExternRef(
    Expr& ref, Module& current_module) {
  if (auto* colon_ref = dynamic_cast<ColonRef*>(&ref); colon_ref != nullptr) {
    std::optional<Import*> import = colon_ref->ResolveImportSubject();
    if (import.has_value()) {
      return import.value();
    }
    return std::nullopt;
  }
  if (auto* name_ref = dynamic_cast<NameRef*>(&ref); name_ref != nullptr) {
    AstNode* definer = name_ref->GetDefiner();
    if (auto* use_tree_entry = dynamic_cast<UseTreeEntry*>(definer);
        use_tree_entry != nullptr) {
      return use_tree_entry;
    }
  }
  return std::nullopt;
}

absl::StatusOr<TypeAndParametricEnv> TypecheckInvocation(
    DeduceCtx* ctx, const Invocation* invocation, const AstEnv& constexpr_env) {
  VLOG(5) << "Typechecking invocation: `" << invocation->ToString() << "` @ "
          << invocation->span().ToString(ctx->file_table());
  XLS_VLOG_LINES(5, ctx->GetFnStackDebugString());

  Expr* callee = invocation->callee();

  Function* caller = ctx->fn_stack().back().f();
  if (auto* name_ref = dynamic_cast<const NameRef*>(callee);
      name_ref != nullptr && IsBuiltinParametricNameRef(name_ref)) {
    return TypecheckParametricBuiltinInvocation(ctx, invocation, caller);
  }

  XLS_ASSIGN_OR_RETURN(
      Function * callee_fn, ResolveFunction(callee, ctx->type_info()),
      _.With([callee, ctx](const absl::Status& s) {
        return TypeInferenceErrorStatus(
            callee->span(), nullptr,
            absl::StrFormat("Cannot resolve callee `%s` to a function; %s",
                            callee->ToString(), s.message()),
            ctx->file_table());
      }));
  XLS_RET_CHECK(callee_fn != nullptr);

  const absl::Span<Expr* const> args = invocation->args();
  std::vector<InstantiateArg> instantiate_args;
  std::vector<std::unique_ptr<Type>> arg_types;
  instantiate_args.reserve(args.size());
  for (Expr* arg : args) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         ctx->DeduceAndResolve(arg));
    arg_types.push_back(type->CloneToUnique());
    instantiate_args.push_back(InstantiateArg{std::move(type), arg->span()});
  }

  // Make a copy; the fn stack can get re-allocated, etc.
  ParametricEnv caller_parametric_env = ctx->GetCurrentParametricEnv();
  absl::flat_hash_map<std::string, InterpValue> caller_parametric_env_map =
      caller_parametric_env.ToMap();

  // We need to deduce a callee relative to its parent module. We still need to
  // record data in the original module/ctx, so we hold on to the parent.
  DeduceCtx* parent_ctx = ctx;
  std::unique_ptr<DeduceCtx> imported_ctx_holder;
  if (auto import_key = IsExternRef(*invocation->callee(), *ctx->module());
      import_key.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        imported_ctx_holder,
        GetImportedDeduceCtx(ctx, invocation, caller_parametric_env,
                             import_key.value()));
    ctx = imported_ctx_holder.get();
  }

  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Type>> param_types,
                       TypecheckFunctionParams(*callee_fn, ctx));

  std::unique_ptr<Type> return_type;
  if (callee_fn->return_type() == nullptr) {
    return_type = TupleType::MakeUnit();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, ctx->Deduce(callee_fn->return_type()));
    XLS_ASSIGN_OR_RETURN(
        return_type,
        UnwrapMetaType(std::move(return_type), callee_fn->return_type()->span(),
                       "function return type", ctx->file_table()));
  }

  FunctionType fn_type(std::move(param_types), std::move(return_type));

  XLS_ASSIGN_OR_RETURN(
      TypeAndParametricEnv callee_tab,
      InstantiateParametricFunction(ctx, parent_ctx, invocation, *callee_fn,
                                    fn_type, instantiate_args));

  // Now that we have the necessary bindings, check for recursion.
  std::vector<FnStackEntry> fn_stack = parent_ctx->fn_stack();
  while (!fn_stack.empty()) {
    if (fn_stack.back().f() == callee_fn &&
        fn_stack.back().parametric_env() == callee_tab.parametric_env) {
      return TypeInferenceErrorStatus(
          invocation->span(), nullptr,
          absl::StrFormat("Recursion detected while typechecking; name: '%s'",
                          callee_fn->identifier()),
          ctx->file_table());
    }
    fn_stack.pop_back();
  }

  FunctionType instantiated_ft{std::move(arg_types),
                               callee_tab.type->CloneToUnique()};
  parent_ctx->type_info()->SetItem(invocation->callee(), instantiated_ft);
  ctx->type_info()->SetItem(callee_fn->name_def(), instantiated_ft);

  // We need to deduce fn body, so we're going to call Deduce, which means we'll
  // need a new stack entry w/the new symbolic bindings.
  TypeInfo* const original_ti = parent_ctx->type_info();
  ctx->AddFnStackEntry(FnStackEntry::Make(
      *callee_fn, callee_tab.parametric_env, invocation,
      callee_fn->proc().has_value() ? WithinProc::kYes : WithinProc::kNo));
  TypeInfo* const derived_type_info = ctx->AddDerivedTypeInfo();

  // We execute this function if we're parametric or a proc. In either case, we
  // want to create a new TypeInfo. The reason for the former is obvious. The
  // reason for the latter is that we need separate constexpr data for every
  // instantiation of a proc. If we didn't create new bindings/a new TypeInfo
  // here, then if we instantiated the same proc 2x from some parent proc, we'd
  // end up with only a single set of constexpr values for proc members.
  XLS_RETURN_IF_ERROR(original_ti->AddInvocationTypeInfo(
      *invocation, caller, caller_parametric_env, callee_tab.parametric_env,
      derived_type_info));

  if (callee_fn->proc().has_value()) {
    Proc* p = callee_fn->proc().value();
    for (auto* member : p->members()) {
      XLS_ASSIGN_OR_RETURN(auto type, ctx->DeduceAndResolve(member));
      ctx->type_info()->SetItem(member, *type);
      ctx->type_info()->SetItem(member->name_def(), *type);
    }
  }

  // Mark params (for proc config fns) or proc members (for proc next fns) as
  // constexpr.
  for (const auto& [k, v] : constexpr_env) {
    ctx->type_info()->NoteConstExpr(ToAstNode(k), v);
    ctx->type_info()->NoteConstExpr(AstEnv::GetNameDefForKey(k), v);
  }

  // Add the new parametric bindings to the constexpr set.
  const auto& bindings_map = callee_tab.parametric_env.ToMap();
  for (ParametricBinding* parametric : callee_fn->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        parametric_binding_type,
        UnwrapMetaType(std::move(parametric_binding_type),
                       parametric->type_annotation()->span(),
                       "parametric binding type", ctx->file_table()));
    if (bindings_map.contains(parametric->identifier())) {
      ctx->type_info()->NoteConstExpr(
          parametric->name_def(), bindings_map.at(parametric->identifier()));
    }
  }

  for (auto* param : callee_fn->params()) {
    std::optional<const Type*> param_type = ctx->type_info()->GetItem(param);
    XLS_RET_CHECK(param_type.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved_type,
                         ctx->Resolve(*param_type.value()));
    ctx->type_info()->SetItem(param, *resolved_type);
    ctx->type_info()->SetItem(param->name_def(), *resolved_type);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> body_type,
                       ctx->Deduce(callee_fn->body()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved_body_type,
                       ctx->Resolve(*body_type));
  XLS_RET_CHECK(!resolved_body_type->IsMeta());

  const Type& annotated_return_type = *callee_tab.type;

  // Assert type consistency between the body and deduced return types.
  if (annotated_return_type != *resolved_body_type) {
    VLOG(5) << "annotated_return_type: " << annotated_return_type
            << " resolved_body_type: " << resolved_body_type->ToString();

    if (callee_fn->tag() == FunctionTag::kProcInit) {
      return ctx->TypeMismatchError(
          callee_fn->body()->span(), nullptr, annotated_return_type,
          callee_fn->body(), *resolved_body_type,
          absl::StrFormat("'next' state param and 'init' types differ."));
    }

    if (callee_fn->tag() == FunctionTag::kProcNext) {
      return ctx->TypeMismatchError(
          callee_fn->body()->span(), nullptr, annotated_return_type,
          callee_fn->body(), *resolved_body_type,
          absl::StrFormat("'next' input and output state types differ."));
    }

    return ctx->TypeMismatchError(
        callee_fn->body()->span(), nullptr, annotated_return_type,
        callee_fn->body(), *resolved_body_type,
        absl::StrFormat("Return type of function body for '%s' did not match "
                        "the annotated return type.",
                        callee_fn->identifier()));
  }

  XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo(derived_type_info));
  ctx->PopFnStackEntry();

  // Implementation note: though we could have all functions have
  // NoteRequiresImplicitToken() be false unless otherwise noted, this helps
  // guarantee we did consider and make a note for every function -- the code
  // is generally complex enough it's nice to have this soundness check.
  if (std::optional<bool> requires_token =
          ctx->type_info()->GetRequiresImplicitToken(*callee_fn);
      !requires_token.has_value()) {
    ctx->type_info()->NoteRequiresImplicitToken(*callee_fn, false);
  }

  return callee_tab;
}

absl::StatusOr<std::vector<std::unique_ptr<Type>>> TypecheckFunctionParams(
    Function& f, DeduceCtx* ctx) {
  CHECK_EQ(f.owner(), ctx->type_info()->module())
      << "function owner: " << f.owner()->name()
      << " vs type info module: " << ctx->type_info()->module()->name();

  {
    ScopedFnStackEntry parametric_env_expr_scope(f, ctx, WithinProc::kNo);

    for (ParametricBinding* parametric : f.parametric_bindings()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_binding_type,
                           ctx->Deduce(parametric->type_annotation()));
      XLS_ASSIGN_OR_RETURN(
          parametric_binding_type,
          UnwrapMetaType(std::move(parametric_binding_type),
                         parametric->type_annotation()->span(),
                         "parametric binding type", ctx->file_table()));

      if (parametric->expr() != nullptr) {
        // TODO(leary): 2020-07-06 Fully document the behavior of parametric
        // function calls in parametric expressions.
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> expr_type,
                             ctx->Deduce(parametric->expr()));
        if (*expr_type != *parametric_binding_type) {
          return ctx->TypeMismatchError(
              parametric->span(), parametric->type_annotation(),
              *parametric_binding_type, parametric->expr(), *expr_type,
              "Annotated type of derived parametric value "
              "did not match inferred type.");
        }
      }
      ctx->type_info()->SetItem(parametric->name_def(),
                                *parametric_binding_type);
    }

    parametric_env_expr_scope.Finish();
  }

  std::vector<std::unique_ptr<Type>> param_types;
  for (Param* param : f.params()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> param_type, ctx->Deduce(param));
    ctx->type_info()->SetItem(param->name_def(), *param_type);
    param_types.push_back(std::move(param_type));
  }

  return param_types;
}

absl::StatusOr<InterpValue> InterpretExpr(
    ImportData* import_data, TypeInfo* type_info, Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(import_data, type_info, expr, env,
                                      /*caller_bindings=*/std::nullopt));
  return BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{});
}

}  // namespace xls::dslx

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
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
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
//
// Returns:
//   An invocation node for the given function when called with an element in
//   the argument array.
static Invocation* CreateElementInvocation(Module* module, const Span& span,
                                           Expr* callee, Expr* arg_array,
                                           AstNode* parent) {
  BuiltinNameDef* name =
      module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32);
  BuiltinTypeAnnotation* annotation =
      module->Make<BuiltinTypeAnnotation>(span, BuiltinType::kU32, name);
  Number* index_number =
      module->Make<Number>(span, "0", NumberKind::kOther, annotation);
  Index* index = module->Make<Index>(span, arg_array, index_number);
  Invocation* result =
      module->Make<Invocation>(span, callee, std::vector<Expr*>{index});
  result->SetParentNonLexical(parent);
  return result;
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceMapInvocation(
    const Invocation* node, DeduceCtx* ctx) {
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
                              /*arg_array=*/args[0], /*parent=*/node->parent());
  XLS_ASSIGN_OR_RETURN(TypeAndParametricEnv tab,
                       ctx->typecheck_invocation()(ctx, element_invocation,
                                                   /*constexpr_env=*/{}));

  const FnStackEntry& caller_fn_entry = ctx->fn_stack().back();
  const ParametricEnv& caller_bindings = caller_fn_entry.parametric_env();
  Function* caller = caller_fn_entry.f();

  std::optional<TypeInfo*> dti = ctx->type_info()->GetInvocationTypeInfo(
      element_invocation, caller_bindings);
  if (dti.has_value()) {
    XLS_RETURN_IF_ERROR(ctx->type_info()->AddInvocationTypeInfo(
        *node, caller, caller_bindings, tab.parametric_env, dti.value()));
  } else {
    XLS_RETURN_IF_ERROR(ctx->type_info()->AddInvocationTypeInfo(
        *node, caller, caller_bindings, tab.parametric_env,
        /*derived_type_info=*/nullptr));
  }

  ArrayType* arg0_array_type = dynamic_cast<ArrayType*>(arg0_type.get());
  return std::make_unique<ArrayType>(tab.type->CloneToUnique(),
                                     arg0_array_type->size());
}

// Resolves "ref" to an AST function.
static absl::StatusOr<Function*> ResolveColonRefToFnForInvocation(
    ColonRef* ref, DeduceCtx* ctx) {
  std::optional<Import*> import = ref->ResolveImportSubject();
  if (!import.has_value()) {
    return TypeInferenceErrorStatus(
        ref->span(), nullptr,
        absl::StrFormat("Colon-reference subject `%s` did not refer to a "
                        "module, so `%s` cannot be invoked.",
                        ToAstNode(ref->subject())->ToString(), ref->ToString()),
        ctx->file_table());
  }
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << ref->ToString();
  std::optional<const ImportedInfo*> imported_info =
      ctx->type_info()->GetImported(*import);
  XLS_RET_CHECK(imported_info.has_value());
  Module* module = imported_info.value()->module;
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

absl::Status AppendArgsForInstantiation(
    const Instantiation* inst, const Expr* callee, absl::Span<Expr* const> args,
    DeduceCtx* ctx, std::vector<InstantiateArg>* instantiate_args) {
  for (Expr* arg : args) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         ctx->DeduceAndResolve(arg));
    VLOG(5) << absl::StreamFormat(
        "AppendArgsForInstantiation; arg: `%s` deduced: `%s` @ %s",
        arg->ToString(), type->ToString(),
        arg->span().ToString(ctx->file_table()));
    XLS_RET_CHECK(!type->IsMeta()) << "parametric arg: " << arg->ToString()
                                   << " type: " << type->ToString();
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

  // Map is special.
  if (IsBuiltinFn(node->callee(), "map")) {
    return DeduceMapInvocation(node, ctx);
  }

  // Gather up the type of all the (actual) arguments.
  std::vector<InstantiateArg> args;
  XLS_RETURN_IF_ERROR(AppendArgsForInstantiation(node, node->callee(),
                                                 node->args(), ctx, &args));

  // Find the callee as a DSLX Function from the expression.
  auto resolve_fn = [](const Instantiation* node,
                       DeduceCtx* ctx) -> absl::StatusOr<Function*> {
    Expr* callee = node->callee();

    Function* fn;
    if (auto* colon_ref = dynamic_cast<ColonRef*>(callee);
        colon_ref != nullptr) {
      XLS_ASSIGN_OR_RETURN(fn,
                           ResolveColonRefToFnForInvocation(colon_ref, ctx));
    } else if (auto* name_ref = dynamic_cast<NameRef*>(callee);
               name_ref != nullptr) {
      AstNode* definer = name_ref->GetDefiner();
      fn = dynamic_cast<Function*>(definer);
      if (fn == nullptr) {
        return TypeInferenceErrorStatus(
            node->callee()->span(), nullptr,
            absl::StrFormat("Invocation callee `%s` is not a function",
                            node->callee()->ToString()),
            ctx->file_table());
      }
    } else {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::StrCat("An invocation callee must be either a name reference "
                       "or a colon reference; instead got: ",
                       AstNodeKindToString(callee->kind())),
          ctx->file_table());
    }
    return fn;
  };

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
  if (!IsBuiltinFn(node->callee())) {
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
    bool callee_needs_implicit_token = callee_opt.value();

    // If the callee function needs an implicit token type (e.g. because it has
    // a fail!() or cover!() operation transitively) then so do we.
    if (callee_needs_implicit_token) {
      UseImplicitToken(ctx);
    }
  }

  return std::move(tab.type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceFormatMacro(const FormatMacro* node,
                                                        DeduceCtx* ctx) {
  int64_t arg_count = OperandsExpectedByFormat(node->format());

  if (arg_count != node->args().size()) {
    return ArgCountMismatchErrorStatus(
        node->span(),
        absl::StrFormat("%s macro expects %d argument(s) from format but has "
                        "%d argument(s)",
                        node->macro(), arg_count, node->args().size()),
        ctx->file_table());
  }

  // Check types of each argument.
  struct Visitor : public TypeVisitor {
   public:
    explicit Visitor(DeduceCtx* ctx, Span span) : ctx_(ctx), span_(span) {}

    absl::Status HandleArray(const ArrayType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleBits(const BitsType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleEnum(const EnumType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleToken(const TokenType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleStruct(const StructType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleTuple(const TupleType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleBitsConstructor(const BitsConstructorType& t) override {
      return absl::OkStatus();
    }
    absl::Status HandleFunction(const FunctionType& t) override {
      return TypeInferenceErrorStatus(
          span_, &t, ": Cannot format an expression with function type",
          ctx_->file_table());
    }
    absl::Status HandleChannel(const ChannelType& t) override {
      return TypeInferenceErrorStatus(
          span_, &t, ": Cannot format an expression with channel type",
          ctx_->file_table());
    }
    absl::Status HandleMeta(const MetaType& t) override {
      return TypeInferenceErrorStatus(
          span_, &t, ": Cannot format an expression with meta type",
          ctx_->file_table());
    }

   private:
    DeduceCtx* ctx_;
    Span span_;
  };

  for (Expr* arg : node->args()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         ctx->DeduceAndResolve(arg));

    Visitor v(ctx, arg->span());
    XLS_RETURN_IF_ERROR(type->Accept(v));
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

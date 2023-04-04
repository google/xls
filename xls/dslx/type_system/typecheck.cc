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

#include "xls/dslx/type_system/typecheck.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/maybe_explain_error.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

// Helper type to place on the stack when we intend to pop off a FnStackEntry
// when done, or expect a caller to pop it off for us. That is, this helps us
// check fn_stack() invariants are as expected.
class ScopedFnStackEntry {
 public:
  // Args:
  //  expect_popped: Indicates that we expect, in the destructor for this scope,
  //    that the entry will have already been popped.
  ScopedFnStackEntry(DeduceCtx* ctx, Module* module, bool expect_popped = false)
      : ctx_(ctx),
        depth_before_(ctx->fn_stack().size()),
        expect_popped_(expect_popped) {
    ctx->AddFnStackEntry(FnStackEntry::MakeTop(module));
  }

  ScopedFnStackEntry(Function* fn, DeduceCtx* ctx, bool expect_popped = false)
      : ctx_(ctx),
        depth_before_(ctx->fn_stack().size()),
        expect_popped_(expect_popped) {
    ctx->AddFnStackEntry(FnStackEntry::Make(fn, ParametricEnv()));
  }

  // Called when we close out a scope. We can't use this object as a scope
  // guard easily because we want to be able to detect if we return an
  // absl::Status early, so we have to manually put end-of-scope calls at usage
  // points.
  void Finish() {
    if (expect_popped_) {
      XLS_CHECK_EQ(ctx_->fn_stack().size(), depth_before_);
    } else {
      int64_t depth_after_push = depth_before_ + 1;
      XLS_CHECK_EQ(ctx_->fn_stack().size(), depth_after_push);
      ctx_->PopFnStackEntry();
    }
  }

 private:
  DeduceCtx* ctx_;
  int64_t depth_before_;
  bool expect_popped_;
};

absl::StatusOr<InterpValue> InterpretExpr(
    ImportData* import_data, TypeInfo* type_info, Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(import_data, type_info, expr, env,
                                      /*caller_bindings=*/std::nullopt));
  return BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{});
}

// Deduces the type for a ParametricBinding (via its type annotation).
static absl::StatusOr<std::unique_ptr<ConcreteType>> ParametricBindingToType(
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

// Checks the function's parametrics' and arguments' types.
absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>> CheckFunctionParams(
    Function* f, DeduceCtx* ctx) {
  for (ParametricBinding* parametric : f->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
    XLS_ASSIGN_OR_RETURN(parametric_binding_type,
                         UnwrapMetaType(std::move(parametric_binding_type),
                                        parametric->type_annotation()->span(),
                                        "parametric binding type"));

    if (parametric->expr() != nullptr) {
      // TODO(leary): 2020-07-06 Fully document the behavior of parametric
      // function calls in parametric expressions.
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                           ctx->Deduce(parametric->expr()));
      if (*expr_type != *parametric_binding_type) {
        return ctx->TypeMismatchError(
            parametric->span(), parametric->type_annotation(),
            *parametric_binding_type, parametric->expr(), *expr_type,
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

absl::StatusOr<TypeAndBindings> CheckParametricBuiltinInvocation(
    DeduceCtx* ctx, const Invocation* invocation, Function* caller) {
  Expr* callee = invocation->callee();
  NameRef* callee_nameref = dynamic_cast<NameRef*>(callee);

  std::vector<std::unique_ptr<ConcreteType>> arg_types;
  std::vector<Span> arg_spans;
  for (Expr* arg : invocation->args()) {
    XLS_ASSIGN_OR_RETURN(auto arg_type, ctx->Deduce(arg));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                         Resolve(*arg_type, ctx));
    arg_types.push_back(std::move(resolved));
    arg_spans.push_back(arg->span());
  }

  if (callee_nameref->identifier() == "fail!" ||
      callee_nameref->identifier() == "cover!") {
    ctx->type_info()->NoteRequiresImplicitToken(caller, true);

    if (callee_nameref->identifier() == "cover!") {
      // Make sure that the coverpoint's identifier is valid in both Verilog
      // and DSLX - notably, we don't support Verilog escaped strings.
      // TODO(rspringer): 2021-05-26: Ensure only one instance of an identifier
      // in a design.
      String* identifier_node = dynamic_cast<String*>(invocation->args()[0]);
      XLS_RET_CHECK(identifier_node != nullptr);
      if (identifier_node->text().empty()) {
        return InvalidIdentifierErrorStatus(
            invocation->span(),
            "An identifier must be specified with a cover! op.");
      }

      std::string identifier = identifier_node->text();
      if (identifier[0] == '\\') {
        return InvalidIdentifierErrorStatus(
            invocation->span(), "Verilog escaped strings are not supported.");
      }

      // We don't support Verilog "escaped strings", so we only have to worry
      // about regular identifier matching.
      if (!RE2::FullMatch(identifier, "[a-zA-Z_][a-zA-Z0-9$_]*")) {
        return InvalidIdentifierErrorStatus(
            invocation->span(),
            "A coverpoint identifier must start with a letter or underscore, "
            "and otherwise consist of letters, digits, underscores, and/or "
            "dollar signs.");
      }
    }
  }

  XLS_VLOG(5) << "Instantiating builtin parametric: "
              << callee_nameref->identifier();
  XLS_ASSIGN_OR_RETURN(
      SignatureFn fsignature,
      GetParametricBuiltinSignature(callee_nameref->identifier()));

  // Callback that signatures can use to request constexpr evaluation of their
  // arguments -- this is a special builtin superpower used by e.g. range().
  auto constexpr_eval = [&](int64_t argno) -> absl::StatusOr<InterpValue> {
    Expr* arg = invocation->args()[argno];

    XLS_ASSIGN_OR_RETURN(
        auto env, MakeConstexprEnv(ctx->import_data(), ctx->type_info(), arg,
                                   ctx->fn_stack().back().parametric_env()));

    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        InterpretExpr(ctx->import_data(), ctx->type_info(), arg, env));
    ctx->type_info()->NoteConstExpr(arg, value);
    return value;
  };

  std::vector<const ConcreteType*> arg_type_ptrs;
  arg_type_ptrs.reserve(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    const std::unique_ptr<ConcreteType>& arg_type = arg_types.at(i);
    if (arg_type->IsMeta()) {
      return TypeInferenceErrorStatus(
          invocation->args().at(i)->span(), arg_type.get(),
          "Argument to built-in function must be a value, not a type.");
    }
    arg_type_ptrs.push_back(arg_type.get());
  }

  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      fsignature(SignatureData{arg_type_ptrs, arg_spans,
                               callee_nameref->identifier(), invocation->span(),
                               /*parametric_bindings=*/{}, constexpr_eval},
                 ctx));

  FunctionType* fn_type = dynamic_cast<FunctionType*>(tab.type.get());
  XLS_RET_CHECK(fn_type != nullptr) << tab.type->ToString();

  const ParametricEnv& fn_parametric_env =
      ctx->fn_stack().back().parametric_env();
  XLS_VLOG(5) << "TypeInfo::AddInvocationCallBindings; type_info: "
              << ctx->type_info() << "; node: `" << invocation->ToString()
              << "`; caller: " << fn_parametric_env
              << "; callee: " << tab.parametric_env;
  ctx->type_info()->AddInvocationCallBindings(invocation,
                                              /*caller=*/fn_parametric_env,
                                              /*callee=*/tab.parametric_env);
  ctx->type_info()->SetItem(invocation->callee(), *fn_type);
  // We don't want to store a type on a BuiltinNameDef.
  if (std::holds_alternative<const NameDef*>(callee_nameref->name_def())) {
    ctx->type_info()->SetItem(ToAstNode(callee_nameref->name_def()), *fn_type);
  }

  ctx->type_info()->SetItem(invocation, fn_type->return_type());

  // fsignature returns a tab w/a fn type, not the fn return type (which is
  // what we actually want). We hack up `tab` to make this consistent with
  // InstantiateFunction.
  tab.type = fn_type->return_type().CloneToUnique();
  return tab;
}

absl::StatusOr<std::unique_ptr<DeduceCtx>> GetImportedDeduceCtx(
    DeduceCtx* ctx, const Invocation* invocation,
    const ParametricEnv& caller_bindings) {
  ColonRef* colon_ref = dynamic_cast<ColonRef*>(invocation->callee());
  ColonRef::Subject subject = colon_ref->subject();
  XLS_RET_CHECK(std::holds_alternative<NameRef*>(subject));
  NameRef* subject_nameref = std::get<NameRef*>(subject);
  AstNode* definer =
      std::get<const NameDef*>(subject_nameref->name_def())->definer();
  Import* import = dynamic_cast<Import*>(definer);

  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported,
                       ctx->type_info()->GetImportedOrError(import));

  XLS_ASSIGN_OR_RETURN(
      TypeInfo * imported_type_info,
      ctx->type_info_owner().New(imported->module, imported->type_info));
  std::unique_ptr<DeduceCtx> imported_ctx =
      ctx->MakeCtx(imported_type_info, imported->module);
  imported_ctx->AddFnStackEntry(FnStackEntry::MakeTop(imported->module));

  return imported_ctx;
}

// Checks a single #[test_proc] construct.
absl::Status CheckTestProc(const TestProc* test_proc, Module* module,
                           DeduceCtx* ctx) {
  Proc* proc = test_proc->proc();
  XLS_VLOG(2) << "Typechecking test proc: " << proc->identifier();

  const std::vector<Param*> config_params = proc->config()->params();
  if (config_params.size() != 1) {
    return TypeInferenceErrorStatus(
        proc->span(), nullptr,
        "Test proc `config` functions should only take a terminator channel.");
  }

  ChannelTypeAnnotation* channel_type =
      dynamic_cast<ChannelTypeAnnotation*>(config_params[0]->type_annotation());
  if (channel_type == nullptr) {
    return TypeInferenceErrorStatus(proc->config()->span(), nullptr,
                                    "Test proc 'config' functions should "
                                    "only take a terminator channel.");
  }
  BuiltinTypeAnnotation* payload_type =
      dynamic_cast<BuiltinTypeAnnotation*>(channel_type->payload());
  if (channel_type->direction() != ChannelDirection::kOut ||
      payload_type == nullptr || payload_type->GetBitCount() != 1) {
    return TypeInferenceErrorStatus(
        proc->config()->span(), nullptr,
        "Test proc 'config' terminator channel must be outgoing "
        "and have boolean payload.");
  }

  const std::vector<Param*>& next_params = proc->next()->params();
  BuiltinTypeAnnotation* builtin_type =
      dynamic_cast<BuiltinTypeAnnotation*>(next_params[0]->type_annotation());
  if (builtin_type == nullptr ||
      builtin_type->builtin_type() != BuiltinType::kToken) {
    return TypeInferenceErrorStatus(
        proc->next()->span(), nullptr,
        "Test proc 'next' functions first arg must be a token.");
  }

  if (proc->IsParametric()) {
    return TypeInferenceErrorStatus(
        proc->span(), nullptr, "Test proc functions cannot be parametric.");
  }

  {
    // The first and only argument to a Proc's config function is the terminator
    // channel. Create it here and mark it constexpr for deduction.
    ScopedFnStackEntry scoped_entry(proc->config(), ctx, false);
    InterpValue terminator(InterpValue::MakeChannel());
    ctx->type_info()->NoteConstExpr(proc->config()->params()[0], terminator);
    XLS_RETURN_IF_ERROR(CheckFunction(proc->config(), ctx));
    scoped_entry.Finish();
  }

  {
    ScopedFnStackEntry scoped_entry(proc->next(), ctx, false);
    XLS_RETURN_IF_ERROR(CheckFunction(proc->next(), ctx));
    scoped_entry.Finish();
    XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                         ctx->type_info()->GetTopLevelProcTypeInfo(proc));

    // Evaluate the init() fn's return type matches the expected state param.
    XLS_RETURN_IF_ERROR(CheckFunction(proc->init(), ctx));
    XLS_RET_CHECK_EQ(proc->next()->params().size(), 2);
    XLS_ASSIGN_OR_RETURN(ConcreteType * state_type,
                         type_info->GetItemOrError(next_params[1]));
    XLS_RET_CHECK(!state_type->IsMeta());

    // TestProcs can't be parameterized, so we don't need to worry about any
    // TypeInfo children, etc.
    XLS_ASSIGN_OR_RETURN(
        ConcreteType * init_metatype,
        type_info->GetItemOrError(proc->init()->return_type()));
    XLS_RET_CHECK(init_metatype->IsMeta());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> init_type,
                         UnwrapMetaType(init_metatype->CloneToUnique(),
                                        proc->init()->return_type()->span(),
                                        "init return type"));

    if (*state_type != *init_type) {
      return TypeInferenceErrorStatus(
          proc->next()->span(), nullptr,
          absl::StrFormat("'next' state param and init types differ: %s vs %s.",
                          state_type->ToString(), init_type->ToString()));
    }
  }

  XLS_VLOG(2) << "Finished typechecking test proc: " << proc->identifier();
  return absl::OkStatus();
}

bool CanTypecheckProc(Proc* p) {
  for (Param* param : p->config()->params()) {
    if (dynamic_cast<ChannelTypeAnnotation*>(param->type_annotation()) ==
        nullptr) {
      XLS_VLOG(3) << "Can't typecheck " << p->identifier() << " at top-level: "
                  << "its `config` function has a non-channel param.";
      return false;
    }
  }

  // Skip test procs (they're typechecked via a different path).
  if (p->parent() != nullptr) {
    return false;
  }

  return true;
}

absl::Status CheckModuleMember(const ModuleMember& member, Module* module,
                               ImportData* import_data, DeduceCtx* ctx) {
  if (std::holds_alternative<Import*>(member)) {
    Import* import = std::get<Import*>(member);
    XLS_ASSIGN_OR_RETURN(
        ModuleInfo * imported,
        DoImport(ctx->typecheck_module(), ImportTokens(import->subject()),
                 import_data, import->span()));
    ctx->type_info()->AddImport(import, &imported->module(),
                                imported->type_info());
  } else if (std::holds_alternative<ConstantDef*>(member) ||
             std::holds_alternative<EnumDef*>(member)) {
    XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(member)).status());
  } else if (std::holds_alternative<Function*>(member)) {
    Function* f = std::get<Function*>(member);
    if (f->IsParametric()) {
      // Typechecking of parametric functions is driven by invocation sites.
      return absl::OkStatus();
    }

    auto maybe_proc = f->proc();
    if (maybe_proc.has_value()) {
      Proc* p = maybe_proc.value();
      if (!CanTypecheckProc(p)) {
        return absl::OkStatus();
      }
    }

    XLS_VLOG(2) << "Typechecking function: " << f->ToString();
    ScopedFnStackEntry scoped_entry(f, ctx, /*expect_popped=*/false);
    XLS_RETURN_IF_ERROR(CheckFunction(f, ctx));
    scoped_entry.Finish();
    XLS_VLOG(2) << "Finished typechecking function: " << f->ToString();
  } else if (std::holds_alternative<Proc*>(member)) {
    // Just skip procs, as we typecheck their config & next functions (see the
    // previous else/if arm).
    return absl::OkStatus();
  } else if (std::holds_alternative<QuickCheck*>(member)) {
    QuickCheck* qc = std::get<QuickCheck*>(member);
    Function* f = qc->f();
    if (f->IsParametric()) {
      // TODO(leary): 2020-08-09 Add support for quickchecking parametric
      // functions.
      return TypeInferenceErrorStatus(
          f->span(), nullptr,
          "Quickchecking parametric functions is unsupported; see "
          "https://github.com/google/xls/issues/81");
    }

    XLS_VLOG(2) << "Typechecking quickcheck function: " << f->ToString();
    ScopedFnStackEntry scoped(f, ctx, false);
    XLS_RETURN_IF_ERROR(CheckFunction(f, ctx));
    scoped.Finish();

    std::optional<const ConcreteType*> quickcheck_f_body_type =
        ctx->type_info()->GetItem(f->body());
    XLS_RET_CHECK(quickcheck_f_body_type.has_value());
    auto u1 = BitsType::MakeU1();
    if (*quickcheck_f_body_type.value() != *u1) {
      return ctx->TypeMismatchError(
          f->span(), f->body(), *quickcheck_f_body_type.value(), nullptr, *u1,
          "Quickcheck functions must return a bool.");
    }

    XLS_VLOG(2) << "Finished typechecking quickcheck function: "
                << f->ToString();
  } else if (std::holds_alternative<StructDef*>(member)) {
    StructDef* struct_def = std::get<StructDef*>(member);
    XLS_VLOG(2) << "Typechecking struct: " << struct_def->ToString();
    ScopedFnStackEntry scoped(ctx, module);
    XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(member)).status());
    scoped.Finish();
    XLS_VLOG(2) << "Finished typechecking struct: " << struct_def->ToString();
  } else if (std::holds_alternative<TestFunction*>(member)) {
    TestFunction* tf = std::get<TestFunction*>(member);
    ScopedFnStackEntry scoped_entry(tf->fn(), ctx, /*expect_popped=*/false);
    XLS_RETURN_IF_ERROR(CheckFunction(tf->fn(), ctx));
    scoped_entry.Finish();
  } else if (std::holds_alternative<TestProc*>(member)) {
    XLS_RETURN_IF_ERROR(
        CheckTestProc(std::get<TestProc*>(member), module, ctx));
  } else if (std::holds_alternative<TypeAlias*>(member)) {
    TypeAlias* type_alias = std::get<TypeAlias*>(member);
    XLS_VLOG(2) << "Typechecking type alias: " << type_alias->ToString();
    ScopedFnStackEntry scoped(ctx, module);
    XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(member)).status());
    scoped.Finish();
    XLS_VLOG(2) << "Finished typechecking type alias: "
                << type_alias->ToString();
  }

  return absl::OkStatus();
}

// Set up parametric constraints and explicit bindings and runs the parametric
// instantiator.
absl::StatusOr<TypeAndBindings> InstantiateParametricFunction(
    DeduceCtx* ctx, DeduceCtx* parent_ctx, const Invocation* invocation,
    Function* callee_fn, const FunctionType& fn_type,
    const std::vector<InstantiateArg>& instantiate_args) {
  const std::vector<ParametricBinding*>& parametric_bindings =
      callee_fn->parametric_bindings();
  absl::flat_hash_map<std::string, InterpValue> explicit_bindings;
  std::vector<ParametricConstraint> parametric_constraints;
  parametric_constraints.reserve(callee_fn->parametric_bindings().size());
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
    XLS_RET_CHECK(std::holds_alternative<Expr*>(eot));
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
        parent_ctx->fn_stack().back().parametric_env(), value,
        value_type.get()));

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

bool IsTypeMismatchStatus(const absl::Status& status) {
  return status.code() == absl::StatusCode::kInvalidArgument &&
         status.message() == "DslxTypeMismatchError";
}

absl::Status MaybeExpandTypeErrorData(absl::Status orig, const DeduceCtx& ctx) {
  if (IsTypeMismatchStatus(orig)) {
    const std::optional<TypeMismatchErrorData>& data =
        ctx.type_mismatch_error_data();
    XLS_RET_CHECK(data.has_value())
        << "Internal error: type mismatch error "
           "was not accompanied by detail data; original status: "
        << orig;
    return MaybeExplainError(data.value());
  }

  return orig;
}

}  // namespace

absl::Status CheckFunction(Function* f, DeduceCtx* ctx) {
  XLS_VLOG(2) << "Typechecking fn: " << f->identifier();

  // Every top-level proc needs its own type info (that's shared between both
  // proc functions). Otherwise, the implicit channels created during top-level
  // Proc typechecking (see `DeduceParam()`) would conflict/override those
  // declared in a TestProc and passed to it.
  TypeInfo* original_ti = ctx->type_info();
  if (f->proc().has_value()) {
    absl::StatusOr<TypeInfo*> proc_ti =
        ctx->type_info()->GetTopLevelProcTypeInfo(f->proc().value());
    if (proc_ti.ok()) {
      XLS_RETURN_IF_ERROR(ctx->PushTypeInfo(proc_ti.value()));
    } else {
      ctx->AddDerivedTypeInfo();
    }
  }

  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<ConcreteType>> param_types,
                       CheckFunctionParams(f, ctx));

  // Second, typecheck the return type of the function.
  // Note: if there is no annotated return type, we assume nil.
  std::unique_ptr<ConcreteType> return_type;
  if (f->return_type() == nullptr) {
    return_type = TupleType::MakeUnit();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, DeduceAndResolve(f->return_type(), ctx));
    XLS_ASSIGN_OR_RETURN(return_type, UnwrapMetaType(std::move(return_type),
                                                     f->return_type()->span(),
                                                     "function return type"));
  }

  // Add proc members to the environment before typechecking the fn body.
  if (f->proc().has_value()) {
    Proc* p = f->proc().value();
    for (auto* param : p->members()) {
      XLS_ASSIGN_OR_RETURN(auto type, DeduceAndResolve(param, ctx));
      ctx->type_info()->SetItem(param, *type);
      ctx->type_info()->SetItem(param->name_def(), *type);
    }
  }

  // Assert type consistency between the body and deduced return types.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       DeduceAndResolve(f->body(), ctx));
  XLS_VLOG(3) << absl::StrFormat("Resolved return type: %s => %s",
                                 return_type->ToString(),
                                 body_type->ToString());
  if (body_type->IsMeta()) {
    return TypeInferenceErrorStatus(f->body()->span(), body_type.get(),
                                    "Types cannot be returned from functions");
  }
  if (*return_type != *body_type) {
    if (f->tag() == Function::Tag::kProcInit) {
      return ctx->TypeMismatchError(
          f->body()->span(), f->body(), *body_type, f->return_type(),
          *return_type,
          absl::StrFormat("'next' state param and 'init' types differ."));
    }

    if (f->tag() == Function::Tag::kProcNext) {
      return ctx->TypeMismatchError(
          f->body()->span(), f->body(), *body_type, f->return_type(),
          *return_type,
          absl::StrFormat("'next' input and output state types differ."));
    }

    return ctx->TypeMismatchError(
        f->body()->span(), f->body(), *body_type, f->return_type(),
        *return_type,
        absl::StrFormat("Return type of function body for '%s' did not match "
                        "the annotated return type.",
                        f->identifier()));
  }

  if (f->tag() != Function::Tag::kNormal) {
    // i.e., if this is a proc function.
    XLS_RETURN_IF_ERROR(original_ti->SetTopLevelProcTypeInfo(f->proc().value(),
                                                             ctx->type_info()));
    XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo());

    // Need to capture the initial value for top-level procs. For spawned procs,
    // DeduceSpawn() handles this.
    Proc* p = f->proc().value();
    Function* init = p->init();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                         ctx->Deduce(init->body()));
    // No need for ParametricEnv; top-level procs can't be parameterized.
    XLS_ASSIGN_OR_RETURN(InterpValue init_value,
                         ConstexprEvaluator::EvaluateToValue(
                             ctx->import_data(), ctx->type_info(),
                             ParametricEnv(), init->body()));
    ctx->type_info()->NoteConstExpr(init->body(), init_value);
  }

  // Implementation note: though we could have all functions have
  // NoteRequiresImplicitToken() be false unless otherwise noted, this helps
  // guarantee we did consider and make a note for every function -- the code
  // is generally complex enough it's nice to have this soundness check.
  if (std::optional<bool> requires_token =
          ctx->type_info()->GetRequiresImplicitToken(f);
      !requires_token.has_value()) {
    ctx->type_info()->NoteRequiresImplicitToken(f, false);
  }

  FunctionType function_type(std::move(param_types), std::move(body_type));
  ctx->type_info()->SetItem(f, function_type);
  ctx->type_info()->SetItem(f->name_def(), function_type);
  return absl::OkStatus();
}

absl::StatusOr<TypeAndBindings> CheckInvocation(
    DeduceCtx* ctx, const Invocation* invocation,
    const absl::flat_hash_map<const Param*, InterpValue>& constexpr_env) {
  XLS_VLOG(3) << "Typechecking invocation: " << invocation->ToString();
  Expr* callee = invocation->callee();

  Function* caller = ctx->fn_stack().back().f();
  if (IsNameParametricBuiltin(callee->ToString())) {
    return CheckParametricBuiltinInvocation(ctx, invocation, caller);
  }

  XLS_ASSIGN_OR_RETURN(Function * callee_fn,
                       ResolveFunction(callee, ctx->type_info()));

  const absl::Span<Expr* const> args = invocation->args();
  std::vector<InstantiateArg> instantiate_args;
  std::vector<std::unique_ptr<ConcreteType>> arg_types;
  instantiate_args.reserve(args.size() + 1);
  if (callee_fn->tag() == Function::Tag::kProcNext) {
    arg_types.push_back(std::make_unique<TokenType>());
    instantiate_args.push_back(
        {std::make_unique<TokenType>(), invocation->span()});
    XLS_RET_CHECK_EQ(invocation->args().size(), 1);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                         DeduceAndResolve(invocation->args()[0], ctx));
    arg_types.push_back(type->CloneToUnique());
    instantiate_args.push_back(
        InstantiateArg{std::move(type), invocation->span()});
  } else {
    for (Expr* arg : args) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                           DeduceAndResolve(arg, ctx));
      arg_types.push_back(type->CloneToUnique());
      instantiate_args.push_back(InstantiateArg{std::move(type), arg->span()});
    }
  }

  // Make a copy; the fn stack can get re-allocated, etc.
  ParametricEnv caller_parametric_env = ctx->fn_stack().back().parametric_env();
  absl::flat_hash_map<std::string, InterpValue> caller_parametric_env_map =
      caller_parametric_env.ToMap();

  // We need to deduce a callee relative to its parent module. We still need to
  // record data in the original module/ctx, so we hold on to the parent.
  DeduceCtx* parent_ctx = ctx;
  std::unique_ptr<DeduceCtx> imported_ctx_holder;
  if (dynamic_cast<ColonRef*>(invocation->callee()) != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        imported_ctx_holder,
        GetImportedDeduceCtx(ctx, invocation, caller_parametric_env));
    ctx = imported_ctx_holder.get();
  }

  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<ConcreteType>> param_types,
                       CheckFunctionParams(callee_fn, ctx));

  std::unique_ptr<ConcreteType> return_type;
  if (callee_fn->return_type() == nullptr) {
    return_type = TupleType::MakeUnit();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, ctx->Deduce(callee_fn->return_type()));
    XLS_ASSIGN_OR_RETURN(
        return_type,
        UnwrapMetaType(std::move(return_type), callee_fn->return_type()->span(),
                       "function return type"));
  }

  FunctionType fn_type(std::move(param_types), std::move(return_type));

  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      InstantiateParametricFunction(ctx, parent_ctx, invocation, callee_fn,
                                    fn_type, instantiate_args));

  // Now that we have the necessary bindings, check for recursion.
  std::vector<FnStackEntry> fn_stack = parent_ctx->fn_stack();
  while (!fn_stack.empty()) {
    if (fn_stack.back().f() == callee_fn &&
        fn_stack.back().parametric_env() == tab.parametric_env) {
      return TypeInferenceErrorStatus(
          invocation->span(), nullptr,
          absl::StrFormat("Recursion detected while typechecking; name: '%s'",
                          callee_fn->identifier()));
    }
    fn_stack.pop_back();
  }

  // We execute this function if we're parametric or a proc. In either case, we
  // want to create a new TypeInfo. The reason for the former is obvious. The
  // reason for the latter is that we need separate constexpr data for every
  // instantiation of a proc. If we didn't create new bindings/a new TypeInfo
  // here, then if we instantiated the same proc 2x from some parent proc, we'd
  // end up with only a single set of constexpr values for proc members.
  parent_ctx->type_info()->AddInvocationCallBindings(
      invocation, caller_parametric_env, tab.parametric_env);

  FunctionType instantiated_ft{std::move(arg_types), tab.type->CloneToUnique()};
  parent_ctx->type_info()->SetItem(invocation->callee(), instantiated_ft);
  ctx->type_info()->SetItem(callee_fn->name_def(), instantiated_ft);

  // We need to deduce fn body, so we're going to call Deduce, which means we'll
  // need a new stack entry w/the new symbolic bindings.
  TypeInfo* original_ti = parent_ctx->type_info();
  ctx->AddFnStackEntry(
      FnStackEntry::Make(callee_fn, tab.parametric_env, invocation));
  ctx->AddDerivedTypeInfo();

  if (callee_fn->proc().has_value()) {
    Proc* p = callee_fn->proc().value();
    for (auto* member : p->members()) {
      XLS_ASSIGN_OR_RETURN(auto type, DeduceAndResolve(member, ctx));
      ctx->type_info()->SetItem(member, *type);
      ctx->type_info()->SetItem(member->name_def(), *type);
    }
  }

  // Mark params (for proc config fns) or proc members (for proc next fns) as
  // constexpr.
  for (const auto& [k, v] : constexpr_env) {
    ctx->type_info()->NoteConstExpr(k, v);
    ctx->type_info()->NoteConstExpr(k->name_def(), v);
  }

  // Add the new parametric bindings to the constexpr set.
  const auto& bindings_map = tab.parametric_env.ToMap();
  for (ParametricBinding* parametric : callee_fn->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
    XLS_ASSIGN_OR_RETURN(parametric_binding_type,
                         UnwrapMetaType(std::move(parametric_binding_type),
                                        parametric->type_annotation()->span(),
                                        "parametric binding type"));
    if (bindings_map.contains(parametric->identifier())) {
      ctx->type_info()->NoteConstExpr(
          parametric->name_def(), bindings_map.at(parametric->identifier()));
    }
  }

  for (auto* param : callee_fn->params()) {
    std::optional<const ConcreteType*> param_type =
        ctx->type_info()->GetItem(param);
    XLS_RET_CHECK(param_type.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_type,
                         Resolve(*param_type.value(), ctx));
    ctx->type_info()->SetItem(param, *resolved_type);
    ctx->type_info()->SetItem(param->name_def(), *resolved_type);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       ctx->Deduce(callee_fn->body()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_body_type,
                       Resolve(*body_type, ctx));
  XLS_RET_CHECK(!resolved_body_type->IsMeta());

  // Assert type consistency between the body and deduced return types.
  if (*tab.type != *resolved_body_type) {
    XLS_VLOG(5) << "tab.type: " << tab.type->ToString()
                << " resolved_body_type: " << resolved_body_type->ToString();
    if (callee_fn->tag() == Function::Tag::kProcInit) {
      return ctx->TypeMismatchError(
          callee_fn->body()->span(), callee_fn->body(), *body_type, nullptr,
          *tab.type,
          absl::StrFormat("'next' state param and 'init' types differ."));
    }

    if (callee_fn->tag() == Function::Tag::kProcNext) {
      return ctx->TypeMismatchError(
          callee_fn->body()->span(), callee_fn->body(), *body_type, nullptr,
          *tab.type,
          absl::StrFormat("'next' input and output state types differ."));
    }

    return ctx->TypeMismatchError(
        callee_fn->body()->span(), callee_fn->body(), *body_type, nullptr,
        *tab.type,
        absl::StrFormat("Return type of function body for '%s' did not match "
                        "the annotated return type.",
                        callee_fn->identifier()));
  }

  original_ti->SetInvocationTypeInfo(invocation, tab.parametric_env,
                                     ctx->type_info());

  XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo());
  ctx->PopFnStackEntry();

  // Implementation note: though we could have all functions have
  // NoteRequiresImplicitToken() be false unless otherwise noted, this helps
  // guarantee we did consider and make a note for every function -- the code
  // is generally complex enough it's nice to have this soundness check.
  if (std::optional<bool> requires_token =
          ctx->type_info()->GetRequiresImplicitToken(callee_fn);
      !requires_token.has_value()) {
    ctx->type_info()->NoteRequiresImplicitToken(callee_fn, false);
  }

  return tab;
}

absl::StatusOr<TypeInfo*> CheckModule(Module* module, ImportData* import_data,
                                      WarningCollector* warnings) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->type_info_owner().New(module));

  auto typecheck_module =
      [import_data, warnings](Module* module) -> absl::StatusOr<TypeInfo*> {
    return CheckModule(module, import_data, warnings);
  };

  DeduceCtx ctx(type_info, module,
                /*deduce_function=*/&Deduce,
                /*typecheck_function=*/&CheckFunction,
                /*typecheck_module=*/typecheck_module,
                /*typecheck_invocation=*/&CheckInvocation, import_data,
                warnings, /*parent=*/nullptr);
  ctx.AddFnStackEntry(FnStackEntry::MakeTop(module));

  for (const ModuleMember& member : module->top()) {
    absl::Status status = CheckModuleMember(member, module, import_data, &ctx);
    if (!status.ok()) {
      return MaybeExpandTypeErrorData(status, ctx);
    }
  }

  return type_info;
}

absl::StatusOr<std::optional<BuiltinType>> GetAsBuiltinType(
    Module* module, TypeInfo* type_info, ImportData* import_data,
    const TypeAnnotation* type) {
  if (auto* builtin_type = dynamic_cast<const BuiltinTypeAnnotation*>(type)) {
    return builtin_type->builtin_type();
  }

  if (auto* array_type = dynamic_cast<const ArrayTypeAnnotation*>(type)) {
    TypeAnnotation* element_type = array_type->element_type();
    auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(element_type);
    if (builtin_type == nullptr) {
      return std::nullopt;
    }

    // If the array size/dim is a scalar < 64b, then the element is really an
    // integral type.
    XLS_ASSIGN_OR_RETURN(
        InterpValue array_dim_value,
        InterpretExpr(import_data, type_info, array_type->dim(), /*env=*/{}));

    if (builtin_type->builtin_type() != BuiltinType::kBits &&
        builtin_type->builtin_type() != BuiltinType::kUN &&
        builtin_type->builtin_type() != BuiltinType::kSN) {
      return std::nullopt;
    }

    XLS_ASSIGN_OR_RETURN(uint64_t array_dim,
                         array_dim_value.GetBitValueUint64());
    if (array_dim_value.IsBits() && array_dim > 0 && array_dim <= 64) {
      return GetBuiltinType(builtin_type->builtin_type() == BuiltinType::kSN,
                            array_dim);
    }
  }

  return std::nullopt;
}

}  // namespace xls::dslx

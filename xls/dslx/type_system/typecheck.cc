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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/maybe_explain_error.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/scoped_fn_stack_entry.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"
#include "xls/dslx/type_system/typecheck_invocation.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/type_system/warn_on_defined_but_unused.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

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
    ScopedFnStackEntry scoped_entry(proc->config(), ctx, WithinProc::kYes);
    InterpValue terminator(InterpValue::MakeChannel());
    ctx->type_info()->NoteConstExpr(proc->config()->params()[0], terminator);
    XLS_RETURN_IF_ERROR(CheckFunction(proc->config(), ctx));
    scoped_entry.Finish();
  }

  {
    ScopedFnStackEntry scoped_entry(proc->next(), ctx, WithinProc::kYes);
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

static bool CanTypecheckProc(Proc* p) {
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

    bool within_proc = false;
    auto maybe_proc = f->proc();
    if (maybe_proc.has_value()) {
      within_proc = true;
      Proc* p = maybe_proc.value();
      if (!CanTypecheckProc(p)) {
        return absl::OkStatus();
      }
    }

    XLS_VLOG(2) << "Typechecking function: `" << f->ToString() << "`";
    ScopedFnStackEntry scoped_entry(
        f, ctx, within_proc ? WithinProc::kYes : WithinProc::kNo);
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
    ScopedFnStackEntry scoped(f, ctx, WithinProc::kNo);
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
    if (tf->fn()->IsParametric()) {
      return TypeInferenceErrorStatus(tf->fn()->span(), nullptr,
                                      "Test functions cannot be parametric.");
    }
    ScopedFnStackEntry scoped_entry(tf->fn(), ctx, WithinProc::kNo);
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
  } else if (std::holds_alternative<ConstAssert*>(member)) {
    XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(member)).status());
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unimplemented node for module-level typechecking: ",
                     ToAstNode(member)->GetNodeTypeName()));
  }

  return absl::OkStatus();
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

  // See if the function is named with a `_test` suffix but not marked with a
  // test annotation -- this is likely to be a user mistake, so we give a
  // warning.
  if (absl::EndsWith(f->identifier(), "_test")) {
    AstNode* parent = f->parent();
    if (parent == nullptr || parent->kind() != AstNodeKind::kTestFunction) {
      ctx->warnings()->Add(
          f->span(), WarningKind::kMisleadingFunctionName,
          absl::StrFormat("Function `%s` ends with `_test` but is "
                          "not marked as a unit test via #[test]",
                          f->identifier()));
    }
  }

  // Every top-level proc needs its own type info (that's shared between both
  // proc functions). Otherwise, the implicit channels created during top-level
  // Proc typechecking (see `DeduceParam()`) would conflict/override those
  // declared in a TestProc and passed to it.
  TypeInfo* original_ti = ctx->type_info();
  TypeInfo* derived_type_info = nullptr;
  if (f->proc().has_value()) {
    absl::StatusOr<TypeInfo*> proc_ti =
        ctx->type_info()->GetTopLevelProcTypeInfo(f->proc().value());
    if (proc_ti.ok()) {
      XLS_RETURN_IF_ERROR(ctx->PushTypeInfo(proc_ti.value()));
      derived_type_info = proc_ti.value();
    } else {
      derived_type_info = ctx->AddDerivedTypeInfo();
    }
  }

  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<ConcreteType>> param_types,
                       TypecheckFunctionParams(f, ctx));

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
    XLS_VLOG(5) << "return type: " << return_type->ToString()
                << " body type: " << body_type->ToString();
    if (f->tag() == FunctionTag::kProcInit) {
      return ctx->TypeMismatchError(
          f->body()->span(), f->body(), *body_type, f->return_type(),
          *return_type,
          absl::StrFormat("'next' state param and 'init' types differ."));
    }

    if (f->tag() == FunctionTag::kProcNext) {
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

  if (return_type->HasParametricDims()) {
    return TypeInferenceErrorStatus(
        f->return_type()->span(), return_type.get(),
        absl::StrFormat(
            "Parametric type being returned from function -- types must be "
            "fully resolved, please fully instantiate the type"));
  }

  // Implementation note: we have to check for defined-but-unused values before
  // we pop derived type info below.
  XLS_RETURN_IF_ERROR(WarnOnDefinedButUnused(f, ctx));

  if (f->tag() != FunctionTag::kNormal) {
    XLS_RET_CHECK(derived_type_info != nullptr);

    // i.e., if this is a proc function.
    XLS_RETURN_IF_ERROR(original_ti->SetTopLevelProcTypeInfo(f->proc().value(),
                                                             ctx->type_info()));
    XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo(derived_type_info));

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
                             ctx->warnings(), ParametricEnv(), init->body()));
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
                /*typecheck_invocation=*/&TypecheckInvocation, import_data,
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
                         array_dim_value.GetBitValueUnsigned());
    if (array_dim_value.IsBits() && array_dim > 0 && array_dim <= 64) {
      return GetBuiltinType(builtin_type->builtin_type() == BuiltinType::kSN,
                            array_dim);
    }
  }

  return std::nullopt;
}

}  // namespace xls::dslx

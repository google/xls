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

#include "xls/dslx/type_system/typecheck_module.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/maybe_explain_error.h"
#include "xls/dslx/type_system/scoped_fn_stack_entry.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"
#include "xls/dslx/type_system/typecheck_function.h"
#include "xls/dslx/type_system/typecheck_invocation.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// Checks a single #[test_proc] construct.
absl::Status CheckTestProc(const TestProc* test_proc, Module* module,
                           DeduceCtx* ctx) {
  Proc* proc = test_proc->proc();
  VLOG(2) << "Typechecking test proc: " << proc->identifier();

  const std::vector<Param*> config_params = proc->config().params();
  if (config_params.size() != 1) {
    return TypeInferenceErrorStatus(
        proc->span(), nullptr,
        "Test proc `config` functions should only take a terminator channel.",
        ctx->file_table());
  }

  ChannelTypeAnnotation* channel_type =
      dynamic_cast<ChannelTypeAnnotation*>(config_params[0]->type_annotation());
  if (channel_type == nullptr) {
    return TypeInferenceErrorStatus(proc->config().span(), nullptr,
                                    "Test proc 'config' functions should "
                                    "only take a terminator channel.",
                                    ctx->file_table());
  }
  BuiltinTypeAnnotation* payload_type =
      dynamic_cast<BuiltinTypeAnnotation*>(channel_type->payload());
  if (channel_type->direction() != ChannelDirection::kOut ||
      payload_type == nullptr || payload_type->GetBitCount() != 1) {
    return TypeInferenceErrorStatus(
        proc->config().span(), nullptr,
        "Test proc 'config' terminator channel must be outgoing "
        "and have boolean payload.",
        ctx->file_table());
  }

  if (proc->IsParametric()) {
    return TypeInferenceErrorStatus(proc->span(), nullptr,
                                    "Test proc functions cannot be parametric.",
                                    ctx->file_table());
  }

  {
    // The first and only argument to a Proc's config function is the terminator
    // channel. Create it here and mark it constexpr for deduction.
    ScopedFnStackEntry scoped_entry(proc->config(), ctx, WithinProc::kYes);
    InterpValue terminator(
        InterpValue::MakeChannelReference(ChannelDirection::kOut));
    ctx->type_info()->NoteConstExpr(proc->config().params()[0], terminator);
    XLS_RETURN_IF_ERROR(TypecheckFunction(proc->config(), ctx));
    scoped_entry.Finish();
  }

  {
    ScopedFnStackEntry scoped_entry(proc->next(), ctx, WithinProc::kYes);
    XLS_RETURN_IF_ERROR(TypecheckFunction(proc->next(), ctx));
    scoped_entry.Finish();
    XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                         ctx->type_info()->GetTopLevelProcTypeInfo(proc));

    // Evaluate the init() fn's return type matches the expected state param.
    XLS_RETURN_IF_ERROR(TypecheckFunction(proc->init(), ctx));
    XLS_RET_CHECK_EQ(proc->next().params().size(), 1);
    XLS_ASSIGN_OR_RETURN(Type * state_type, type_info->GetItemOrError(
                                                proc->next().params().back()));
    XLS_RET_CHECK(!state_type->IsMeta());

    // TestProcs can't be parameterized, so we don't need to worry about any
    // TypeInfo children, etc.
    XLS_ASSIGN_OR_RETURN(Type * init_metatype,
                         type_info->GetItemOrError(proc->init().return_type()));
    XLS_RET_CHECK(init_metatype->IsMeta());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> init_type,
                         UnwrapMetaType(init_metatype->CloneToUnique(),
                                        proc->init().return_type()->span(),
                                        "init return type", ctx->file_table()));

    if (*state_type != *init_type) {
      return TypeInferenceErrorStatus(
          proc->next().span(), nullptr,
          absl::StrFormat("'next' state param and init types differ: %s vs %s.",
                          state_type->ToString(), init_type->ToString()),
          ctx->file_table());
    }
  }

  VLOG(2) << "Finished typechecking test proc: " << proc->identifier();
  return absl::OkStatus();
}

static bool CanTypecheckProc(Proc* p) {
  for (Param* param : p->config().params()) {
    if (dynamic_cast<ChannelTypeAnnotation*>(param->type_annotation()) ==
        nullptr) {
      VLOG(3) << "Can't typecheck " << p->identifier() << " at top-level: "
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

// Note that this (local) routine will only be applied to unparameterized
// `proc`s -- parameterized procs get instantiated via `spawn` statements and
// associated statements will be typechecked at those instantiation points.
static absl::Status TypecheckProcStmts(Proc* p, DeduceCtx* ctx) {
  for (ProcStmt& stmt : p->stmts()) {
    XLS_RETURN_IF_ERROR(
        absl::visit(Visitor{// These are typechecked elsewhere in the flow.
                            [](Function* n) { return absl::OkStatus(); },
                            [](ProcMember* n) { return absl::OkStatus(); },

                            [&](auto* n) { return ctx->Deduce(n).status(); }},
                    stmt));
  }
  return absl::OkStatus();
}

static absl::Status TypecheckQuickcheck(QuickCheck* qc, DeduceCtx* ctx) {
  Function* quickcheck_f_ptr = qc->fn();
  XLS_RET_CHECK(quickcheck_f_ptr != nullptr);
  Function& quickcheck_f = *quickcheck_f_ptr;

  if (quickcheck_f.IsParametric()) {
    // TODO(leary): 2020-08-09 Add support for quickchecking parametric
    // functions.
    return TypeInferenceErrorStatus(
        quickcheck_f.span(), nullptr,
        "Quickchecking parametric functions is unsupported; see "
        "https://github.com/google/xls/issues/81",
        ctx->file_table());
  }

  VLOG(2) << "Typechecking quickcheck function: " << quickcheck_f.ToString();
  ScopedFnStackEntry scoped(quickcheck_f, ctx, WithinProc::kNo);
  XLS_RETURN_IF_ERROR(TypecheckFunction(quickcheck_f, ctx));
  scoped.Finish();

  std::optional<const Type*> quickcheck_f_body_type =
      ctx->type_info()->GetItem(quickcheck_f.body());
  XLS_RET_CHECK(quickcheck_f_body_type.has_value());
  auto u1 = BitsType::MakeU1();
  if (*quickcheck_f_body_type.value() != *u1) {
    return ctx->TypeMismatchError(quickcheck_f.span(), quickcheck_f.body(),
                                  *quickcheck_f_body_type.value(), nullptr, *u1,
                                  "Quickcheck functions must return a bool.");
  }

  VLOG(2) << "Finished typechecking quickcheck function: "
          << quickcheck_f.ToString();
  return absl::OkStatus();
}

absl::Status TypecheckModuleMember(const ModuleMember& member, Module* module,
                                   ImportData* import_data, DeduceCtx* ctx) {
  VLOG(5) << "TypecheckModuleMember; member: `" << ToAstNode(member)->ToString()
          << "`";
  XLS_RET_CHECK_EQ(ctx->fn_stack().size(), 1);
  XLS_RET_CHECK_EQ(ctx->fn_stack().back().f(), nullptr);

  XLS_RETURN_IF_ERROR(absl::visit(
      Visitor{
          [import_data, ctx](Import* import) -> absl::Status {
            XLS_ASSIGN_OR_RETURN(
                ModuleInfo * imported,
                DoImport(ctx->typecheck_module(),
                         ImportTokens(import->subject()), import_data,
                         import->span(), import_data->vfs()));

            ctx->type_info()->AddImport(import, &imported->module(),
                                        imported->type_info());
            return absl::OkStatus();
          },
          [](Use* use) -> absl::Status {
            return absl::UnimplementedError(
                "`use` statements are not yet supported for typechecking");
          },
          [ctx](ConstantDef* member) -> absl::Status {
            return ctx->Deduce(ToAstNode(member)).status();
          },
          [ctx](EnumDef* member) -> absl::Status {
            return ctx->Deduce(ToAstNode(member)).status();
          },
          [ctx](Function* f_ptr) -> absl::Status {
            XLS_RET_CHECK(f_ptr != nullptr);
            Function& f = *f_ptr;

            if (f.IsParametric()) {
              // Typechecking of parametric functions is driven by invocation
              // sites.
              return absl::OkStatus();
            }

            bool within_proc = false;
            auto maybe_proc = f.proc();
            if (maybe_proc.has_value()) {
              within_proc = true;
              Proc* p = maybe_proc.value();
              if (!CanTypecheckProc(p)) {
                return absl::OkStatus();
              }
              VLOG(5) << "Determined function was within proc `"
                      << p->identifier()
                      << "` but proc is able to be typechecked";
              XLS_RETURN_IF_ERROR(TypecheckProcStmts(p, ctx));
            }

            VLOG(2) << "Typechecking function: `" << f.ToString() << "`";

            // Note: we push a function stack entry on top of the "top" entry
            // for the module as we go in to typecheck this function.
            ScopedFnStackEntry scoped_entry(
                f, ctx, within_proc ? WithinProc::kYes : WithinProc::kNo);
            XLS_RETURN_IF_ERROR(TypecheckFunction(f, ctx));
            scoped_entry.Finish();

            VLOG(2) << "Finished typechecking function: " << f.ToString();
            return absl::OkStatus();
          },
          [](Proc* member) -> absl::Status {
            // Just skip non-impl-style procs, as we typecheck their config &
            // next functions (see the previous else/if arm).
            return absl::OkStatus();
          },
          [ctx](QuickCheck* qc) -> absl::Status {
            return TypecheckQuickcheck(qc, ctx);
          },
          [ctx](StructDef* struct_def) -> absl::Status {
            VLOG(2) << "Typechecking struct: " << struct_def->ToString();
            XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(struct_def)).status());
            VLOG(2) << "Finished typechecking struct: "
                    << struct_def->ToString();
            return absl::OkStatus();
          },
          [ctx](TestFunction* tf) -> absl::Status {
            if (tf->fn().IsParametric()) {
              return TypeInferenceErrorStatus(
                  tf->fn().span(), nullptr,
                  "Test functions cannot be parametric.", ctx->file_table());
            }

            // Note: we push a function stack entry on top of the "top" entry
            // for the module as we go in to typecheck this test function.
            ScopedFnStackEntry scoped_entry(tf->fn(), ctx, WithinProc::kNo);
            XLS_RETURN_IF_ERROR(TypecheckFunction(tf->fn(), ctx));
            scoped_entry.Finish();
            return absl::OkStatus();
          },
          [module, ctx](TestProc* member) -> absl::Status {
            return CheckTestProc(member, module, ctx);
          },
          [ctx](TypeAlias* type_alias) -> absl::Status {
            VLOG(2) << "Typechecking type alias: " << type_alias->ToString();
            XLS_RETURN_IF_ERROR(ctx->Deduce(type_alias).status());
            VLOG(2) << "Finished typechecking type alias: "
                    << type_alias->ToString();
            return absl::OkStatus();
          },
          [ctx](ConstAssert* member) -> absl::Status {
            return ctx->Deduce(ToAstNode(member)).status();
          },
          [ctx](Impl* member) -> absl::Status {
            return ctx->Deduce(ToAstNode(member)).status();
          },
          [ctx](ProcDef* proc_def) -> absl::Status {
            VLOG(2) << "Typechecking impl-style proc: " << proc_def->ToString();
            XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(proc_def)).status());
            VLOG(2) << "Finished typechecking impl-style proc: "
                    << proc_def->ToString();
            return absl::OkStatus();
          },
          [](VerbatimNode* verbatim_node) -> absl::Status {
            return absl::OkStatus();
          }},
      member));

  XLS_RET_CHECK_EQ(ctx->fn_stack().size(), 1);
  XLS_RET_CHECK_EQ(ctx->fn_stack().back().f(), nullptr);
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
    return MaybeExplainError(data.value(), ctx.file_table());
  }

  return orig;
}

}  // namespace

absl::StatusOr<TypeInfo*> TypecheckModule(Module* module,
                                          ImportData* import_data,
                                          WarningCollector* warnings) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->type_info_owner().New(module));

  auto typecheck_module =
      [import_data, warnings](Module* module) -> absl::StatusOr<TypeInfo*> {
    return TypecheckModule(module, import_data, warnings);
  };

  DeduceCtx ctx(type_info, module,
                /*deduce_function=*/&Deduce,
                /*typecheck_function=*/&TypecheckFunction,
                /*typecheck_module=*/typecheck_module,
                /*typecheck_invocation=*/&TypecheckInvocation, import_data,
                warnings, /*parent=*/nullptr);

  XLS_RET_CHECK(ctx.fn_stack().empty());
  ctx.AddFnStackEntry(FnStackEntry::MakeTop(module));
  XLS_RET_CHECK_EQ(ctx.fn_stack().back().f(), nullptr);

  for (const ModuleMember& member : module->top()) {
    absl::Status status =
        TypecheckModuleMember(member, module, import_data, &ctx);
    if (!status.ok()) {
      return MaybeExpandTypeErrorData(status, ctx);
    }
    // Should just be the module-top entry on the function stack when we're
    // done with each member.
    XLS_RET_CHECK_EQ(ctx.fn_stack().size(), 1);
    XLS_RET_CHECK_EQ(ctx.fn_stack().back().f(), nullptr);
  }

  return type_info;
}

}  // namespace xls::dslx

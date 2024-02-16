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

#include "xls/dslx/type_system/typecheck_function.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_invocation.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/type_system/warn_on_defined_but_unused.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

// Sees if the function is named with a `_test` suffix but not marked with a
// test annotation -- this is likely to be a user mistake, so we give a warning.
void WarnIfConfusinglyNamedLikeTest(Function& f, DeduceCtx* ctx) {
  if (!absl::EndsWith(f.identifier(), "_test")) {
    return;
  }
  AstNode* parent = f.parent();
  if (parent == nullptr || parent->kind() != AstNodeKind::kTestFunction) {
    ctx->warnings()->Add(
        f.span(), WarningKind::kMisleadingFunctionName,
        absl::StrFormat("Function `%s` ends with `_test` but is "
                        "not marked as a unit test via #[test]",
                        f.identifier()));
  }
}

}  // namespace

absl::Status TypecheckFunction(Function& f, DeduceCtx* ctx) {
  XLS_VLOG(2) << "Typechecking fn: " << f.identifier();

  WarnIfConfusinglyNamedLikeTest(f, ctx);

  // Every top-level proc needs its own type info (that's shared between both
  // proc functions). Otherwise, the implicit channels created during top-level
  // Proc typechecking (see `DeduceParam()`) would conflict/override those
  // declared in a TestProc and passed to it.
  TypeInfo* original_ti = ctx->type_info();
  TypeInfo* derived_type_info = nullptr;
  if (f.proc().has_value()) {
    absl::StatusOr<TypeInfo*> proc_ti =
        ctx->type_info()->GetTopLevelProcTypeInfo(f.proc().value());
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
  if (f.return_type() == nullptr) {
    return_type = TupleType::MakeUnit();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, DeduceAndResolve(f.return_type(), ctx));
    XLS_ASSIGN_OR_RETURN(return_type, UnwrapMetaType(std::move(return_type),
                                                     f.return_type()->span(),
                                                     "function return type"));
  }

  // Add proc members to the environment before typechecking the fn body.
  if (f.proc().has_value()) {
    Proc* p = f.proc().value();
    for (auto* param : p->members()) {
      XLS_ASSIGN_OR_RETURN(auto type, DeduceAndResolve(param, ctx));
      ctx->type_info()->SetItem(param, *type);
      ctx->type_info()->SetItem(param->name_def(), *type);
    }
  }

  // Assert type consistency between the body and deduced return types.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       DeduceAndResolve(f.body(), ctx));
  XLS_VLOG(3) << absl::StrFormat("Resolved return type: %s => %s",
                                 return_type->ToString(),
                                 body_type->ToString());
  if (body_type->IsMeta()) {
    return TypeInferenceErrorStatus(f.body()->span(), body_type.get(),
                                    "Types cannot be returned from functions");
  }
  if (*return_type != *body_type) {
    XLS_VLOG(5) << "return type: " << return_type->ToString()
                << " body type: " << body_type->ToString();
    if (f.tag() == FunctionTag::kProcInit) {
      return ctx->TypeMismatchError(
          f.body()->span(), f.body(), *body_type, f.return_type(), *return_type,
          absl::StrFormat("'next' state param and 'init' types differ."));
    }

    if (f.tag() == FunctionTag::kProcNext) {
      return ctx->TypeMismatchError(
          f.body()->span(), f.body(), *body_type, f.return_type(), *return_type,
          absl::StrFormat("'next' input and output state types differ."));
    }

    return ctx->TypeMismatchError(
        f.body()->span(), f.body(), *body_type, f.return_type(), *return_type,
        absl::StrFormat("Return type of function body for '%s' did not match "
                        "the annotated return type.",
                        f.identifier()));
  }

  if (return_type->HasParametricDims()) {
    return TypeInferenceErrorStatus(
        f.return_type()->span(), return_type.get(),
        absl::StrFormat(
            "Parametric type being returned from function -- types must be "
            "fully resolved, please fully instantiate the type"));
  }

  // Implementation note: we have to check for defined-but-unused values before
  // we pop derived type info below.
  XLS_RETURN_IF_ERROR(WarnOnDefinedButUnused(f, ctx));

  if (f.tag() != FunctionTag::kNormal) {
    XLS_RET_CHECK(derived_type_info != nullptr);

    // i.e., if this is a proc function.
    XLS_RETURN_IF_ERROR(original_ti->SetTopLevelProcTypeInfo(f.proc().value(),
                                                             ctx->type_info()));
    XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo(derived_type_info));

    // Need to capture the initial value for top-level procs. For spawned procs,
    // DeduceSpawn() handles this.
    Proc* p = f.proc().value();
    const Function& init = p->init();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                         ctx->Deduce(init.body()));
    // No need for ParametricEnv; top-level procs can't be parameterized.
    XLS_ASSIGN_OR_RETURN(InterpValue init_value,
                         ConstexprEvaluator::EvaluateToValue(
                             ctx->import_data(), ctx->type_info(),
                             ctx->warnings(), ParametricEnv(), init.body()));
    ctx->type_info()->NoteConstExpr(init.body(), init_value);
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
  ctx->type_info()->SetItem(&f, function_type);
  ctx->type_info()->SetItem(f.name_def(), function_type);

  return absl::OkStatus();
}

}  // namespace xls::dslx

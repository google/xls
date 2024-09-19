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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/scoped_fn_stack_entry.h"
#include "xls/dslx/type_system/type.h"
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

// Checks that an actual `TypeDim` for one parametric argument matches that of a
// formal type (e.g. a `TypeDim` in an actual return type of a function vs. the
// counterpart `TypeDim` in its declared return type).
absl::Status TypecheckParametric(
    const TypeDim& actual_dim, const TypeDim& formal_dim,
    const TypeRefTypeAnnotation* formal_containing_type, const Span& error_span,
    DeduceCtx* ctx) {
  // Note that, in the case of a struct type with a nominal dim that is an
  // actual constant, as in
  //    `const FOO = u32:3;`
  //     fn f() -> S<FOO> { ... }`
  // the value of the `TypeDim` in that instance of `S` is a `ParametricSymbol`
  // referencing `FOO`, which there is no good way to evaluate. We can't tie
  // that back to the `NameDef`, due to `ParametricExpression` being like a
  // separate AST. So for now, we only attempt to check dims that have been
  // reduced to an `InterpValue`.
  absl::StatusOr<int64_t> actual_dim_value = actual_dim.GetAsInt64();
  absl::StatusOr<int64_t> formal_dim_value = formal_dim.GetAsInt64();
  if (actual_dim_value.ok() && formal_dim_value.ok() &&
      actual_dim_value != formal_dim_value) {
    return TypeInferenceErrorStatusForAnnotation(
        error_span, formal_containing_type,
        absl::StrFormat(
            "Parametric argument of the returned value does not match the "
            "function return type. Expected %s; got %s.",
            formal_dim.ToString(), actual_dim.ToString()),
        ctx->file_table());
  }
  return absl::OkStatus();
}

// Checks that the parametric values in the actual type of a struct match the
// formal type (e.g., the actual return value of a function vs. its declared
// return type).
absl::Status TypecheckStructParametrics(
    const StructType& actual_type, const TypeRefTypeAnnotation* formal_type,
    const Span& error_span, DeduceCtx* ctx) {
  const absl::flat_hash_map<std::string, TypeDim>& actual_nominal_dims =
      actual_type.nominal_type_dims_by_identifier();
  const std::vector<ExprOrType>& formal_parametrics =
      formal_type->parametrics();
  // There may be fewer actual parametrics specified than formal ones, in
  // which case the later formal ones are derived via expressions and not
  // relevant to this check. If there are too many actual ones, that is caught
  // elsewhere.
  const size_t dim_count_to_check =
      std::min(actual_nominal_dims.size(), formal_parametrics.size());
  for (int i = 0; i < dim_count_to_check; i++) {
    Expr* formal_type_expr = std::get<Expr*>(formal_parametrics[i]);
    XLS_ASSIGN_OR_RETURN(TypeDim formal_dim,
                         DimToConcreteUsize(formal_type_expr, ctx));
    const auto actual_dim_it = actual_nominal_dims.find(
        actual_type.nominal_type().parametric_bindings()[i]->identifier());
    if (actual_dim_it != actual_nominal_dims.end()) {
      const TypeDim& actual_dim = actual_dim_it->second;
      XLS_RETURN_IF_ERROR(TypecheckParametric(actual_dim, formal_dim,
                                              formal_type, error_span, ctx));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status TypecheckFunction(Function& f, DeduceCtx* ctx) {
  VLOG(2) << "Typechecking fn: " << f.identifier();
  XLS_VLOG_LINES(2, ctx->GetFnStackDebugString());

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
      VLOG(5) << "Typchecking fn; " << f.identifier()
              << "; found proc-level type info for proc: "
              << f.proc().value()->identifier();
      XLS_RETURN_IF_ERROR(ctx->PushTypeInfo(proc_ti.value()));
      derived_type_info = proc_ti.value();
    } else {
      VLOG(5) << "Typchecking fn; " << f.identifier()
              << "; creating proc-level type info for proc: "
              << f.proc().value()->identifier();
      derived_type_info = ctx->AddDerivedTypeInfo();
    }
  }

  VLOG(2) << "Typechecking fn: " << f.identifier() << "; starting params";
  XLS_VLOG_LINES(2, ctx->GetFnStackDebugString());

  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Type>> param_types,
                       TypecheckFunctionParams(f, ctx));

  // Second, typecheck the return type of the function.
  // Note: if there is no annotated return type, we assume nil.
  std::unique_ptr<Type> return_type;
  if (f.return_type() == nullptr) {
    return_type = TupleType::MakeUnit();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, ctx->DeduceAndResolve(f.return_type()));
    XLS_ASSIGN_OR_RETURN(
        return_type,
        UnwrapMetaType(std::move(return_type), f.return_type()->span(),
                       "function return type", ctx->file_table()));
  }

  // Add proc members to the environment before typechecking the fn body.
  if (f.proc().has_value()) {
    Proc* p = f.proc().value();
    for (auto* param : p->members()) {
      XLS_ASSIGN_OR_RETURN(auto type, ctx->DeduceAndResolve(param));
      ctx->type_info()->SetItem(param, *type);
      ctx->type_info()->SetItem(param->name_def(), *type);
    }
  }

  // Assert type consistency between the body and deduced return types.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> body_type,
                       ctx->DeduceAndResolve(f.body()));
  VLOG(3) << absl::StrFormat("Resolved return type: %s => %s",
                             return_type->ToString(), body_type->ToString());
  if (body_type->IsMeta()) {
    return TypeInferenceErrorStatus(f.body()->span(), body_type.get(),
                                    "Types cannot be returned from functions",
                                    ctx->file_table());
  }
  if (*return_type != *body_type) {
    VLOG(5) << "return type: " << return_type->ToString()
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

    auto type_mismatch_error_msg = absl::StrFormat(
        "Return type of function body for '%s' did not match "
        "the annotated return type.",
        f.identifier());
    if (body_type->IsUnit() && f.body()->trailing_semi()) {
      absl::StrAppend(&type_mismatch_error_msg,
                      " Did you intend to add a trailing semicolon to the last "
                      "expression in the function body? If the last expression "
                      "is terminated with a semicolon, it is discarded, and "
                      "the function implicitly returns ().");
    }

    return ctx->TypeMismatchError(f.body()->span(), f.body(), *body_type,
                                  f.return_type(), *return_type,
                                  type_mismatch_error_msg);
  }

  if (return_type->HasParametricDims()) {
    return TypeInferenceErrorStatus(
        f.return_type()->span(), return_type.get(),
        absl::StrFormat(
            "Parametric type being returned from function -- types must be "
            "fully resolved, please fully instantiate the type"),
        ctx->file_table());
  }
  if (return_type->IsStruct() &&
      !body_type->AsStruct().nominal_type_dims_by_identifier().empty()) {
    XLS_RETURN_IF_ERROR(TypecheckStructParametrics(
        body_type->AsStruct(),
        down_cast<const TypeRefTypeAnnotation*>(f.return_type()),
        f.body()->span(), ctx));
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
    Function& init = p->init();
    ScopedFnStackEntry init_entry(init, ctx, WithinProc::kYes);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, ctx->Deduce(init.body()));
    init_entry.Finish();
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

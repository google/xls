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

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

bool ContainsProc(const Type& type) {
  if (type.IsArray()) {
    const Type& element_type = type.AsArray().element_type();
    return element_type.IsProc() || ContainsProc(element_type);
  }
  if (type.IsTuple()) {
    const TupleType& tuple = type.AsTuple();
    for (const std::unique_ptr<Type>& member_type : tuple.members()) {
      if (member_type->IsProc() || ContainsProc(*member_type)) {
        return true;
      }
    }
  }
  return false;
}

// Warn folks if it's not following
// https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
void WarnOnInappropriateMemberName(std::string_view member_name,
                                   const Span& span, const Module& module,
                                   DeduceCtx* ctx) {
  if (!IsAcceptablySnakeCase(member_name) &&
      !module.annotations().contains(
          ModuleAnnotation::kAllowNonstandardMemberNaming)) {
    ctx->warnings()->Add(
        span, WarningKind::kMemberNaming,
        absl::StrFormat("Standard style is snake_case for struct member names; "
                        "got: `%s`",
                        member_name));
  }
}

}  // namespace

absl::Status TypecheckStructDefBase(const StructDefBase* node, DeduceCtx* ctx) {
  for (const ParametricBinding* parametric : node->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
    XLS_ASSIGN_OR_RETURN(parametric_binding_type,
                         UnwrapMetaType(std::move(parametric_binding_type),
                                        parametric->type_annotation()->span(),
                                        "parametric binding type annotation",
                                        ctx->file_table()));
    if (parametric->expr() != nullptr) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> expr_type,
                           ctx->Deduce(parametric->expr()));
      if (*expr_type != *parametric_binding_type) {
        return ctx->TypeMismatchError(
            node->span(), parametric->expr(), *expr_type,
            parametric->type_annotation(), *parametric_binding_type,
            "Annotated type of "
            "parametric value did not match inferred type.");
      }
    }
    ctx->type_info()->SetItem(parametric->name_def(), *parametric_binding_type);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<Type>>> DeduceStructDefBaseMembers(
    const StructDefBase* node, DeduceCtx* ctx,
    absl::AnyInvocable<absl::Status(DeduceCtx* ctx, const Span&, const Type&)>
        validator) {
  std::vector<std::unique_ptr<Type>> members;
  for (const auto* member : node->members()) {
    WarnOnInappropriateMemberName(member->name(), member->name_def()->span(),
                                  *node->owner(), ctx);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> concrete,
                         ctx->DeduceAndResolve(member->type()));
    XLS_ASSIGN_OR_RETURN(
        concrete, UnwrapMetaType(std::move(concrete), member->type()->span(),
                                 "struct member type", ctx->file_table()));
    XLS_RETURN_IF_ERROR(validator(ctx, member->name_def()->span(), *concrete));
    members.push_back(std::move(concrete));
  }
  return members;
}

absl::Status ValidateStructMember(DeduceCtx* ctx, const Span& span,
                                  const Type& type) {
  if (type.IsProc() || ContainsProc(type)) {
    return TypeInferenceErrorStatus(span, nullptr,
                                    "Structs cannot contain procs as members.",
                                    ctx->file_table());
  }
  return absl::OkStatus();
}

absl::Status ValidateProcMember(DeduceCtx*, const Span&, const Type&) {
  return absl::OkStatus();
}

}  // namespace xls::dslx

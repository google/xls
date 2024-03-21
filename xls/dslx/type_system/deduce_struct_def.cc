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

#include "xls/dslx/type_system/deduce_struct_def.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

// Forward declaration from sister implementation file as we are recursively
// bound to the central deduce-and-resolve routine in our node deduction
// routines.
//
// TODO(cdleary): 2024-01-16 We can break this circular resolution with a
// virtual function on DeduceCtx when we get things refactored nicely.
extern absl::StatusOr<std::unique_ptr<Type>> DeduceAndResolve(
    const AstNode* node, DeduceCtx* ctx);

// Warn folks if it's not following
// https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
static void WarnOnInappropriateMemberName(std::string_view member_name,
                                          const Span& span,
                                          const Module& module,
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

absl::StatusOr<std::unique_ptr<Type>> DeduceStructDef(const StructDef* node,
                                                      DeduceCtx* ctx) {
  for (const ParametricBinding* parametric : node->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
    XLS_ASSIGN_OR_RETURN(parametric_binding_type,
                         UnwrapMetaType(std::move(parametric_binding_type),
                                        parametric->type_annotation()->span(),
                                        "parametric binding type annotation"));
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

  std::vector<std::unique_ptr<Type>> members;
  for (const auto& [name_span, name, type] : node->members()) {
    WarnOnInappropriateMemberName(name, name_span, *node->owner(), ctx);

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> concrete,
                         DeduceAndResolve(type, ctx));
    XLS_ASSIGN_OR_RETURN(concrete,
                         UnwrapMetaType(std::move(concrete), type->span(),
                                        "struct member type"));
    members.push_back(std::move(concrete));
  }
  auto wrapped = std::make_unique<StructType>(std::move(members), *node);
  auto result = std::make_unique<MetaType>(std::move(wrapped));
  ctx->type_info()->SetItem(node->name_def(), *result);
  VLOG(5) << absl::StreamFormat("Deduced type for struct %s => %s",
                                node->ToString(), result->ToString());
  return result;
}

}  // namespace xls::dslx

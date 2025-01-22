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

#include "xls/dslx/type_system/deduce_ctx.h"

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

namespace {

absl::StatusOr<std::unique_ptr<Type>> ResolveViaEnv(
    const Type& type, const ParametricEnv& parametric_env) {
  VLOG(10) << "ResolveViaEnv; type: " << type.ToString()
           << " parametric_env: " << parametric_env.ToString();
  ParametricExpression::Env env;
  for (const auto& [k, v] : parametric_env.bindings()) {
    env[k] = v;
  }

  return type.MapSize([&](const TypeDim& dim) -> absl::StatusOr<TypeDim> {
    if (std::holds_alternative<TypeDim::OwnedParametric>(dim.value())) {
      const auto& parametric = std::get<TypeDim::OwnedParametric>(dim.value());
      return TypeDim(parametric->Evaluate(env));
    }
    return dim;
  });
}

}  // namespace

std::string FnStackEntry::ToReprString(const FileTable& file_table) const {
  if (f_ != nullptr) {
    return absl::StrFormat(
        "FnStackEntry{.f = `%s` @ %s, .name = \"%s\", .parametric_env = %s}",
        f_->identifier(), f_->span().ToString(file_table), name_,
        parametric_env_.ToString());
  }
  return absl::StrFormat("FnStackEntry{\"%s\", %s}", name_,
                         parametric_env_.ToString());
}

DeduceCtx::DeduceCtx(TypeInfo* type_info, Module* module,
                     DeduceFn deduce_function,
                     TypecheckFunctionFn typecheck_function,
                     TypecheckModuleFn typecheck_module,
                     TypecheckInvocationFn typecheck_invocation,
                     ImportData* import_data, WarningCollector* warnings,
                     DeduceCtx* parent)
    : type_info_(type_info),
      module_(module),
      deduce_function_(std::move(deduce_function)),
      typecheck_function_(std::move(typecheck_function)),
      typecheck_module_(std::move(typecheck_module)),
      typecheck_invocation_(std::move(typecheck_invocation)),
      import_data_(import_data),
      warnings_(warnings),
      parent_(parent) {}

absl::Status DeduceCtx::TypeMismatchError(
    Span mismatch_span, const AstNode* lhs_node, const Type& lhs,
    const AstNode* rhs_node, const Type& rhs, std::string message) {
  XLS_RET_CHECK(!type_mismatch_error_data_.has_value())
      << "internal error: nested type mismatch error";
  VLOG(10) << "TypeMismatchError; lhs_node: "
           << (lhs_node != nullptr ? lhs_node->ToString() : "null")
           << " rhs_node: "
           << (rhs_node != nullptr ? rhs_node->ToString() : "null");
  DeduceCtx* top = this;
  while (top->parent_ != nullptr) {
    top = top->parent_;
  }
  top->type_mismatch_error_data_ = TypeMismatchErrorData{
      mismatch_span,       lhs_node,          lhs.CloneToUnique(), rhs_node,
      rhs.CloneToUnique(), std::move(message)};
  return absl::InvalidArgumentError("DslxTypeMismatchError");
}

// Helper that converts the symbolic bindings to a parametric expression
// environment (for parametric evaluation).
ParametricExpression::Env ToParametricEnv(const ParametricEnv& parametric_env) {
  ParametricExpression::Env env;
  for (const ParametricEnvItem& binding : parametric_env.bindings()) {
    env[binding.identifier] = binding.value;
  }
  return env;
}

std::unique_ptr<DeduceCtx> DeduceCtx::MakeCtx(TypeInfo* new_type_info,
                                              Module* new_module) {
  return std::make_unique<DeduceCtx>(new_type_info, new_module,
                                     deduce_function_, typecheck_function_,
                                     typecheck_module_, typecheck_invocation_,
                                     import_data_, warnings_, /*parent=*/this);
}

std::unique_ptr<DeduceCtx> DeduceCtx::MakeCtxWithSameFnStack(
    TypeInfo* new_type_info, Module* new_module) {
  auto result = MakeCtx(new_type_info, new_module);
  result->fn_stack_ = fn_stack_;
  return result;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceCtx::Deduce(const AstNode* node) {
  XLS_RET_CHECK(!fn_stack().empty());
  XLS_RET_CHECK(deduce_function_ != nullptr);
  XLS_RET_CHECK_EQ(node->owner(), type_info()->module())
      << "node: `" << node->ToString() << "` is from module "
      << node->owner()->name()
      << " vs type info is for module: " << type_info()->module()->name();
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> result,
                       deduce_function_(node, this));

  if (dynamic_cast<const TypeAnnotation*>(node) != nullptr) {
    XLS_RET_CHECK(result->IsMeta())
        << node->ToString() << " @ "
        << SpanToString(node->GetSpan(), file_table());
  }

  return result;
}

TypeInfo* DeduceCtx::AddDerivedTypeInfo() {
  type_info_ = type_info_owner().New(module(), /*parent=*/type_info_).value();
  return type_info_;
}

absl::Status DeduceCtx::PushTypeInfo(TypeInfo* ti) {
  XLS_RET_CHECK_EQ(ti->parent(), type_info_);
  type_info_ = ti;
  return absl::OkStatus();
}

absl::Status DeduceCtx::PopDerivedTypeInfo(TypeInfo* expect_popped) {
  XLS_RET_CHECK(type_info_->parent() != nullptr);

  // Check that the type info we're popping is the one we expected to pop.
  XLS_RET_CHECK_EQ(type_info_, expect_popped);

  type_info_ = type_info_->parent();
  return absl::OkStatus();
}

void DeduceCtx::AddFnStackEntry(FnStackEntry entry) {
  fn_stack_.push_back(std::move(entry));
}

std::optional<FnStackEntry> DeduceCtx::PopFnStackEntry() {
  if (fn_stack_.empty()) {
    return std::nullopt;
  }
  FnStackEntry result = fn_stack_.back();
  fn_stack_.pop_back();
  return result;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceCtx::Resolve(
    const Type& type) const {
  XLS_RET_CHECK(!fn_stack().empty());
  const FnStackEntry& entry = fn_stack().back();
  const ParametricEnv& fn_parametric_env = entry.parametric_env();
  return ResolveViaEnv(type, fn_parametric_env);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceCtx::DeduceAndResolve(
    const AstNode* node) {
  VLOG(10) << "DeduceAndResolve; node: `" << node->ToString() << "`";
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> deduced, Deduce(node));
  return Resolve(*deduced);
}

std::string DeduceCtx::GetFnStackDebugString() const {
  std::stringstream ss;
  ss << absl::StreamFormat("== Function Stack for DeduceCtx %p\n", this);
  for (const FnStackEntry& fse : fn_stack_) {
    ss << "  " << fse.ToReprString(file_table()) << "\n";
  }
  return ss.str();
}

}  // namespace xls::dslx

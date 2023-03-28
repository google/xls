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

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "xls/common/string_to_int.h"

namespace xls::dslx {

std::string FnStackEntry::ToReprString() const {
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
    Span mismatch_span, const AstNode* lhs_node, const ConcreteType& lhs,
    const AstNode* rhs_node, const ConcreteType& rhs, std::string message) {
  XLS_RET_CHECK(!type_mismatch_error_data_.has_value())
      << "internal error: nested type mismatch error";
  DeduceCtx* top = this;
  while (top->parent_ != nullptr) {
    top = top->parent_;
  }
  top->type_mismatch_error_data_ = TypeMismatchErrorData{
      mismatch_span,       lhs_node, lhs.CloneToUnique(), rhs_node,
      rhs.CloneToUnique(), message};
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

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceCtx::Deduce(
    const AstNode* node) {
  XLS_RET_CHECK(deduce_function_ != nullptr);
  XLS_RET_CHECK_EQ(node->owner(), type_info()->module())
      << "node: `" << node->ToString() << "` from module "
      << node->owner()->name()
      << " vs type info module: " << type_info()->module()->name();
  return deduce_function_(node, this);
}

void DeduceCtx::AddDerivedTypeInfo() {
  type_info_ = type_info_owner().New(module(), /*parent=*/type_info_).value();
}

absl::Status DeduceCtx::PushTypeInfo(TypeInfo* ti) {
  XLS_RET_CHECK_EQ(ti->parent(), type_info_);
  type_info_ = ti;
  return absl::OkStatus();
}

absl::Status DeduceCtx::PopDerivedTypeInfo() {
  XLS_RET_CHECK(type_info_->parent() != nullptr);
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

}  // namespace xls::dslx

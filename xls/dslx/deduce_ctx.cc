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

#include "xls/dslx/deduce_ctx.h"

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xls/common/string_to_int.h"

namespace xls::dslx {

std::string FnStackEntry::ToReprString() const {
  return absl::StrFormat("FnStackEntry{\"%s\", %s}", name,
                         symbolic_bindings.ToString());
}

bool FnStackEntry::Matches(const Function* f) const {
  return f->identifier() == name && f->owner() == module;
}

DeduceCtx::DeduceCtx(const std::shared_ptr<TypeInfo>& type_info,
                     const std::shared_ptr<Module>& module,
                     DeduceFn deduce_function,
                     TypecheckFunctionFn typecheck_function,
                     TypecheckFn typecheck_module,
                     absl::Span<std::string const> additional_search_paths,
                     ImportCache* import_cache)
    : type_info_(type_info),
      module_(module),
      deduce_function_(std::move(XLS_DIE_IF_NULL(deduce_function))),
      typecheck_function_(std::move(typecheck_function)),
      typecheck_module_(std::move(typecheck_module)),
      additional_search_paths_(additional_search_paths.begin(),
                               additional_search_paths.end()),
      import_cache_(import_cache) {}

// Helper that converts the symbolic bindings to a parametric expression
// environment (for parametric evaluation).
ParametricExpression::Env ToParametricEnv(
    const SymbolicBindings& symbolic_bindings) {
  ParametricExpression::Env env;
  for (const SymbolicBinding& binding : symbolic_bindings.bindings()) {
    env[binding.identifier] = binding.value;
  }
  return env;
}

absl::Status TypeInferenceErrorStatus(const Span& span,
                                      const ConcreteType* type,
                                      absl::string_view message) {
  std::string type_str = "<>";
  if (type != nullptr) {
    type_str = type->ToString();
  }
  return absl::InternalError(absl::StrFormat(
      "TypeInferenceError: %s %s %s", span.ToString(), type_str, message));
}

absl::Status XlsTypeErrorStatus(const Span& span, const ConcreteType& lhs,
                                const ConcreteType& rhs,
                                absl::string_view message) {
  return absl::InternalError(absl::StrFormat("XlsTypeError: %s %s %s %s",
                                             span.ToString(), lhs.ToString(),
                                             rhs.ToString(), message));
}

NodeAndUser ParseTypeMissingErrorMessage(absl::string_view s) {
  (void)absl::ConsumePrefix(&s, "TypeMissingError: ");
  std::vector<absl::string_view> pieces =
      absl::StrSplit(s, absl::MaxSplits(' ', 2));
  XLS_CHECK_EQ(pieces.size(), 3);
  int64 node = 0;
  if (pieces[0] != "(nil)") {
    node = StrTo64Base(pieces[0], 16).value();
  }
  int64 user = 0;
  if (pieces[1] != "(nil)") {
    user = StrTo64Base(pieces[1], 16).value();
  }
  return {absl::bit_cast<AstNode*>(node), absl::bit_cast<AstNode*>(user)};
}

absl::Status TypeMissingErrorStatus(AstNode* node, AstNode* user) {
  return absl::InternalError(absl::StrFormat(
      "TypeMissingError: %p %p AST node is missing a corresponding type: %s",
      node, user, node->ToString()));
}

bool IsTypeMissingErrorStatus(const absl::Status& status) {
  return !status.ok() &&
         absl::StartsWith(status.message(), "TypeMissingError:");
}

absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         absl::string_view message) {
  return absl::InternalError(absl::StrFormat("ArgCountMismatchError: %s %s",
                                             span.ToString(), message));
}

}  // namespace xls::dslx

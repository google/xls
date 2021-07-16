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
  return absl::StrFormat("FnStackEntry{\"%s\", %s}", name_,
                         symbolic_bindings_.ToString());
}

bool FnStackEntry::Matches(const Function* f) const { return f == function_; }

DeduceCtx::DeduceCtx(
    TypeInfo* type_info, Module* module, DeduceFn deduce_function,
    TypecheckFunctionFn typecheck_function, TypecheckFn typecheck_module,
    absl::Span<const std::filesystem::path> additional_search_paths,
    ImportData* import_data)
    : type_info_(type_info),
      module_(module),
      deduce_function_(std::move(XLS_DIE_IF_NULL(deduce_function))),
      typecheck_function_(std::move(typecheck_function)),
      typecheck_module_(std::move(typecheck_module)),
      additional_search_paths_(additional_search_paths.begin(),
                               additional_search_paths.end()),
      import_data_(import_data) {}

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
  return absl::InvalidArgumentError(absl::StrFormat(
      "TypeInferenceError: %s %s %s", span.ToString(), type_str, message));
}

absl::Status XlsTypeErrorStatus(const Span& span, const ConcreteType& lhs,
                                const ConcreteType& rhs,
                                absl::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrFormat("XlsTypeError: %s %s vs %s: %s", span.ToString(),
                      lhs.ToErrorString(), rhs.ToErrorString(), message));
}

NodeAndUser ParseTypeMissingErrorMessage(absl::string_view s) {
  (void)absl::ConsumePrefix(&s, "TypeMissingError: ");
  std::vector<absl::string_view> pieces =
      absl::StrSplit(s, absl::MaxSplits(' ', 3));
  XLS_CHECK_EQ(pieces.size(), 4);
  int64_t node = 0;
  if (pieces[1] != "(nil)") {
    node = StrTo64Base(pieces[1], 16).value();
  }
  int64_t user = 0;
  if (pieces[2] != "(nil)") {
    user = StrTo64Base(pieces[2], 16).value();
  }
  return {absl::bit_cast<AstNode*>(node), absl::bit_cast<AstNode*>(user)};
}

absl::Status TypeMissingErrorStatus(AstNode* node, AstNode* user) {
  std::string span_string;
  if (user != nullptr) {
    span_string = SpanToString(user->GetSpan()) + " ";
  } else if (node != nullptr) {
    span_string = SpanToString(node->GetSpan()) + " ";
  }
  return absl::InternalError(
      absl::StrFormat("TypeMissingError: %s%p %p internal error: AST node is "
                      "missing a corresponding type: %s (%s) defined @ %s",
                      span_string, node, user, node->ToString(),
                      node->GetNodeTypeName(), SpanToString(node->GetSpan())));
}

bool IsTypeMissingErrorStatus(const absl::Status& status) {
  return !status.ok() &&
         absl::StartsWith(status.message(), "TypeMissingError:");
}

absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "ArgCountMismatchError: %s %s", span.ToString(), message));
}

absl::Status InvalidIdentifierErrorStatus(const Span& span,
                                          absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "InvalidIdentifierError: %s %s", span.ToString(), message));
}

absl::flat_hash_map<std::string, InterpValue> MakeConstexprEnv(
    Expr* node, const SymbolicBindings& symbolic_bindings,
    TypeInfo* type_info) {
  XLS_CHECK_EQ(node->owner(), type_info->module())
      << "expr `" << node->ToString()
      << "` from module: " << node->owner()->name()
      << " vs type info module: " << type_info->module()->name();
  XLS_VLOG(5) << "Creating constexpr environment for node: "
              << node->ToString();
  absl::flat_hash_map<std::string, InterpValue> env;
  absl::flat_hash_map<std::string, InterpValue> values;

  for (auto [id, value] : symbolic_bindings.ToMap()) {
    env.insert({id, value});
  }

  // Collect all the freevars that are constexpr.
  //
  // TODO(https://github.com/google/xls/issues/333): 2020-03-11 We'll want the
  // expression to also be able to constexpr evaluate local non-integral values,
  // like constant tuple definitions and such. We'll need to extend the
  // constexpr ability to full InterpValues to accomplish this.
  //
  // E.g. fn main(x: u32) -> ... { const B = u32:20; x[:B] }
  FreeVariables freevars = node->GetFreeVariables();
  XLS_VLOG(5) << "freevars for " << node->ToString() << ": "
              << freevars.GetFreeVariableCount();
  for (ConstRef* const_ref : freevars.GetConstRefs()) {
    ConstantDef* constant_def = const_ref->GetConstantDef();
    XLS_VLOG(5) << "analyzing constant reference: " << const_ref->ToString()
                << " def: " << constant_def->ToString();
    absl::optional<InterpValue> value =
        type_info->GetConstExpr(constant_def->value());
    if (!value.has_value()) {
      // Could be a tuple or similar, not part of the (currently integral-only)
      // constexpr environment.
      XLS_VLOG(5) << "Could not find constexpr value for constant def: `"
                  << constant_def->ToString() << "` @ " << constant_def->value()
                  << " in " << type_info;
      continue;
    }

    XLS_VLOG(5) << "freevar env record: " << const_ref->identifier() << " => "
                << value->ToString();
    env.insert({const_ref->identifier(), *value});
  }

  return env;
}

}  // namespace xls::dslx

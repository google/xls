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

namespace xls::dslx {

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

}  // namespace xls::dslx

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
#include "xls/dslx/errors.h"

namespace xls::dslx {

absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         std::string_view message) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "ArgCountMismatchError: %s %s", span.ToString(), message));
}

absl::Status FailureErrorStatus(const Span& span, std::string_view message) {
  return absl::InternalError(absl::StrFormat(
      "FailureError: %s The program being interpreted failed! %s",
      span.ToString(), message));
}

absl::Status InvalidIdentifierErrorStatus(const Span& span,
                                          std::string_view message) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "InvalidIdentifierError: %s %s", span.ToString(), message));
}

absl::Status TypeInferenceErrorStatus(const Span& span,
                                      const ConcreteType* type,
                                      std::string_view message) {
  std::string type_str = "<>";
  if (type != nullptr) {
    type_str = type->ToString();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "TypeInferenceError: %s %s %s", span.ToString(), type_str, message));
}

absl::Status TypeMissingErrorStatus(const AstNode* node, const AstNode* user) {
  std::string span_string;
  if (user != nullptr) {
    span_string = SpanToString(user->GetSpan()) + " ";
  } else if (node != nullptr) {
    span_string = SpanToString(node->GetSpan()) + " ";
  }
  return absl::InternalError(
      absl::StrFormat("TypeMissingError: %s%p %p internal error: AST node is "
                      "missing a corresponding type: %s (%s) defined @ %s. "
                      "This may be due to recursion, which is not supported.",
                      span_string, node, user, node->ToString(),
                      node->GetNodeTypeName(), SpanToString(node->GetSpan())));
}

absl::Status XlsTypeErrorStatus(const Span& span, const ConcreteType& lhs,
                                const ConcreteType& rhs,
                                std::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrFormat("XlsTypeError: %s %s vs %s: %s", span.ToString(),
                      lhs.ToErrorString(), rhs.ToErrorString(), message));
}

}  // namespace xls::dslx

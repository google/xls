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

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_record.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

template <typename TypeOrAnnotation>
absl::Status TypeInferenceErrorStatusInternal(
    const Span& span, const TypeOrAnnotation* type_or_annotation,
    std::string_view message, const FileTable& file_table) {
  std::string type_str;
  if (type_or_annotation != nullptr) {
    type_str = type_or_annotation->ToString() + " ";
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("TypeInferenceError: %s %s%s", span.ToString(file_table),
                      type_str, message));
}

}  // namespace

absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         std::string_view message,
                                         const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "ArgCountMismatchError: %s %s", span.ToString(file_table), message));
}

absl::Status FailureErrorStatus(const Span& span, std::string_view message,
                                const FileTable& file_table) {
  return absl::InternalError(absl::StrFormat(
      "FailureError: %s The program being interpreted failed!%s%s",
      span.ToString(file_table),
      message.empty() || message[0] == '\n' ? "" : " ", message));
}

absl::Status ProofErrorStatus(const Span& span, std::string_view message,
                              const FileTable& file_table) {
  return absl::InternalError(absl::StrFormat(
      "ProofError: %s Failed to prove the property!%s%s",
      span.ToString(file_table),
      message.empty() || message[0] == '\n' ? "" : " ", message));
}

absl::Status InvalidIdentifierErrorStatus(const Span& span,
                                          std::string_view message,
                                          const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "InvalidIdentifierError: %s %s", span.ToString(file_table), message));
}

absl::Status TypeInferenceErrorStatus(const Span& span, const Type* type,
                                      std::string_view message,
                                      const FileTable& file_table) {
  return TypeInferenceErrorStatusInternal(span, type, message, file_table);
}

absl::Status TypeInferenceErrorStatusForAnnotation(
    const Span& span, const TypeAnnotation* type_annotation,
    std::string_view message, const FileTable& file_table) {
  return TypeInferenceErrorStatusInternal(span, type_annotation, message,
                                          file_table);
}

absl::Status SignednessMismatchErrorStatus(const TypeAnnotation* annotation1,
                                           const TypeAnnotation* annotation2,
                                           const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::Substitute(
      "TypeInferenceError: signed vs. unsigned mismatch: $0 at $1 vs. $2 at $3",
      annotation1->ToString(), annotation1->span().ToString(file_table),
      annotation2->ToString(), annotation2->span().ToString(file_table)));
}

absl::Status SignednessMismatchErrorStatus(const Type& type1, const Type& type2,
                                           const Span& span1, const Span& span2,
                                           const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::Substitute(
      "TypeInferenceError: signed vs. unsigned mismatch: $0 at $1 vs. $2 at $3",
      type1.ToString(), span1.ToString(file_table), type2.ToString(),
      span2.ToString(file_table)));
}

absl::Status BitCountMismatchErrorStatus(const TypeAnnotation* annotation1,
                                         const TypeAnnotation* annotation2,
                                         const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::Substitute(
      "TypeInferenceError: size mismatch: $0 at $1 vs. $2 at $3",
      annotation1->ToString(), annotation1->span().ToString(file_table),
      annotation2->ToString(), annotation2->span().ToString(file_table)));
}

absl::Status BitCountMismatchErrorStatus(const Type& type1, const Type& type2,
                                         const Span& span1, const Span& span2,
                                         const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::Substitute(
      "TypeInferenceError: size mismatch: $0 at $1 vs. $2 at $3",
      type1.ToString(), span1.ToString(file_table), type2.ToString(),
      span2.ToString(file_table)));
}

absl::Status TypeMismatchErrorStatus(const TypeAnnotation* annotation1,
                                     const TypeAnnotation* annotation2,
                                     const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::Substitute(
      "TypeInferenceError: type mismatch: $0 at $1 vs. $2 at $3",
      annotation1->ToString(), annotation1->span().ToString(file_table),
      annotation2->ToString(), annotation2->span().ToString(file_table)));
}

absl::Status TypeMismatchErrorStatus(const Type& type1, const Type& type2,
                                     const Span& span1, const Span& span2,
                                     const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::Substitute(
      "TypeInferenceError: type mismatch: $0 at $1 vs. $2 at $3",
      type1.ToString(), span1.ToString(file_table), type2.ToString(),
      span2.ToString(file_table)));
}

absl::Status TypeMissingErrorStatus(const AstNode& node, const AstNode* user,
                                    const FileTable& file_table) {
  std::string span_string = user == nullptr
                                ? SpanToString(node.GetSpan(), file_table)
                                : SpanToString(user->GetSpan(), file_table);

  return absl::InternalError(absl::StrFormat(
      "TypeMissingError: %s node: %p user: %p internal error: AST node is "
      "missing a corresponding type: %s %p (kind: %s) defined @ %s. "
      "This may be due to recursion, which is not supported.",
      span_string, &node, user, node.ToString(), &node, node.GetNodeTypeName(),
      SpanToString(node.GetSpan(), file_table)));
}

absl::Status RecursiveImportErrorStatus(const Span& nested_import,
                                        const Span& earlier_import,
                                        absl::Span<const ImportRecord> cycle,
                                        const FileTable& file_table) {
  std::string cycle_str = absl::StrJoin(
      cycle, " imports\n    ", [](std::string* out, const ImportRecord& r) {
        absl::StrAppend(out, std::string{r.imported});
      });

  // Display the entry translation unit specially.
  std::string earlier_import_str;
  if (earlier_import.empty()) {
    earlier_import_str =
        absl::StrCat(earlier_import.GetFilename(file_table), " (entry)");
  } else {
    earlier_import_str = earlier_import.ToString(file_table);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("RecursiveImportError: %s import cycle detected, import "
                      "cycles are not allowed:\n  previous import @ %s\n  "
                      "subsequent (nested) import @ %s\n  cycle:\n    %s",
                      nested_import.ToString(file_table), earlier_import_str,
                      nested_import.ToString(file_table), cycle_str));
}

absl::Status CheckedCastErrorStatus(const Span& span,
                                    const InterpValue& from_value,
                                    const Type* to_type,
                                    const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "CheckedCastError: %s unable to cast value %s to type %s without "
      "truncation.",
      span.ToString(file_table), from_value.ToString(), to_type->ToString()));
}

absl::Status NotConstantErrorStatus(const Span& span, const Expr* expr,
                                    const FileTable& file_table) {
  return absl::InvalidArgumentError(
      absl::StrFormat("NotConstantError: %s expr `%s` is not a constexpr.",
                      span.ToString(file_table), expr->ToString()));
}

absl::Status RedefinedNameErrorStatus(const Span& span, const AstNode* expr,
                                      std::string_view name,
                                      const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "RedefinedNameErrorStatus: %s duplicated name `%s` in `%s` is found.",
      span.ToString(file_table), name, expr->ToString()));
}

absl::Status UndefinedNameErrorStatus(const Span& span, const AstNode* expr,
                                      std::string_view name,
                                      const FileTable& file_table) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "UndefinedNameErrorStatus: %s name `%s` in `%s` is undefined.",
      span.ToString(file_table), name, expr->ToString()));
}

}  // namespace xls::dslx

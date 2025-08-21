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
#ifndef XLS_DSLX_ERRORS_H_
#define XLS_DSLX_ERRORS_H_

#include <string>
#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_record.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"

// Specialized error types that can be encountered during DSLX evaluation.

namespace xls::dslx {

// Returned when an incorrect number of args is given to a function
// invocation [in a design].
absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         std::string_view message,
                                         const FileTable& file_table);

// Returned when interpretation of a design fails.
absl::Status FailureErrorStatus(const Span& span, std::string_view message,
                                const FileTable& file_table);

// Variant of `FailureErrorStatus` for  when interpretation of a design
// fails on assertion.
absl::Status FailureErrorStatusForAssertion(const Span& span,
                                            std::string_view label,
                                            std::string_view message,
                                            const FileTable& file_table);

// Returned when proof of a property fails.
absl::Status ProofErrorStatus(const Span& span, std::string_view message,
                              const FileTable& file_table);

// Returned when an invalid identifier (invalid at some position in the
// compilation chain, DSLX, IR, or Verilog) is encountered.
absl::Status InvalidIdentifierErrorStatus(const Span& span,
                                          std::string_view message,
                                          const FileTable& file_table);

// To be raised when an error occurs during type inference.
absl::Status TypeInferenceErrorStatus(const Span& span, const Type* type,
                                      std::string_view message,
                                      const FileTable& file_table);

// Variant of `TypeInferenceErrorStatus` for when there is only a
// `type_annotation` available, or its readability is better than that of the
// corresponding `Type` (e.g. due to erasure of parametric bindings).
absl::Status TypeInferenceErrorStatusForAnnotation(
    const Span& span, const TypeAnnotation* type_annotation,
    std::string_view message, const FileTable& file_table);

// Variant of `TypeInferenceErrorStatus` for when the signedness of one type
// annotation is expected to match another and doesn't.
absl::Status SignednessMismatchErrorStatus(const TypeAnnotation* annotation1,
                                           const TypeAnnotation* annotation2,
                                           const FileTable& file_table);

// Variant of `SignednessMismatchErrorStatus` for when concrete types rather
// than annotations are available.
absl::Status SignednessMismatchErrorStatus(const Type& type1, const Type& type2,
                                           const Span& span1, const Span& span2,
                                           const FileTable& file_table);

// Variant of `TypeInferenceErrorStatus` for when the bit count of one type
// annotation is expected to match another and doesn't.
absl::Status BitCountMismatchErrorStatus(const TypeAnnotation* annotation1,
                                         const TypeAnnotation* annotation2,
                                         const FileTable& file_table);

// Variant of `BitCountMismatchErrorStatus` for when concrete types rather than
// annotations are available.
absl::Status BitCountMismatchErrorStatus(const Type& type1, const Type& type2,
                                         const Span& span1, const Span& span2,
                                         const FileTable& file_table);

// Variant of `TypeInferenceErrorStatus` for when the overall type of one type
// annotation is expected to match another and doesn't (e.g. one is bits and the
// other is a tuple).
absl::Status TypeMismatchErrorStatus(const TypeAnnotation* annotation1,
                                     const TypeAnnotation* annotation2,
                                     const FileTable& file_table);

// Variant of `TypeInferenceErrorStatus` for when concrete types instead of
// annotations are available.
absl::Status TypeMismatchErrorStatus(const Type& type1, const Type& type2,
                                     const Span& span1, const Span& span2,
                                     const FileTable& file_table);

// Creates a TypeMissingError status value referencing the given node (which has
// its type missing) and user (which found that its type was missing).
absl::Status TypeMissingErrorStatus(const AstNode& node, const AstNode* user,
                                    const FileTable& file_table);

// To be raised when a recursive import is detected.
absl::Status RecursiveImportErrorStatus(const Span& nested_import,
                                        const Span& earlier_import,
                                        absl::Span<const ImportRecord> cycle,
                                        const FileTable& file_table);

// To be raised when a checked_cast is unable to cast without truncation.
absl::Status CheckedCastErrorStatus(const Span& span,
                                    const InterpValue& from_value,
                                    const Type* to_type,
                                    const FileTable& file_table);

// To be raised when an expression is expected to be a constant while it is
// actually not.
absl::Status NotConstantErrorStatus(const Span& span, const Expr* expr,
                                    const FileTable& file_table);

// To be raised when using a duplicated name where not allowed.
absl::Status RedefinedNameErrorStatus(const Span& span, const AstNode* expr,
                                      std::string_view name,
                                      const FileTable& file_table);

// To be raised when using a name that is not defined.
absl::Status UndefinedNameErrorStatus(const Span& span, const AstNode* expr,
                                      std::string_view name,
                                      const FileTable& file_table);

// To be raised when the start is greater than the end in a range.
absl::Status RangeStartGreaterThanEndErrorStatus(const Span& span,
                                                 const Range* range,
                                                 const InterpValue& start,
                                                 const InterpValue& end,
                                                 const FileTable& file_table);

// To be raised when the size of a range is greater than the maximum value
// representable by u32.
absl::Status RangeTooLargeErrorStatus(const Span& span, const Range* range,
                                      const InterpValue& size,
                                      const FileTable& file_table);

// Extracts the assertion label from an error message.
// Intended for use with errors returned by `FailureErrorStatusForAssertion()`.
std::optional<std::string> GetAssertionLabelFromError(
    const absl::Status& status);

}  // namespace xls::dslx

#endif  // XLS_DSLX_ERRORS_H_

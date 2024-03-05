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

#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_record.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"

// Specialized error types that can be encountered during DSLX evaluation.

namespace xls::dslx {

// Returned when an incorrect number of args is given to a function
// invocation [in a design].
absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         std::string_view message);

// Returned when interpretation of a design fails.
absl::Status FailureErrorStatus(const Span& span, std::string_view message);

// Returned when an invalid identifier (invalid at some position in the
// compilation chain, DSLX, IR, or Verilog) is encountered.
absl::Status InvalidIdentifierErrorStatus(const Span& span,
                                          std::string_view message);

// To be raised when an error occurs during type inference.
absl::Status TypeInferenceErrorStatus(const Span& span,
                                      const ConcreteType* type,
                                      std::string_view message);

// Creates a TypeMissingError status value referencing the given node (which has
// its type missing) and user (which found that its type was missing).
absl::Status TypeMissingErrorStatus(const AstNode& node, const AstNode* user);

// To be raised when a recursive import is detected.
absl::Status RecursiveImportErrorStatus(const Span& nested_import,
                                        const Span& earlier_import,
                                        absl::Span<const ImportRecord> cycle);

// To be raised when a checked_cast is unable to cast without truncation.
absl::Status CheckedCastErrorStatus(const Span& span,
                                    const InterpValue& from_value,
                                    const ConcreteType* to_type);

}  // namespace xls::dslx

#endif  // XLS_DSLX_ERRORS_H_

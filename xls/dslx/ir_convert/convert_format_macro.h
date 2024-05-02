// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_IR_CONVERT_CONVERT_FORMAT_MACRO_H_
#define XLS_DSLX_IR_CONVERT_CONVERT_FORMAT_MACRO_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/function_builder.h"

namespace xls::dslx {

// Converts a trace_fmt! macro AST node to IR. This is involved enough that we
// break it out -- it needs to deal with communicating types to the IR tracing
// layer by flattening the structural data that we have at the DSLX level into
// appropriate format strings.
//
// Args:
//  node: The format macro DSLX AST node being converted to an IR trace
//    operation.
//  entry_token: The token that precedes the trace operation.
//  control_predicate: The predicate that says whether the trace operation
//    should execute.
//  arg_vals: Evaluated IR values for the argument expressions (to the node).
//  current_type_info: Type information that contains the node.
//  function_builder: Function builder that we should place the IR trace
//    operation on.
//
// Returns the trace (IR operation) token.
absl::StatusOr<BValue> ConvertFormatMacro(const FormatMacro& node,
                                          const BValue& entry_token,
                                          const BValue& control_predicate,
                                          absl::Span<const BValue> arg_vals,
                                          const TypeInfo& current_type_info,
                                          BuilderBase& function_builder);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_CONVERT_FORMAT_MACRO_H_

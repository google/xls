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

// Library to assist the lowering of IR Nodes down to Verilog.

#ifndef XLS_CODEGEN_NODE_EXPRESSIONS_H_
#define XLS_CODEGEN_NODE_EXPRESSIONS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/vast.h"
#include "xls/ir/node.h"

namespace xls {
namespace verilog {

// Returns true if the given operand number of the given node must be a
// reference to a named variable/net rather than an arbitrary expression. As a
// illustrative and motivating example for this functionality, consider the
// following IR:
//
//  a: bits[8] = ...
//  b: bits[8] = ...
//  a_plus_b: bits[8] = add(a, b)
//  slice: bits[4] = bit_slice(a_plus_b, start=0, width=4)
//
// A naive Verilog lowering might yield:
//
//  slice = (a+b)[3:0];
//
// However, this is invalid Verilog. The slice operand must be a reference to a
// named variable/net like so:
//
//  a_plus_b = a + b;
//  slice = a_plus_b[3:0];
//
// To support this Verilog-imposed constraint OperandMustBeNamedReference would
// return true for a kBitSlice nodes (for operand_no value of zero, it's only
// operand) and the generator would be responsible for declaring and assigning
// an intermediate value.
bool OperandMustBeNamedReference(Node* node, int64_t operand_no);

// Returns a VAST expression which computes the given node with the given
// inputs. If the node cannot be expressed as an expression (e.g., CountedFor),
// then an error is returned.
absl::StatusOr<Expression*> NodeToExpression(
    Node* node, absl::Span<Expression* const> inputs, VerilogFile* file,
    const CodegenOptions& options);

// Returns true if the expression for the given Node should be inlined into its
// uses. Expressions are inlined if they are unary and have very terse
// representations in verilog (e.g., bit-slice and not). This is purely to
// reduce the verbosity of the Verilog and not required for correctness.
bool ShouldInlineExpressionIntoMultipleUses(Node* node);

// Return an expression which indexes into an array with the given indices as
// described by the given ArrayIndex operation.
absl::StatusOr<IndexableExpression*> ArrayIndexExpression(
    IndexableExpression* array, absl::Span<Expression* const> indices,
    ArrayIndex* array_index, const CodegenOptions& options);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_NODE_EXPRESSIONS_H_

// Copyright 2020 Google LLC
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

#ifndef XLS_IR_IR_INTERPRETER_H_
#define XLS_IR_IR_INTERPRETER_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_interpreter_stats.h"

namespace xls {
namespace ir_interpreter {

// Executes the function and returns the calculated output(s) (i.e. perform
// behavioral simulation) by interpreting each node in the IR graph.
absl::StatusOr<Value> RunKwargs(
    Function* function, const absl::flat_hash_map<std::string, Value>& args,
    InterpreterStats* stats = nullptr);

// As above, but with positional arguments.
absl::StatusOr<Value> Run(Function* function, absl::Span<const Value> args,
                          InterpreterStats* stats = nullptr);

// Evaluates the given node. All operands of the nodes must be literal which are
// used in the evaluation.
absl::StatusOr<Value> EvaluateNodeWithLiteralOperands(Node* node);

// Evaluates the given nodes using the given operand values.
absl::StatusOr<Value> EvaluateNode(
    Node* node, absl::Span<const Value* const> operand_values);

}  // namespace ir_interpreter
}  // namespace xls

#endif  // XLS_IR_IR_INTERPRETER_H_

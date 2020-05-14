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

#ifndef THIRD_PARTY_XLS_IR_IR_INTERPRETER_H_
#define THIRD_PARTY_XLS_IR_IR_INTERPRETER_H_

#include "absl/types/span.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_interpreter_stats.h"

namespace xls {
namespace ir_interpreter {

// Executes the function and returns the calculated output(s) (i.e. perform
// behavioral simulation) by interpreting each node in the IR graph.
xabsl::StatusOr<Value> RunKwargs(
    Function* function, const absl::flat_hash_map<std::string, Value>& args,
    InterpreterStats* stats = nullptr);

// As above, but with positional arguments.
xabsl::StatusOr<Value> Run(Function* function, absl::Span<const Value> args,
                           InterpreterStats* stats = nullptr);

// Evaluates the given node. All operands of the nodes must be literal which are
// used in the evaluation.
xabsl::StatusOr<Value> EvaluateNodeWithLiteralOperands(Node* node);

// Evaluates the given nodes using the given operand values.
xabsl::StatusOr<Value> EvaluateNode(
    Node* node, absl::Span<const Value* const> operand_values);

}  // namespace ir_interpreter
}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_IR_INTERPRETER_H_

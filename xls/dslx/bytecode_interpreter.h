// Copyright 2021 The XLS Authors
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
#ifndef XLS_DSLX_BYTECODE_INTERPRETER_H_
#define XLS_DSLX_BYTECODE_INTERPRETER_H_

#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/bytecode_emitter.h"

namespace xls::dslx {

// Bytecode interpreter for DSLX. Accepts sequence of "bytecode" "instructions"
// and a set of initial environmental bindings (key/value pairs) and executes
// until end result.
//
// TODO(rspringer): Finish adding the rest of the opcodes, etc.
class BytecodeInterpreter {
 public:
  static absl::StatusOr<InterpValue> Interpret(
      const std::vector<Bytecode>& bytecode, std::vector<InterpValue>* slots) {
    BytecodeInterpreter interp(bytecode, slots);
    XLS_RETURN_IF_ERROR(interp.Run());
    return interp.stack_.back();
  }

  const std::vector<InterpValue>& stack() { return stack_; }

 private:
  BytecodeInterpreter(const std::vector<Bytecode>& bytecode,
                      std::vector<InterpValue>* slots)
      : bytecode_(bytecode), slots_(slots) {}

  absl::Status Run();

  absl::StatusOr<int64_t> EvalInstruction(int64_t pc, const Bytecode& bytecode);

  absl::Status EvalAdd(const Bytecode& bytecode);
  absl::Status EvalCall(const Bytecode& bytecode);
  absl::Status EvalCreateTuple(const Bytecode& bytecode);
  absl::Status EvalEq(const Bytecode& bytecode);
  absl::Status EvalExpandTuple(const Bytecode& bytecode);
  absl::Status EvalLoad(const Bytecode& bytecode);
  absl::Status EvalLiteral(const Bytecode& bytecode);
  absl::Status EvalStore(const Bytecode& bytecode);
  std::optional<int64_t> EvalJumpRelIf(int64_t pc, const Bytecode& bytecode);

  absl::Status RunBuiltinFn(const Bytecode& bytecode, Builtin builtin);
  absl::Status RunBuiltinAssertEq(const Bytecode& bytecode);

  InterpValue Pop() {
    InterpValue v = std::move(stack_.back());
    stack_.pop_back();
    return v;
  }

  const std::vector<Bytecode>& bytecode_;
  std::vector<InterpValue> stack_;
  std::vector<InterpValue>* slots_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_INTERPRETER_H_

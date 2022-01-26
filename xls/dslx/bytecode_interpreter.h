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
#include "xls/dslx/bytecode.h"
#include "xls/dslx/import_data.h"

namespace xls::dslx {

// Bytecode interpreter for DSLX. Accepts sequence of "bytecode" "instructions"
// and a set of initial environmental bindings (key/value pairs) and executes
// until end result.
//
// TODO(rspringer): Finish adding the rest of the opcodes, etc.
class BytecodeInterpreter {
 public:
  static absl::StatusOr<InterpValue> Interpret(
      ImportData* import_data, BytecodeFunction* bf,
      const std::vector<InterpValue>& params) {
    BytecodeInterpreter interp(import_data, bf);
    XLS_RETURN_IF_ERROR(interp.Run(params));
    return interp.stack_.back();
  }

  const std::vector<InterpValue>& stack() { return stack_; }

 private:
  struct Frame {
    int64_t pc;
    std::vector<InterpValue> slots;
    BytecodeFunction* bf;
  };

  BytecodeInterpreter(ImportData* import_data, BytecodeFunction* bf)
      : import_data_(import_data) {
    frames_.emplace_back(Frame{0, {}, bf});
  }

  absl::Status Run(const std::vector<InterpValue>& params);

  // Runs the next instruction in the current frame. Returns an error if called
  // when the PC is already pointing to the end of the bytecode.
  absl::Status EvalNextInstruction();

  absl::Status EvalAdd(const Bytecode& bytecode);
  absl::Status EvalAnd(const Bytecode& bytecode);
  absl::Status EvalCall(const Bytecode& bytecode);
  absl::Status EvalCast(const Bytecode& bytecode);
  absl::Status EvalConcat(const Bytecode& bytecode);
  absl::Status EvalCreateArray(const Bytecode& bytecode);
  absl::Status EvalCreateTuple(const Bytecode& bytecode);
  absl::Status EvalDiv(const Bytecode& bytecode);
  absl::Status EvalEq(const Bytecode& bytecode);
  absl::Status EvalExpandTuple(const Bytecode& bytecode);
  absl::Status EvalGe(const Bytecode& bytecode);
  absl::Status EvalGt(const Bytecode& bytecode);
  absl::Status EvalIndex(const Bytecode& bytecode);
  absl::Status EvalInvert(const Bytecode& bytecode);
  absl::Status EvalLe(const Bytecode& bytecode);
  absl::Status EvalLiteral(const Bytecode& bytecode);
  absl::Status EvalLoad(const Bytecode& bytecode);
  absl::Status EvalLogicalAnd(const Bytecode& bytecode);
  absl::Status EvalLogicalOr(const Bytecode& bytecode);
  absl::Status EvalLt(const Bytecode& bytecode);
  absl::Status EvalMul(const Bytecode& bytecode);
  absl::Status EvalNe(const Bytecode& bytecode);
  absl::Status EvalNegate(const Bytecode& bytecode);
  absl::Status EvalOr(const Bytecode& bytecode);
  absl::Status EvalShll(const Bytecode& bytecode);
  absl::Status EvalShrl(const Bytecode& bytecode);
  absl::Status EvalSlice(const Bytecode& bytecode);
  absl::Status EvalStore(const Bytecode& bytecode);
  absl::Status EvalSub(const Bytecode& bytecode);
  absl::Status EvalWidthSlice(const Bytecode& bytecode);
  absl::Status EvalXor(const Bytecode& bytecode);

  absl::Status EvalBinop(const std::function<absl::StatusOr<InterpValue>(
                             const InterpValue& lhs, const InterpValue& rhs)>
                             op);
  absl::StatusOr<BytecodeFunction*> GetBytecodeFn(Module* module,
                                                  Function* function);
  absl::StatusOr<std::optional<int64_t>> EvalJumpRelIf(
      int64_t pc, const Bytecode& bytecode);
  absl::Status RunBuiltinFn(const Bytecode& bytecode, Builtin builtin);
  absl::Status RunBuiltinAssertEq(const Bytecode& bytecode);

  absl::StatusOr<InterpValue> Pop();

  ImportData* import_data_;
  std::vector<InterpValue> stack_;

  std::vector<Frame> frames_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_INTERPRETER_H_

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
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {

// Bytecode interpreter for DSLX. Accepts sequence of "bytecode" "instructions"
// and a set of initial environmental bindings (key/value pairs) and executes
// until end result.
//
// TODO(rspringer): Finish adding the rest of the opcodes, etc.
class BytecodeInterpreter {
 public:
  // Takes ownership of `args`.
  static absl::StatusOr<InterpValue> Interpret(ImportData* import_data,
                                               BytecodeFunction* bf,
                                               std::vector<InterpValue> args);

  const std::vector<InterpValue>& stack() { return stack_; }

 private:
  // Represents a frame on the function stack: holds the program counter, local
  // storage, and instructions to execute.
  class Frame {
   public:
    // `bindings` will hold the bindings used to instantiate the
    // BytecodeFunction's source Function, if it is parametric.
    // `bf_holder` is only for storing the pointer to ephemeral functions, e.g.,
    // those generated on-the-fly from interpreting the `map` operation.
    // For other cases, the BytecodeCache will own BytecodeFunction storage.
    Frame(BytecodeFunction* bf, std::vector<InterpValue> args,
          const TypeInfo* type_info,
          const absl::optional<SymbolicBindings>& bindings,
          std::unique_ptr<BytecodeFunction> bf_holder = nullptr);

    int64_t pc() const { return pc_; }
    void set_pc(int64_t pc) { pc_ = pc; }
    void IncrementPc() { pc_++; }
    std::vector<InterpValue>& slots() { return slots_; }
    BytecodeFunction* bf() const { return bf_; }
    const TypeInfo* type_info() const { return type_info_; }
    const absl::optional<SymbolicBindings>& bindings() const {
      return bindings_;
    }

    void StoreSlot(Bytecode::SlotIndex slot_index, InterpValue value);

   private:
    int64_t pc_;
    std::vector<InterpValue> slots_;
    BytecodeFunction* bf_;
    const TypeInfo* type_info_;
    absl::optional<SymbolicBindings> bindings_;
    std::unique_ptr<BytecodeFunction> bf_holder_;
  };

  BytecodeInterpreter(ImportData* import_data, BytecodeFunction* bf,
                      std::vector<InterpValue> args);

  absl::Status Run();

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
  absl::Status EvalDup(const Bytecode& bytecode);
  absl::Status EvalEq(const Bytecode& bytecode);
  absl::Status EvalExpandTuple(const Bytecode& bytecode);
  absl::Status EvalFail(const Bytecode& bytecode);
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
  absl::Status EvalMatchArm(const Bytecode& bytecode);
  absl::Status EvalMul(const Bytecode& bytecode);
  absl::Status EvalNe(const Bytecode& bytecode);
  absl::Status EvalNegate(const Bytecode& bytecode);
  absl::Status EvalOr(const Bytecode& bytecode);
  absl::Status EvalPop(const Bytecode& bytecode);
  absl::Status EvalShl(const Bytecode& bytecode);
  absl::Status EvalShr(const Bytecode& bytecode);
  absl::Status EvalSlice(const Bytecode& bytecode);
  absl::Status EvalStore(const Bytecode& bytecode);
  absl::Status EvalSub(const Bytecode& bytecode);
  absl::Status EvalSwap(const Bytecode& bytecode);
  absl::Status EvalTrace(const Bytecode& bytecode);
  absl::Status EvalWidthSlice(const Bytecode& bytecode);
  absl::Status EvalXor(const Bytecode& bytecode);

  absl::Status EvalUnop(
      const std::function<absl::StatusOr<InterpValue>(const InterpValue& arg)>&
          op);
  absl::Status EvalBinop(
      const std::function<absl::StatusOr<InterpValue>(
          const InterpValue& lhs, const InterpValue& rhs)>& op);
  absl::StatusOr<BytecodeFunction*> GetBytecodeFn(
      Function* function, Invocation* invocation,
      const absl::optional<SymbolicBindings>& caller_bindings);
  absl::StatusOr<std::optional<int64_t>> EvalJumpRelIf(
      int64_t pc, const Bytecode& bytecode);

  // TODO(rspringer): 2022-02-14: Builtins should probably go in their own file,
  // likely after removing the old interpreter.
  absl::Status RunBuiltinFn(const Bytecode& bytecode, Builtin builtin);
  absl::Status RunBinaryBuiltin(std::function<absl::StatusOr<InterpValue>(
                                    const InterpValue& a, const InterpValue& b)>
                                    fn);
  absl::Status RunTernaryBuiltin(
      std::function<absl::StatusOr<InterpValue>(
          const InterpValue& a, const InterpValue& b, const InterpValue& c)>
          fn);
  absl::Status RunBuiltinAddWithCarry(const Bytecode& bytecode);
  absl::Status RunBuiltinAndReduce(const Bytecode& bytecode);
  absl::Status RunBuiltinAssertEq(const Bytecode& bytecode);
  absl::Status RunBuiltinAssertLt(const Bytecode& bytecode);
  absl::Status RunBuiltinBitSlice(const Bytecode& bytecode);
  absl::Status RunBuiltinBitSliceUpdate(const Bytecode& bytecode);
  absl::Status RunBuiltinClz(const Bytecode& bytecode);
  absl::Status RunBuiltinCtz(const Bytecode& bytecode);
  absl::Status RunBuiltinEnumerate(const Bytecode& bytecode);
  absl::Status RunBuiltinGate(const Bytecode& bytecode);
  absl::Status RunBuiltinMap(const Bytecode& bytecode);
  absl::Status RunBuiltinOneHot(const Bytecode& bytecode);
  absl::Status RunBuiltinOneHotSel(const Bytecode& bytecode);
  absl::Status RunBuiltinOrReduce(const Bytecode& bytecode);
  absl::Status RunBuiltinRange(const Bytecode& bytecode);
  absl::Status RunBuiltinRev(const Bytecode& bytecode);
  absl::Status RunBuiltinSignex(const Bytecode& bytecode);
  absl::Status RunBuiltinSlice(const Bytecode& bytecode);
  absl::Status RunBuiltinUpdate(const Bytecode& bytecode);
  absl::Status RunBuiltinXorReduce(const Bytecode& bytecode);

  absl::StatusOr<std::string> TraceDataToString(
      const Bytecode::TraceData& trace_data);
  absl::StatusOr<bool> MatchArmEqualsInterpValue(
      Frame* frame, const Bytecode::MatchArmItem& item,
      const InterpValue& value);

  absl::StatusOr<InterpValue> Pop();

  ImportData* import_data_;
  std::vector<InterpValue> stack_;

  std::vector<Frame> frames_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_INTERPRETER_H_

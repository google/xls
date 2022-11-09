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

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/bytecode.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {

class ProcInstance;

// Represents a frame on the function stack: holds the program counter, local
// storage, and instructions to execute.
class Frame {
 public:
  // `bindings` will hold the bindings used to instantiate the
  // BytecodeFunction's source Function, if it is parametric.
  // `bf_holder` is only for storing the pointer to ephemeral functions, e.g.,
  // those generated on-the-fly from interpreting the `map` operation.
  // `initial_args` holds the set of args used to initially construct the frame
  // (i.e., the arguments to this function). This is necessary for comparing
  // results to a reference, e.g., the JIT via BytecodeInterpreter's
  // post_fn_eval_hook.
  // For other cases, the BytecodeCache will own BytecodeFunction storage.
  Frame(BytecodeFunction* bf, std::vector<InterpValue> args,
        const TypeInfo* type_info,
        const std::optional<SymbolicBindings>& bindings,
        std::vector<InterpValue> initial_args,
        std::unique_ptr<BytecodeFunction> bf_holder = nullptr);

  int64_t pc() const { return pc_; }
  void set_pc(int64_t pc) { pc_ = pc; }
  void IncrementPc() { pc_++; }
  std::vector<InterpValue>& slots() { return slots_; }
  BytecodeFunction* bf() const { return bf_; }
  const TypeInfo* type_info() const { return type_info_; }
  const std::optional<SymbolicBindings>& bindings() const { return bindings_; }
  const std::vector<InterpValue>& initial_args() { return initial_args_; }

  void StoreSlot(Bytecode::SlotIndex slot_index, InterpValue value);

 private:
  int64_t pc_;
  std::vector<InterpValue> slots_;
  BytecodeFunction* bf_;
  const TypeInfo* type_info_;
  std::optional<SymbolicBindings> bindings_;
  std::vector<InterpValue> initial_args_;

  std::unique_ptr<BytecodeFunction> bf_holder_;
};

// Bytecode interpreter for DSLX. Accepts sequence of "bytecode" "instructions"
// and a set of initial environmental bindings (key/value pairs) and executes
// until end result.
class BytecodeInterpreter {
 public:
  // Function signature for a "post function-evaluation hook" -- this is invoked
  // after a function is evaluated by the interpreter. This is useful for e.g.
  // externally-implementing and hooking-in comparison to the JIT execution
  // mode.
  using PostFnEvalHook = std::function<absl::Status(
      const Function* f, absl::Span<const InterpValue> args,
      const SymbolicBindings*, const InterpValue& got)>;
  virtual ~BytecodeInterpreter() {}

  // Takes ownership of `args`.
  static absl::StatusOr<InterpValue> Interpret(
      ImportData* import_data, BytecodeFunction* bf,
      const std::vector<InterpValue>& args,
      PostFnEvalHook post_fn_eval_hook = nullptr);
  absl::Status InitFrame(BytecodeFunction* bf,
                         const std::vector<InterpValue>& args,
                         const TypeInfo* type_info);

  const std::vector<InterpValue>& stack() { return stack_; }

  // Helper for converting a trace format string to its result given a stack
  // state.
  static absl::StatusOr<std::string> TraceDataToString(
      const Bytecode::TraceData& trace_data, std::vector<InterpValue>& stack);

 protected:
  BytecodeInterpreter(ImportData* import_data, BytecodeFunction* bf);
  static absl::StatusOr<std::unique_ptr<BytecodeInterpreter>> CreateUnique(
      ImportData* import_data, BytecodeFunction* bf,
      const std::vector<InterpValue>& args);
  std::vector<Frame>& frames() { return frames_; }
  ImportData* import_data() { return import_data_; }
  absl::Status Run(PostFnEvalHook post_fn_eval_hook = nullptr);

 private:
  friend class ProcInstance;

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
  absl::Status EvalRange(const Bytecode& bytecode);
  absl::Status EvalRecv(const Bytecode& bytecode);
  absl::Status EvalRecvNonBlocking(const Bytecode& bytecode);
  absl::Status EvalSend(const Bytecode& bytecode);
  absl::Status EvalShl(const Bytecode& bytecode);
  absl::Status EvalShr(const Bytecode& bytecode);
  absl::Status EvalSlice(const Bytecode& bytecode);
  // TODO(rspringer): 2022-04-12: Rather than use inheritance here, consider
  // injecting a Proc/Spawn strategy function/lambda into the interpreter.
  virtual absl::Status EvalSpawn(const Bytecode& bytecode) {
    return absl::UnimplementedError(
        "BytecodeInterpreter does not support spawning procs. "
        "ProcConfigBytecodeInterpreter should be used for proc network "
        "config.");
  }
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
      Function* function, const Invocation* invocation,
      const std::optional<SymbolicBindings>& caller_bindings);
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
  absl::Status RunBuiltinPrioritySel(const Bytecode& bytecode);
  absl::Status RunBuiltinOrReduce(const Bytecode& bytecode);
  absl::Status RunBuiltinRange(const Bytecode& bytecode);
  absl::Status RunBuiltinRev(const Bytecode& bytecode);
  absl::Status RunBuiltinSignex(const Bytecode& bytecode);
  absl::Status RunBuiltinSlice(const Bytecode& bytecode);
  absl::Status RunBuiltinSMulp(const Bytecode& bytecode);
  absl::Status RunBuiltinUMulp(const Bytecode& bytecode);
  absl::Status RunBuiltinUpdate(const Bytecode& bytecode);
  absl::Status RunBuiltinXorReduce(const Bytecode& bytecode);

  // Common handler for the range bytecode and builtin range() fn.
  absl::Status RangeInternal();

  absl::StatusOr<bool> MatchArmEqualsInterpValue(
      Frame* frame, const Bytecode::MatchArmItem& item,
      const InterpValue& value);

  static absl::StatusOr<InterpValue> Pop(std::vector<InterpValue>& stack);
  absl::StatusOr<InterpValue> Pop() { return Pop(stack_); }

  ImportData* import_data_;
  std::vector<InterpValue> stack_;

  std::vector<Frame> frames_;
};

// Specialization of BytecodeInterpreter for executing Proc `config` functions.
// These are special b/c they define a tree of ProcInstances that we need to
// collect at the end so we can "tick" them. Only this class, unlike
// BytecodeInterpreter, can process `spawn` nodes.
class ProcConfigBytecodeInterpreter : public BytecodeInterpreter {
 public:
  // Channels have state, so we can't just copy args. We're guaranteed that args
  // will live for the duration of network initialization, so storing references
  // is safe.
  // `config_args` is not moved, in order to allow callers to specify channels
  // without losing their handles to them.
  // `proc_instances` is an out-param into which instantiated ProcInstances
  // should be placed.
  static absl::Status InitializeProcNetwork(
      ImportData* import_data, TypeInfo* type_info, Proc* root_proc,
      InterpValue terminator, std::vector<ProcInstance>* proc_instances);

  virtual ~ProcConfigBytecodeInterpreter() = default;

  // Implementation of Spawn handling common to both InitializeProcNetwork
  // and EvalSpawn. `next_args` should not include Proc members or the
  // obligatory Token; they're added to the arg list internally.
  static absl::Status EvalSpawn(ImportData* import_data,
                                const TypeInfo* type_info,
                                const std::optional<SymbolicBindings>& bindings,
                                std::optional<const Spawn*> maybe_spawn,
                                Proc* proc,
                                const std::vector<InterpValue>& config_args,
                                std::vector<ProcInstance>* proc_instances);

 private:
  ProcConfigBytecodeInterpreter(ImportData* import_data, BytecodeFunction* bf,
                                std::vector<ProcInstance>* proc_instances);

  absl::Status EvalSpawn(const Bytecode& bytecode) override;

  std::vector<ProcInstance>* proc_instances_;
};

// A ProcInstance is an instantiation of a Proc.
// ProcInstance : Proc :: Object : Class, roughly.
class ProcInstance {
 public:
  ProcInstance(Proc* proc, std::unique_ptr<BytecodeInterpreter> interpreter,
               std::unique_ptr<BytecodeFunction> next_fn,
               std::vector<InterpValue> next_args)
      : proc_(proc),
        interpreter_(std::move(interpreter)),
        next_fn_(std::move(next_fn)),
        next_args_(std::move(next_args)) {}

  // Executes a single "tick" of the ProcInstance. Returns success if the
  // ProcInstance fully completed execution of its `next` function _OR_ if the
  // ProcInstance is blocked trying to receive on an empty channel. IOW, this
  // will only return non-OK in case of a fatal error.
  absl::Status Run();

 private:
  Proc* proc_;
  std::unique_ptr<BytecodeInterpreter> interpreter_;
  std::unique_ptr<BytecodeFunction> next_fn_;
  std::vector<InterpValue> next_args_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_INTERPRETER_H_

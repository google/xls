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
#ifndef XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_H_
#define XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/frame.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

class ProcInstance;

using PostFnEvalHook = std::function<absl::Status(
    const Function* f, absl::Span<const InterpValue> args, const ParametricEnv*,
    const InterpValue& got)>;

// Trace hook which logs trace messages to INFO.
inline void InfoLoggingTraceHook(const FileTable& file_table,
                                 const Span& source_location,
                                 std::string_view message) {
  XLS_LOG_LINES_LOC(INFO, message, source_location.GetFilename(file_table),
                    source_location.start().GetHumanLineno());
}

// Stores name and use-location of a blocked channel, for error messaging.
struct BlockedChannelInfo {
  std::string name;
  Span span;
};

// Bytecode interpreter for DSLX. Accepts sequence of "bytecode" "instructions"
// and a set of initial environmental bindings (key/value pairs) and executes
// until end result.
class BytecodeInterpreter {
 public:
  static absl::StatusOr<InterpValue> Interpret(
      ImportData* import_data, BytecodeFunction* bf,
      const std::vector<InterpValue>& args,
      const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions());

  virtual ~BytecodeInterpreter() = default;

  absl::Status InitFrame(BytecodeFunction* bf,
                         absl::Span<const InterpValue> args,
                         const TypeInfo* type_info);

  // Helper for converting a trace format string to its result given a stack
  // state.
  //
  // Note that this mutates the stack by popping off values consumed by the
  // trace data.
  static absl::StatusOr<std::string> TraceDataToString(
      const Bytecode::TraceData& trace_data, InterpreterStack& stack);

  const FileTable& file_table() const { return import_data_->file_table(); }

 protected:
  BytecodeInterpreter(ImportData* import_data,
                      const BytecodeInterpreterOptions& options);

  // Creates a new interpreter object with an initialized entry frame.
  static absl::StatusOr<std::unique_ptr<BytecodeInterpreter>> CreateUnique(
      ImportData* import_data, BytecodeFunction* bf,
      const std::vector<InterpValue>& args,
      const BytecodeInterpreterOptions& options);

  const InterpreterStack& stack() const { return stack_; }

  std::vector<Frame>& frames() { return frames_; }
  ImportData* import_data() { return import_data_; }
  const BytecodeInterpreterOptions& options() const { return options_; }
  const std::optional<BlockedChannelInfo>& blocked_channel_info() const {
    return blocked_channel_info_;
  }

  // Sets `progress_made` to true (if not null) if at least a single bytecode
  // executed.  Progress can be stalled on blocked receive operations.
  absl::Status Run(bool* progress_made = nullptr);

 private:
  friend class ProcInstance;

  // Runs the next instruction in the current frame. Returns an error if called
  // when the PC is already pointing to the end of the bytecode.
  absl::Status EvalNextInstruction();

  absl::Status EvalAdd(const Bytecode& bytecode, bool is_signed);
  absl::Status EvalSub(const Bytecode& bytecode, bool is_signed);
  absl::Status EvalMul(const Bytecode& bytecode, bool is_signed);

  absl::Status EvalAnd(const Bytecode& bytecode);
  absl::Status EvalCall(const Bytecode& bytecode);
  absl::Status EvalCast(const Bytecode& bytecode, bool is_checked = false);
  absl::Status EvalConcat(const Bytecode& bytecode);
  absl::Status EvalCreateArray(const Bytecode& bytecode);
  absl::Status EvalCreateTuple(const Bytecode& bytecode);
  absl::Status EvalDecode(const Bytecode& bytecode);
  absl::Status EvalDiv(const Bytecode& bytecode);
  absl::Status EvalDup(const Bytecode& bytecode);
  absl::Status EvalEq(const Bytecode& bytecode);
  absl::Status EvalExpandTuple(const Bytecode& bytecode);
  absl::Status EvalFail(const Bytecode& bytecode);
  absl::Status EvalGe(const Bytecode& bytecode);
  absl::Status EvalGt(const Bytecode& bytecode);
  absl::Status EvalIndex(const Bytecode& bytecode);
  absl::Status EvalTupleIndex(const Bytecode& bytecode);
  absl::Status EvalInvert(const Bytecode& bytecode);
  absl::Status EvalLe(const Bytecode& bytecode);
  absl::Status EvalLiteral(const Bytecode& bytecode);
  absl::Status EvalLoad(const Bytecode& bytecode);
  absl::Status EvalLogicalAnd(const Bytecode& bytecode);
  absl::Status EvalLogicalOr(const Bytecode& bytecode);
  absl::Status EvalLt(const Bytecode& bytecode);
  absl::Status EvalMatchArm(const Bytecode& bytecode);
  absl::Status EvalMod(const Bytecode& bytecode);
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
  absl::Status EvalSwap(const Bytecode& bytecode);
  absl::Status EvalTrace(const Bytecode& bytecode);
  absl::Status EvalWidthSlice(const Bytecode& bytecode);
  absl::Status EvalXor(const Bytecode& bytecode);

  absl::Status EvalBinop(
      const std::function<absl::StatusOr<InterpValue>(
          const InterpValue& lhs, const InterpValue& rhs)>& op);

  absl::StatusOr<BytecodeFunction*> GetBytecodeFn(
      Function& function, const Invocation* invocation,
      const ParametricEnv& caller_bindings);

  absl::StatusOr<std::optional<int64_t>> EvalJumpRelIf(
      int64_t pc, const Bytecode& bytecode);

  // TODO(rspringer): 2022-02-14: Builtins should probably go in their own file,
  // likely after removing the old interpreter.
  absl::Status RunBuiltinFn(const Bytecode& bytecode, Builtin builtin);

  // Map is a very special builtin function at the moment and so lives within
  // the bytecode interpreter -- currently it emits new bytecode *on the fly*
  // and evaluates it.
  //
  // TODO(cdleary): 2023-07-12 We should be able to emit all bytecode objects
  // necessary at emission time and make it a question more "invoke like" normal
  // runtime execution.
  absl::Status RunBuiltinMap(const Bytecode& bytecode);

  absl::StatusOr<bool> MatchArmEqualsInterpValue(
      Frame* frame, const Bytecode::MatchArmItem& item,
      const InterpValue& value);

  absl::StatusOr<InterpValue> Pop() { return stack_.Pop(); }

  ImportData* const import_data_;
  InterpreterStack stack_;
  std::vector<Frame> frames_;

  BytecodeInterpreterOptions options_;

  // This field is set to the name of the blocked channel when a receive is
  // blocked. This is reset (and potentially set again) each time the Run method
  // executes.
  // TODO(meheff): 2023/02/14 A better way of handling this is by definining a
  // separate continuation data structure which encapsulates the entire
  // execution state including this value.
  std::optional<BlockedChannelInfo> blocked_channel_info_;
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
      const InterpValue& terminator, std::vector<ProcInstance>* proc_instances,
      const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions());

  ~ProcConfigBytecodeInterpreter() override = default;

  // Implementation of Spawn handling common to both InitializeProcNetwork
  // and EvalSpawn. `next_args` should not include Proc members or the
  // obligatory Token; they're added to the arg list internally.
  static absl::Status EvalSpawn(
      ImportData* import_data, const TypeInfo* type_info,
      const std::optional<ParametricEnv>& caller_bindings,
      const std::optional<ParametricEnv>& callee_bindings,
      std::optional<const Bytecode::SpawnFunctions*> spawn_funcs, Proc* proc,
      absl::Span<const InterpValue> config_args,
      std::vector<ProcInstance>* proc_instances,
      const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions());

 private:
  ProcConfigBytecodeInterpreter(ImportData* import_data,
                                std::vector<ProcInstance>* proc_instances,
                                const BytecodeInterpreterOptions& options);

  absl::Status EvalSpawn(const Bytecode& bytecode) override;

  std::vector<ProcInstance>* proc_instances_;
};

// The execution state that a proc may be left in after a call to
// ProcInstance::Run.
enum class ProcExecutionState : uint8_t {
  // The proc tick completed.
  kCompleted,
  // The proc tick was blocked on a blocking receive.
  kBlockedOnReceive,
};

// Data structure holding the result of a single call to ProcInstance::Run.
struct ProcRunResult {
  ProcExecutionState execution_state;

  // If tick state is kBlockedOnReceive this field holds the name and usage
  // location of the blocked channel.
  std::optional<BlockedChannelInfo> blocked_channel_info;

  // Whether any progress was made (at least one instruction was executed).
  bool progress_made;
};

// A ProcInstance is an instantiation of a Proc.
// ProcInstance : Proc :: Object : Class, roughly.
class ProcInstance {
 public:
  ProcInstance(Proc* proc, std::unique_ptr<BytecodeInterpreter> interpreter,
               std::unique_ptr<BytecodeFunction> next_fn,
               std::vector<InterpValue> next_args, const TypeInfo* type_info)
      : proc_(proc),
        interpreter_(std::move(interpreter)),
        next_fn_(std::move(next_fn)),
        next_args_(std::move(next_args)),
        type_info_(type_info) {}

  // Executes a single "tick" of the ProcInstance.
  absl::StatusOr<ProcRunResult> Run();
  Proc* proc() const { return proc_; }

 private:
  Proc* proc_;
  std::unique_ptr<BytecodeInterpreter> interpreter_;
  std::unique_ptr<BytecodeFunction> next_fn_;
  std::vector<InterpValue> next_args_;
  const TypeInfo* type_info_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_H_

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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/frame.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

using PostFnEvalHook = std::function<absl::Status(
    const Function* f, absl::Span<const InterpValue> args, const ParametricEnv*,
    const InterpValue& got)>;

// A data structure which records trace and assert message in the DSLX
// interpreter.
class DslxInterpreterEvents {
 public:
  virtual ~DslxInterpreterEvents() = default;

  virtual void AddTraceStatementMessage(const FileTable& file_table,
                                        const Span& source_location,
                                        std::string_view msg);
  virtual void AddTraceCallMessage(const FileTable& file_table,
                                   const Span& source_location,
                                   std::string_view function_name,
                                   absl::Span<const InterpValue> args,
                                   int64_t call_depth,
                                   FormatPreference format_preference);
  virtual void AddTraceCallReturnMessage(const FileTable& file_table,
                                         const Span& source_location,
                                         std::string_view function_name,
                                         int64_t call_depth,
                                         FormatPreference format_preference,
                                         const InterpValue& return_value);
  virtual void AddTraceChannelMessage(
      const FileTable& file_table, const Span& source_location,
      std::string_view channel_name, const InterpValue& value,
      ChannelDirection direction,
      const ValueFormatDescriptor& format_descriptor);
  virtual void AddAssertMessage(const FileTable& file_table,
                                const Span& source_location,
                                std::string_view msg);
  std::vector<std::string> GetTraceMessageStrings() const;
  std::vector<std::string> GetAssertMessageStrings() const;

  static DslxInterpreterEvents FromProto(const EvaluatorEventsProto& proto);
  const EvaluatorEventsProto& AsProto() const { return proto_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DslxInterpreterEvents& events) {
    for (const std::string& s : events.GetTraceMessageStrings()) {
      absl::Format(&sink, "%s\n", s);
    }
    for (const std::string& s : events.GetAssertMessageStrings()) {
      absl::Format(&sink, "%s\n", s);
    }
  }
  bool operator==(const DslxInterpreterEvents& other) const;

  bool operator!=(const DslxInterpreterEvents& other) const {
    return !(*this == other);
  }

 protected:
  // Method called for each trace message string that is recorded. Can be
  // overridden.
  virtual void NoteTraceMessageString(const FileTable& file_table,
                                      const Span& source_location,
                                      std::string_view msg) {}

  EvaluatorEventsProto proto_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const DslxInterpreterEvents& events) {
  return os << absl::StrCat(events);
}

// A DslxInterpreterEvents implementation that logs trace messages to INFO.
class InfoLoggingDslxInterpreterEvents : public DslxInterpreterEvents {
 public:
  ~InfoLoggingDslxInterpreterEvents() override = default;

 protected:
  void NoteTraceMessageString(const FileTable& file_table,
                              const Span& source_location,
                              std::string_view msg) override;
};

// Stores name and use-location of a blocked channel, for error messaging.
struct BlockedChannelInfo {
  std::string name;
  Span span;
};

// A FIFO which backs channel instances in the bytecode interpreter.
class InterpValueChannel {
 public:
  InterpValueChannel(int64_t id, std::optional<int64_t> depth)
      : id_(id), depth_(depth) {}
  InterpValueChannel(const InterpValueChannel&) = delete;
  InterpValueChannel(InterpValueChannel&&) = default;

  int64_t GetId() const { return id_; }
  bool IsEmpty() const { return queue_.empty(); }
  bool IsFull() const {
    return depth_.has_value() && (queue_.size() >= *depth_);
  }
  int64_t GetSize() const { return queue_.size(); }
  InterpValue Read() {
    InterpValue result = std::move(queue_.front());
    queue_.pop_front();
    return result;
  }
  void Write(InterpValue v) { queue_.push_back(std::move(v)); }

 private:
  const int64_t id_;
  std::optional<int64_t> depth_;
  std::deque<InterpValue> queue_;
};

// A collection of all channel objects used by a proc network (elaboration).
class InterpValueChannelManager {
 public:
  explicit InterpValueChannelManager(int64_t size = 0) : channels_(size) {}
  virtual ~InterpValueChannelManager() = default;

  int64_t size() const { return channels_.size(); }
  virtual InterpValueChannel& GetChannel(
      const TypeInfo* ti, const InterpValue::ChannelReference& channel_ref) = 0;

  // Allocates a channel using the next available ID. This is for legacy procs.
  // `depth` sets the FIFO capacity; nullopt means unbounded.
  virtual InterpValueChannel& AllocateLegacyChannel(
      std::optional<int64_t> depth = std::nullopt) = 0;

  // Allocates a channel using the passed in ID, which should have been provided
  // by type inference. The definer is the `Param` or `ChannelDecl` node where
  // the channel originates (`Param` only for top boundary channels).
  // `depth` sets the FIFO capacity; nullopt means unbounded.
  virtual absl::StatusOr<InterpValueChannel*> AllocateChannel(
      int64_t channel_id, const AstNode* definer,
      std::optional<int64_t> depth) = 0;

 protected:
  absl::flat_hash_map<int64_t, std::unique_ptr<InterpValueChannel>> channels_;
};

class LegacyChannelManager : public InterpValueChannelManager {
 public:
  InterpValueChannel& AllocateLegacyChannel(
      std::optional<int64_t> depth = std::nullopt) override {
    const int64_t channel_id = channels_.size();
    return *channels_
                .emplace(channel_id, std::make_unique<InterpValueChannel>(
                                         channel_id, depth))
                .first->second;
  }

  absl::StatusOr<InterpValueChannel*> AllocateChannel(
      int64_t, const AstNode*,
      std::optional<int64_t>) override {
    return absl::UnimplementedError(
        "This channel manager is for legacy procs and does not implement "
        "allocation with external IDs.");
  }

  InterpValueChannel& GetChannel(
      const TypeInfo*,
      const InterpValue::ChannelReference& channel_ref) override {
    CHECK(channel_ref.GetChannelId().has_value());
    return *channels_[*channel_ref.GetChannelId()];
  }
};

class ProcDefChannelManager : public InterpValueChannelManager {
 public:
  InterpValueChannel& AllocateLegacyChannel(
      std::optional<int64_t> = std::nullopt) override {
    CHECK(false) << "This channel manager is for impl-style procs and does not "
                    "implement legacy allocation.";
  }

  absl::StatusOr<InterpValueChannel*> AllocateChannel(
      int64_t channel_id, const AstNode* definer,
      std::optional<int64_t> depth) override {
    VLOG(5) << "Allocating channel ID " << channel_id
            << " from definer: " << definer->ToString();
    XLS_RET_CHECK(definer->kind() == AstNodeKind::kParam ||
                  definer->kind() == AstNodeKind::kChannelDecl);
    auto& value = channels_[channel_id];
    if (value == nullptr) {
      value = std::make_unique<InterpValueChannel>(channel_id, depth);
      definers_[channel_id] = definer;
    } else if (definers_.at(channel_id) != definer) {
      return absl::AlreadyExistsError(absl::StrCat(
          "There is already a channel allocated to ID: ", channel_id));
    }
    return value.get();
  }

  // When a ProcDef constructor forwards a channel to another ProcDef
  // constructor, this function should be used to associate the original channel
  // ID with the Param that is the definer of the forwarded ChannelReference.
  void Forward(int64_t channel_id, const TypeInfo* ti, const Param* param) {
    InterpValueChannel* channel = channels_.at(channel_id).get();
    forwarded_channels_.insert_or_assign(std::make_pair(ti, param), channel);
  }

  InterpValueChannel& GetChannel(
      const TypeInfo* ti,
      const InterpValue::ChannelReference& channel_ref) override {
    CHECK(channel_ref.GetChannelId().has_value());
    const auto it_by_id = channels_.find(*channel_ref.GetChannelId());
    if (it_by_id != channels_.end()) {
      return *it_by_id->second;
    }

    CHECK(channel_ref.GetDefiner().has_value());
    return *forwarded_channels_.at(
        std::make_pair(ti, *channel_ref.GetDefiner()));
  }

 private:
  absl::flat_hash_map<int64_t, const AstNode*> definers_;
  absl::flat_hash_map<std::pair<const TypeInfo*, const AstNode*>,
                      InterpValueChannel*>
      forwarded_channels_;
};

// Bytecode interpreter for DSLX. Accepts sequence of "bytecode" "instructions"
// and a set of initial environmental bindings (key/value pairs) and executes
// until end result.
class BytecodeInterpreter {
 public:
  friend class ProcInstance;

  static absl::StatusOr<InterpValue> Interpret(
      ImportData* import_data, BytecodeFunction* bf,
      const std::vector<InterpValue>& args,
      std::optional<InterpValueChannelManager*> channel_manager = std::nullopt,
      const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions(),
      std::optional<DslxInterpreterEvents*> events = std::nullopt);

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
  const BytecodeInterpreterOptions& options() const { return options_; }
  const InterpreterStack& stack() const { return stack_; }
  const std::optional<BlockedChannelInfo>& blocked_channel_info() const {
    return blocked_channel_info_;
  }
  const std::optional<ProcId>& proc_id() const { return proc_id_; }

  void SetStateValue(const NameDef* state_element, InterpValue value) {
    state_values_.insert_or_assign(state_element, value);
  }

  // Sets `progress_made` to true (if not null) if at least a single bytecode
  // executed.  Progress can be stalled on blocked receive operations.
  absl::Status Run(bool* progress_made = nullptr);

  // Creates a new interpreter object with an initialized entry frame.
  static absl::StatusOr<std::unique_ptr<BytecodeInterpreter>> CreateUnique(
      ImportData* import_data, const std::optional<ProcId>& proc_id,
      BytecodeFunction* bf, const std::vector<InterpValue>& args,
      std::optional<InterpValueChannelManager*> channel_manager,
      const BytecodeInterpreterOptions& options,
      std::optional<DslxInterpreterEvents*> events = std::nullopt);

 protected:
  BytecodeInterpreter(
      ImportData* import_data, const std::optional<ProcId>& proc_id,
      std::optional<InterpValueChannelManager*> channel_manager,
      const BytecodeInterpreterOptions& options,
      std::optional<DslxInterpreterEvents*> events = std::nullopt);

  std::vector<Frame>& frames() { return frames_; }
  ImportData* import_data() { return import_data_; }

  // Pops `count` arguments to a function or spawn, assuming they were pushed in
  // left-to-right order and must be popped in right-to-left order.
  absl::StatusOr<std::vector<InterpValue>> PopArgsRightToLeft(size_t count);

  // Formats the name of the given `channel` for logging via the trace hook. The
  // name is qualified with the proc instantiation context.
  std::string FormatChannelNameForTracing(const Bytecode::ChannelData& channel);

 private:
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
  absl::Status EvalRead(const Bytecode& bytecode);
  absl::Status EvalWrite(const Bytecode& bytecode);
  absl::Status EvalStore(const Bytecode& bytecode);
  absl::Status EvalSwap(const Bytecode& bytecode);
  absl::Status EvalTraceFmt(const Bytecode& bytecode);
  absl::Status EvalTraceArg(const Bytecode& bytecode);
  absl::Status EvalWidthSlice(const Bytecode& bytecode);
  absl::Status EvalXor(const Bytecode& bytecode);

  absl::Status EvalBinop(
      const std::function<absl::StatusOr<InterpValue>(
          const InterpValue& lhs, const InterpValue& rhs)>& op);

  absl::StatusOr<BytecodeFunction*> GetBytecodeFn(
      const Function& function, const Invocation* invocation,
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

  RolloverEvent CreateRolloverEvent(const Bytecode& bytecode,
                                    const InterpValue& lhs,
                                    const InterpValue& rhs);

  ImportData* const import_data_;
  const std::optional<ProcId> proc_id_;

  InterpreterStack stack_;
  std::vector<Frame> frames_;
  std::optional<InterpValueChannelManager*> channel_manager_;
  BytecodeInterpreterOptions options_;

  // This field is set to the name of the blocked channel when a receive is
  // blocked. This is reset (and potentially set again) each time the Run method
  // executes.
  // TODO(meheff): 2023/02/14 A better way of handling this is by defining a
  // separate continuation data structure which encapsulates the entire
  // execution state including this value.
  std::optional<BlockedChannelInfo> blocked_channel_info_;
  std::optional<DslxInterpreterEvents*> events_;
  absl::flat_hash_map<const NameDef*, InterpValue> state_values_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_H_

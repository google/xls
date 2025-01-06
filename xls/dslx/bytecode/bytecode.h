// Copyright 2022 The XLS Authors
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
#ifndef XLS_DSLX_BYTECODE_BYTECODE_H_
#define XLS_DSLX_BYTECODE_BYTECODE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/strong_int.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {

// Defines a single "instruction" for the DSLX bytecode interpreter: an opcode
// and optional accessory data (load/store value name, function call target).
class Bytecode {
 public:
  // In these descriptions, "TOS1" refers to the second-to-top-stack element and
  // "TOS0" refers to the top stack element.
  enum class Op : uint8_t {
    // Adds the top two values on the stack.
    kUAdd,
    kSAdd,
    // Performs a bitwise AND of the top two values on the stack.
    kAnd,
    // Invokes the function given in the Bytecode's data argument. Arguments are
    // given on the stack with deeper elements being earlier in the arg list
    // (rightmost arg is TOS0 because we evaluate args left-to-right).
    kCall,
    // Casts the element on top of the stack to the type given in the optional
    // arg.
    kCast,
    // Casts the element on top of the stack to the type given in the optional
    // arg.  Check that no information is lost.
    kCheckedCast,
    // Concatenates TOS1 and TOS0, with TOS1 comprising the most significant
    // bits of the result.
    kConcat,
    // Combines the top N values on the stack (N given in the data argument)
    // into an array.
    kCreateArray,
    // Creates an N-tuple (N given in the data argument) from the values on the
    // stack.
    kCreateTuple,
    // Decodes the element on top of the stack to a one-hot of the type given as
    // the parametric arg.
    kDecode,
    // Divides the N-1th value on the stack by the Nth value.
    kDiv,
    // Determines remainder of division of the N-1th value by the Nth value.
    kMod,
    // Expands the N-tuple on the top of the stack by one level, placing leading
    // elements at the top of the stack. In other words, expanding the tuple
    // `(a, (b, c))` will result in a stack of `(b, c), a`, where `a` is on top
    // of the stack.
    kExpandTuple,
    // Compares TOS1 to TOS0, storing true if TOS1 == TOS0.
    kEq,
    // Duplicates the value at TOS0 to be the new TOS0.
    kDup,
    // Terminates the current program with a failure status. Consumes as many
    // values from the stack as are specified in the `TraceData` data member.
    kFail,
    // Compares TOS1 to TOS0, storing true if TOS1 >= TOS0.
    kGe,
    // Compares TOS1 to TOS0, storing true if TOS1 > TOS0.
    kGt,
    // Selects the TOS0'th element of the array- or tuple-typed value at TOS1.
    kIndex,
    // Same as kIndex but runtime checks that the value TOS1 is a tuple.
    kTupleIndex,
    // Inverts the bits of TOS0.
    kInvert,
    // Indicates a jump destination PC for control flow integrity checking.
    // (Note that there's no actual execution logic for this opcode.)
    kJumpDest,
    // Unconditional jump (relative).
    kJumpRel,
    // Pops the entry at the top of the stack and jumps (relative) if it is
    // true, otherwise PC proceeds as normal (i.e. PC = PC + 1).
    kJumpRelIf,
    // Compares TOS1 to TOS0, storing true if TOS1 <= TOS0.
    kLe,
    // Pushes the literal in the data argument onto the stack.
    kLiteral,
    // Loads the value from the data-arg-specified slot and pushes it onto the
    // stack.
    kLoad,
    // Performs a logical AND of TOS1 and TOS0.
    kLogicalAnd,
    // Performs a logical OR of TOS1 and TOS0.
    kLogicalOr,
    // Compares TOS1 to B, storing true if TOS1 < TOS0.
    kLt,
    // Evaluates the item at TOS0 and pushes `true` on the stack if it's
    // equivalent to the MatchArmItem (defined below) held in the optional data
    // member.
    kMatchArm,
    // Multiplies the top two values on the stack.
    kUMul,
    kSMul,
    // Compares TOS1 to B, storing true if TOS1 != TOS0.
    kNe,
    // Performs two's complement negation of TOS0.
    kNegate,
    // Performs a bitwise OR of the top two values on the stack.
    kOr,
    // Pops the value at TOS0.
    kPop,
    // Creates an array of values [TOS1, TOS0).
    kRange,
    // Pulls TOS0 (a condition) and TOS1 (a channel).
    // If TOS0 is true, then
    //   pulls a value off of the channel or "blocks"
    //   if empty: terminates execution at the opcode's PC. The interpreter can
    //    be resumed/retried if/when a value becomes available.
    // else
    //   a tuple containing a tuple and zero value is pushed on the stack.
    kRecv,
    // Pulls a value off of the channel at TOS0, but does not block if empty.
    // A tuple containing
    //   0. A token.
    //   1. The value pulled off (or a zero value if the channel is empty).
    //     and
    //   2. A valid flag (false if the channel is empty).
    // is pushed on the stack.
    kRecvNonBlocking,
    // Inserts the value at TOS0 into the channel at TOS1.
    kSend,
    // Performs a left shift of the second-to-top stack element by the
    // top element's number.
    kShl,
    // Performs a right shift of the second-to-top stack element by the
    // top element's number. If TOS1 is signed, the shift will be arithmetic,
    // otherwise it'll be logical.
    kShr,
    // Slices out a subset of the bits-typed value on TOS2,
    // starting at index TOS1 with bitwidth at TOS0.
    // Note: the start index and the bitwidth should both be non-negative.
    kSlice,
    // Creates a new proc interpreter using the data in the optional data member
    // (as a `SpawnData`).
    kSpawn,
    // Stores the value at stack top into the arg-data-specified slot.
    kStore,
    // Subtracts the Nth value from the N-1th value on the stack.
    kUSub,
    kSSub,
    // Swaps TOS0 and TOS1 on the stack.
    kSwap,
    // Prints information about the given arguments to the terminal.
    kTraceFmt,
    // Prints a single operand argument to the terminal and acts as identity
    // function (i.e. places the operand value back on the stack).
    kTraceArg,
    // Slices out TOS0 bits of the array- or bits-typed value on TOS2,
    // starting at index TOS1.
    kWidthSlice,
    // Performs a bitwise XOR of the top two values on the stack.
    kXor,
  };

  // Indicates the amount by which the PC should be adjusted.
  // Used by kJumpRel and kJumpRelIf opcodes.
  XLS_DEFINE_STRONG_INT_TYPE(JumpTarget, int64_t);

  // Indicates the size of a data structure; used by kCreateArray and
  // kCreateTuple opcodes.
  XLS_DEFINE_STRONG_INT_TYPE(NumElements, int64_t);

  // Indicates the index into which to store or from which to load a value. Used
  // by kLoad and kStore opcodes.
  XLS_DEFINE_STRONG_INT_TYPE(SlotIndex, int64_t);

  // Data needed to resolve a potentially parametric Function invocation to
  // its concrete implementation.
  class InvocationData {
   public:
    InvocationData(const Invocation* invocation,
                   std::optional<ParametricEnv> caller_bindings,
                   std::optional<ParametricEnv> callee_bindings)
        : invocation_(invocation),
          caller_bindings_(std::move(caller_bindings)),
          callee_bindings_(std::move(callee_bindings)) {}

    const Invocation* invocation() const { return invocation_; }
    const std::optional<ParametricEnv>& caller_bindings() const {
      return caller_bindings_;
    }
    const std::optional<ParametricEnv>& callee_bindings() const {
      return callee_bindings_;
    }

   private:
    const Invocation* invocation_;
    std::optional<ParametricEnv> caller_bindings_;
    std::optional<ParametricEnv> callee_bindings_;
  };

  // Encapsulates an element in a MatchArm's NameDefTree. For literals, a
  // this is an InterpValue. For NameRefs, this is the associated SlotIndex. For
  // NameDefs (i.e., assignments to a name from the value to match), this is the
  // SlotIndex to which to store, and for wildcards, this is a simple "matches
  // anything" flag.
  class MatchArmItem {
   public:
    static MatchArmItem MakeInterpValue(const InterpValue& interp_value);
    static MatchArmItem MakeLoad(SlotIndex slot_index);
    static MatchArmItem MakeStore(SlotIndex slot_index);
    static MatchArmItem MakeRange(InterpValue start, InterpValue limit);
    static MatchArmItem MakeTuple(std::vector<MatchArmItem> elements);
    static MatchArmItem MakeWildcard();
    static MatchArmItem MakeRestOfTuple();

    enum class Kind : uint8_t {
      kInterpValue,
      kLoad,
      kRange,
      kStore,
      kTuple,
      kWildcard,
      kRestOfTuple,
    };

    struct RangeData {
      InterpValue start;
      InterpValue limit;
    };

    absl::StatusOr<InterpValue> interp_value() const;
    absl::StatusOr<SlotIndex> slot_index() const;
    absl::StatusOr<RangeData> range() const;
    absl::StatusOr<std::vector<MatchArmItem>> tuple_elements() const;
    Kind kind() const { return kind_; }

    std::string ToString() const;

   private:
    explicit MatchArmItem(Kind kind);
    MatchArmItem(Kind kind, std::variant<InterpValue, SlotIndex, RangeData,
                                         std::vector<MatchArmItem>>
                                data);

    Kind kind_;
    std::optional<std::variant<InterpValue, SlotIndex, RangeData,
                               std::vector<MatchArmItem>>>
        data_;
  };

  // Represents the functions that are provided by a `Spawn` node, to decouple
  // the `Spawn` node from the bytecode interpreter.
  struct SpawnFunctions {
    Invocation* config;
    Invocation* next;
  };

  // Information necessary to spawn a child proc. The data here is used to drive
  // evaluation and/or translation of the proc's component functions.
  // TODO(https://github.com/google/xls/issues/608): Reduce the AST surface
  // exposed here to just Functions.
  class SpawnData {
   public:
    SpawnData(SpawnFunctions spawn_functions, Proc* proc,
              InterpValue initial_state,
              std::optional<ParametricEnv> caller_bindings,
              std::optional<ParametricEnv> callee_bindings)
        : spawn_functions_(spawn_functions),
          proc_(proc),
          initial_state_(std::move(initial_state)),
          caller_bindings_(std::move(caller_bindings)),
          callee_bindings_(std::move(callee_bindings)) {}

    const SpawnFunctions& spawn_functions() const { return spawn_functions_; }
    Proc* proc() const { return proc_; }
    const InterpValue& initial_state() const { return initial_state_; }
    const std::optional<ParametricEnv>& caller_bindings() const {
      return caller_bindings_;
    }
    const std::optional<ParametricEnv>& callee_bindings() const {
      return callee_bindings_;
    }

   private:
    SpawnFunctions spawn_functions_;

    // The proc itself.
    Proc* proc_;

    // The initial state of the new proc.
    InterpValue initial_state_;

    // Note: can't store a pointer, since the underlying storage isn't
    // guaranteed to be stable.
    std::optional<ParametricEnv> caller_bindings_;
    std::optional<ParametricEnv> callee_bindings_;
  };

  // Information necessary to run a trace operation.
  class TraceData {
   public:
    TraceData(const TraceData& other) = delete;
    TraceData(TraceData&& other) = default;
    TraceData& operator=(TraceData&& other) = default;

    TraceData(std::vector<FormatStep> steps,
              std::vector<ValueFormatDescriptor> value_fmt_descs)
        : steps_(std::move(steps)),
          value_fmt_descs_(std::move(value_fmt_descs)) {}

    absl::Span<const FormatStep> steps() const { return steps_; }
    absl::Span<const ValueFormatDescriptor> value_fmt_descs() const {
      return value_fmt_descs_;
    }

   private:
    std::vector<FormatStep> steps_;

    // For default formatting of struct operands we hold metadata that allows us
    // to format them in more detail (struct name, fields, etc).
    std::vector<ValueFormatDescriptor> value_fmt_descs_;
  };

  // Information necessary for channel operations.
  class ChannelData {
   public:
    ChannelData(const ChannelData& other) = delete;
    ChannelData(ChannelData&& other) = default;
    ChannelData& operator=(ChannelData&& other) = default;

    ChannelData(std::string_view channel_name,
                std::unique_ptr<Type> payload_type,
                ValueFormatDescriptor value_fmt_desc)
        : channel_name_(channel_name),
          payload_type_(std::move(payload_type)),
          value_fmt_desc_(std::move(value_fmt_desc)) {}

    std::string_view channel_name() const { return channel_name_; }
    const Type& payload_type() const { return *payload_type_; }
    const ValueFormatDescriptor& value_fmt_desc() const {
      return value_fmt_desc_;
    }

   private:
    std::string channel_name_;
    std::unique_ptr<Type> payload_type_;
    ValueFormatDescriptor value_fmt_desc_;
  };

  using Data = std::variant<InterpValue, JumpTarget, NumElements, SlotIndex,
                            std::unique_ptr<Type>, InvocationData, MatchArmItem,
                            SpawnData, TraceData, ChannelData>;

  static Bytecode MakeDup(Span span);
  static Bytecode MakeIndex(Span span);
  static Bytecode MakeTupleIndex(Span span);
  static Bytecode MakeInvert(Span span);
  static Bytecode MakeJumpDest(Span span);
  static Bytecode MakeJumpRelIf(Span span, JumpTarget target);
  static Bytecode MakeJumpRel(Span span, JumpTarget target);
  static Bytecode MakeLiteral(
      Span span, InterpValue literal,
      std::optional<ValueFormatDescriptor> = std::nullopt);
  static Bytecode MakeLoad(Span span, SlotIndex slot_index);
  static Bytecode MakeLogicalOr(Span span);
  static Bytecode MakeMatchArm(Span span, MatchArmItem item);
  static Bytecode MakePop(Span span);
  static Bytecode MakeRecv(Span span, ChannelData channel_data);
  static Bytecode MakeRecvNonBlocking(Span span, ChannelData channel_data);
  static Bytecode MakeRange(Span span);
  static Bytecode MakeStore(Span span, SlotIndex slot_index);
  static Bytecode MakeSpawn(Span span, SpawnData spawn_data);
  static Bytecode MakeSwap(Span span);
  static Bytecode MakeSend(Span span, ChannelData channel_data);

  // TODO(rspringer): 2022-02-14: These constructors end up being pretty
  // verbose. Consider a builder?
  // Creates an operation w/o any accessory data. The span is present for
  // reporting error source location.
  Bytecode(Span source_span, Op op)
      : source_span_(std::move(source_span)), op_(op), data_(std::nullopt) {}

  // Creates an operation with associated string or InterpValue data.
  Bytecode(
      Span source_span, Op op, std::optional<Data> data,
      std::optional<ValueFormatDescriptor> format_descriptor = std::nullopt)
      : source_span_(std::move(source_span)),
        op_(op),
        data_(std::move(data)),
        format_descriptor_(std::move(format_descriptor)) {}

  // Not copyable, but move-able.
  Bytecode(const Bytecode& other) = delete;
  Bytecode(Bytecode&& other) = default;
  Bytecode& operator=(Bytecode&& other) = default;

  Span source_span() const { return source_span_; }
  Op op() const { return op_; }
  const std::optional<Data>& data() const { return data_; }

  bool has_data() const { return data_.has_value(); }

  absl::StatusOr<InvocationData> invocation_data() const;
  absl::StatusOr<JumpTarget> jump_target() const;
  absl::StatusOr<const MatchArmItem*> match_arm_item() const;
  absl::StatusOr<NumElements> num_elements() const;
  absl::StatusOr<SlotIndex> slot_index() const;
  absl::StatusOr<const SpawnData*> spawn_data() const;
  absl::StatusOr<const TraceData*> trace_data() const;
  absl::StatusOr<const ChannelData*> channel_data() const;
  absl::StatusOr<const Type*> type_data() const;
  absl::StatusOr<InterpValue> value_data() const;

  std::string ToString(const FileTable& file_table,
                       bool source_locs = false) const;

  // Value used as an integer data placeholder in jumps before their
  // target/amount has become known during bytecode emission.
  static constexpr JumpTarget kPlaceholderJumpAmount = JumpTarget(-1);

  // Used for patching up e.g. jump targets.
  //
  // That is, if you're going to jump forward over some code, but you don't know
  // how big that code is yet (in terms of bytecodes), you emit the jump with
  // the kPlaceholderJumpAmount and later, once the code to jump over has been
  // emitted, you go back and make it jump over the right (measured) amount.
  // All jump amounts are relative to the current PC.
  //
  // Note: kPlaceholderJumpAmount is used as a canonical placeholder for things
  // that should be patched.
  void PatchJumpTarget(int64_t value);

  const std::optional<ValueFormatDescriptor>& format_descriptor() const {
    return format_descriptor_;
  }

 private:
  Span source_span_;
  Op op_;
  std::optional<Data> data_;

  // For numbers and literals, this field carries the numeric format of original
  // text input. This enables better messages in assert_eq!, for example.
  std::optional<ValueFormatDescriptor> format_descriptor_;
};

std::string OpToString(Bytecode::Op op);

// Holds all the bytecode implementing a function along with useful metadata.
class BytecodeFunction {
 public:
  // Args:
  //  module: The function's containing module -- we need the function's
  //    containing module in order to get the root TypeInfo for top-level
  //    BytecodeFunctions.
  //  source_fn: may be nullptr for ephemeral functions, such as those created
  //    for realizing `match` ops.
  //
  // Note: this is an O(N) operation where N is the number of ops in the
  // bytecode.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> Create(
      const Module* owner, const Function* source_fn, const TypeInfo* type_info,
      std::vector<Bytecode> bytecode);

  const Module* owner() const { return owner_; }
  const Function* source_fn() const { return source_fn_; }
  const TypeInfo* type_info() const { return type_info_; }
  const std::vector<Bytecode>& bytecodes() const { return bytecodes_; }

  // Creates and returns a [caller-owned] copy of the internal bytecodes.
  std::vector<Bytecode> CloneBytecodes() const;

 private:
  BytecodeFunction(const Module* owner, const Function* source_fn,
                   const TypeInfo* type_info, std::vector<Bytecode> bytecode);

  const Module* owner_;
  const Function* source_fn_;
  const TypeInfo* type_info_;
  std::vector<Bytecode> bytecodes_;
};

// Converts the given sequence of bytecodes to a more human-readable string,
// source_locs indicating whether source locations are annotated on the bytecode
// lines.
std::string BytecodesToString(absl::Span<const Bytecode> bytecodes,
                              bool source_locs, const FileTable& file_table);

// Converts a string as given by BytecodesToString(..., /*source_locs=*/false)
// into a bytecode sequence; e.g. for testing.
absl::StatusOr<std::vector<Bytecode>> BytecodesFromString(
    std::string_view text);

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_H_

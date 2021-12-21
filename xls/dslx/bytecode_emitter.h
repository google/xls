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
#ifndef XLS_DSLX_BYTECODE_EMITTER_H_
#define XLS_DSLX_BYTECODE_EMITTER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Defines a single "instruction" for the DSLX bytecode interpreter: an opcode
// and optional accessory data (load/store value name, function call target).
class Bytecode {
 public:
  // In these descriptions, "TOS1" refers to the second-to-top-stack element and
  // "TOS0" refers to the top stack element.
  enum class Op {
    // Adds the top two values on the stack.
    kAdd,
    // Performs a bitwise AND of the top two values on the stack.
    kAnd,
    // Invokes the function given in the Bytecode's data argument.
    kCall,
    // Concatenates TOS1 and TOS0, with TOS1 comprising the most significant
    // bits of the result.
    kConcat,
    // Combines the top N values on the stack (N given in the data argument)
    // into an array.
    kCreateArray,
    // Creates an N-tuple (N given in the data argument) from the values on the
    // stack.
    kCreateTuple,
    // Divides the N-1th value on the stack by the Nth value.
    kDiv,
    // Expands the N-tuple on the top of the stack by one level, placing leading
    // elements at the top of the stack. In other words, expanding the tuple
    // `(a, (b, c))` will result in a stack of `(b, c), a`, where `a` is on top
    // of the stack.
    kExpandTuple,
    // Compares TOS1 to TOS0, storing true if TOS1 == TOS0.
    kEq,
    // Compares TOS1 to TOS0, storing true if TOS1 >= TOS0.
    kGe,
    // Compares TOS1 to TOS0, storing true if TOS1 > TOS0.
    kGt,
    // Inverts the bits of TOS0.
    kInvert,
    // Unconditional jump (relative).
    kJumpRel,
    // Pops the entry at the top of the stack and jumps (relative) if it is
    // true, otherwise PC proceeds as normal (i.e. PC = PC + 1).
    kJumpRelIf,
    // Indicates a jump destination PC for control flow integrity checking.
    // (Note that there's no actual execution logic for this opcode.)
    kJumpDest,
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
    // Multiplies the top two values on the stack.
    kMul,
    // Compares TOS1 to B, storing true if TOS1 != TOS0.
    kNe,
    // Performs two's complement negation of TOS0.
    kNegate,
    // Performs a bitwise OR of the top two values on the stack.
    kOr,
    // Performs a logical left shift of the second-to-top stack element by the
    // top element's number.
    kShll,
    // Performs a logical right shift of the second-to-top stack element by the
    // top element's number.
    kShrl,
    // Stores the value at stack top into the arg-data-specified slot.
    kStore,
    // Subtracts the Nth value from the N-1th value on the stack.
    kSub,
    // Performs a bitwise XOR of the top two values on the stack.
    kXor,
  };

  using Data = absl::variant<int64_t, InterpValue>;

  // Creates an operation w/o any accessory data. The span is present for
  // reporting error source location.
  Bytecode(Span source_span, Op op)
      : source_span_(source_span), op_(op), data_(absl::nullopt) {}

  // Creates an operation with associated string or InterpValue data.
  Bytecode(Span source_span, Op op, absl::optional<Data> data)
      : source_span_(source_span), op_(op), data_(data) {}

  Span source_span() const { return source_span_; }
  Op op() const { return op_; }
  absl::optional<Data> data() const { return data_; }

  bool has_data() const { return data_.has_value(); }

  absl::StatusOr<int64_t> integer_data() const {
    if (!has_data()) {
      return absl::InvalidArgumentError("Bytecode does not hold data.");
    }
    if (!absl::holds_alternative<int64_t>(data_.value())) {
      return absl::InvalidArgumentError("Bytecode data is not an integer.");
    }
    return absl::get<int64_t>(data_.value());
  }

  absl::StatusOr<InterpValue> value_data() const {
    if (!has_data()) {
      return absl::InvalidArgumentError("Bytecode does not hold data.");
    }
    if (!absl::holds_alternative<InterpValue>(data_.value())) {
      return absl::InvalidArgumentError("Bytecode data is not an InterpValue.");
    }
    return absl::get<InterpValue>(data_.value());
  }

  std::string ToString(bool source_locs = true) const;

  // Value used as an integer data placeholder in jumps before their
  // target/amount has become known during bytecode emission.
  static constexpr int64_t kPlaceholderJumpAmount = -1;

  // Used for patching up e.g. jump targets.
  //
  // That is, if you're going to jump forward over some code, but you don't know
  // how big that code is yet (in terms of bytecodes), you emit the jump with
  // the kPlaceholderJumpAmount and later, once the code to jump over has been
  // emitted, you go back and make it jump over the right (measured) amount.
  //
  // Note: kPlaceholderJumpAmount is used as a canonical placeholder for things
  // that should be patched.
  void Patch(int64_t value) {
    XLS_CHECK(data_.has_value());
    XLS_CHECK_EQ(absl::get<int64_t>(data_.value()), kPlaceholderJumpAmount);
    data_ = value;
  }

 private:
  Span source_span_;
  Op op_;
  absl::optional<Data> data_;
};

// Converts the given sequence of bytecodes to a more human-readable string,
// source_locs indicating whether source locations are annotated on the bytecode
// lines.
std::string BytecodesToString(absl::Span<const Bytecode> bytecodes,
                              bool source_locs);

// Converts a string as given by BytecodesToString(..., /*source_locs=*/false)
// into a bytecode sequence; e.g. for testing.
absl::StatusOr<std::vector<Bytecode>> BytecodesFromString(
    absl::string_view text);

// Translates a DSLX expression tree into a linear sequence of bytecode
// (bytecodes?).
// TODO(rspringer): Handle the rest of the Expr node types.
class BytecodeEmitter : public ExprVisitor {
 public:
  BytecodeEmitter(
      ImportData* import_data, TypeInfo* type_info,
      absl::flat_hash_map<const NameDef*, int64_t>* namedef_to_slot);
  ~BytecodeEmitter();
  absl::StatusOr<std::vector<Bytecode>> Emit(Function* f);

 private:
  void HandleArray(Array* node) override;
  void HandleAttr(Attr* node) override { DefaultHandler(node); }
  void HandleBinop(Binop* node) override;
  void HandleCarry(Carry* node) override { DefaultHandler(node); }
  void HandleCast(Cast* node) override { DefaultHandler(node); }
  void HandleChannelDecl(ChannelDecl* node) override { DefaultHandler(node); }
  void HandleColonRef(ColonRef* node) override { DefaultHandler(node); }
  void HandleConstRef(ConstRef* node) override { DefaultHandler(node); }
  void HandleFor(For* node) override { DefaultHandler(node); }
  void HandleFormatMacro(FormatMacro* node) override { DefaultHandler(node); }
  void HandleIndex(Index* node) override { DefaultHandler(node); }
  void HandleInvocation(Invocation* node) override;
  void HandleJoin(Join* node) override { DefaultHandler(node); }
  void HandleLet(Let* node) override;
  void HandleMatch(Match* node) override { DefaultHandler(node); }
  void HandleNameRef(NameRef* node) override;
  void HandleNumber(Number* node) override;
  void HandleRecv(Recv* node) override { DefaultHandler(node); }
  void HandleRecvIf(RecvIf* node) override { DefaultHandler(node); }
  void HandleSend(Send* node) override { DefaultHandler(node); }
  void HandleSendIf(SendIf* node) override { DefaultHandler(node); }
  void HandleSpawn(Spawn* node) override { DefaultHandler(node); }
  void HandleString(String* node) override { DefaultHandler(node); }
  void HandleStructInstance(StructInstance* node) override {
    DefaultHandler(node);
  }
  void HandleSplatStructInstance(SplatStructInstance* node) override {
    DefaultHandler(node);
  }
  void HandleTernary(Ternary* node) override;
  void HandleUnop(Unop* node) override;
  void HandleWhile(While* node) override { DefaultHandler(node); }
  void HandleXlsTuple(XlsTuple* node) override;

  void DefaultHandler(Expr* node) {
    status_ = absl::UnimplementedError(
        absl::StrFormat("Unhandled node kind: %s: %s", node->GetNodeTypeName(),
                        node->ToString()));
  }

  void DestructureLet(NameDefTree* tree);

  ImportData* import_data_;
  TypeInfo* type_info_;

  absl::Status status_;
  std::vector<Bytecode> bytecode_;
  absl::flat_hash_map<const NameDef*, int64_t>* namedef_to_slot_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_EMITTER_H_

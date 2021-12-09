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
  enum class Op {
    kAdd,
    kCall,
    kEq,
    kLoad,
    kLiteral,
    kStore,
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

 private:
  Span source_span_;
  Op op_;
  absl::optional<Data> data_;
};

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
  void HandleArray(Array* node) override { DefaultHandler(node); }
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
  void HandleTernary(Ternary* node) override { DefaultHandler(node); }
  void HandleUnop(Unop* node) override { DefaultHandler(node); }
  void HandleWhile(While* node) override { DefaultHandler(node); }
  void HandleXlsTuple(XlsTuple* node) override { DefaultHandler(node); }

  void DefaultHandler(Expr* node) {
    status_ = absl::UnimplementedError(
        absl::StrFormat("Unhandled node kind: %s: %s", node->GetNodeTypeName(),
                        node->ToString()));
  }

  ImportData* import_data_;
  TypeInfo* type_info_;

  absl::Status status_;
  std::vector<Bytecode> bytecode_;
  absl::flat_hash_map<const NameDef*, int64_t>* namedef_to_slot_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_EMITTER_H_

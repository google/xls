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
#include "xls/dslx/bytecode.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Translates a DSLX expression tree into a linear sequence of bytecode
// (bytecodes?).
// TODO(rspringer): Handle the rest of the Expr node types.
class BytecodeEmitter : public ExprVisitor {
 public:
  // `caller_bindings` contains the symbolic bindings associated with the
  // _caller_ of `f`, if any, and is used to determine the symbolic bindings for
  // `f` itself. It will be nullopt for non-parametric functions.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> Emit(
      ImportData* import_data, const TypeInfo* type_info, const Function* f,
      const std::optional<SymbolicBindings>& caller_bindings);

  // TODO(rspringer): 2022-03-16: I think we can delete `caller_bindings`.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitExpression(
      ImportData* import_data, const TypeInfo* type_info, const Expr* expr,
      const absl::flat_hash_map<std::string, InterpValue>& env,
      const std::optional<SymbolicBindings>& caller_bindings);

  // Emits a function, just as the above, but reserves the first N slots for
  // the given proc members.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitProcNext(
      ImportData* import_data, const TypeInfo* type_info, const Function* f,
      const std::optional<SymbolicBindings>& caller_bindings,
      const std::vector<NameDef*>& proc_members);

 private:
  BytecodeEmitter(ImportData* import_data, const TypeInfo* type_info,
                  const std::optional<SymbolicBindings>& caller_bindings);
  ~BytecodeEmitter();
  absl::Status Init(const Function* f);

  // Adds the given bytecode to the program.
  void Add(Bytecode bytecode) { bytecode_.push_back(std::move(bytecode)); }
  absl::Status HandleArray(const Array* node) override;
  absl::Status HandleAttr(const Attr* node) override;
  absl::Status HandleBinop(const Binop* node) override;
  absl::Status HandleBlock(const Block* node) override;
  absl::Status HandleCast(const Cast* node) override;
  absl::Status HandleChannelDecl(const ChannelDecl* node) override;
  absl::Status HandleColonRef(const ColonRef* node) override;
  absl::StatusOr<InterpValue> HandleColonRefInternal(const ColonRef* node);
  absl::Status HandleConstantArray(const ConstantArray* node) override;
  absl::Status HandleConstRef(const ConstRef* node) override;
  absl::Status HandleFor(const For* node) override;
  absl::Status HandleFormatMacro(const FormatMacro* node) override;
  absl::Status HandleIndex(const Index* node) override;
  absl::Status HandleInvocation(const Invocation* node) override;
  absl::Status HandleJoin(const Join* node) override;
  absl::Status HandleLet(const Let* node) override;
  absl::Status HandleMatch(const Match* node) override;
  absl::Status HandleNameRef(const NameRef* node) override;
  absl::StatusOr<std::variant<InterpValue, Bytecode::SlotIndex>>
  HandleNameRefInternal(const NameRef* node);
  absl::Status HandleNumber(const Number* node) override;
  absl::StatusOr<InterpValue> HandleNumberInternal(const Number* node);
  absl::Status HandleRange(const Range* node) override;
  absl::Status HandleRecv(const Recv* node) override;
  absl::Status HandleRecvIf(const RecvIf* node) override;
  absl::Status HandleRecvIfNonBlocking(const RecvIfNonBlocking* node) override;
  absl::Status HandleRecvNonBlocking(const RecvNonBlocking* node) override;
  absl::Status HandleSend(const Send* node) override;
  absl::Status HandleSendIf(const SendIf* node) override;
  absl::Status HandleSpawn(const Spawn* node) override;
  absl::Status HandleString(const String* node) override;
  absl::Status HandleStructInstance(const StructInstance* node) override;
  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* node) override;
  absl::Status HandleTernary(const Ternary* node) override;
  absl::Status HandleTupleIndex(const TupleIndex* node) override;
  absl::Status HandleUnop(const Unop* node) override;
  absl::Status HandleUnrollFor(const UnrollFor* node) override;
  absl::Status HandleXlsTuple(const XlsTuple* node) override;

  absl::Status CastArrayToBits(Span span, ArrayType* from_array,
                               BitsType* to_bits);
  absl::Status CastBitsToArray(Span span, BitsType* from_bits,
                               ArrayType* to_array);

  // Given a TypeDef, determines the EnumDef to which it refers.
  absl::StatusOr<EnumDef*> ResolveTypeDefToEnum(const TypeInfo* type_info,
                                                TypeDef* type_def);

  absl::StatusOr<InterpValue> HandleColonRefToEnum(const ColonRef* colon_ref,
                                                   EnumDef* enum_def,
                                                   const TypeInfo* type_info);
  absl::StatusOr<InterpValue> HandleColonRefToValue(Module* module,
                                                    const ColonRef* colon_ref);

  absl::StatusOr<Bytecode::MatchArmItem> HandleNameDefTreeExpr(
      NameDefTree* tree);

  void DestructureLet(NameDefTree* tree);

  ImportData* import_data_;
  const TypeInfo* type_info_;
  const std::optional<SymbolicBindings>& caller_bindings_;

  std::vector<Bytecode> bytecode_;
  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_EMITTER_H_

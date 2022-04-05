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
      const absl::optional<SymbolicBindings>& caller_bindings);

  // TODO(rspringer): 2022-03-16: I think we can delete `caller_bindings`.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitExpression(
      ImportData* import_data, const TypeInfo* type_info, const Expr* expr,
      const absl::flat_hash_map<std::string, InterpValue>& env,
      const absl::optional<SymbolicBindings>& caller_bindings);

  // Emits a function, just as the above, but reserves the first N slots for
  // the given proc members.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitProcNext(
      ImportData* import_data, const TypeInfo* type_info, const Function* f,
      const absl::optional<SymbolicBindings>& caller_bindings,
      const std::vector<NameDef*>& proc_members);

 private:
  BytecodeEmitter(ImportData* import_data, const TypeInfo* type_info,
                  const absl::optional<SymbolicBindings>& caller_bindings);
  ~BytecodeEmitter();
  absl::Status Init(const Function* f);

  // Adds the given bytecode to the program.
  void Add(Bytecode bytecode) { bytecode_.push_back(std::move(bytecode)); }
  void HandleArray(const Array* node) override;
  void HandleAttr(const Attr* node) override;
  void HandleBinop(const Binop* node) override;
  void HandleCast(const Cast* node) override;
  void HandleChannelDecl(const ChannelDecl* node) override;
  void HandleColonRef(const ColonRef* node) override;
  absl::StatusOr<InterpValue> HandleColonRefInternal(const ColonRef* node);
  void HandleConstRef(const ConstRef* node) override;
  void HandleFor(const For* node) override;
  void HandleFormatMacro(const FormatMacro* node) override;
  void HandleIndex(const Index* node) override;
  void HandleInvocation(const Invocation* node) override;
  void HandleJoin(const Join* node) override { DefaultHandler(node); }
  void HandleLet(const Let* node) override;
  void HandleMatch(const Match* node) override;
  void HandleNameRef(const NameRef* node) override;
  absl::StatusOr<absl::variant<InterpValue, Bytecode::SlotIndex>>
  HandleNameRefInternal(const NameRef* node);
  void HandleNumber(const Number* node) override;
  absl::StatusOr<InterpValue> HandleNumberInternal(const Number* node);
  void HandleRecv(const Recv* node) override;
  void HandleRecvIf(const RecvIf* node) override { DefaultHandler(node); }
  void HandleSend(const Send* node) override;
  void HandleSendIf(const SendIf* node) override { DefaultHandler(node); }
  void HandleSpawn(const Spawn* node) override { DefaultHandler(node); }
  void HandleString(const String* node) override;
  void HandleStructInstance(const StructInstance* node) override;
  void HandleSplatStructInstance(const SplatStructInstance* node) override;
  void HandleTernary(const Ternary* node) override;
  void HandleUnop(const Unop* node) override;
  void HandleXlsTuple(const XlsTuple* node) override;

  void DefaultHandler(const Expr* node) {
    status_ = absl::UnimplementedError(
        absl::StrFormat("Unhandled node kind: %s: %s", node->GetNodeTypeName(),
                        node->ToString()));
  }

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
  const absl::optional<SymbolicBindings>& caller_bindings_;

  absl::Status status_;
  std::vector<Bytecode> bytecode_;
  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_EMITTER_H_

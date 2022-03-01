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
      absl::optional<const SymbolicBindings*> caller_bindings);

 private:
  BytecodeEmitter(ImportData* import_data, const TypeInfo* type_info,
                  absl::optional<const SymbolicBindings*> caller_bindings);
  ~BytecodeEmitter();
  absl::Status Init(const Function* f);

  void HandleArray(Array* node) override;
  void HandleAttr(Attr* node) override;
  void HandleBinop(Binop* node) override;
  void HandleCarry(Carry* node) override { DefaultHandler(node); }
  void HandleCast(Cast* node) override;
  void HandleChannelDecl(ChannelDecl* node) override { DefaultHandler(node); }
  void HandleColonRef(ColonRef* node) override;
  void HandleConstRef(ConstRef* node) override;
  void HandleFor(For* node) override;
  void HandleFormatMacro(FormatMacro* node) override;
  void HandleIndex(Index* node) override;
  void HandleInvocation(Invocation* node) override;
  void HandleJoin(Join* node) override { DefaultHandler(node); }
  void HandleLet(Let* node) override;
  void HandleMatch(Match* node) override;
  void HandleNameRef(NameRef* node) override;
  void HandleNumber(Number* node) override;
  void HandleRecv(Recv* node) override { DefaultHandler(node); }
  void HandleRecvIf(RecvIf* node) override { DefaultHandler(node); }
  void HandleSend(Send* node) override { DefaultHandler(node); }
  void HandleSendIf(SendIf* node) override { DefaultHandler(node); }
  void HandleSpawn(Spawn* node) override { DefaultHandler(node); }
  void HandleString(String* node) override;
  void HandleStructInstance(StructInstance* node) override;
  void HandleSplatStructInstance(SplatStructInstance* node) override;
  void HandleTernary(Ternary* node) override;
  void HandleUnop(Unop* node) override;
  void HandleWhile(While* node) override { DefaultHandler(node); }
  void HandleXlsTuple(XlsTuple* node) override;

  void DefaultHandler(Expr* node) {
    status_ = absl::UnimplementedError(
        absl::StrFormat("Unhandled node kind: %s: %s", node->GetNodeTypeName(),
                        node->ToString()));
  }

  absl::Status CastArrayToBits(Span span, ArrayType* from_array,
                               BitsType* to_bits);
  absl::Status CastBitsToArray(Span span, BitsType* from_bits,
                               ArrayType* to_array);

  // Finds either the Module or EnumDef that a ColonRef ultimately refers to.
  absl::StatusOr<absl::variant<Module*, EnumDef*>> ResolveColonRefSubject(
      ColonRef* node);

  // Given a TypeDef, determines the EnumDef to which it refers.
  absl::StatusOr<EnumDef*> ResolveTypeDefToEnum(Module* module,
                                                TypeDef* type_def);

  absl::StatusOr<Bytecode> HandleColonRefToEnum(ColonRef* colon_ref,
                                                EnumDef* enum_def,
                                                TypeInfo* type_info);
  absl::StatusOr<Bytecode> HandleColonRefToValue(Module* module,
                                                 ColonRef* colon_ref);

  void HandleNameDefTreeExpr(NameDefTree* tree);

  void DestructureLet(NameDefTree* tree);

  ImportData* import_data_;
  const TypeInfo* type_info_;
  absl::optional<const SymbolicBindings*> caller_bindings_;

  absl::Status status_;
  std::vector<Bytecode> bytecode_;
  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_EMITTER_H_

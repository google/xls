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
#ifndef XLS_DSLX_BYTECODE_BYTECODE_EMITTER_H_
#define XLS_DSLX_BYTECODE_BYTECODE_EMITTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

struct BytecodeEmitterOptions {
  // The format preference to use when one is not otherwise specified.
  FormatPreference format_preference;
};

// Translates a DSLX expression tree into a linear sequence of bytecodes.
class BytecodeEmitter : public ExprVisitor {
 public:
  // `caller_bindings` contains the symbolic bindings associated with the
  // _caller_ of `f`, if any, and is used to determine the symbolic bindings for
  // `f` itself. It will be std::nullopt for non-parametric functions.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> Emit(
      ImportData* import_data, const TypeInfo* type_info, const Function& f,
      const std::optional<ParametricEnv>& caller_bindings,
      const BytecodeEmitterOptions& options = BytecodeEmitterOptions());

  // TODO(rspringer): 2022-03-16: I think we can delete `caller_bindings`.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitExpression(
      ImportData* import_data, const TypeInfo* type_info, const Expr* expr,
      const absl::flat_hash_map<std::string, InterpValue>& env,
      const std::optional<ParametricEnv>& caller_bindings,
      const BytecodeEmitterOptions& options = BytecodeEmitterOptions());

  // Emits a function, just as the above, but reserves the first N slots for
  // the given proc members.
  static absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitProcNext(
      ImportData* import_data, const TypeInfo* type_info, const Function& f,
      const std::optional<ParametricEnv>& caller_bindings,
      const std::vector<NameDef*>& proc_members,
      const BytecodeEmitterOptions& options = BytecodeEmitterOptions());

 private:
  BytecodeEmitter(ImportData* import_data, const TypeInfo* type_info,
                  const std::optional<ParametricEnv>& caller_bindings,
                  const BytecodeEmitterOptions& options);
  ~BytecodeEmitter() override;

  // Initializes namedef-to-slot mapping.
  absl::Status Init(const Function& f);

  // Precondition: node must be Bits typed.
  absl::StatusOr<bool> IsBitsTypeNodeSigned(const AstNode* node) const;

  // Adds the given bytecode to the program.
  void Add(Bytecode bytecode) { bytecode_.push_back(std::move(bytecode)); }
  absl::Status HandleArray(const Array* node) override;
  absl::Status HandleAttr(const Attr* node) override;
  absl::Status HandleBinop(const Binop* node) override;
  absl::Status HandleCast(const Cast* node) override;
  absl::Status HandleChannelDecl(const ChannelDecl* node) override;
  absl::Status HandleColonRef(const ColonRef* node) override;
  absl::StatusOr<InterpValue> HandleColonRefInternal(const ColonRef* node);
  absl::Status HandleConstAssert(const ConstAssert* node) override;
  absl::Status HandleConstantArray(const ConstantArray* node) override;
  absl::Status HandleConstRef(const ConstRef* node) override;
  absl::Status HandleFor(const For* node) override;
  absl::Status HandleFormatMacro(const FormatMacro* node) override;
  absl::Status HandleZeroMacro(const ZeroMacro* node) override;
  absl::Status HandleAllOnesMacro(const AllOnesMacro* node) override;
  absl::Status HandleIndex(const Index* node) override;
  absl::Status HandleInvocation(const Invocation* node) override;
  absl::Status HandleCastImpl(const Invocation* node);
  absl::Status HandleLet(const Let* node) override;
  absl::Status HandleMatch(const Match* node) override;
  absl::Status HandleNameRef(const NameRef* node) override;
  absl::StatusOr<std::variant<InterpValue, Bytecode::SlotIndex>>
  HandleNameRefInternal(const NameRef* node);
  absl::Status HandleNumber(const Number* node) override;
  struct FormattedInterpValue {
    InterpValue value;
    std::optional<ValueFormatDescriptor> format_descriptor;
  };
  absl::StatusOr<FormattedInterpValue> HandleNumberInternal(const Number* node);
  absl::Status HandleRange(const Range* node) override;
  absl::Status HandleSpawn(const Spawn* node) override;
  absl::Status HandleString(const String* node) override;
  absl::Status HandleStructInstance(const StructInstance* node) override;
  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* node) override;
  absl::Status HandleConditional(const Conditional* node) override;
  absl::Status HandleStatementBlock(const StatementBlock* node) override;
  absl::Status HandleTupleIndex(const TupleIndex* node) override;
  absl::Status HandleUnop(const Unop* node) override;
  absl::Status HandleUnrollFor(const UnrollFor* node) override;
  absl::Status HandleXlsTuple(const XlsTuple* node) override;

  absl::Status HandleBuiltinDecode(const Invocation* node);
  absl::Status HandleBuiltinCheckedCast(const Invocation* node);
  absl::Status HandleBuiltinWideningCast(const Invocation* node);
  absl::Status HandleBuiltinSend(const Invocation* node);
  absl::Status HandleBuiltinSendIf(const Invocation* node);
  absl::Status HandleBuiltinRecv(const Invocation* node);
  absl::Status HandleBuiltinRecvIf(const Invocation* node);
  absl::Status HandleBuiltinRecvNonBlocking(const Invocation* node);
  absl::Status HandleBuiltinRecvIfNonBlocking(const Invocation* node);
  absl::Status HandleBuiltinJoin(const Invocation* node);
  absl::Status HandleBuiltinToken(const Invocation* node);

  absl::StatusOr<InterpValue> HandleColonRefToEnum(const ColonRef* colon_ref,
                                                   EnumDef* enum_def,
                                                   const TypeInfo* type_info);
  absl::StatusOr<InterpValue> HandleColonRefToValue(Module* module,
                                                    const ColonRef* colon_ref);

  absl::StatusOr<Bytecode::MatchArmItem> HandleNameDefTreeExpr(
      NameDefTree* tree, Type* type = nullptr);

  absl::Status DestructureLet(NameDefTree* tree,
                              std::variant<Type*, int64_t> type_or_size);

  const FileTable& file_table() const { return import_data_->file_table(); }

  ImportData* import_data_;
  const TypeInfo* type_info_;
  const std::optional<ParametricEnv>& caller_bindings_;
  BytecodeEmitterOptions options_;

  std::vector<Bytecode> bytecode_;
  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot_;
  int64_t next_slotno_ = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_EMITTER_H_

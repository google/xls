// Copyright 2024 The XLS Authors
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

#ifndef XLS_CODEGEN_VAST__DSLX_BUILDER_H_
#define XLS_CODEGEN_VAST__DSLX_BUILDER_H_

#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls {

absl::StatusOr<dslx::InterpValue> InterpretExpr(dslx::ImportData& import_data,
                                                dslx::TypeInfo& type_info,
                                                dslx::Expr* expr);

// Helper class for building a DSLX translation in `TranslateVastToDslx`. This
// attempts to separate most of the DSLX-related logic from AST traversal.
class DslxBuilder {
 public:
  DslxBuilder(dslx::Module& module, std::string_view dslx_stdlib_path,
              absl::flat_hash_map<verilog::Expression*, verilog::DataType*>
                  vast_type_map,
              dslx::WarningCollector& warnings);

  dslx::NameDef* MakeNameDef(const dslx::Span& span, std::string_view name);

  absl::StatusOr<dslx::NameRef*> MakeNameRef(const dslx::Span& span,
                                             std::string_view name);

  // Creates a name ref with a cast, if necessary, to the equivalent of the
  // inferred VAST type in the type map.
  absl::StatusOr<dslx::Expr*> MakeNameRefAndMaybeCast(
      verilog::Expression* vast_expr, const dslx::Span& span,
      std::string_view name);

  // Registers a VAST typedef, and what it maps to in DSLX, for later lookup via
  // `FindTypedef`.
  void AddTypedef(verilog::Typedef* type_def, dslx::TypeDefinition dslx_type);

  // Finds a previously-registered typedef.
  absl::StatusOr<dslx::TypeDefinition> FindTypedef(
      verilog::TypedefType* typedef_type);

  // Finds a previously-registered Verilog typedef for an enum.
  absl::StatusOr<verilog::Typedef*> ReverseEnumTypedef(verilog::Enum* enum_def);

  // Handles the work of importing a module into the current module, for cases
  // where SV built-in functions need to dispatch to DSLX functions.
  absl::StatusOr<dslx::Import*> GetOrImportModule(
      const dslx::ImportTokens& import_tokens);

  // Returns `expr` casted, if necessary, to the equivalent of the inferred VAST
  // type for `vast_expr` in the type map. Pass `true` for
  // `cast_enum_to_builtin` in rare contexts where Verilog allows an enum and
  // DSLX doesn't  (e.g. concat operands).
  absl::StatusOr<dslx::Expr*> MaybeCastToInferredVastType(
      verilog::Expression* vast_expr, dslx::Expr* expr,
      bool cast_enum_to_builtin = false);

  // Returns `expr` casted, if necessary, to the equivalent of the specified
  // `vast_type`. If `cast_enum_to_builtin` is true, then the corresponding DSLX
  // built-in type will be used for any VAST enum type.
  absl::StatusOr<dslx::Expr*> MaybeCast(verilog::DataType* vast_type,
                                        dslx::Expr* expr,
                                        bool cast_enum_to_builtin = false);

  // Converts an integer or array `vast_type` into the most appropriate DSLX
  // type s32/u32, sN/uN, etc.
  dslx::TypeAnnotation* VastTypeToDslxTypeForCast(const dslx::Span& span,
                                                  verilog::DataType* vast_type,
                                                  bool force_builtin = false);

  dslx::Unop* HandleUnaryOperator(const dslx::Span& span,
                                  dslx::UnopKind unop_kind, dslx::Expr* arg);

  absl::StatusOr<dslx::Expr*> HandleIntegerExponentiation(
      const dslx::Span& span, dslx::Expr* lhs, dslx::Expr* rhs);

  dslx::Number* HandleConstVal(const dslx::Span& span, const Bits& bits,
                               FormatPreference format_preference,
                               verilog::DataType* vast_type,
                               dslx::TypeAnnotation* force_dslx_type);

  absl::StatusOr<dslx::Expr*> ConvertMaxToWidth(verilog::Expression* vast_value,
                                                dslx::Expr* dslx_value);

  absl::StatusOr<dslx::ConstantDef*> HandleConstantDecl(
      const dslx::Span& span, verilog::Module* module,
      verilog::Parameter* parameter, std::string_view name, dslx::Expr* expr);

  void SetRefTargetModule(verilog::VastNode* target, verilog::Module* module) {
    ref_target_to_module_.emplace(target, module);
  }

  absl::StatusOr<verilog::Module*> FindRefTargetModule(
      verilog::VastNode* target) const;

  // Returns the inferred type for `expr` from the type map.
  absl::StatusOr<verilog::DataType*> GetVastDataType(
      verilog::Expression* expr) const;

  // Returns the final, formatted DSLX.
  absl::StatusOr<std::string> FormatModule();

  dslx::ImportData& import_data() { return import_data_; }
  dslx::DeduceCtx& deduce_ctx() { return deduce_ctx_; }
  dslx::TypeInfo& type_info() { return *type_info_; }
  dslx::Module& module() { return module_; }

 private:
  // Returns `expr` casted, whether necessary or not, to the equivalent of the
  // specified `vast_type`.
  absl::StatusOr<dslx::Expr*> Cast(verilog::DataType* vast_type,
                                   dslx::Expr* expr,
                                   bool force_builtin = false);

  std::optional<std::string> GenerateSizeCommentIfNotObvious(
      verilog::DataType* data_type, bool compute_size_if_struct);

  dslx::Module& module_;
  const std::string dslx_stdlib_path_;
  dslx::ImportData import_data_;
  dslx::WarningCollector warnings_;
  dslx::TypeInfo* type_info_;

  dslx::DeduceCtx deduce_ctx_;
  dslx::InterpBindings bindings_;

  const absl::flat_hash_map<verilog::Expression*, verilog::DataType*>
      vast_type_map_;
  absl::flat_hash_map<std::string, dslx::NameDef*> name_to_namedef_;
  absl::flat_hash_map<std::string, dslx::TypeDefinition>
      typedefs_by_loc_string_;
  absl::flat_hash_map<verilog::DataType*, verilog::Typedef*> reverse_typedefs_;
  absl::flat_hash_map<verilog::VastNode*, verilog::Module*>
      ref_target_to_module_;

  // Comments describing the sizes of types and values of constants are
  // generated here while building the DSLX AST, and actually applied to the
  // DSLX when FormatModule() is invoked at the end, because the AST doesn't
  // store comments.
  absl::flat_hash_map<std::string, std::string> type_def_comments_;
  absl::flat_hash_map<std::string, std::string> constant_def_comments_;
  // The outer map is the struct typedef name, and the inner key is the member
  // name.
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::string>>
      struct_member_comments_;
};

}  // namespace xls

#endif  // XLS_CODEGEN_VAST__DSLX_BUILDER_H_

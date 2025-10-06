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

#include "xls/codegen/vast/dslx_builder.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/codegen/vast/dslx_type_fixer.h"
#include "xls/codegen/vast/fold_vast_constants.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/type_inference_error_handler.h"
#include "xls/dslx/type_system_v2/typecheck_module_v2.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

std::string GetTypeDefName(const dslx::TypeDefinition& type_def) {
  return absl::visit(Visitor{
                         [&](const dslx::ColonRef* n) { return n->ToString(); },
                         [&](const dslx::UseTreeEntry* n) {
                           return n->GetLeafNameDef().value()->identifier();
                         },
                         [&](const auto* n) { return n->identifier(); },
                     },
                     type_def);
}

dslx::CommentData CommentAfter(const dslx::Span& span,
                               const std::string& comment) {
  dslx::Span comment_span(
      span.limit().BumpCol(),
      dslx::Pos(span.fileno(), span.limit().lineno() + 1, 0));
  return dslx::CommentData{.span = comment_span, .text = comment};
}

dslx::CommentData CommentAfter(const dslx::AstNode* node,
                               const std::string& comment) {
  std::optional<dslx::Span> span = node->GetSpan();
  QCHECK(span.has_value());
  return CommentAfter(*span, comment);
}

dslx::CommentData CommentAtBeginning(const dslx::AstNode* node,
                                     const std::string& comment) {
  std::optional<dslx::Span> span = node->GetSpan();
  QCHECK(span.has_value());
  dslx::Span comment_span(
      span->start(), dslx::Pos(span->fileno(), span->start().lineno() + 1, 0));
  return dslx::CommentData{.span = comment_span, .text = comment};
}

dslx::TypeInfo* GetTypeInfoOrDie(dslx::ImportData& import_data,
                                 dslx::Module* module) {
  absl::StatusOr<dslx::TypeInfo*> result =
      import_data.type_info_owner().New(module);
  CHECK_OK(result);
  return *result;
}

std::vector<std::filesystem::path> WrapOptionalPathInVector(
    const std::optional<std::filesystem::path>& path) {
  if (path.has_value()) {
    return {*path};
  }
  return {};
}

}  // namespace

absl::StatusOr<dslx::InterpValue> InterpretExpr(dslx::ImportData& import_data,
                                                dslx::TypeInfo& type_info,
                                                dslx::Expr* expr) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::BytecodeFunction> bf,
                       dslx::BytecodeEmitter::EmitExpression(
                           &import_data, &type_info, expr, /*env=*/{},
                           /*caller_bindings=*/std::nullopt));
  return dslx::BytecodeInterpreter::Interpret(&import_data, bf.get(),
                                              /*args=*/{});
}

std::string DslxResolver::GetNamespacedName(
    dslx::Module& module, std::string_view name,
    std::optional<verilog::Module*> vast_module) {
  std::string vast_module_name = module.name();
  if (vast_module.has_value()) {
    vast_module_name = (*vast_module)->name();
  }
  return vast_module_name == main_module_name_
             ? std::string(name)
             : absl::StrCat(vast_module_name, "_", name);
}

dslx::NameDef* DslxResolver::MakeNameDef(
    DslxBuilder& builder, const dslx::Span& span, std::string_view name,
    std::optional<verilog::VastNode*> vast_node,
    std::optional<verilog::Module*> vast_module) {
  const std::string namespaced_name =
      vast_node.has_value() &&
              dynamic_cast<verilog::EnumMember*>(*vast_node) != nullptr
          ? GetNamespacedName(builder.module(), name)
          : GetNamespacedName(builder.module(), name, vast_module);
  const std::string name_in_dslx_code =
      generate_combined_dslx_module_ ? namespaced_name : std::string(name);
  VLOG(3) << "MakeNameDef; span: " << span.ToString(builder.file_table())
          << " name: `" << namespaced_name << "`";
  auto* name_def = builder.module().Make<dslx::NameDef>(span, name_in_dslx_code,
                                                        /*definer=*/nullptr);
  namespaced_name_to_namedef_.emplace(namespaced_name, name_def);
  std::optional<std::string> loc_string;
  if (vast_node.has_value() && vast_module.has_value()) {
    loc_string = loc_string = (*vast_node)->loc().ToString();
    defining_modules_by_loc_string_.emplace(*loc_string, *vast_module);
  }
  return name_def;
}

absl::StatusOr<dslx::Expr*> DslxResolver::MakeNameRef(
    DslxBuilder& builder, const dslx::Span& span, std::string_view name,
    verilog::VastNode* target) {
  std::optional<verilog::Module*> vast_module;
  const auto module_it =
      defining_modules_by_loc_string_.find(target->loc().ToString());
  if (module_it != defining_modules_by_loc_string_.end()) {
    vast_module = module_it->second;
  }
  const std::string namespaced_name =
      GetNamespacedName(builder.module(), name, vast_module);
  const std::string name_in_dslx_code =
      generate_combined_dslx_module_ ? namespaced_name : std::string(name);
  VLOG(3) << "MakeNameRef; span: " << span.ToString(builder.file_table())
          << " name: `" << name_in_dslx_code << "`" << " vast module: "
          << (vast_module.has_value() ? (*vast_module)->name() : "none");
  const auto it = namespaced_name_to_namedef_.find(namespaced_name);
  if (it == namespaced_name_to_namedef_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Reference to undefined name: ", name_in_dslx_code));
  }
  if (generate_combined_dslx_module_ || !vast_module.has_value() ||
      (*vast_module)->name() == builder.module().name()) {
    return builder.module().Make<dslx::NameRef>(span, name_in_dslx_code,
                                                it->second);
  }
  return builder.CreateColonRef(span, (*vast_module)->name(),
                                name_in_dslx_code);
}

absl::StatusOr<dslx::ColonRef*> DslxBuilder::CreateColonRef(
    const dslx::Span& span, std::string_view module_name,
    std::string_view name) {
  XLS_ASSIGN_OR_RETURN(
      dslx::Import * import,
      GetOrImportModule(dslx::ImportTokens({std::string(module_name)})));
  auto* module_ref = module().Make<dslx::NameRef>(
      span, std::string(module_name), &import->name_def());
  return module().Make<dslx::ColonRef>(span, module_ref, std::string(name));
}

dslx::ColonRef::Subject DslxResolver::NameRefToColonRefSubject(
    dslx::Expr* ref) {
  if (auto* name_ref = dynamic_cast<dslx::NameRef*>(ref); name_ref) {
    return name_ref;
  }
  return down_cast<dslx::ColonRef*>(ref);
}

void DslxResolver::AddTypedef(dslx::Module& module, verilog::Module* definer,
                              verilog::Typedef* type_def,
                              dslx::TypeDefinition dslx_type) {
  // Note that we use the loc as a key because it's resilient to the creation
  // of derivative `DataType` objects by VAST type inference, e.g. during
  // constant folding. This way we can look up a DSLX typedef by either the
  // original `verilog::Typedef` object or one that is an artifact of type
  // inference.
  const std::string loc_string = type_def->loc().ToString();
  typedefs_by_loc_string_.emplace(loc_string, dslx_type);
  defining_modules_by_loc_string_.emplace(loc_string, definer);
  if (dynamic_cast<verilog::Enum*>(type_def->data_type())) {
    reverse_typedefs_.emplace(type_def->data_type(), type_def);
  }
}

absl::StatusOr<dslx::TypeDefinition> DslxResolver::FindTypedef(
    DslxBuilder& builder, verilog::TypedefType* typedef_type) {
  const std::string loc_string = typedef_type->type_def()->loc().ToString();
  const auto it = typedefs_by_loc_string_.find(loc_string);
  if (it == typedefs_by_loc_string_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Typedef ", typedef_type->type_def()->GetName(), " at ",
                     typedef_type->loc().ToString(),
                     " has not been associated with a DSLX type."));
  }
  if (!generate_combined_dslx_module_) {
    const auto definer_it = defining_modules_by_loc_string_.find(loc_string);
    QCHECK(definer_it != defining_modules_by_loc_string_.end());
    if (definer_it->second->name() != builder.module().name()) {
      VLOG(3) << "Found external typedef "
              << typedef_type->type_def()->GetName();
      return builder.CreateColonRef(dslx::Span(), definer_it->second->name(),
                                    typedef_type->type_def()->GetName());
    }
  }
  VLOG(3) << "Found internal typedef " << typedef_type->type_def()->GetName();
  return it->second;
}

absl::StatusOr<verilog::Typedef*> DslxResolver::ReverseEnumTypedef(
    verilog::Enum* enum_def) {
  const auto it = reverse_typedefs_.find(enum_def);
  if (it == reverse_typedefs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Data type ", enum_def->Emit(nullptr),
                     " has not been associated with a typedef."));
  }
  return it->second;
}

DslxBuilder::DslxBuilder(
    std::string_view main_module_name, DslxResolver* resolver,
    const std::optional<std::filesystem::path>& additional_search_path,
    std::string_view dslx_stdlib_path,
    const absl::flat_hash_map<verilog::Expression*, verilog::DataType*>&
        vast_type_map,
    dslx::WarningCollector& warnings)
    : additional_search_paths_(
          WrapOptionalPathInVector(additional_search_path)),
      dslx_stdlib_path_(dslx_stdlib_path),
      import_data_(
          dslx::CreateImportData(dslx_stdlib_path_, additional_search_paths_,
                                 /*enabled_warnings=*/dslx::kDefaultWarningsSet,
                                 std::make_unique<dslx::RealFilesystem>())),
      module_(std::string(main_module_name), /*fs_path=*/std::nullopt,
              import_data_.file_table()),
      resolver_(resolver),
      warnings_(warnings),
      type_info_(GetTypeInfoOrDie(import_data_, &module_)),
      vast_type_map_(vast_type_map) {}

absl::StatusOr<dslx::Expr*> DslxBuilder::MakeNameRefAndCast(
    verilog::Expression* expr, const dslx::Span& span, std::string_view name,
    verilog::VastNode* target) {
  XLS_ASSIGN_OR_RETURN(dslx::Expr * ref,
                       resolver_->MakeNameRef(*this, span, name, target));
  return CastToInferredVastType(expr, ref);
}

void DslxBuilder::AddTypedef(verilog::Module* definer,
                             verilog::Typedef* type_def,
                             dslx::TypeDefinition dslx_type) {
  resolver_->AddTypedef(module(), definer, type_def, dslx_type);
  std::optional<std::string> comment = GenerateSizeCommentIfNotObvious(
      type_def->data_type(), /*compute_size_if_struct=*/true);
  if (comment.has_value()) {
    const std::string type_def_name = GetTypeDefName(dslx_type);
    type_def_comments_.emplace(type_def_name, *comment);
    if (auto* struct_def =
            dynamic_cast<verilog::Struct*>(type_def->data_type());
        struct_def != nullptr) {
      absl::flat_hash_map<std::string, std::string> member_comments;
      for (const verilog::Def* def : struct_def->members()) {
        std::optional<std::string> member_comment =
            GenerateSizeCommentIfNotObvious(def->data_type(),
                                            /*compute_size_if_struct=*/false);
        if (member_comment.has_value()) {
          member_comments.emplace(def->GetName(), *member_comment);
        }
      }
      if (!member_comments.empty()) {
        struct_member_comments_.emplace(type_def_name,
                                        std::move(member_comments));
      }
    }
  }
}

absl::StatusOr<dslx::ConstantDef*> DslxBuilder::HandleConstantDecl(
    const dslx::Span& span, verilog::Module* vast_module,
    verilog::Parameter* parameter, std::string_view name, dslx::Expr* expr) {
  auto* name_def =
      resolver_->MakeNameDef(*this, span, name, parameter, vast_module);

  dslx::TypeAnnotation* type_annotation = nullptr;
  if (parameter->def() && parameter->def()->data_type() &&
      parameter->def()->data_type()->IsUserDefined()) {
    XLS_ASSIGN_OR_RETURN(
        type_annotation,
        VastTypeToDslxTypeForCast(dslx::Span(), parameter->def()->data_type()));
  }

  auto* constant_def =
      module().Make<dslx::ConstantDef>(span, name_def, type_annotation, expr,
                                       /*is_public=*/true);
  name_def->set_definer(constant_def);

  XLS_RETURN_IF_ERROR(
      module().AddTop(constant_def, /*make_collision_error=*/nullptr));

  // Add a comment with the value, if it is not obvious and can be folded.
  if (parameter->rhs() != nullptr && !parameter->rhs()->IsLiteral()) {
    absl::StatusOr<int64_t> folded_value =
        verilog::FoldEntireVastExpr(parameter->rhs(), vast_type_map_);
    absl::StatusOr<verilog::DataType*> folded_type =
        GetVastDataType(parameter->rhs());
    if (folded_value.ok() && folded_type.ok()) {
      absl::StatusOr<dslx::TypeAnnotation*> type =
          VastTypeToDslxTypeForCast(dslx::Span(), *folded_type,
                                    /*force_builtin=*/true);
      // This can't fail because it's pre-folded.
      QCHECK_OK(type);
      const std::string type_str = (*type)->ToString();
      constant_def_comments_.emplace(
          name_def->identifier(),
          *folded_value >= 0x100000
              ? absl::StrFormat(" %s:0x%x", type_str, *folded_value)
              : absl::StrFormat(" %s:%d", type_str, *folded_value));
    }
  }
  return constant_def;
}

absl::StatusOr<dslx::Number*> DslxBuilder::HandleConstVal(
    const dslx::Span& span, const Bits& bits,
    FormatPreference format_preference, verilog::DataType* vast_type,
    dslx::TypeAnnotation* dslx_type) {
  if (dslx_type == nullptr) {
    XLS_ASSIGN_OR_RETURN(dslx_type, VastTypeToDslxTypeForCast(span, vast_type));
  }
  return module().Make<dslx::Number>(span,
                                     BitsToString(bits, format_preference),
                                     dslx::NumberKind::kOther, dslx_type);
}

absl::StatusOr<dslx::Expr*> DslxBuilder::ConvertMaxToWidth(
    verilog::Expression* vast_value, dslx::Expr* dslx_value) {
  dslx::BuiltinType builtin_type = dslx::BuiltinType::kU32;
  const dslx::Span& span = dslx_value->span();
  dslx::BuiltinTypeAnnotation* type_annot =
      module().Make<dslx::BuiltinTypeAnnotation>(
          span, builtin_type, module().GetOrCreateBuiltinNameDef(builtin_type));
  verilog::DataType* unsigned_int_type =
      vast_value->file()->Make<verilog::IntegerType>(SourceInfo(),
                                                     /*signed=*/false);
  // If we would be producing `foo - 1 + 1`, just extract the `foo`.
  auto* binop = dynamic_cast<dslx::Binop*>(dslx_value);
  if (binop == nullptr) {
    auto* cast = dynamic_cast<dslx::Cast*>(dslx_value);
    if (cast != nullptr) {
      binop = dynamic_cast<dslx::Binop*>(cast->expr());
    }
  }
  if (binop != nullptr && binop->binop_kind() == dslx::BinopKind::kSub) {
    auto* rhs_number = dynamic_cast<dslx::Number*>(binop->rhs());
    if (rhs_number != nullptr) {
      absl::StatusOr<uint64_t> rhs_value =
          rhs_number->GetAsUint64(file_table());
      if (rhs_value.ok() && *rhs_value == 1) {
        return Cast(unsigned_int_type, binop->lhs());
      }
    }
  }
  // Add 1 if really necessary.
  XLS_ASSIGN_OR_RETURN(dslx::Expr * casted_max,
                       Cast(unsigned_int_type, dslx_value));
  dslx::Number* one = module().Make<dslx::Number>(
      span, "1", dslx::NumberKind::kOther, type_annot);
  return module().Make<dslx::Binop>(span, dslx::BinopKind::kAdd, casted_max,
                                    one, span);
}

dslx::Unop* DslxBuilder::HandleUnaryOperator(const dslx::Span& span,
                                             dslx::UnopKind unop_kind,
                                             dslx::Expr* arg) {
  // Note it uses the same span for the whole node and the operand;
  // the only time the operand span is used is for formatting, and
  // this node won't be used for formatting.
  return module().Make<dslx::Unop>(span, dslx::UnopKind::kNegate, arg, span);
}

absl::StatusOr<dslx::Expr*> DslxBuilder::HandleIntegerExponentiation(
    const dslx::Span& span, dslx::Expr* lhs, dslx::Expr* rhs,
    verilog::Expression* vast_rhs) {
  XLS_ASSIGN_OR_RETURN(int64_t rhs_value,
                       verilog::FoldEntireVastExpr(vast_rhs, vast_type_map_));
  if (rhs_value < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("RHS of pow() cannot be negative: ", rhs_value));
  }

  XLS_ASSIGN_OR_RETURN(verilog::DataType * vast_rhs_type,
                       GetVastDataType(vast_rhs));
  XLS_ASSIGN_OR_RETURN(int64_t rhs_bit_count,
                       vast_rhs_type->FlatBitCountAsInt64());
  auto* element_type = module().Make<dslx::BuiltinTypeAnnotation>(
      span, dslx::BuiltinType::kUN,
      module().GetOrCreateBuiltinNameDef(dslx::BuiltinType::kUN));
  auto* array_dim =
      module().Make<dslx::Number>(span, absl::StrCat(rhs_bit_count),
                                  dslx::NumberKind::kOther, /*type=*/nullptr);
  auto* new_rhs_type =
      module().Make<dslx::ArrayTypeAnnotation>(span, element_type, array_dim);
  rhs = module().Make<dslx::Cast>(span, rhs, new_rhs_type);

  XLS_ASSIGN_OR_RETURN(dslx::Import * std,
                       GetOrImportModule(dslx::ImportTokens({"std"})));
  auto* colon_ref = module().Make<dslx::ColonRef>(
      span, module().Make<dslx::NameRef>(span, "std", &std->name_def()), "pow");
  auto* invocation = module().Make<dslx::Invocation>(
      span, colon_ref, std::vector<dslx::Expr*>({lhs, rhs}));
  return invocation;
}

absl::StatusOr<dslx::Import*> DslxBuilder::GetOrImportModule(
    const dslx::ImportTokens& import_tokens) {
  XLS_RET_CHECK(!import_tokens.pieces().empty());
  std::string tail = import_tokens.pieces().back();
  std::optional<dslx::ModuleMember*> module_or =
      module().FindMemberWithName(tail);
  if (module_or.has_value()) {
    if (!std::holds_alternative<dslx::Import*>(*module_or.value())) {
      return absl::InternalError(
          absl::StrCat("Found symbol \"", tail, "\", but wasn't an import!"));
    }
    return std::get<dslx::Import*>(*module_or.value());
  }

  if (!module().FindMemberWithName(tail)) {
    VLOG(2) << "Could not find module member that represents import of: "
            << import_tokens.ToString();
    XLS_ASSIGN_OR_RETURN(dslx::ModuleInfo * mod_info,
                         dslx::DoImport(
                             [this](std::unique_ptr<dslx::Module> module,
                                    std::filesystem::path path) {
                               return dslx::TypecheckModuleV2(
                                   std::move(module), path, &import_data_,
                                   &warnings_, nullptr, nullptr);
                             },
                             import_tokens, &import_data_, dslx::Span::Fake(),
                             import_data_.vfs()));

    auto* name_def = resolver_->MakeNameDef(*this, dslx::Span::Fake(), tail);
    VLOG(2) << "Creating import node via name definition `"
            << name_def->ToString() << "`";
    auto* import = module().Make<dslx::Import>(
        dslx::Span::Fake(), import_tokens.pieces(), *name_def, std::nullopt);
    name_def->set_definer(import);
    XLS_RETURN_IF_ERROR(
        module().AddTop(import, /*make_collision_error=*/nullptr));
    type_info_->AddImport(import, &mod_info->module(), mod_info->type_info());
    import_data_.GetOrCreateTopLevelBindings(&module()).AddModule(
        tail, &mod_info->module());
  }

  std::optional<dslx::ModuleMember*> std_or = module().FindMemberWithName(tail);
  if (!std_or.has_value()) {
    return absl::InternalError(
        absl::StrCat("Unable to find \"", tail, "\" after successful import!"));
  }

  if (!std::holds_alternative<dslx::Import*>(**std_or)) {
    return absl::InternalError(
        absl::StrCat("Module member \"", tail, "\" should be Import."));
  }

  return std::get<dslx::Import*>(*std_or.value());
}

absl::StatusOr<dslx::Expr*> DslxBuilder::CastToInferredVastType(
    verilog::Expression* vast_expr, dslx::Expr* expr,
    bool force_cast_user_defined) {
  XLS_ASSIGN_OR_RETURN(verilog::DataType * vast_type,
                       GetVastDataType(vast_expr));
  return Cast(vast_type, expr, force_cast_user_defined);
}

absl::StatusOr<dslx::Expr*> DslxBuilder::Cast(verilog::DataType* vast_type,
                                              dslx::Expr* expr,
                                              bool force_cast_user_defined) {
  if (!vast_type->FlatBitCountAsInt64().ok()) {
    VLOG(2) << "Warning: cannot insert a cast of expr: " << expr->ToString()
            << " to type: " << vast_type->Emit(nullptr)
            << " because the VAST type does not have a bit count; this is "
               "generally OK if the width is not statically computed.";
    return expr;
  }
  if (vast_type->IsUserDefined()) {
    const auto* typedef_type =
        dynamic_cast<const verilog::TypedefType*>(vast_type);
    // Integer type aliases should really be casted, but enums should not.
    if (typedef_type &&
        !typedef_type->type_def()->data_type()->IsUserDefined()) {
      return CastInternal(vast_type, expr);
    }

    if (force_cast_user_defined) {
      return CastInternal(vast_type, expr);
    }

    return expr;
  }

  return CastInternal(vast_type, expr);
}

absl::StatusOr<dslx::TypeAnnotation*> DslxBuilder::VastTypeToDslxTypeForCast(
    const dslx::Span& span, verilog::DataType* vast_type, bool force_builtin) {
  // If it's a typedef, then use the DSLX counterpart.
  if (auto* typedef_type = dynamic_cast<verilog::TypedefType*>(vast_type);
      typedef_type && !force_builtin) {
    absl::StatusOr<dslx::TypeDefinition> target_type =
        resolver_->FindTypedef(*this, typedef_type);
    if (target_type.ok()) {
      return module().Make<dslx::TypeRefTypeAnnotation>(
          span, module().Make<dslx::TypeRef>(span, *target_type),
          /*parametrics=*/std::vector<dslx::ExprOrType>{});
    }
    VLOG(2) << "Casting to typedef " << typedef_type->type_def()->GetName()
            << ", which is not associated with a DSLX type; using raw type.";
  }
  // Try to use a concrete built-in type.
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, vast_type->FlatBitCountAsInt64());
  if (bit_count <= dslx::kConcreteBuiltinTypeLimit) {
    absl::StatusOr<dslx::BuiltinType> concrete_type =
        dslx::GetBuiltinType(vast_type->is_signed(), bit_count);
    if (concrete_type.ok()) {
      return module().Make<dslx::BuiltinTypeAnnotation>(
          span, *concrete_type,
          module().GetOrCreateBuiltinNameDef(*concrete_type));
    }
  }
  // If the number of bits is too high or unknown, declare it as sN or uN.
  dslx::BuiltinType builtin_type =
      vast_type->is_signed() ? dslx::BuiltinType::kSN : dslx::BuiltinType::kUN;
  auto* builtin_type_annot = module().Make<dslx::BuiltinTypeAnnotation>(
      span, builtin_type, module().GetOrCreateBuiltinNameDef(builtin_type));
  return module().Make<dslx::ArrayTypeAnnotation>(
      span, builtin_type_annot,
      module().Make<dslx::Number>(span, std::to_string(bit_count),
                                  dslx::NumberKind::kOther,
                                  /*type_annotation=*/nullptr));
}

std::optional<std::string> DslxBuilder::GenerateSizeCommentIfNotObvious(
    verilog::DataType* data_type, bool compute_size_if_struct) {
  if (auto* typedef_type = dynamic_cast<verilog::TypedefType*>(data_type);
      typedef_type) {
    data_type = typedef_type->BaseType();
  }
  auto* struct_def = dynamic_cast<verilog::Struct*>(data_type);
  // Filter out struct sizes if desired.
  if (struct_def != nullptr && !compute_size_if_struct) {
    return std::nullopt;
  }
  // Filter out obvious sizes.
  if (data_type->FlatBitCountAsInt64().ok() &&
      !(struct_def != nullptr && struct_def->members().size() > 1)) {
    return std::nullopt;
  }
  absl::StatusOr<verilog::DataType*> folded_type =
      FoldVastConstants(data_type, vast_type_map_);
  if (folded_type.ok()) {
    absl::StatusOr<int64_t> bit_count = (*folded_type)->FlatBitCountAsInt64();
    if (bit_count.ok()) {
      if (dynamic_cast<verilog::Struct*>(data_type) != nullptr) {
        return absl::StrFormat(" %d bits", *bit_count);
      }
      absl::StatusOr<dslx::TypeAnnotation*> type_annot =
          VastTypeToDslxTypeForCast(dslx::Span(), *folded_type,
                                    /*force_builtin=*/true);
      // This can't fail because we have guaranteed the bit count is ok.
      QCHECK_OK(type_annot);
      return absl::StrFormat(" %s", (*type_annot)->ToString());
    }
  }
  return std::nullopt;
}

absl::StatusOr<dslx::Expr*> DslxBuilder::CastInternal(
    verilog::DataType* vast_type, dslx::Expr* expr, bool force_builtin) {
  XLS_ASSIGN_OR_RETURN(
      dslx::TypeAnnotation * type,
      VastTypeToDslxTypeForCast(expr->span(), vast_type, force_builtin));
  return module().Make<dslx::Cast>(expr->span(), expr, type);
}

absl::StatusOr<verilog::DataType*> DslxBuilder::GetVastDataType(
    verilog::Expression* expr) const {
  const auto it = vast_type_map_.find(expr);
  if (it == vast_type_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No Verilog type inferred for: ", expr->Emit(nullptr)));
  }
  return it->second;
}

absl::StatusOr<dslx::TypecheckedModule> DslxBuilder::RoundTrip(
    const dslx::Module& module, std::string_view path,
    dslx::ImportData& import_data,
    dslx::TypeInferenceErrorHandler error_handler) {
  const std::string text = module.ToString();
  dslx::Fileno fileno = import_data.file_table().GetOrCreate(path);
  dslx::Scanner scanner(import_data.file_table(), fileno, text);
  dslx::Parser parser(module_.name(), &scanner);
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule parsed_module,
      ParseAndTypecheck(text, path, module_.name(), &import_data,
                        /*comments=*/nullptr,
                        /*force_version=*/dslx::TypeInferenceVersion::kVersion2,
                        /*options=*/dslx::ConvertOptions{}, error_handler),
      _ << "Failed to parse and typecheck module:\n"
        << text);
  return parsed_module;
}

dslx::ImportData DslxBuilder::CreateImportData() {
  return dslx::CreateImportData(dslx_stdlib_path_, additional_search_paths_,
                                /*enabled_warnings=*/dslx::kDefaultWarningsSet,
                                std::make_unique<dslx::RealFilesystem>());
}

absl::StatusOr<std::string> DslxBuilder::FormatModule() {
  for (const dslx::ModuleMember& member : module_.top()) {
    if (std::holds_alternative<dslx::ConstantDef*>(member)) {
      // Allow non-standard constant names, because we do not canonicalize the
      // original SV names to SCREAMING_SNAKE_CASE in the conversion process.
      module_.AddAttribute(
          dslx::ModuleAttribute::kAllowNonstandardConstantNaming);

      // Similar for members, they are not in SV going to be consistency named
      // in `snake_case` convention.
      module_.AddAttribute(
          dslx::ModuleAttribute::kAllowNonstandardMemberNaming);
      break;
    }
  }

  // Perform an initial type inference run, then do cleanup informed by type
  // inference, like removal of dead casts.
  dslx::ImportData initial_import_data = CreateImportData();
  const std::string file_name = module_.name() + ".x";
  std::unique_ptr<DslxTypeFixer> fixer =
      CreateDslxTypeFixer(module_, import_data_);
  XLS_ASSIGN_OR_RETURN(dslx::TypecheckedModule initial_module,
                       RoundTrip(module_, file_name, initial_import_data,
                                 fixer->GetErrorHandler()));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<dslx::Module> module_with_errors_fixed,
      CloneModule(*initial_module.module,
                  fixer->GetErrorFixReplacer(initial_module.type_info)));

  dslx::ImportData fix_import_data = CreateImportData();
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule fixed_and_typechecked_module,
      RoundTrip(*module_with_errors_fixed, file_name, fix_import_data));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<dslx::Module> simplified_module,
      CloneModule(
          *fixed_and_typechecked_module.module,
          fixer->GetSimplifyReplacer(fixed_and_typechecked_module.type_info)));

  // We now need to round-trip the module to text and back to AST, without the
  // comments, in order for the nodes to get spans accurately representing the
  // DSLX as opposed to the source Verilog. We then position the comments
  // relative to the appropriate spans.
  dslx::ImportData import_data = CreateImportData();
  XLS_ASSIGN_OR_RETURN(dslx::TypecheckedModule parsed_module,
                       RoundTrip(*simplified_module, file_name, import_data));

  std::vector<dslx::CommentData> comment_data;
  for (const auto& [type_name, comment] : type_def_comments_) {
    absl::StatusOr<dslx::TypeDefinition> type_def =
        parsed_module.module->GetTypeDefinition(type_name);
    if (type_def.ok()) {
      const dslx::AstNode* node =
          absl::visit([&](auto* n) -> dslx::AstNode* { return n; }, *type_def);
      if (std::holds_alternative<dslx::StructDef*>(*type_def) ||
          std::holds_alternative<dslx::EnumDef*>(*type_def)) {
        // For types with members, the comment looks better before the first
        // member, and this also has the nice effect of preventing overzealous
        // collapsing of the type to one line.
        comment_data.push_back(CommentAtBeginning(node, comment));
        // Structs may then have comments at the end of specific members.
        const auto member_comments = struct_member_comments_.find(type_name);
        if (member_comments != struct_member_comments_.end()) {
          for (const dslx::StructMemberNode* member :
               std::get<dslx::StructDef*>(*type_def)->members()) {
            const auto member_comment =
                member_comments->second.find(member->name());
            if (member_comment != member_comments->second.end()) {
              comment_data.push_back(
                  CommentAfter(member->span(), member_comment->second));
            }
          }
        }
      } else {
        // Simple typedefs get comments at the end.
        comment_data.push_back(CommentAfter(node, comment));
      }
    } else {
      // This does not normally happen at all.
      VLOG(2) << "Could not add comment for " << type_name
              << " because it was not found after round-tripping the DSLX.";
    }
  }
  for (const auto& [constant_name, comment] : constant_def_comments_) {
    absl::StatusOr<dslx::ConstantDef*> constant_def =
        parsed_module.module->GetConstantDef(constant_name);
    if (constant_def.ok()) {
      comment_data.push_back(CommentAfter(*constant_def, comment));
    } else {
      // This does not normally happen at all.
      VLOG(2) << "Could not add comment for " << constant_name
              << " because it was not found after round-tripping the DSLX.";
    }
  }
  dslx::Comments comments = dslx::Comments::Create(comment_data);
  return AutoFmt(import_data.vfs(), *parsed_module.module, comments,
                 /*text_width=*/100);
}

}  // namespace xls

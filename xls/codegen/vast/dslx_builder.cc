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
#include "xls/codegen/vast/fold_vast_constants.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_function.h"
#include "xls/dslx/type_system/typecheck_invocation.h"
#include "xls/dslx/type_system/typecheck_module.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

absl::StatusOr<bool> IsNegative(const dslx::InterpValue& value) {
  if (!value.IsSigned()) {
    return false;
  }
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
  auto zero = dslx::InterpValue::MakeSBits(bit_count, 0);
  XLS_ASSIGN_OR_RETURN(auto lt_zero, value.Lt(zero));
  XLS_ASSIGN_OR_RETURN(int64_t lt_zero_bit_value, lt_zero.GetBitValueViaSign());
  return lt_zero_bit_value == 1;
}

std::string GetTypeDefName(const dslx::TypeDefinition& type_def) {
  return absl::visit(
      Visitor{
          [&](const dslx::TypeAlias* n) { return n->identifier(); },
          [&](const dslx::StructDef* n) { return n->identifier(); },
          [&](const dslx::EnumDef* n) { return n->identifier(); },
          [&](const dslx::ColonRef* n) { return n->ToString(); },
      },
      type_def);
}

dslx::CommentData CommentAfter(const dslx::Span& span,
                               const std::string& comment) {
  dslx::Span comment_span(
      span.limit().BumpCol(),
      dslx::Pos(span.filename(), span.limit().lineno() + 1, 0));
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
      span->start(),
      dslx::Pos(span->filename(), span->start().lineno() + 1, 0));
  return dslx::CommentData{.span = comment_span, .text = comment};
}

dslx::TypeInfo* GetTypeInfoOrDie(dslx::ImportData& import_data,
                                 dslx::Module* module) {
  absl::StatusOr<dslx::TypeInfo*> result =
      import_data.type_info_owner().New(module);
  CHECK_OK(result);
  return *result;
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

DslxBuilder::DslxBuilder(
    dslx::Module& module, std::string_view dslx_stdlib_path,
    absl::flat_hash_map<verilog::Expression*, verilog::DataType*> vast_type_map,
    dslx::WarningCollector& warnings)
    : module_(module),
      dslx_stdlib_path_(dslx_stdlib_path),
      import_data_(dslx::CreateImportData(
          dslx_stdlib_path,
          /*additional_search_paths=*/{},
          /*enabled_warnings=*/dslx::kDefaultWarningsSet)),
      warnings_(warnings),
      type_info_(GetTypeInfoOrDie(import_data_, &module_)),
      deduce_ctx_(
          type_info_, &module, dslx::Deduce, &dslx::TypecheckFunction,
          [this](dslx::Module*) {
            return dslx::TypecheckModule(&module_, &import_data_, &warnings_);
          },
          dslx::TypecheckInvocation, &import_data_, &warnings_,
          /*parent=*/nullptr),
      vast_type_map_(std::move(vast_type_map)) {
  deduce_ctx_.fn_stack().push_back(dslx::FnStackEntry::MakeTop(&module));
}

dslx::NameDef* DslxBuilder::MakeNameDef(const dslx::Span& span,
                                        std::string_view name) {
  VLOG(3) << "MakeNameDef; span: " << span << " name: `" << name << "`";
  auto* name_def = module().Make<dslx::NameDef>(span, std::string(name),
                                                /*definer=*/nullptr);
  name_to_namedef_[name] = name_def;
  return name_def;
}

absl::StatusOr<dslx::NameRef*> DslxBuilder::MakeNameRef(const dslx::Span& span,
                                                        std::string_view name) {
  const auto it = name_to_namedef_.find(name);
  if (it == name_to_namedef_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Reference to undefined name: ", name));
  }
  return module().Make<dslx::NameRef>(span, std::string(name), it->second);
}

absl::StatusOr<dslx::Expr*> DslxBuilder::MakeNameRefAndMaybeCast(
    verilog::Expression* expr, const dslx::Span& span, std::string_view name) {
  XLS_ASSIGN_OR_RETURN(dslx::NameRef * ref, MakeNameRef(span, name));
  return MaybeCastToInferredVastType(expr, ref);
}

void DslxBuilder::AddTypedef(verilog::Typedef* type_def,
                             dslx::TypeDefinition dslx_type) {
  // Note that we use the loc as a key because it's resilient to the creation
  // of derivative `DataType` objects by VAST type inference, e.g. during
  // constant folding. This way we can look up a DSLX typedef by either the
  // original `verilog::Typedef` object or one that is an artifact of type
  // inference.
  typedefs_by_loc_string_.emplace(type_def->loc().ToString(), dslx_type);
  if (dynamic_cast<verilog::Enum*>(type_def->data_type())) {
    reverse_typedefs_.emplace(type_def->data_type(), type_def);
  }
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

absl::StatusOr<dslx::TypeDefinition> DslxBuilder::FindTypedef(
    verilog::TypedefType* typedef_type) {
  const auto it =
      typedefs_by_loc_string_.find(typedef_type->type_def()->loc().ToString());
  if (it == typedefs_by_loc_string_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Typedef ", typedef_type->type_def()->GetName(), " at ",
                     typedef_type->loc().ToString(),
                     " has not been associated with a DSLX type."));
  }
  return it->second;
}

absl::StatusOr<verilog::Typedef*> DslxBuilder::ReverseEnumTypedef(
    verilog::Enum* enum_def) {
  const auto it = reverse_typedefs_.find(enum_def);
  if (it == reverse_typedefs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Data type ", enum_def->Emit(nullptr),
                     " has not been associated with a typedef."));
  }
  return it->second;
}

absl::StatusOr<dslx::ConstantDef*> DslxBuilder::HandleConstantDecl(
    const dslx::Span& span, verilog::Module* vast_module,
    verilog::Parameter* parameter, std::string_view name, dslx::Expr* expr) {
  auto* name_def = MakeNameDef(span, name);
  auto* constant_def = module().Make<dslx::ConstantDef>(
      span, name_def, /*type_annotation=*/nullptr, expr,
      /*is_public=*/true);
  name_def->set_definer(constant_def);

  XLS_RETURN_IF_ERROR(
      module().AddTop(constant_def, /*make_collision_error=*/nullptr));

  // Note: historically, the propagation of errors was incorrectly dropped by
  // the caller, hiding the fact that the logic from here down is a nice-to-have
  // and historically fails e.g. in ConcatOfEnums.
  auto deduced_type = deduce_ctx().Deduce(expr);
  absl::StatusOr<dslx::InterpValue> value =
      InterpretExpr(import_data(), type_info(), expr);
  if (deduced_type.ok() && value.ok()) {
    bindings_.AddValue(name_def->identifier(), *value);
    import_data_.GetOrCreateTopLevelBindings(&module()).AddValue(
        name_def->identifier(), *value);
  }

  auto def_deduced_type = deduce_ctx().Deduce(constant_def);
  if (!def_deduced_type.ok()) {
    VLOG(2) << "Failed to deduce constant def type: "
            << def_deduced_type.status();
  }
  SetRefTargetModule(parameter, vast_module);
  // Add a comment with the value, if it is not obvious and can be folded.
  if (parameter->rhs() != nullptr && !parameter->rhs()->IsLiteral()) {
    absl::StatusOr<int64_t> folded_value =
        verilog::FoldEntireVastExpr(parameter->rhs(), vast_type_map_);
    absl::StatusOr<verilog::DataType*> folded_type =
        GetVastDataType(parameter->rhs());
    if (folded_value.ok() && folded_type.ok()) {
      const std::string type_str =
          VastTypeToDslxTypeForCast(dslx::Span(), *folded_type,
                                    /*force_builtin=*/true)
              ->ToString();
      constant_def_comments_.emplace(
          name, *folded_value >= 0x100000
                    ? absl::StrFormat(" %s:0x%x", type_str, *folded_value)
                    : absl::StrFormat(" %s:%d", type_str, *folded_value));
    }
  }
  return constant_def;
}

dslx::Number* DslxBuilder::HandleConstVal(
    const dslx::Span& span, const Bits& bits,
    FormatPreference format_preference, verilog::DataType* vast_type,
    dslx::TypeAnnotation* force_dslx_type) {
  return module().Make<dslx::Number>(
      span, BitsToString(bits, format_preference), dslx::NumberKind::kOther,
      force_dslx_type != nullptr ? force_dslx_type
                                 : VastTypeToDslxTypeForCast(span, vast_type));
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
  if (auto* binop = dynamic_cast<dslx::Binop*>(dslx_value);
      binop && binop->binop_kind() == dslx::BinopKind::kSub) {
    auto* rhs_number = dynamic_cast<dslx::Number*>(binop->rhs());
    if (rhs_number != nullptr) {
      absl::StatusOr<uint64_t> rhs_value = rhs_number->GetAsUint64();
      if (rhs_value.ok() && *rhs_value == 1) {
        return MaybeCast(unsigned_int_type, binop->lhs());
      }
    }
  }
  // Add 1 if really necessary.
  XLS_ASSIGN_OR_RETURN(dslx::Expr * casted_max,
                       MaybeCast(unsigned_int_type, dslx_value));
  dslx::Number* one = module().Make<dslx::Number>(
      span, "1", dslx::NumberKind::kOther, type_annot);
  return module().Make<dslx::Binop>(span, dslx::BinopKind::kAdd, casted_max,
                                    one);
}

dslx::Unop* DslxBuilder::HandleUnaryOperator(const dslx::Span& span,
                                             dslx::UnopKind unop_kind,
                                             dslx::Expr* arg) {
  return module().Make<dslx::Unop>(span, dslx::UnopKind::kNegate, arg);
}

absl::StatusOr<dslx::Expr*> DslxBuilder::HandleIntegerExponentiation(
    const dslx::Span& span, dslx::Expr* lhs, dslx::Expr* rhs) {
  // TODO(b/330575305): 2022-03-16: Switch the model to deduce after
  // constructing a value instead of deducing immediately beforehand.
  XLS_RETURN_IF_ERROR(deduce_ctx().Deduce(lhs).status());
  XLS_RETURN_IF_ERROR(deduce_ctx().Deduce(rhs).status());
  XLS_ASSIGN_OR_RETURN(dslx::InterpValue lhs_value,
                       InterpretExpr(import_data(), type_info(), lhs));
  if (!lhs_value.HasBits()) {
    return absl::InvalidArgumentError("pow() LHS isn't bits-typed.");
  }

  XLS_ASSIGN_OR_RETURN(dslx::InterpValue rhs_value,
                       InterpretExpr(import_data(), type_info(), rhs));
  if (!rhs_value.HasBits()) {
    return absl::InvalidArgumentError("pow() RHS isn't bits-typed.");
  }

  XLS_ASSIGN_OR_RETURN(bool is_negative, IsNegative(rhs_value));
  if (is_negative) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RHS of pow() cannot be negative: ", rhs_value.ToString()));
  }

  auto* element_type = module().Make<dslx::BuiltinTypeAnnotation>(
      span, dslx::BuiltinType::kUN,
      module().GetOrCreateBuiltinNameDef(dslx::BuiltinType::kUN));
  XLS_ASSIGN_OR_RETURN(int64_t rhs_bit_count, rhs_value.GetBitCount());
  auto* array_dim =
      module().Make<dslx::Number>(span, absl::StrCat(rhs_bit_count),
                                  dslx::NumberKind::kOther, /*type=*/nullptr);
  auto* new_rhs_type =
      module().Make<dslx::ArrayTypeAnnotation>(span, element_type, array_dim);
  rhs = module().Make<dslx::Cast>(span, rhs, new_rhs_type);

  std::string power_fn = lhs_value.IsUBits() ? "upow" : "spow";
  XLS_ASSIGN_OR_RETURN(dslx::Import * std,
                       GetOrImportModule(dslx::ImportTokens({"std"})));
  auto* colon_ref = module().Make<dslx::ColonRef>(
      span, module().Make<dslx::NameRef>(span, "std", &std->name_def()),
      power_fn);
  auto* invocation = module().Make<dslx::Invocation>(
      span, colon_ref, std::vector<dslx::Expr*>({lhs, rhs}));
  dslx::ParametricEnv parametric_env(
      absl::flat_hash_map<std::string, dslx::InterpValue>{
          {"N", dslx::InterpValue::MakeUBits(32, rhs_bit_count)}});
  XLS_RETURN_IF_ERROR(type_info().AddInvocationTypeInfo(
      *invocation, /*caller=*/nullptr,
      /*caller_env=*/deduce_ctx().GetCurrentParametricEnv(),
      /*callee_env=*/parametric_env,
      /*derived_type_info=*/nullptr));
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
    dslx::Span span{dslx::Pos(), dslx::Pos()};
    XLS_ASSIGN_OR_RETURN(dslx::ModuleInfo * mod_info,
                         dslx::DoImport(
                             [this](dslx::Module* module) {
                               return dslx::TypecheckModule(
                                   module, &import_data_, &warnings_);
                             },
                             import_tokens, &import_data_, dslx::Span::Fake()));

    auto* name_def = MakeNameDef(span, tail);
    auto* import = module().Make<dslx::Import>(span, import_tokens.pieces(),
                                               *name_def, std::nullopt);
    name_def->set_definer(import);
    XLS_RETURN_IF_ERROR(
        module().AddTop(import, /*make_collision_error=*/nullptr));
    deduce_ctx().type_info()->AddImport(import, &mod_info->module(),
                                        mod_info->type_info());
    import_data_.GetOrCreateTopLevelBindings(&module()).AddModule(
        tail, &mod_info->module());
    bindings_.AddModule(tail, &mod_info->module());
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

absl::StatusOr<dslx::Expr*> DslxBuilder::MaybeCastToInferredVastType(
    verilog::Expression* vast_expr, dslx::Expr* expr,
    bool cast_enum_to_builtin) {
  XLS_ASSIGN_OR_RETURN(verilog::DataType * vast_type,
                       GetVastDataType(vast_expr));
  return MaybeCast(vast_type, expr, cast_enum_to_builtin);
}

absl::StatusOr<dslx::Expr*> DslxBuilder::MaybeCast(verilog::DataType* vast_type,
                                                   dslx::Expr* expr,
                                                   bool cast_enum_to_builtin) {
  if (auto* cast = dynamic_cast<dslx::Cast*>(expr); cast) {
    // Avoid doing something like `(foo as s32) as u32`, which can happen
    // because we coerce array sizes to u32 on the DSLX translation side.
    absl::StatusOr<dslx::Expr*> smaller_onion =
        MaybeCast(vast_type, cast->expr());
    if (smaller_onion.ok()) {
      return smaller_onion;
    }
  }
  if (!vast_type->FlatBitCountAsInt64().ok()) {
    VLOG(2) << "Warning: cannot insert a cast of expr: " << expr->ToString()
            << " to type: " << vast_type->Emit(nullptr)
            << " because the VAST type does not have a bit count; this is "
               "generally OK if the width is not statically computed.";
    return expr;
  }
  absl::StatusOr<std::unique_ptr<dslx::Type>> deduced_dslx_type =
      deduce_ctx().Deduce(expr);
  if (!deduced_dslx_type.ok()) {
    VLOG(2) << "Warning: Pessimistically inserting a cast of expr: "
            << expr->ToString() << " to type: " << vast_type->Emit(nullptr)
            << " because the DSLX type cannot be deduced. This may happen if a "
               "parameter value has a system function call.";
    return Cast(vast_type, expr);
  }
  if ((*deduced_dslx_type)->HasEnum() &&
      (cast_enum_to_builtin || !vast_type->IsUserDefined())) {
    // DSLX considers enum and values to mismatch operands of an equivalent
    // built-in type. VAST type inference will say in that case that they are
    // both the generic type, and we need the cast to make DSLX comply.
    return Cast(vast_type, expr, cast_enum_to_builtin);
  }
  XLS_ASSIGN_OR_RETURN(dslx::TypeDim deduced_dslx_dim,
                       (*deduced_dslx_type)->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t deduced_dslx_bit_count,
                       deduced_dslx_dim.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(int64_t verilog_bit_count,
                       vast_type->FlatBitCountAsInt64());
  if (deduced_dslx_bit_count != verilog_bit_count) {
    return Cast(vast_type, expr);
  }
  absl::StatusOr<bool> deduced_dslx_signed =
      dslx::IsSigned(**deduced_dslx_type);
  if (!deduced_dslx_signed.ok() ||
      *deduced_dslx_signed != vast_type->is_signed()) {
    return Cast(vast_type, expr);
  }
  return expr;
}

dslx::TypeAnnotation* DslxBuilder::VastTypeToDslxTypeForCast(
    const dslx::Span& span, verilog::DataType* vast_type, bool force_builtin) {
  // If it's a typedef, then use the DSLX counterpart.
  if (auto* typedef_type = dynamic_cast<verilog::TypedefType*>(vast_type);
      typedef_type && !force_builtin) {
    absl::StatusOr<dslx::TypeDefinition> target_type =
        FindTypedef(typedef_type);
    if (target_type.ok()) {
      return module().Make<dslx::TypeRefTypeAnnotation>(
          span, module().Make<dslx::TypeRef>(span, *target_type),
          /*parametrics=*/std::vector<dslx::ExprOrType>{});
    }
    VLOG(2) << "Casting to typedef " << typedef_type->type_def()->GetName()
            << ", which is not associated with a DSLX type; using raw type.";
  }
  // Try to use a concrete built-in type.
  absl::StatusOr<int64_t> bit_count = vast_type->FlatBitCountAsInt64();
  if (bit_count.ok() && (*bit_count <= dslx::kConcreteBuiltinTypeLimit)) {
    absl::StatusOr<dslx::BuiltinType> concrete_type =
        dslx::GetBuiltinType(vast_type->is_signed(), *bit_count);
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
      module().Make<dslx::Number>(span, std::to_string(*bit_count),
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
      dslx::TypeAnnotation* type_annot = VastTypeToDslxTypeForCast(
          dslx::Span(), *folded_type, /*force_builtin=*/true);
      return absl::StrFormat(" %s", type_annot->ToString());
    }
  }
  return std::nullopt;
}

absl::StatusOr<dslx::Expr*> DslxBuilder::Cast(verilog::DataType* vast_type,
                                              dslx::Expr* expr,
                                              bool force_builtin) {
  return module().Make<dslx::Cast>(
      expr->span(), expr,
      VastTypeToDslxTypeForCast(expr->span(), vast_type, force_builtin));
}

absl::StatusOr<verilog::Module*> DslxBuilder::FindRefTargetModule(
    verilog::VastNode* target) const {
  const auto it = ref_target_to_module_.find(target);
  if (it == ref_target_to_module_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No ref target module for: ", target->Emit(nullptr)));
  }
  return it->second;
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

absl::StatusOr<std::string> DslxBuilder::FormatModule() {
  for (const dslx::ModuleMember& member : module_.top()) {
    if (std::holds_alternative<dslx::ConstantDef*>(member)) {
      // Allow non-standard constant names, because we do not canonicalize the
      // original SV names to SCREAMING_SNAKE_CASE in the conversion process.
      module_.AddAnnotation(
          dslx::ModuleAnnotation::kAllowNonstandardConstantNaming);

      // Similar for members, they are not in SV going to be consistency named
      // in `snake_case` convention.
      module_.AddAnnotation(
          dslx::ModuleAnnotation::kAllowNonstandardMemberNaming);
      break;
    }
  }
  // We now need to round-trip the module to text and back to AST, without the
  // comments, in order for the nodes to get spans accurately representing the
  // DSLX as opposed to the source Verilog. We then position the comments
  // relative to the appropriate spans.
  const std::string text = module_.ToString();
  const std::string file_name = module_.name() + ".x";
  dslx::Scanner scanner(file_name, text);
  dslx::Parser parser(module_.name(), &scanner, /*options=*/{});
  auto import_data =
      dslx::CreateImportData(dslx_stdlib_path_,
                             /*additional_search_paths=*/{},
                             /*enabled_warnings=*/dslx::kDefaultWarningsSet);
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule parsed_module,
      ParseAndTypecheck(text, file_name, module_.name(), &import_data));
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
          for (const dslx::StructMember& member :
               std::get<dslx::StructDef*>(*type_def)->members()) {
            const auto member_comment =
                member_comments->second.find(member.name);
            if (member_comment != member_comments->second.end()) {
              comment_data.push_back(
                  CommentAfter(member.GetSpan(), member_comment->second));
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
  return AutoFmt(*parsed_module.module, dslx::Comments::Create(comment_data),
                 /*text_width=*/100);
}

}  // namespace xls

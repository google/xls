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

#include "xls/codegen/vast/translate_vast_to_dslx.h"

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/dslx_builder.h"
#include "xls/codegen/vast/infer_vast_types.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/casts.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/trait_visitor.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

// Returns true if the given string is reserved in DSLX.
bool IsDslxReserved(const std::string& identifier) {
  if (dslx::BuiltinTypeFromString(identifier).ok()) {
    return true;
  }
  return dslx::KeywordFromString(identifier).has_value();
}

dslx::Pos CreateNodePos(const SourceInfo& source_info) {
  return dslx::Pos(dslx::Fileno(0), source_info.locations[0].lineno().value(),
                   source_info.locations[0].colno().value());
}

dslx::Span CreateNodeSpan(verilog::VastNode* node) {
  dslx::Pos pos = node == nullptr ? CreateNodePos(SourceInfo())
                                  : CreateNodePos(node->loc());
  return dslx::Span(pos, pos);
}

// A helper class that translates VAST trees to DSLX.
class VastToDslxTranslator {
 public:
  VastToDslxTranslator(
      verilog::Module* main_vast_module, bool generate_combined_dslx_module,
      const std::optional<std::filesystem::path>& additional_search_path,
      std::string_view dslx_stdlib_path,
      absl::flat_hash_map<verilog::Expression*, verilog::DataType*>
          vast_type_map)
      : generate_combined_dslx_module_(generate_combined_dslx_module),
        additional_search_path_(additional_search_path),
        dslx_stdlib_path_(dslx_stdlib_path),
        warnings_(dslx::kDefaultWarningsSet),
        resolver_(main_vast_module->name(), generate_combined_dslx_module),
        vast_type_map_(std::move(vast_type_map)) {
    if (generate_combined_dslx_module) {
      dslx_builder_ = CreateDslxBuilder(main_vast_module);
    }
  }

  std::unique_ptr<DslxBuilder> CreateDslxBuilder(verilog::Module* vast_module) {
    return std::make_unique<DslxBuilder>(
        vast_module->name(), &resolver_, additional_search_path_,
        dslx_stdlib_path_, vast_type_map_, warnings_);
  }

  absl::Status TranslateModule(verilog::Module* vast_module) {
    current_vast_module_ = vast_module;
    if (!generate_combined_dslx_module_) {
      dslx_builder_ = CreateDslxBuilder(vast_module);
    }
    for (verilog::ModuleMember member : vast_module->top()->members()) {
      verilog::VastNode* node = nullptr;
      if (std::holds_alternative<verilog::VerilogFunction*>(member)) {
        node = std::get<verilog::VerilogFunction*>(member);
      } else if (std::holds_alternative<verilog::Typedef*>(member)) {
        node = std::get<verilog::Typedef*>(member);
      } else if (std::holds_alternative<verilog::Parameter*>(member)) {
        node = std::get<verilog::Parameter*>(member);
      }
      if (node) {
        absl::Status status =
            TranslateByNodeType(node, &VastToDslxTranslator::TranslateFunction,
                                &VastToDslxTranslator::TranslateTypedef,
                                &VastToDslxTranslator::TranslateParameter)
                .status();
        if (!status.ok()) {
          VLOG(2) << "Could not translate: " << node->Emit(nullptr)
                  << "; status: " << status;
        }
      }
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> GetDslxModuleText() {
    return dslx_builder_->FormatModule();
  }

 private:
  // Function parameters are the only nodes of this type as of yet encountered.
  absl::StatusOr<dslx::Param*> TranslateFunctionArgument(verilog::Def* def) {
    dslx::NameDef* name_def = resolver_.MakeNameDef(
        *dslx_builder_, CreateNodeSpan(def), def->GetName());
    XLS_ASSIGN_OR_RETURN(dslx::TypeAnnotation * annot,
                         TranslateType(def->data_type()));
    auto* param = module().Make<dslx::Param>(name_def, annot);
    name_def->set_definer(param);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::Type> type,
                         deduce_ctx().Deduce(param));
    deduce_ctx().type_info()->SetItem(param->name_def(), *type);
    return param;
  }

  absl::StatusOr<dslx::Expr*> TranslateBinop(verilog::BinaryInfix* op) {
    XLS_ASSIGN_OR_RETURN(dslx::Expr * lhs, TranslateExpression(op->lhs()));
    XLS_ASSIGN_OR_RETURN(dslx::Expr * rhs, TranslateExpression(op->rhs()));
    dslx::Span span = CreateNodeSpan(op);
    dslx::BinopKind kind;
    dslx::Expr* result = nullptr;
    switch (op->kind()) {
      case verilog::OperatorKind::kEq:
        kind = dslx::BinopKind::kEq;
        break;
      case verilog::OperatorKind::kNe:
        kind = dslx::BinopKind::kNe;
        break;
      case verilog::OperatorKind::kGe:
        kind = dslx::BinopKind::kGe;
        break;
      case verilog::OperatorKind::kGt:
        kind = dslx::BinopKind::kGt;
        break;
      case verilog::OperatorKind::kLe:
        kind = dslx::BinopKind::kLe;
        break;
      case verilog::OperatorKind::kLt:
        kind = dslx::BinopKind::kLt;
        break;
      case verilog::OperatorKind::kAdd:
        kind = dslx::BinopKind::kAdd;
        break;
      case verilog::OperatorKind::kSub:
        kind = dslx::BinopKind::kSub;
        break;
      case verilog::OperatorKind::kMul:
        kind = dslx::BinopKind::kMul;
        break;
      case verilog::OperatorKind::kDiv:
        kind = dslx::BinopKind::kDiv;
        break;
      case verilog::OperatorKind::kPower: {
        XLS_ASSIGN_OR_RETURN(
            result, dslx_builder_->HandleIntegerExponentiation(span, lhs, rhs));
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unhandled binop kind: ", op->kind()));
    }

    if (!result) {
      XLS_ASSIGN_OR_RETURN(auto lhs_type, deduce_ctx().Deduce(lhs));
      XLS_ASSIGN_OR_RETURN(auto rhs_type, deduce_ctx().Deduce(rhs));

      if (*lhs_type != *rhs_type) {
        // Generally, compatible but not identical types (like an enum value vs.
        // generic value) will be coerced to be identical via VAST type
        // inference and the insertion of casts based on that during translation
        // of the LHS and RHS. In the event that this somehow gets it wrong or
        // fails to appease DSLX deduction, we can't keep the expr.
        return absl::InvalidArgumentError(absl::StrFormat(
            "Cannot translate binop \"%s\": arguments have different types: "
            "%s vs %s",
            op->Emit(nullptr), lhs_type->ToString(), rhs_type->ToString()));
      }
      // Note it uses the same span for the whole node and the operand;
      // the only time the operand span is used is for formatting, and
      // this node won't be used for formatting.
      result = module().Make<dslx::Binop>(span, kind, lhs, rhs, span);
    }
    return dslx_builder_->MaybeCastToInferredVastType(op, result);
  }

  absl::StatusOr<dslx::Expr*> TranslateConcat(verilog::Concat* concat) {
    if (concat->args().empty()) {
      return absl::InvalidArgumentError("Empty concat cannot be translated.");
    }
    dslx::Expr* result = nullptr;
    for (verilog::Expression* next : concat->args()) {
      // DSLX wants concat operands to be both arrays or both bits. To satisfy
      // DSLX:
      // - We must coerce enums to bits here.
      // - Typedefs to bits will work fine with normal VAST-DSLX casting rules.
      // - Structs will fail because you can't even cast them to bits.
      // See https://github.com/google/xls/issues/1498.
      XLS_ASSIGN_OR_RETURN(dslx::Expr * dslx_expr, TranslateExpression(next));
      XLS_ASSIGN_OR_RETURN(dslx_expr,
                           dslx_builder_->MaybeCastToInferredVastType(
                               next, dslx_expr, /*cast_enum_to_builtin=*/true));
      XLS_ASSIGN_OR_RETURN(auto expr_type, deduce_ctx().Deduce(dslx_expr));
      bool is_signed = false;
      int64_t size = 0;
      auto* enum_type = dynamic_cast<dslx::EnumType*>(expr_type.get());
      if (enum_type) {
        is_signed = enum_type->is_signed();
        XLS_ASSIGN_OR_RETURN(size, enum_type->GetTotalBitCount()->GetAsInt64());
      } else {
        dslx::BitsType* bits_type =
            dynamic_cast<dslx::BitsType*>(expr_type.get());
        if (bits_type == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("Cannot translate concat \"%s\": all arguments must "
                           "be bits-typed.",
                           concat->Emit(nullptr)));
        }
        is_signed = bits_type->is_signed();
        XLS_ASSIGN_OR_RETURN(size, bits_type->size().GetAsInt64());
      }
      if (is_signed) {
        dslx::Span span = dslx_expr->span();
        dslx::TypeAnnotation* annot =
            module().Make<dslx::BuiltinTypeAnnotation>(
                span, dslx::BuiltinType::kBits,
                module().GetOrCreateBuiltinNameDef(dslx::BuiltinType::kBits));
        auto* dim = module().Make<dslx::Number>(
            span, absl::StrCat(size), dslx::NumberKind::kOther, nullptr);
        annot = module().Make<dslx::ArrayTypeAnnotation>(dslx_expr->span(),
                                                         annot, dim);
        dslx_expr =
            module().Make<dslx::Cast>(dslx_expr->span(), dslx_expr, annot);
      }
      dslx::Span span = CreateNodeSpan(concat);
      // Note it uses the same span for the whole node and the operand;
      // the only time the operand span is used is for formatting, and
      // this node won't be used for formatting.
      result = result == nullptr
                   ? dslx_expr
                   : module().Make<dslx::Binop>(span, dslx::BinopKind::kConcat,
                                                result, dslx_expr, span);
    }
    return result;
  }

  absl::StatusOr<dslx::Expr*> TranslateLiteral(verilog::Literal* literal) {
    absl::StatusOr<verilog::DataType*> vast_type =
        dslx_builder_->GetVastDataType(literal);
    if (!vast_type.ok()) {
      // Array dims that are literals do not get in the general type inference
      // map.
      XLS_ASSIGN_OR_RETURN(auto literal_map, InferVastTypes(literal));
      const auto it = literal_map.find(literal);
      QCHECK(it != literal_map.end());
      vast_type = it->second;
    }
    dslx::TypeAnnotation* force_dslx_type = nullptr;
    if ((*vast_type)->IsUserDefined()) {
      XLS_ASSIGN_OR_RETURN(force_dslx_type, TranslateType(*vast_type));
    }
    return dslx_builder_->HandleConstVal(CreateNodeSpan(literal),
                                         literal->bits(), literal->format(),
                                         *vast_type, force_dslx_type);
  }

  absl::StatusOr<dslx::AstNode*> TranslateParameter(
      verilog::Parameter* parameter) {
    XLS_ASSIGN_OR_RETURN(dslx::Expr * expr,
                         TranslateExpression(parameter->rhs()));
    return dslx_builder_->HandleConstantDecl(CreateNodeSpan(parameter),
                                             current_vast_module_, parameter,
                                             parameter->GetName(), expr);
  }

  absl::StatusOr<dslx::EnumDef*> TranslateEnumWithName(
      verilog::Enum* vast_enum, dslx::NameDef* name_def,
      std::string_view vast_name) {
    XLS_ASSIGN_OR_RETURN(dslx::TypeAnnotation * enum_type_annotation,
                         TranslateType(vast_enum->BaseType()));
    XLS_ASSIGN_OR_RETURN(
        std::vector<dslx::EnumMember> members,
        TranslateEnumMembers(vast_enum->members(), enum_type_annotation));
    dslx::EnumDef* enum_def = module().Make<dslx::EnumDef>(
        CreateNodeSpan(vast_enum), name_def, enum_type_annotation, members,
        /*is_public=*/true);
    name_def->set_definer(enum_def);
    enum_def->set_extern_type_name(vast_name);
    XLS_RETURN_IF_ERROR(
        module().AddTop(enum_def, /*make_collision_error=*/nullptr));
    XLS_RETURN_IF_ERROR(deduce_ctx().Deduce(enum_def).status());
    return enum_def;
  }

  absl::StatusOr<dslx::StructDef*> TranslateStructWithName(
      verilog::Struct* vast_struct, dslx::NameDef* name_def,
      std::string_view vast_name) {
    std::vector<dslx::StructMemberNode*> members;
    members.reserve(vast_struct->members().size());
    for (verilog::Def* def : vast_struct->members()) {
      XLS_ASSIGN_OR_RETURN(dslx::TypeAnnotation * type,
                           TranslateType(def->data_type()));
      std::string member_name = def->GetName();
      while (IsDslxReserved(member_name)) {
        member_name = absl::StrCat("_", member_name);
      }
      dslx::Span span = CreateNodeSpan(def);
      auto* name_def =
          module().Make<dslx::NameDef>(span, member_name, /*definer=*/nullptr);
      auto* struct_member =
          module().Make<dslx::StructMemberNode>(span, name_def, span, type);
      members.push_back(struct_member);
    }
    dslx::StructDef* struct_def = module().Make<dslx::StructDef>(
        CreateNodeSpan(vast_struct), name_def,
        /*parametric_bindings=*/std::vector<dslx::ParametricBinding*>(),
        members, /*is_public=*/true);
    name_def->set_definer(struct_def);
    struct_def->set_extern_type_name(vast_name);
    XLS_RETURN_IF_ERROR(
        module().AddTop(struct_def, /*make_collision_error=*/nullptr));
    return struct_def;
  }

  absl::StatusOr<dslx::AstNode*> TranslateTypedef(verilog::Typedef* type_def) {
    dslx::NameDef* name_def = resolver_.MakeNameDef(
        *dslx_builder_, CreateNodeSpan(type_def), type_def->GetName(), type_def,
        current_vast_module_);
    verilog::DataType* data_type = type_def->data_type();
    if (auto* struct_def = dynamic_cast<verilog::Struct*>(data_type);
        struct_def) {
      XLS_ASSIGN_OR_RETURN(
          dslx::StructDef * translated_struct,
          TranslateStructWithName(struct_def, name_def, SvName(type_def)));
      dslx_builder_->AddTypedef(current_vast_module_, type_def,
                                translated_struct);
      return translated_struct;
    }
    if (auto* enum_def = dynamic_cast<verilog::Enum*>(data_type); enum_def) {
      XLS_ASSIGN_OR_RETURN(
          dslx::EnumDef * translated_enum,
          TranslateEnumWithName(enum_def, name_def, SvName(type_def)));
      dslx_builder_->AddTypedef(current_vast_module_, type_def,
                                translated_enum);
      return translated_enum;
    }
    XLS_ASSIGN_OR_RETURN(dslx::TypeAnnotation * type_annot,
                         TranslateType(data_type));
    auto* type_alias = module().Make<dslx::TypeAlias>(
        CreateNodeSpan(type_def), *name_def, *type_annot, /*is_public=*/true);
    type_alias->set_extern_type_name(SvName(type_def));
    dslx_builder_->AddTypedef(current_vast_module_, type_def, type_alias);
    XLS_RETURN_IF_ERROR(
        module().AddTop(type_alias, /*make_collision_error=*/nullptr));
    return type_alias;
  }

  absl::StatusOr<dslx::TypeAnnotation*> TranslateType(
      verilog::DataType* data_type) {
    dslx::Span span = CreateNodeSpan(data_type);
    if (dynamic_cast<verilog::IntegerType*>(data_type)) {
      return module().Make<dslx::BuiltinTypeAnnotation>(
          span, dslx::BuiltinType::kS32,
          module().GetOrCreateBuiltinNameDef(dslx::BuiltinType::kS32));
    }
    if (auto* typedef_type = dynamic_cast<verilog::TypedefType*>(data_type);
        typedef_type) {
      XLS_ASSIGN_OR_RETURN(dslx::TypeDefinition target_type,
                           resolver_.FindTypedef(*dslx_builder_, typedef_type));
      return module().Make<dslx::TypeRefTypeAnnotation>(
          span, module().Make<dslx::TypeRef>(span, target_type),
          /*parametrics=*/std::vector<dslx::ExprOrType>{});
    }
    dslx::TypeAnnotation* annot = module().Make<dslx::BuiltinTypeAnnotation>(
        CreateNodeSpan(data_type), dslx::BuiltinType::kBits,
        module().GetOrCreateBuiltinNameDef(dslx::BuiltinType::kBits));
    if (dynamic_cast<verilog::ScalarType*>(data_type)) {
      // Handle bits -> bits[1]. We emit all bits types as bits[N], and we
      // need some index even if num_dims == 0.
      auto* dslx_one = module().Make<dslx::Number>(
          annot->span(), "1", dslx::NumberKind::kOther, nullptr);
      return module().Make<dslx::ArrayTypeAnnotation>(annot->span(), annot,
                                                      dslx_one);
    }
    std::vector<verilog::Expression*> dims;
    bool dims_are_max = false;
    if (auto* bit_vector_type =
            dynamic_cast<verilog::BitVectorType*>(data_type);
        bit_vector_type) {
      dims_are_max = bit_vector_type->max().has_value();
      if (dims_are_max) {
        QCHECK(bit_vector_type->max().has_value());
        dims.push_back(*bit_vector_type->max());
      } else {
        QCHECK(bit_vector_type->width().has_value());
        dims.push_back(*bit_vector_type->width());
      }
    } else if (auto* array_type =
                   dynamic_cast<verilog::ArrayTypeBase*>(data_type);
               array_type) {
      // TODO(b/338397279): Consider changing VAST to somehow just put all the
      // dims together.
      if (auto* wrapped_bit_vector =
              dynamic_cast<verilog::BitVectorType*>(array_type->element_type());
          wrapped_bit_vector) {
        if (!wrapped_bit_vector->max().has_value() ||
            !array_type->dims_are_max()) {
          return absl::InvalidArgumentError(
              "Wrapped bit vectors are only supported if all dimensions are "
              "max values.");
        }
        dims_are_max = true;
        dims.push_back(*wrapped_bit_vector->max());
      } else {
        XLS_ASSIGN_OR_RETURN(annot, TranslateType(array_type->element_type()));
        dims_are_max = array_type->dims_are_max();
      }
      dims.insert(dims.begin(), array_type->dims().rbegin(),
                  array_type->dims().rend());
    }
    for (verilog::Expression* dim : dims) {
      XLS_ASSIGN_OR_RETURN(annot,
                           ArrayDimToTypeAnnot(dim, dims_are_max, annot));
    }
    return annot;
  }

  absl::StatusOr<std::vector<dslx::EnumMember>> TranslateEnumMembers(
      absl::Span<verilog::EnumMember* const> vast_members,
      dslx::TypeAnnotation* enum_type_annotation) {
    std::vector<dslx::EnumMember> members;
    members.reserve(members.size());
    int next_auto_value = 0;
    for (verilog::EnumMember* member : vast_members) {
      dslx::NameDef* name_def = resolver_.MakeNameDef(
          *dslx_builder_, CreateNodeSpan(member), member->GetName(), member,
          current_vast_module_);
      if (member->rhs()) {
        XLS_ASSIGN_OR_RETURN(dslx::Expr * constant_value,
                             TranslateExpression(member->rhs()));
        // According to 5.7.1 in the SV spec, an unannotated number will have at
        // least 32 bits, but inside an enum, we know the _actual_ size, so
        // apply it here.
        auto* number_value = dynamic_cast<dslx::Number*>(constant_value);
        if (number_value != nullptr) {
          number_value->SetTypeAnnotation(enum_type_annotation);
        }
        auto* expr = down_cast<dslx::Expr*>(constant_value);
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::Type> member_expr_type,
                             deduce_ctx().Deduce(expr));
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::Type> enum_type,
                             deduce_ctx().Deduce(enum_type_annotation));
        XLS_ASSIGN_OR_RETURN(
            enum_type,
            UnwrapMetaType(std::move(enum_type), enum_type_annotation->span(),
                           "enum type", file_table()));

        XLS_ASSIGN_OR_RETURN(bool member_signedness,
                             IsSigned(*member_expr_type));
        XLS_ASSIGN_OR_RETURN(bool enum_signedness, IsSigned(*enum_type));

        // We need a cast here, for example, if the member is an unsigned
        // expression (e.g. a concatenation) but the enum type is signed.
        if (member_signedness != enum_signedness) {
          expr = module().Make<dslx::Cast>(CreateNodeSpan(member), expr,
                                           enum_type_annotation);
        }
        members.push_back({name_def, expr});
      } else {
        members.push_back(
            {name_def, module().Make<dslx::Number>(
                           name_def->span(), absl::StrCat(next_auto_value),
                           dslx::NumberKind::kOther, enum_type_annotation)});
      }
      ++next_auto_value;
    }
    return members;
  }

  absl::StatusOr<std::vector<dslx::Expr*>> TranslateFunctionCallArgs(
      absl::Span<verilog::Expression* const> vast_args) {
    std::vector<dslx::Expr*> args;
    args.reserve(vast_args.size());
    for (verilog::Expression* arg : vast_args) {
      XLS_ASSIGN_OR_RETURN(dslx::Expr * expr, TranslateExpression(arg));
      args.push_back(expr);
    }
    return args;
  }

  absl::StatusOr<dslx::Expr*> TranslateFunctionCall(
      verilog::VerilogFunctionCall* call) {
    dslx::Span span = CreateNodeSpan(call);
    XLS_ASSIGN_OR_RETURN(
        dslx::Expr * name_ref,
        resolver_.MakeNameRef(*dslx_builder_, span, call->func()->name(),
                              call->func()));
    XLS_ASSIGN_OR_RETURN(std::vector<dslx::Expr*> args,
                         TranslateFunctionCallArgs(call->args()));

    return module().Make<dslx::Invocation>(span, name_ref, args);
  }

  absl::StatusOr<dslx::AstNode*> TranslateFunction(
      verilog::VerilogFunction* function) {
    XLS_ASSIGN_OR_RETURN(
        dslx::TypeAnnotation * fn_type,
        TranslateType(function->return_value_def()->data_type()));
    std::vector<dslx::Param*> params;
    params.reserve(function->arguments().size());
    for (verilog::Def* def : function->arguments()) {
      XLS_ASSIGN_OR_RETURN(dslx::Param * arg, TranslateFunctionArgument(def));
      params.push_back(arg);
    }

    if (function->statement_block()->statements().size() != 1) {
      return absl::InvalidArgumentError(
          "Function translation is currently limited to one-statement "
          "functions.");
    }

    XLS_ASSIGN_OR_RETURN(
        dslx::Statement * stmt,
        TranslateStatement(function->statement_block()->statements()[0]));
    dslx::StatementBlock* block = module().Make<dslx::StatementBlock>(
        CreateNodeSpan(function), std::vector<dslx::Statement*>{stmt},
        /*trailing_semi=*/false);
    dslx::NameDef* fn_name =
        resolver_.MakeNameDef(*dslx_builder_, CreateNodeSpan(function),
                              function->name(), function, current_vast_module_);
    auto* fn = module().Make<dslx::Function>(
        CreateNodeSpan(function), fn_name,
        /*parametric_bindings=*/std::vector<dslx::ParametricBinding*>(), params,
        fn_type, block, dslx::FunctionTag::kNormal,
        /*is_public=*/true);
    XLS_RETURN_IF_ERROR(module().AddTop(fn, /*make_collision_error=*/nullptr));
    fn_name->set_definer(fn);
    return fn;
  }

  absl::StatusOr<dslx::Statement*> TranslateReturnStatement(
      verilog::ReturnStatement* statement) {
    XLS_ASSIGN_OR_RETURN(dslx::Expr * expr,
                         TranslateExpression(statement->expr()));
    return module().Make<dslx::Statement>(expr);
  }

  absl::StatusOr<dslx::Statement*> TranslateStatement(
      verilog::Statement* statement) {
    return TranslateByNodeType(statement,
                               &VastToDslxTranslator::TranslateReturnStatement);
  }

  absl::StatusOr<dslx::Expr*> TranslateTernary(verilog::Ternary* ternary) {
    XLS_ASSIGN_OR_RETURN(dslx::Expr * test,
                         TranslateExpression(ternary->test()));
    XLS_ASSIGN_OR_RETURN(dslx::Expr * consequent,
                         TranslateExpression(ternary->consequent()));
    XLS_ASSIGN_OR_RETURN(dslx::Expr * alternate,
                         TranslateExpression(ternary->alternate()));
    return MakeTernary(&module(), CreateNodeSpan(ternary), test, consequent,
                       alternate);
  }

  absl::StatusOr<dslx::Expr*> TranslateSystemFunctionCall(
      verilog::SystemFunctionCall* vast_call) {
    std::vector<dslx::Expr*> args;
    if (vast_call->args().has_value()) {
      XLS_ASSIGN_OR_RETURN(args,
                           TranslateFunctionCallArgs(*(vast_call->args())));
    }

    // There's no enumeration for these calls, so string matching it is.
    // Once we have more builtins to call, we'll obv. have to factor this out.
    if (strcmp(vast_call->name().c_str(), "clog2") == 0) {
      XLS_ASSIGN_OR_RETURN(
          dslx::Import * std,
          dslx_builder_->GetOrImportModule(dslx::ImportTokens({"std"})));

      XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::Type> ct,
                           deduce_ctx().Deduce(args[0]));

      dslx::Span span = CreateNodeSpan(vast_call);
      dslx::BitsType* bt = down_cast<dslx::BitsType*>(ct.get());
      auto* ubits_type = module().Make<dslx::BuiltinTypeAnnotation>(
          span, dslx::BuiltinType::kUN,
          module().GetOrCreateBuiltinNameDef(dslx::BuiltinType::kUN));

      XLS_ASSIGN_OR_RETURN(int64_t bit_width, bt->size().GetAsInt64());
      auto* bits_size = module().Make<dslx::Number>(
          span, absl::StrCat(bit_width), dslx::NumberKind::kOther, nullptr);
      if (bt->is_signed()) {
        auto* unsigned_type = module().Make<dslx::ArrayTypeAnnotation>(
            span, ubits_type, bits_size);
        args[0] = module().Make<dslx::Cast>(span, args[0], unsigned_type);
      }
      auto* name_ref =
          module().Make<dslx::NameRef>(span, "std", &std->name_def());
      auto* fn_ref = module().Make<dslx::ColonRef>(span, name_ref, "clog2");
      dslx::Expr* result = module().Make<dslx::Invocation>(span, fn_ref, args);

      XLS_ASSIGN_OR_RETURN(result, dslx_builder_->MaybeCastToInferredVastType(
                                       vast_call, result));
      XLS_RETURN_IF_ERROR(deduce_ctx().Deduce(result).status());
      return result;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported system function: ", vast_call->name()));
  }

  absl::StatusOr<dslx::Expr*> TranslateParameterRef(
      verilog::ParameterRef* ref) {
    return dslx_builder_->MakeNameRefAndMaybeCast(ref, CreateNodeSpan(ref),
                                                  ref->parameter()->GetName(),
                                                  ref->parameter());
  }

  absl::StatusOr<dslx::Expr*> TranslateEnumMemberRef(
      verilog::EnumMemberRef* ref) {
    dslx::Span span = CreateNodeSpan(ref);
    XLS_ASSIGN_OR_RETURN(verilog::Typedef * type_def,
                         resolver_.ReverseEnumTypedef(ref->enum_def()));
    XLS_ASSIGN_OR_RETURN(dslx::Expr * type_def_ref,
                         resolver_.MakeNameRef(*dslx_builder_, span,
                                               type_def->GetName(), type_def));
    return dslx_builder_->MaybeCastToInferredVastType(
        ref, module().Make<dslx::ColonRef>(
                 span, resolver_.NameRefToColonRefSubject(type_def_ref),
                 ref->member()->GetName()));
  }

  absl::StatusOr<dslx::Expr*> TranslateLogicRef(verilog::LogicRef* ref) {
    return dslx_builder_->MakeNameRefAndMaybeCast(ref, CreateNodeSpan(ref),
                                                  ref->GetName(), ref->def());
  }

  absl::StatusOr<dslx::Expr*> TranslateUnary(verilog::Unary* op) {
    auto span = CreateNodeSpan(op);
    XLS_ASSIGN_OR_RETURN(dslx::Expr * arg, TranslateExpression(op->arg()));
    dslx::Expr* result;
    switch (op->kind()) {
      case verilog::OperatorKind::kNegate:
        result = dslx_builder_->HandleUnaryOperator(
            span, dslx::UnopKind::kNegate, arg);
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported unary operator kind: ", op->kind()));
    }
    return dslx_builder_->MaybeCastToInferredVastType(op, result);
  }

  absl::StatusOr<dslx::Expr*> TranslateExpression(verilog::Expression* expr) {
    return TranslateByNodeType(
        expr, &VastToDslxTranslator::TranslateLiteral,
        &VastToDslxTranslator::TranslateBinop,
        &VastToDslxTranslator::TranslateUnary,
        &VastToDslxTranslator::TranslateParameterRef,
        &VastToDslxTranslator::TranslateEnumMemberRef,
        &VastToDslxTranslator::TranslateLogicRef,
        &VastToDslxTranslator::TranslateConcat,
        &VastToDslxTranslator::TranslateTernary,
        &VastToDslxTranslator::TranslateSystemFunctionCall,
        &VastToDslxTranslator::TranslateFunctionCall);
  }

  absl::StatusOr<dslx::TypeAnnotation*> ArrayDimToTypeAnnot(
      verilog::Expression* vast_dim, bool dim_is_max,
      dslx::TypeAnnotation* base_type) {
    XLS_ASSIGN_OR_RETURN(dslx::Expr * dim, TranslateExpression(vast_dim));
    // Reduce the width to a constant if possible.
    dslx::TraitVisitor trait_visitor;
    XLS_RETURN_IF_ERROR(dim->AcceptExpr(&trait_visitor));
    dslx::Span span = dim->span();
    if (!trait_visitor.name_refs().empty()) {
      if (dim_is_max) {
        XLS_ASSIGN_OR_RETURN(dim,
                             dslx_builder_->ConvertMaxToWidth(vast_dim, dim));
      }
      return module().Make<dslx::ArrayTypeAnnotation>(span, base_type, dim);
    }
    XLS_RETURN_IF_ERROR(deduce_ctx().Deduce(dim).status());
    XLS_ASSIGN_OR_RETURN(dslx::InterpValue range_value,
                         InterpretExpr(import_data(), type_info(), dim));
    if (range_value.IsUnit()) {
      return absl::InternalError("Expected range to be unit");
    }
    XLS_ASSIGN_OR_RETURN(int64_t int_value, range_value.GetBitValueViaSign());
    if (dim_is_max) {
      ++int_value;
    }
    dslx::Number* bit_width = module().Make<dslx::Number>(
        span, absl::StrCat(int_value), dslx::NumberKind::kOther, nullptr);
    return module().Make<dslx::ArrayTypeAnnotation>(span, base_type, bit_width);
  }

  std::string SvName(const verilog::Typedef* type_def,
                     std::optional<std::string> module_name = std::nullopt) {
    std::string real_module_name =
        module_name ? *module_name : current_vast_module_->name();
    return absl::StrCat(real_module_name, "::", type_def->GetName());
  }

  template <typename Result, typename T>
  using TranslateFn = absl::StatusOr<Result> (VastToDslxTranslator::*)(T*);

  template <typename Result, typename T>
  bool TranslateIfNodeTypeMatches(verilog::VastNode* node,
                                  TranslateFn<Result, T> fn,
                                  absl::StatusOr<Result>* out) {
    if (auto* obj = dynamic_cast<T*>(node); obj) {
      *out = (this->*fn)(obj);
      return true;
    }
    return false;
  }

  template <typename Result, typename... Ts>
  absl::StatusOr<Result> TranslateByNodeType(verilog::VastNode* node,
                                             TranslateFn<Result, Ts>... fs) {
    absl::StatusOr<Result> result;
    bool dispatched = (TranslateIfNodeTypeMatches(node, fs, &result) || ...);
    if (!dispatched) {
      return absl::InvalidArgumentError(absl::StrCat(
          "No appropriate translation function for: ", node->Emit(nullptr)));
    }
    return result;
  }

  // TODO(b/338397279): 2024-03-26 These members are law-of-demeter'd from the
  // DSLX builder as a temporary measure to start eliminating them -- ideally we
  // push DSLX building facilities into the DslxBuilder (increasingly over
  // time as we refactor) such that these are just opaque handles we don't
  // need to use in any depth.

  dslx::Module& module() { return dslx_builder_->module(); }
  dslx::DeduceCtx& deduce_ctx() { return dslx_builder_->deduce_ctx(); }
  dslx::TypeInfo& type_info() { return dslx_builder_->type_info(); }
  dslx::ImportData& import_data() { return dslx_builder_->import_data(); }
  dslx::FileTable& file_table() { return import_data().file_table(); }

  const bool generate_combined_dslx_module_;
  const std::optional<std::filesystem::path> additional_search_path_;
  const std::string dslx_stdlib_path_;
  dslx::WarningCollector warnings_;
  DslxResolver resolver_;
  absl::flat_hash_map<verilog::Expression*, verilog::DataType*> vast_type_map_;
  // The builder for the current module or the combined module.
  std::unique_ptr<DslxBuilder> dslx_builder_;

  // The current VAST module being traversed.
  verilog::Module* current_vast_module_ = nullptr;
};

absl::StatusOr<std::string> TranslateVastToDslxInternal(
    bool generate_combined_dslx_module,
    std::optional<std::filesystem::path> out_dir_path,
    std::string_view dslx_stdlib_path,
    const std::vector<std::filesystem::path>& verilog_paths_in_order,
    const absl::flat_hash_map<std::filesystem::path,
                              std::unique_ptr<verilog::VerilogFile>>&
        verilog_files,
    std::vector<std::string>* warnings) {
  DCHECK(!dslx_stdlib_path.empty());
  DCHECK(!verilog_paths_in_order.empty());
  DCHECK(!verilog_files.empty());
  // The cross-module imports that are done in non-combined mode only work if we
  // output actual files as we go.
  DCHECK(generate_combined_dslx_module || out_dir_path.has_value());
  const std::filesystem::path& target_path =
      verilog_paths_in_order[verilog_paths_in_order.size() - 1];

  auto it = verilog_files.find(target_path);
  if (it == verilog_files.end()) {
    return absl::NotFoundError(
        absl::StrCat("Verilog parser did not return a tree for target: ",
                     target_path.string()));
  }
  verilog::VerilogFile* target_file = it->second.get();
  if (target_file->members().empty() ||
      !std::holds_alternative<verilog::Module*>(target_file->members()[0])) {
    return absl::NotFoundError(
        absl::StrCat("No module found in target file: ", target_path.string()));
  }
  verilog::Module* target_module =
      std::get<verilog::Module*>(target_file->members()[0]);
  std::vector<verilog::VerilogFile*> verilog_file_vector;
  for (const std::filesystem::path& path : verilog_paths_in_order) {
    const auto file_it = verilog_files.find(path);
    if (file_it != verilog_files.end()) {
      verilog_file_vector.push_back(file_it->second.get());
    }
  }
  XLS_ASSIGN_OR_RETURN(auto vast_types, InferVastTypes(verilog_file_vector));
  auto translator = std::make_unique<VastToDslxTranslator>(
      target_module, generate_combined_dslx_module,
      /*additional_search_path=*/out_dir_path, dslx_stdlib_path,
      std::move(vast_types));
  // Preserve the ordering in the DSLX output.
  for (const std::filesystem::path& path : verilog_paths_in_order) {
    const auto file_it = verilog_files.find(path);
    if (file_it != verilog_files.end()) {
      const verilog::VerilogFile& file = *file_it->second;
      VLOG(2) << "Translating source file: " << path;
      for (verilog::FileMember member : file.members()) {
        if (std::holds_alternative<verilog::Module*>(member)) {
          verilog::Module* vast_module = std::get<verilog::Module*>(member);
          XLS_RETURN_IF_ERROR(translator->TranslateModule(vast_module));
          if (!generate_combined_dslx_module) {
            XLS_ASSIGN_OR_RETURN(std::string dslx_code,
                                 translator->GetDslxModuleText());
            std::filesystem::path x_path =
                *out_dir_path / absl::StrCat(vast_module->name(), ".x");
            VLOG(2) << "Writing DSLX module contents for VAST module "
                    << vast_module->name() << " to " << x_path;
            XLS_RETURN_IF_ERROR(SetFileContents(x_path, dslx_code));
          }
        }
      }
    }
  }
  return translator->GetDslxModuleText();
}

}  // namespace

absl::Status TranslateVastToDslx(
    std::filesystem::path out_dir_path, std::string_view dslx_stdlib_path,
    const std::vector<std::filesystem::path>& verilog_paths_in_order,
    const absl::flat_hash_map<std::filesystem::path,
                              std::unique_ptr<verilog::VerilogFile>>&
        verilog_files,
    std::vector<std::string>* warnings) {
  return TranslateVastToDslxInternal(
             /*generate_combined_dslx_module=*/false, out_dir_path,
             dslx_stdlib_path, verilog_paths_in_order, verilog_files, warnings)
      .status();
}

absl::StatusOr<std::string> TranslateVastToCombinedDslx(
    std::string_view dslx_stdlib_path,
    const std::vector<std::filesystem::path>& verilog_paths_in_order,
    const absl::flat_hash_map<std::filesystem::path,
                              std::unique_ptr<verilog::VerilogFile>>&
        verilog_files,
    std::vector<std::string>* warnings) {
  return TranslateVastToDslxInternal(/*generate_combined_dslx_module=*/true,
                                     /*out_dir_path=*/std::nullopt,
                                     dslx_stdlib_path, verilog_paths_in_order,
                                     verilog_files, warnings);
}

}  // namespace xls

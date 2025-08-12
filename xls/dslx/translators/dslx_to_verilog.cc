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

#include "xls/dslx/translators/dslx_to_verilog.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/bits.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/source_location.h"

namespace xls::dslx {

namespace {

const Type* UnboxMetaTypes(const Type* type) {
  while (type->IsMeta()) {
    type = type->AsMeta().wrapped().get();
  }
  return type;
}

// Obtains the concrete Type from an AstNode.
absl::StatusOr<Type*> GetActualType(const AstNode* node, TypeInfo* type_info,
                                    ImportData* import_data) {
  std::optional<Type*> type = type_info->GetItem(node);

  if (!type.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Unable to locate concrete type for param %s in %s",
                        node->ToInlineString(),
                        node->GetSpan()
                            .value_or(FakeSpan())
                            .ToString(import_data->file_table())));
  }

  XLS_RET_CHECK(type.value()->IsMeta());
  return type.value()->AsMeta().wrapped().get();
}

std::optional<AstNode*> GetTypeDefinition(
    const TypeAnnotation* type_annotation) {
  if (auto* ta = dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation)) {
    return TypeDefinitionToAstNode(ta->type_ref()->type_definition());
  }
  return std::nullopt;
}

// Obtain a typedef identifier for the given TypeAnnotation, named
//  1. <dslx_type_name> for DSLX type references.
//  2. <function_name>_<parameter_name>_t for anonymous types (everything else).
std::string GetVerilogTypedefIdentifier(const TypeAnnotation* type_annotation,
                                        std::string_view function_name,
                                        std::string_view param_name) {
  if (dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation)) {
    return type_annotation->ToString();
  }

  return absl::StrCat(function_name, "_", param_name, "_t");
}

// For type definitions, returns the name given to the type.
// Note: unlike nominal type name which uses deduced types that chases through
// aliases, this returns the name of a specific type definition, so you might
// get the name of an alias of an otherwise unnamed type.
std::optional<std::string_view> TypeDefinitionIdentifier(
    const TypeInfo::TypeSource& resolved_type_definition) {
  return absl::visit(
      Visitor{
          [](TypeAlias* alias) -> std::optional<std::string_view> {
            return alias->name_def().identifier();
          },
          [](StructDef* struct_def) -> std::optional<std::string_view> {
            return struct_def->name_def()->identifier();
          },
          [](ProcDef* proc_def) -> std::optional<std::string_view> {
            return proc_def->name_def()->identifier();
          },
          [](EnumDef* enum_def) -> std::optional<std::string_view> {
            return enum_def->name_def()->identifier();
          },
      },
      resolved_type_definition.definition);
}
}  // namespace

absl::StatusOr<std::pair<std::vector<int64_t>, verilog::DataType*>>
DslxTypeToVerilogManager::GetArrayDimsAndBaseType(
    const Type* type, const ArrayTypeAnnotation* array_type_annotation,
    ImportData* import_data) {
  type = UnboxMetaTypes(type);

  // Check if this "array" is actually a bits type.
  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type);
      bits_like.has_value()) {
    XLS_ASSIGN_OR_RETURN(int64_t size, bits_like->size.GetAsInt64());

    verilog::DataType* base_type =
        file_->Make<verilog::ScalarType>(SourceInfo());
    std::vector<int64_t> dims;
    if (size > 1) {
      dims.push_back(size);
    }
    return std::make_pair(dims, base_type);
  }

  const ArrayType& array_type = type->AsArray();

  // Get size.
  XLS_ASSIGN_OR_RETURN(int64_t size, array_type.size().GetAsInt64());

  // Check if element type is a bits type.
  if (std::optional<BitsLikeProperties> bits_like =
          GetBitsLike(array_type.element_type());
      bits_like.has_value()) {
    verilog::DataType* base_type =
        file_->Make<verilog::ScalarType>(SourceInfo());
    XLS_ASSIGN_OR_RETURN(int64_t bits_size, bits_like->size.GetAsInt64());
    return std::make_pair(std::vector<int64_t>{size, bits_size}, base_type);
  }

  // Check if element type is an array type as well
  if (auto element_type_annotation = dynamic_cast<const ArrayTypeAnnotation*>(
          array_type_annotation->element_type())) {
    XLS_ASSIGN_OR_RETURN(auto ret, GetArrayDimsAndBaseType(
                                       &array_type.element_type(),
                                       element_type_annotation, import_data));

    ret.first.insert(ret.first.begin(), size);

    return ret;
  }

  // This is a "single" dimension array with element of non-bits type.
  XLS_ASSIGN_OR_RETURN(verilog::DataType * element_type,
                       TypeAnnotationToVastType(
                           &array_type.element_type(),
                           array_type_annotation->element_type(), import_data));

  std::vector<int64_t> dims{size};
  return std::make_pair(dims, element_type);
}

absl::StatusOr<verilog::DataType*>
DslxTypeToVerilogManager::TypeAnnotationToVastType(
    const Type* type, const TypeAnnotation* type_annotation,
    ImportData* import_data) {
  VLOG(3) << "Converting TypeAnnotation " << type_annotation->ToString()
          << " with concrete Type to Verilog: " << *type;

  if (auto ta = dynamic_cast<const BuiltinTypeAnnotation*>(type_annotation)) {
    int64_t size = ta->GetBitCount();

    if (size == 1) {
      return file_->Make<verilog::ScalarType>(SourceInfo());
    }

    return file_->Make<verilog::BitVectorType>(SourceInfo(), size, false);
  }

  if (auto ta = dynamic_cast<const ArrayTypeAnnotation*>(type_annotation)) {
    XLS_ASSIGN_OR_RETURN((auto [dims, data_type]),
                         GetArrayDimsAndBaseType(type, ta, import_data));

    if (dims.empty()) {
      return data_type;
    }

    // Unpacked arrays of unpacked arrays are not supported.
    if (dynamic_cast<verilog::UnpackedArrayType*>(data_type) != nullptr) {
      return absl::UnimplementedError(
          absl::StrFormat("DslxTypeToVerilogManager: Unpacked array of "
                          "unpacked array not supported, type annotation: %s",
                          type_annotation->ToString()));
    }

    return file_->Make<verilog::PackedArrayType>(SourceInfo(), data_type, dims,
                                                 /*dims_are_max=*/false);
  }

  if (auto ta = dynamic_cast<const TupleTypeAnnotation*>(type_annotation)) {
    const TupleType& tuple_type = type->AsTuple();

    // Tuple types in DSLX are converted to verilog structs.
    std::vector<verilog::Def*> struct_members;

    for (int64_t i = 0; i < tuple_type.size(); ++i) {
      const TypeAnnotation* element_type_annotation = ta->members().at(i);
      const Type& element_type = tuple_type.GetMemberType(i);
      XLS_ASSIGN_OR_RETURN(
          verilog::DataType * element_data_type,
          TypeAnnotationToVastType(&element_type, element_type_annotation,
                                   import_data));

      struct_members.push_back(file_->Make<verilog::Def>(
          SourceInfo(), absl::StrFormat("index_%d", i),
          element_data_type->IsUserDefined() ? verilog::DataKind::kUser
                                             : verilog::DataKind::kLogic,
          element_data_type));
    }

    return file_->Make<verilog::Struct>(SourceInfo(), struct_members);
  }

  if (auto ta = dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation)) {
    AnyNameDef name_def =
        TypeDefinitionGetNameDef(ta->type_ref()->type_definition());
    std::string identifier = absl::visit(
        Visitor{[](const NameDef* name_def) { return name_def->identifier(); },
                [](const BuiltinNameDef* name_def) {
                  return name_def->identifier();
                }},
        name_def);
    return TypeDefinitionToVastType(ta->type_ref()->type_definition(),
                                    import_data, identifier);
  }

  return absl::InternalError(absl::StrFormat(
      "TypeAnnotation Misc %s not supported by DslxTypeToVerilogManager",
      type_annotation->ToString()));
}

absl::StatusOr<verilog::DataType*>
DslxTypeToVerilogManager::TypeDefinitionToVastType(
    const TypeDefinition& type_definition, ImportData* import_data,
    std::optional<std::string_view> identifier) {
  AstNode* type_definition_node = TypeDefinitionToAstNode(type_definition);
  XLS_ASSIGN_OR_RETURN(
      TypeInfo * type_info,
      import_data->GetRootTypeInfoForNode(type_definition_node));
  XLS_ASSIGN_OR_RETURN(
      Type * type, GetActualType(type_definition_node, type_info, import_data));
  XLS_ASSIGN_OR_RETURN(
      const TypeInfo::TypeSource resolved_type_definition_source,
      type_info->ResolveTypeDefinition(type_definition));

  std::optional<std::string_view> type_definition_name =
      TypeDefinitionIdentifier(resolved_type_definition_source);
  if (!identifier.has_value()) {
    identifier = type_definition_name;
  }
  XLS_RET_CHECK(identifier.has_value());

  std::string typedef_identifier =
      typedef_name_uniquer_->GetSanitizedUniqueName(*identifier);

  auto iter = converted_types_.find(type_definition_node);
  if (type_definition_name.has_value() && iter != converted_types_.end()) {
    return iter->second;
  }

  VLOG(3) << "Converting TypeDefinition " << type_definition_node->ToString()
          << " with concrete Type to Verilog: " << *type;

  XLS_ASSIGN_OR_RETURN(
      verilog::DataType * data_type,
      absl::visit(
          Visitor{
              [&](TypeAlias* alias) -> absl::StatusOr<verilog::DataType*> {
                XLS_ASSIGN_OR_RETURN(
                    TypeInfo * alias_type_info,
                    import_data->GetRootTypeInfoForNode(alias));
                std::optional<Type*> alias_type =
                    alias_type_info->GetItem(alias);
                XLS_RET_CHECK(alias_type.has_value()) << absl::StrFormat(
                    "Unable to locate concrete type for alias %s in %s",
                    alias->name_def().identifier(),
                    alias->GetSpan()
                        .value_or(FakeSpan())
                        .ToString(import_data->file_table()));
                return TypeAnnotationToVastType(
                    *alias_type, &alias->type_annotation(), import_data);
              },
              [&](StructDef* struct_def) -> absl::StatusOr<verilog::DataType*> {
                std::vector<verilog::Def*> vast_struct_members;

                XLS_RET_CHECK(type->IsStruct());
                const StructType& struct_type = type->AsStruct();

                VLOG(3) << "Converting struct type to Verilog: " << struct_type
                        << " size " << struct_type.size();

                for (int64_t i = 0; i < struct_type.size(); ++i) {
                  std::string_view member_name = struct_type.GetMemberName(i);
                  const TypeAnnotation* member_type_annotation =
                      struct_def->members().at(i)->type();
                  const Type& member_type = struct_type.GetMemberType(i);

                  XLS_ASSIGN_OR_RETURN(
                      verilog::DataType * element_data_type,
                      TypeAnnotationToVastType(
                          &member_type, member_type_annotation, import_data));

                  vast_struct_members.push_back(file_->Make<verilog::Def>(
                      SourceInfo(), member_name,
                      element_data_type->IsUserDefined()
                          ? verilog::DataKind::kUser
                          : verilog::DataKind::kLogic,
                      element_data_type));
                }

                return file_->Make<verilog::Struct>(SourceInfo(),
                                                    vast_struct_members);
              },
              [&](ProcDef* proc_def) -> absl::StatusOr<verilog::DataType*> {
                return absl::InternalError(absl::StrFormat(
                    "TypeAnnotation ProcDef %s not supported by "
                    "DslxTypeToVerilogManager",
                    proc_def->ToString()));
              },
              [&](UseTreeEntry* use_tree_entry)
                  -> absl::StatusOr<verilog::DataType*> {
                return absl::UnimplementedError(absl::StrFormat(
                    "TypeAnnotation UseTreeEntry %s not supported by "
                    "DslxTypeToVerilogManager",
                    use_tree_entry->ToString()));
              },
              [&](EnumDef* enum_def) -> absl::StatusOr<verilog::DataType*> {
                XLS_RET_CHECK(type->IsEnum());
                const EnumType& enum_type = type->AsEnum();
                auto it = converted_types_.find(&enum_type.nominal_type());
                if (it != converted_types_.end()) {
                  return it->second;
                }

                XLS_ASSIGN_OR_RETURN(TypeDim dim, enum_type.GetTotalBitCount());
                XLS_ASSIGN_OR_RETURN(int64_t size, dim.GetAsInt64());

                verilog::DataType* vast_enum_data_type;
                if (size == 1) {
                  vast_enum_data_type =
                      file_->Make<verilog::ScalarType>(SourceInfo());
                } else {
                  vast_enum_data_type = file_->Make<verilog::BitVectorType>(
                      SourceInfo(), size, false);
                }

                verilog::Enum* vast_enum_def = file_->Make<verilog::Enum>(
                    SourceInfo(), verilog::DataKind::kLogic,
                    vast_enum_data_type);

                for (int64_t i = 0; i < enum_def->values().size(); ++i) {
                  const std::string& member_name = enum_def->GetMemberName(i);
                  const InterpValue& member_val = enum_type.members().at(i);

                  XLS_ASSIGN_OR_RETURN(Bits member_val_as_bits,
                                       member_val.GetBits());
                  verilog::Literal* vast_literal =
                      file_->Literal(member_val_as_bits, SourceInfo());
                  vast_enum_def->AddMember(member_name, vast_literal,
                                           SourceInfo());
                }

                return vast_enum_def;
              },
          },
          resolved_type_definition_source.definition));

  // Add typedef to the verilog file.
  top_pkg_->Add<verilog::BlankLine>(SourceInfo());
  top_pkg_->Add<verilog::Comment>(
      SourceInfo(),
      absl::StrFormat("DSLX Type: %s",
                      TypeDefinitionToAstNode(type_definition)->ToString()));

  verilog::Typedef* typedef_ = top_pkg_->Add<verilog::Typedef>(
      SourceInfo(), file_->Make<verilog::Def>(SourceInfo(), typedef_identifier,
                                              data_type->IsUserDefined()
                                                  ? verilog::DataKind::kUser
                                                  : verilog::DataKind::kLogic,
                                              data_type));
  verilog::DataType* typedef_type =
      file_->Make<verilog::TypedefType>(SourceInfo(), typedef_);
  converted_types_.insert({type_definition_node, typedef_type});
  return typedef_type;
}

DslxTypeToVerilogManager::DslxTypeToVerilogManager(
    std::string_view package_name)
    : file_(std::make_unique<verilog::VerilogFile>(
          verilog::FileType::kSystemVerilog)),
      typedef_name_uniquer_(std::make_unique<NameUniquer>("__")) {
  top_pkg_ = file_->AddVerilogPackage(package_name, SourceInfo());
}

absl::Status DslxTypeToVerilogManager::AddTypeForFunctionParam(
    dslx::Function* func, dslx::ImportData* import_data,
    std::string_view param_name,
    std::optional<std::string_view> verilog_type_name) {
  if (!func->parametric_bindings().empty()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unable to convert function %s with parametric bindings",
        func->identifier()));
  }

  XLS_ASSIGN_OR_RETURN(TypeInfo * func_type_info,
                       import_data->GetRootTypeInfoForNode(func));

  XLS_ASSIGN_OR_RETURN(Param * param, func->GetParamByName(param_name));

  TypeAnnotation* type_annotation = param->type_annotation();
  XLS_ASSIGN_OR_RETURN(
      Type * type, GetActualType(type_annotation, func_type_info, import_data));

  std::string typedef_identifier =
      verilog_type_name.has_value()
          ? std::string(*verilog_type_name)
          : GetVerilogTypedefIdentifier(type_annotation, func->identifier(),
                                        param->identifier());

  return AddTypeToVerilogPackage(type, type_annotation, func_type_info,
                                 import_data, typedef_identifier);
}

absl::Status DslxTypeToVerilogManager::AddTypeForFunctionOutput(
    dslx::Function* func, dslx::ImportData* import_data,
    std::optional<std::string_view> verilog_type_name) {
  if (!func->parametric_bindings().empty()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unable to convert function %s with parametric bindings",
        func->identifier()));
  }

  XLS_ASSIGN_OR_RETURN(TypeInfo * func_type_info,
                       import_data->GetRootTypeInfoForNode(func));

  // Create a typedef for the return type, named
  //  1. <function_name>_out_t for anonymous types.
  //  2. <dslx_type_name> for DSLX type references.
  TypeAnnotation* return_type_annotation = func->return_type();
  XLS_ASSIGN_OR_RETURN(
      Type * return_type,
      GetActualType(return_type_annotation, func_type_info, import_data));

  std::string typedef_identifier =
      verilog_type_name.has_value()
          ? std::string(*verilog_type_name)
          : GetVerilogTypedefIdentifier(return_type_annotation,
                                        func->identifier(), "out");

  return AddTypeToVerilogPackage(return_type, return_type_annotation,
                                 func_type_info, import_data,
                                 typedef_identifier);
}

absl::Status DslxTypeToVerilogManager::AddTypeForTypeDefinition(
    const dslx::TypeDefinition& def, dslx::ImportData* import_data,
    std::optional<std::string_view> verilog_type_name) {
  AstNode* node = TypeDefinitionToAstNode(def);
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->GetRootTypeInfoForNode(node));
  std::string identifier = std::string(verilog_type_name.value_or(""));
  if (identifier.empty()) {
    AnyNameDef name_def = TypeDefinitionGetNameDef(def);
    identifier = absl::visit(
        Visitor{[](const NameDef* name_def) { return name_def->identifier(); },
                [](const BuiltinNameDef* name_def) {
                  return name_def->identifier();
                }},
        name_def);
  }
  XLS_ASSIGN_OR_RETURN(Type * tpe, GetActualType(node, type_info, import_data));
  return AddTypeToVerilogPackage(tpe, def, import_data, identifier);
}

absl::Status DslxTypeToVerilogManager::AddTypeToVerilogPackage(
    Type* type, const TypeDefinition& type_definition, ImportData* import_data,
    std::string_view typedef_identifier) {
  // Filter out unsupported interface types.
  if (type->HasParametricDims()) {
    return absl::InternalError(
        absl::StrFormat("Interface type %s should not be parametric for "
                        "DslxTypeToVerilogManager.",
                        type->ToString()));
  }
  if (type->HasToken()) {
    return absl::UnimplementedError(
        absl::StrFormat("Interface type %s containing tokens not supported.",
                        type->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(TypeDim type_dim, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t type_bit_count, type_dim.GetAsInt64());
  if (type_bit_count == 0) {
    return absl::UnimplementedError(absl::StrFormat(
        "Zero sized interface type %s not supported.", type->ToString()));
  }

  return TypeDefinitionToVastType(type_definition, import_data,
                                  typedef_identifier)
      .status();
}

absl::Status DslxTypeToVerilogManager::AddTypeToVerilogPackage(
    Type* type, TypeAnnotation* type_annotation, TypeInfo* type_info,
    ImportData* import_data, std::string_view typedef_identifier) {
  // Filter out unsupported interface types.
  if (type->HasParametricDims()) {
    return absl::InternalError(
        absl::StrFormat("Interface type %s should not be parametric for "
                        "DslxTypeToVerilogManager.",
                        type->ToString()));
  }
  if (type->HasToken()) {
    return absl::UnimplementedError(
        absl::StrFormat("Interface type %s containing tokens not supported.",
                        type->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(TypeDim type_dim, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t type_bit_count, type_dim.GetAsInt64());
  if (type_bit_count == 0) {
    return absl::UnimplementedError(absl::StrFormat(
        "Zero sized interface type %s not supported.", type->ToString()));
  }

  // Add typedef to the verilog file.
  XLS_ASSIGN_OR_RETURN(
      verilog::DataType * data_type,
      TypeAnnotationToVastType(type, type_annotation, import_data));

  // Check after converting to a VAST type because converting to VAST may create
  // a new typedef.
  if (std::optional<AstNode*> node = GetTypeDefinition(type_annotation);
      node.has_value()) {
    auto iter = converted_types_.find(*node);
    // If there's an existing typedef with the same name, return early.
    // We need to check the name of the typedef because the typedef identifier
    // can get a user-defined name.
    if (iter != converted_types_.end() &&
        static_cast<verilog::TypedefType*>(iter->second)
                ->type_def()
                ->GetName() == typedef_identifier) {
      return absl::OkStatus();
    }
  }

  top_pkg_->Add<verilog::BlankLine>(SourceInfo());
  top_pkg_->Add<verilog::Comment>(
      SourceInfo(), absl::StrFormat("DSLX Type: %s", type->ToString()));
  verilog::Typedef* typedef_ = top_pkg_->Add<verilog::Typedef>(
      SourceInfo(), file_->Make<verilog::Def>(SourceInfo(), typedef_identifier,
                                              data_type->IsUserDefined()
                                                  ? verilog::DataKind::kUser
                                                  : verilog::DataKind::kLogic,
                                              data_type));

  verilog::DataType* typedef_type =
      file_->Make<verilog::TypedefType>(SourceInfo(), typedef_);
  converted_types_.insert({type_annotation, typedef_type});

  return absl::OkStatus();
}

}  // namespace xls::dslx

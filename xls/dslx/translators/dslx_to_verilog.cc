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
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/bits.h"
#include "xls/ir/source_location.h"

namespace xls::dslx {

namespace {

// Obtains the concrete Type from a TypeAnnotation.
absl::StatusOr<Type*> GetActualTypeForParameter(
    const TypeAnnotation* type_annotation, TypeInfo* type_info,
    ImportData* import_data, std::string_view param_name = "") {
  std::optional<Type*> type = type_info->GetItem(type_annotation);

  if (!type.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Unable to locate concrete type for param %s in %s", param_name,
        type_annotation->span().ToString(import_data->file_table())));
  }

  CHECK(type.value()->IsMeta());
  return type.value()->AsMeta().wrapped().get();
}

// Obtain a typedef identifier for the given TypeAnnotation, named
//  1. <dslx_type_name> for DSLX type references.
//  2. <function_name>_<parameter_name>_t for anonymous types (everything else).
std::string GetVerilogTypedefIdentifier(const TypeAnnotation* type_annotation,
                                        std::string_view function_name,
                                        std::string_view param_name) {
  if (auto ta = dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation)) {
    return type_annotation->ToString();
  }

  return absl::StrCat(function_name, "_", param_name, "_t");
}

}  // namespace

DslxTypeToVerilogManager::DslxTypeToVerilogManager(
    std::string_view package_name)
    : file_(std::make_unique<verilog::VerilogFile>(
          verilog::FileType::kSystemVerilog)) {
  top_pkg_ = file_->AddVerilogPackage(package_name, SourceInfo());
}

absl::Status DslxTypeToVerilogManager::AddTypeForFunctionParam(
    dslx::Function* func, dslx::ImportData* import_data,
    std::string_view param_name, std::string_view verilog_type_name) {
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
      Type * type, GetActualTypeForParameter(type_annotation, func_type_info,
                                             import_data, param->ToString()));

  std::string typedef_identifier =
      verilog_type_name.empty()
          ? GetVerilogTypedefIdentifier(type_annotation, func->identifier(),
                                        param->identifier())
          : std::string(verilog_type_name);

  return AddTypeToVerilogPackage(type, type_annotation, typedef_identifier);
}

absl::Status DslxTypeToVerilogManager::AddTypeForFunctionOutput(
    dslx::Function* func, dslx::ImportData* import_data,
    std::string_view verilog_type_name) {
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
      GetActualTypeForParameter(return_type_annotation, func_type_info,
                                import_data, "<return_type>"));

  std::string typedef_identifier =
      verilog_type_name.empty()
          ? GetVerilogTypedefIdentifier(return_type_annotation,
                                        func->identifier(), "out")
          : std::string(verilog_type_name);

  return AddTypeToVerilogPackage(return_type, return_type_annotation,
                                 typedef_identifier);
}

absl::Status DslxTypeToVerilogManager::AddTypeToVerilogPackage(
    Type* type, TypeAnnotation* type_annotation,
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

  // Add typedef to the verilog file.
  top_pkg_->Add<verilog::BlankLine>(SourceInfo());
  top_pkg_->Add<verilog::Comment>(
      SourceInfo(), absl::StrFormat("DSLX Type: %s", type->ToString()));

  XLS_ASSIGN_OR_RETURN(verilog::DataType * data_type,
                       TypeAnnotationToVastType(type, type_annotation));

  top_pkg_->Add<verilog::Typedef>(
      SourceInfo(), file_->Make<verilog::Def>(SourceInfo(), typedef_identifier,
                                              data_type->IsUserDefined()
                                                  ? verilog::DataKind::kUser
                                                  : verilog::DataKind::kLogic,
                                              data_type));

  return absl::OkStatus();
}

absl::StatusOr<verilog::DataType*>
DslxTypeToVerilogManager::TypeAnnotationToVastType(
    const Type* type, const TypeAnnotation* type_annotation) {
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
    XLS_ASSIGN_OR_RETURN(auto dims_and_type, GetArrayDimsAndBaseType(type, ta));

    std::vector<int64_t>& dims = dims_and_type.first;
    verilog::DataType* data_type = dims_and_type.second;

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

    return file_->Make<verilog::UnpackedArrayType>(
        SourceInfo(), dims_and_type.second,
        absl::MakeSpan(dims_and_type.first));
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
          TypeAnnotationToVastType(&element_type, element_type_annotation));

      struct_members.push_back(file_->Make<verilog::Def>(
          SourceInfo(), absl::StrFormat("index_%d", i),
          element_data_type->IsUserDefined() ? verilog::DataKind::kUser
                                             : verilog::DataKind::kLogic,
          element_data_type));
    }

    return file_->Make<verilog::Struct>(SourceInfo(), struct_members);
  }

  if (auto ta = dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation)) {
    return absl::visit(
        Visitor{
            [&](TypeAlias* alias) -> absl::StatusOr<verilog::DataType*> {
              return TypeAnnotationToVastType(type, &alias->type_annotation());
            },
            [&](StructDef* struct_def) -> absl::StatusOr<verilog::DataType*> {
              std::vector<verilog::Def*> vast_struct_members;

              CHECK(type->IsStruct());
              const StructType& struct_type = type->AsStruct();

              VLOG(3) << "Converting struct type to Verilog: " << struct_type
                      << " size " << struct_type.size();

              for (int64_t i = 0; i < struct_type.size(); ++i) {
                std::string_view member_name = struct_type.GetMemberName(i);
                const TypeAnnotation* member_type_annotation =
                    struct_def->members().at(i)->type();
                const Type& member_type = struct_type.GetMemberType(i);

                XLS_ASSIGN_OR_RETURN(verilog::DataType * element_data_type,
                                     TypeAnnotationToVastType(
                                         &member_type, member_type_annotation));

                vast_struct_members.push_back(
                    file_->Make<verilog::Def>(SourceInfo(), member_name,
                                              element_data_type->IsUserDefined()
                                                  ? verilog::DataKind::kUser
                                                  : verilog::DataKind::kLogic,
                                              element_data_type));
              }

              return file_->Make<verilog::Struct>(SourceInfo(),
                                                  vast_struct_members);
            },
            [&](EnumDef* enum_def) -> absl::StatusOr<verilog::DataType*> {
              CHECK(type->IsEnum());

              const EnumType& enum_type = type->AsEnum();

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
                  SourceInfo(), verilog::DataKind::kLogic, vast_enum_data_type);

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
            [&](ColonRef*) -> absl::StatusOr<verilog::DataType*> {
              return absl::InternalError(
                  absl::StrFormat("TypeAnnotation ColonRef %s not yet "
                                  "supported by DslxTypeToVerilogManager",
                                  type_annotation->ToString()));
            },
            [&](ProcDef*) -> absl::StatusOr<verilog::DataType*> {
              return absl::InternalError(
                  absl::StrFormat("TypeAnnotation ProcDef %s not supported by "
                                  "DslxTypeToVerilogManager",
                                  type_annotation->ToString()));
            },
        },
        ta->type_ref()->type_definition());
  }

  return absl::InternalError(absl::StrFormat(
      "TypeAnnotation Misc %s not supported by DslxTypeToVerilogManager",
      type_annotation->ToString()));
}

absl::StatusOr<std::pair<std::vector<int64_t>, verilog::DataType*>>
DslxTypeToVerilogManager::GetArrayDimsAndBaseType(
    const Type* type, const ArrayTypeAnnotation* array_type_annotation) {
  // Check if this "array" is actualy a bits type.
  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type);
      bits_like.has_value()) {
    XLS_ASSIGN_OR_RETURN(int64_t size, bits_like->size.GetAsInt64());

    verilog::DataType* base_type = nullptr;
    if (size == 1) {
      base_type = file_->Make<verilog::ScalarType>(SourceInfo());
    } else {
      base_type =
          file_->Make<verilog::BitVectorType>(SourceInfo(), size, false);
    }

    return std::make_pair(std::vector<int64_t>(), base_type);
  }

  const ArrayType& array_type = type->AsArray();

  // Get size.
  XLS_ASSIGN_OR_RETURN(int64_t size, array_type.size().GetAsInt64());

  // Check if element type is an array type as well
  if (auto element_type_annotation = dynamic_cast<const ArrayTypeAnnotation*>(
          array_type_annotation->element_type())) {
    XLS_ASSIGN_OR_RETURN(auto ret,
                         GetArrayDimsAndBaseType(&array_type.element_type(),
                                                 element_type_annotation));

    ret.first.insert(ret.first.begin(), size);

    return ret;
  }

  // This is a "single" dimension array.
  // Get the element type.
  XLS_ASSIGN_OR_RETURN(
      verilog::DataType * element_type,
      TypeAnnotationToVastType(&array_type.element_type(),
                               array_type_annotation->element_type()));

  std::vector<int64_t> dims{size};
  return std::make_pair(dims, element_type);
}

}  // namespace xls::dslx

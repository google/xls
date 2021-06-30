// Copyright 2020 The XLS Authors
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

#include "xls/dslx/cpp_transpiler.h"

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "xls/common/case_converters.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {
namespace {

// Just a helper to avoid long signatures.
struct TranspileData {
  Module* module;
  TypeInfo* type_info;
  ImportData* import_data;
  absl::Span<const std::filesystem::path> addl_search_paths;
};

absl::StatusOr<std::string> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition);

absl::StatusOr<std::string> TranspileColonRef(const TranspileData& xpile_data,
                                              const ColonRef* colon_ref) {
  return absl::UnimplementedError("TranspileColonRef not yet implemented.");
}

absl::StatusOr<std::string> TranspileEnumDef(const TranspileData& xpile_data,
                                             const EnumDef* enum_def) {
  constexpr absl::string_view kTemplate = "enum class %s {\n%s\n};";
  constexpr absl::string_view kMemberTemplate = "  %s = %s,";

  std::vector<std::string> members;
  for (const EnumMember& member : enum_def->values()) {
    auto typecheck_fn = [&xpile_data](Module* module) {
      return CheckModule(module, xpile_data.import_data,
                         xpile_data.addl_search_paths);
    };

    // TODO(rspringer): 2021-06-09 Support parametric values.
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Interpreter::InterpretExpr(xpile_data.module, xpile_data.type_info,
                                   typecheck_fn, xpile_data.addl_search_paths,
                                   xpile_data.import_data, {}, member.value));
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
    if (bit_count > 64) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "While transpiling enum %s: "
          "Only values up to 64b in size are currently supported. "
          "Member: %s (%d)",
          enum_def->identifier(), member.name_def->identifier(), bit_count));
    }

    std::string identifier =
        std::string(1, 'k') + Camelize(member.name_def->identifier());

    std::string val_str;
    if (value.IsSigned()) {
      XLS_ASSIGN_OR_RETURN(int64_t int_val, value.GetBitValueInt64());
      val_str = absl::StrCat(int_val);
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t int_val, value.GetBitValueUint64());
      val_str = absl::StrCat(int_val);
    }
    members.push_back(absl::StrFormat(kMemberTemplate, identifier, val_str));
  }

  return absl::StrFormat(kTemplate, enum_def->identifier(),
                         absl::StrJoin(members, "\n"));
}

absl::StatusOr<std::string> TypeAnnotationToString(
    const TranspileData& xpile_data, const TypeAnnotation* annot);

absl::StatusOr<std::string> BuiltinTypeAnnotationToString(
    const TranspileData& xpile_data, const BuiltinTypeAnnotation* annot) {
  int bit_count = annot->GetBitCount();
  std::string prefix;
  if (!annot->GetSignedness()) {
    prefix = "u";
  }
  if (bit_count <= 8) {
    return prefix + "int8_t";
  } else if (bit_count <= 16) {
    return prefix + "int16_t";
  } else if (bit_count <= 32) {
    return prefix + "int32_t";
  } else if (bit_count <= 64) {
    return prefix + "int64_t";
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Only bit types up to 64b wide are currently supported: ",
                     annot->ToString()));
  }
}

absl::StatusOr<std::string> ArrayTypeAnnotationToString(
    const TranspileData& xpile_data, const ArrayTypeAnnotation* annot) {
  XLS_ASSIGN_OR_RETURN(
      std::string element_type,
      TypeAnnotationToString(xpile_data, annot->element_type()));

  uint64_t dim_int;
  if (auto* number = dynamic_cast<Number*>(annot->dim())) {
    XLS_ASSIGN_OR_RETURN(dim_int, number->GetAsUint64());
  } else {
    auto typecheck_fn = [&xpile_data](Module* module) {
      return CheckModule(module, xpile_data.import_data,
                         xpile_data.addl_search_paths);
    };
    XLS_ASSIGN_OR_RETURN(
        InterpValue dim_value,
        Interpreter::InterpretExpr(xpile_data.module, xpile_data.type_info,
                                   typecheck_fn, xpile_data.addl_search_paths,
                                   xpile_data.import_data, {}, annot->dim()));
    // TODO(rspringer): Handle multidimensional arrays.
    if (!dim_value.IsBits()) {
      return absl::UnimplementedError(
          "Multidimensional arrays aren't yet supported.");
    }
    XLS_ASSIGN_OR_RETURN(dim_int, dim_value.GetBitValueUint64());
  }
  return absl::StrCat(element_type, "[", dim_int, "]");
}

absl::StatusOr<std::string> TupleTypeAnnotationToString(
    const TranspileData& xpile_data, const TupleTypeAnnotation* annot) {
  std::vector<std::string> elements;
  for (const auto& member : annot->members()) {
    XLS_ASSIGN_OR_RETURN(std::string element,
                         TypeAnnotationToString(xpile_data, member));
    elements.push_back(element);
  }
  return absl::StrCat("std::tuple<", absl::StrJoin(elements, ", "), ">");
}

absl::StatusOr<std::string> TypeAnnotationToString(
    const TranspileData& xpile_data, const TypeAnnotation* annot) {
  if (const BuiltinTypeAnnotation* builtin =
          dynamic_cast<const BuiltinTypeAnnotation*>(annot);
      builtin != nullptr) {
    return BuiltinTypeAnnotationToString(xpile_data, builtin);
  } else if (const ArrayTypeAnnotation* array =
                 dynamic_cast<const ArrayTypeAnnotation*>(annot);
             array != nullptr) {
    return ArrayTypeAnnotationToString(xpile_data, array);
  } else if (const TupleTypeAnnotation* tuple =
                 dynamic_cast<const TupleTypeAnnotation*>(annot);
             tuple != nullptr) {
    return TupleTypeAnnotationToString(xpile_data, tuple);
  } else if (const TypeRefTypeAnnotation* type_ref =
                 dynamic_cast<const TypeRefTypeAnnotation*>(annot);
             type_ref != nullptr) {
    return type_ref->ToString();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown TypeAnnotation kind: ", annot->ToString()));
  }
}

absl::StatusOr<std::string> TranspileTypeDef(const TranspileData& xpile_data,
                                             const TypeDef* type_def) {
  XLS_ASSIGN_OR_RETURN(
      std::string annot_str,
      TypeAnnotationToString(xpile_data, type_def->type_annotation()));
  return absl::StrFormat("using %s = %s;", Camelize(type_def->identifier()),
                         annot_str);
}

absl::StatusOr<std::string> TranspileStructDef(const StructDef* struct_def) {
  return absl::UnimplementedError("TranspileStructDef not yet implemented.");
}

absl::StatusOr<std::string> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition) {
  return absl::visit(Visitor{[&](const TypeDef* type_def) {
                               return TranspileTypeDef(xpile_data, type_def);
                             },
                             [](const StructDef* struct_def) {
                               return TranspileStructDef(struct_def);
                             },
                             [&](const EnumDef* enum_def) {
                               return TranspileEnumDef(xpile_data, enum_def);
                             },
                             [&](const ColonRef* colon_ref) {
                               return TranspileColonRef(xpile_data, colon_ref);
                             }},
                     type_definition);
}

}  // namespace

absl::StatusOr<std::string> TranspileToCpp(
    Module* module, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  std::vector<std::string> results;
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->GetRootTypeInfo(module));
  struct TranspileData xpile_data {
    module, type_info, import_data, additional_search_paths
  };

  // Don't need to worry aboot ordering, since constexpr eval does it for us.
  for (const TypeDefinition& def : module->GetTypeDefinitions()) {
    XLS_ASSIGN_OR_RETURN(std::string result,
                         TranspileSingleToCpp(xpile_data, def));
    results.push_back(result);
  }

  return absl::StrJoin(results, "\n");
}

}  // namespace xls::dslx

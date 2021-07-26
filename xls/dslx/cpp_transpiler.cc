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
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
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
};

absl::StatusOr<Sources> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition);

absl::StatusOr<Sources> TranspileColonRef(const TranspileData& xpile_data,
                                          const ColonRef* colon_ref) {
  return absl::UnimplementedError("TranspileColonRef not yet implemented.");
}

absl::StatusOr<Sources> TranspileEnumDef(const TranspileData& xpile_data,
                                         const EnumDef* enum_def) {
  constexpr absl::string_view kTemplate = "enum class %s {\n%s\n};";
  constexpr absl::string_view kMemberTemplate = "  %s = %s,";

  std::vector<std::string> members;
  for (const EnumMember& member : enum_def->values()) {
    auto typecheck_fn = [&xpile_data](Module* module) {
      return CheckModule(module, xpile_data.import_data);
    };

    // TODO(rspringer): 2021-06-09 Support parametric values.
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Interpreter::InterpretExpr(xpile_data.module, xpile_data.type_info,
                                   typecheck_fn, xpile_data.import_data, {},
                                   member.value));
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

  return Sources{absl::StrFormat(kTemplate, enum_def->identifier(),
                                 absl::StrJoin(members, "\n")),
                 ""};
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
      return CheckModule(module, xpile_data.import_data);
    };
    XLS_ASSIGN_OR_RETURN(
        InterpValue dim_value,
        Interpreter::InterpretExpr(xpile_data.module, xpile_data.type_info,
                                   typecheck_fn, xpile_data.import_data, {},
                                   annot->dim()));
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

absl::StatusOr<Sources> TranspileTypeDef(const TranspileData& xpile_data,
                                         const TypeDef* type_def) {
  XLS_ASSIGN_OR_RETURN(
      std::string annot_str,
      TypeAnnotationToString(xpile_data, type_def->type_annotation()));
  return Sources{absl::StrFormat("using %s = %s;",
                                 Camelize(type_def->identifier()), annot_str),
                 ""};
}

absl::StatusOr<std::string> SetStructMember(const TranspileData& xpile_data,
                                            absl::string_view object_name,
                                            absl::string_view field_name,
                                            int element_index,
                                            TypeAnnotation* type) {
  if (auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    return absl::StrFormat(
        "    %s.%s = elements[%d].ToBits().ToUint64().value();", object_name,
        field_name, element_index);
  } else if (auto* array_type = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    TypeAnnotation* element_type = array_type->element_type();
    // If the array size/dim is a scalar < 64b, then the element is really an
    // integral type.
    auto typecheck_fn = [&xpile_data](Module* module) {
      return CheckModule(module, xpile_data.import_data);
    };
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Interpreter::InterpretExpr(xpile_data.module, xpile_data.type_info,
                                   typecheck_fn, xpile_data.import_data, {},
                                   array_type->dim()));
    auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(element_type);
    if (builtin_type != nullptr && value.IsBits() &&
        value.GetBitValueUint64().value() < 64) {
      return absl::StrFormat(
          "    %s.%s = elements[%d].ToBits().ToUint64().value();", object_name,
          field_name, element_index);
    }

    return absl::UnimplementedError("Proper arrays not yet supported.");
  }

  return absl::UnimplementedError(
      "Only builtin types are currently supported.");
}

absl::StatusOr<std::string> StructMemberToValue(const TranspileData& xpile_data,
                                                absl::string_view member_name,
                                                TypeAnnotation* type) {
  // Because the input DSLX must be in decl order, the translators for any types
  // we encounter here must have already been defined, so we can reference them
  // without worry.
  constexpr absl::string_view kToValueTemplate = R"(    %s;
    elements.push_back(%s_value);)";

  std::string setter;
  if (auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    setter =
        absl::StrFormat("Value %s_value(%cBits(%s, /*bit_count=*/%d))",
                        member_name, builtin_type->GetSignedness() ? 'S' : 'U',
                        member_name, builtin_type->GetBitCount());
  } else {
    return absl::UnimplementedError(
        "Only builtin types are currently supported.");
  }

  return absl::StrFormat(kToValueTemplate, setter, member_name);
}

std::string GenerateOutputOperator(const TranspileData& xpile_data,
                                   const StructDef* struct_def) {
  constexpr absl::string_view kOutputOperatorTemplate =
      R"(std::ostream& operator<<(std::ostream& os, const $0& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
$1
  os << ")\n";
  return os;
})";
  std::vector<std::string> members;
  for (int i = 0; i < struct_def->members().size(); i++) {
    members.push_back(absl::StrFormat(
        R"(  os << "  %s: " << elements[%d].ToString() << "\n";)",
        struct_def->members()[i].first->identifier(), i));
  }

  return absl::Substitute(kOutputOperatorTemplate, struct_def->identifier(),
                          absl::StrJoin(members, "\n"));
}

// Should performance become an issue, optimizing struct layouts by reordering
// (packing?) struct members could be considered.
absl::StatusOr<Sources> TranspileStructDef(const TranspileData& xpile_data,
                                           const StructDef* struct_def) {
  // Move impl to .cc
  // $0: name.
  // $1: element count.
  // $2: FromValue element setters.
  // $3: ToValue element setters.
  // $4: Member vars decls.
  constexpr absl::string_view kStructTemplate = R"(struct $0 {
  static absl::StatusOr<$0> FromValue(const Value& value) {
    absl::Span<const xls::Value> elements = value.elements();
    if (elements.size() != $1) {
      return absl::InvalidArgumentError(
          "$0::FromValue input must be a $1-tuple.");
    }

    $0 result;
$2
    return result;
  }

  Value ToValue() const {
    std::vector<Value> elements;
$3
    return Value::Tuple(elements);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const $0& data);

$4
};
)";

  std::string struct_body;
  std::vector<std::string> setters;
  std::vector<std::string> to_values;
  std::vector<std::string> member_decls;
  for (int i = 0; i < struct_def->members().size(); i++) {
    std::string member_name = struct_def->members()[i].first->identifier();
    TypeAnnotation* type = struct_def->members()[i].second;

    XLS_ASSIGN_OR_RETURN(
        std::string setter,
        SetStructMember(xpile_data, "result", member_name, i, type));
    setters.push_back(setter);

    XLS_ASSIGN_OR_RETURN(std::string to_value,
                         StructMemberToValue(xpile_data, member_name, type));
    to_values.push_back(to_value);

    XLS_ASSIGN_OR_RETURN(std::string type_str,
                         TypeAnnotationToString(xpile_data, type));
    member_decls.push_back(absl::StrFormat("  %s %s;", type_str, member_name));
  }

  // for each elem, collect decl, ToValue, FromValue, operator<< decl,
  // and operator<< impl.
  std::string header = absl::Substitute(
      kStructTemplate, struct_def->identifier(), struct_def->members().size(),
      absl::StrJoin(setters, "\n"), absl::StrJoin(to_values, "\n"),
      absl::StrJoin(member_decls, "\n"));
  std::string body = GenerateOutputOperator(xpile_data, struct_def);
  return Sources{header, body};
}

absl::StatusOr<Sources> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition) {
  return absl::visit(Visitor{[&](const TypeDef* type_def) {
                               return TranspileTypeDef(xpile_data, type_def);
                             },
                             [&](const StructDef* struct_def) {
                               return TranspileStructDef(xpile_data,
                                                         struct_def);
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

absl::StatusOr<Sources> TranspileToCpp(Module* module,
                                       ImportData* import_data) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->GetRootTypeInfo(module));
  struct TranspileData xpile_data {
    module, type_info, import_data
  };

  std::vector<std::string> header;
  std::vector<std::string> body;
  for (const TypeDefinition& def : module->GetTypeDefinitions()) {
    XLS_ASSIGN_OR_RETURN(Sources result, TranspileSingleToCpp(xpile_data, def));
    header.push_back(result.header);
    body.push_back(result.body);
  }

  return Sources{absl::StrJoin(header, "\n"), absl::StrJoin(body, "\n")};
}

}  // namespace xls::dslx

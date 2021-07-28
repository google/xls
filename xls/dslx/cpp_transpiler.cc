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
#include "absl/types/variant.h"
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
  // TODO(rspringer): Handle TypeRef to TypeDef to TypeDef to ... to
  // EnumDef.
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

absl::StatusOr<std::string> GenerateScalarFromValue(
    absl::string_view src_element, absl::string_view dst_element,
    BuiltinType builtin_type, int indent_level) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, GetBuiltinTypeSignedness(builtin_type));
  return absl::StrFormat("%s%s = %s.ToBits().To%snt64().value();",
                         std::string(indent_level * 2, ' '), dst_element,
                         src_element, is_signed ? "I" : "Ui");
}

absl::StatusOr<std::string> SetArrayFromValue(const TranspileData& xpile_data,
                                              ArrayTypeAnnotation* array_type,
                                              absl::string_view src_element,
                                              absl::string_view dst_element,
                                              int indent_level) {
  constexpr absl::string_view kTemplate =
      R"(%sfor (int i = 0; i < %d; i++) {
%s
%s})";

  auto typecheck_fn = [&xpile_data](Module* module) {
    return CheckModule(module, xpile_data.import_data);
  };
  XLS_ASSIGN_OR_RETURN(
      InterpValue array_dim_value,
      Interpreter::InterpretExpr(
          xpile_data.module, xpile_data.type_info, typecheck_fn,
          xpile_data.import_data, {}, array_type->dim(), nullptr,
          xpile_data.type_info->GetItem(array_type->dim()).value()));
  if (array_dim_value.IsArray()) {
    return absl::UnimplementedError(
        "Only single-dimensional arrays are currently supported.");
  }

  TypeAnnotation* element_type = array_type->element_type();
  std::string setter;
  XLS_ASSIGN_OR_RETURN(absl::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, element_type));
  if (as_builtin_type.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        setter,
        GenerateScalarFromValue(absl::StrCat(src_element, ".element(i)"),
                                absl::StrCat(dst_element, "[i]"),
                                as_builtin_type.value(), indent_level + 1));
  } else {
    return absl::UnimplementedError(
        "Only scalars are currently supported as array elements.");
  }

  std::string indent(indent_level * 2, ' ');
  return absl::StrFormat(kTemplate, indent,
                         array_dim_value.GetBitValueUint64().value(), setter,
                         indent);
}

absl::StatusOr<std::string> SetStructMemberFromValue(
    const TranspileData& xpile_data, absl::string_view object_name,
    absl::string_view field_name, int element_index, TypeAnnotation* type,
    int indent_level) {
  XLS_ASSIGN_OR_RETURN(absl::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, type));
  if (as_builtin_type.has_value()) {
    return GenerateScalarFromValue(
        absl::StrCat("elements[", element_index, "]"),
        absl::StrCat(object_name, ".", field_name), as_builtin_type.value(),
        indent_level);
  } else if (auto* array_type = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    // GetAsBuiltinType covers the integral case above.
    return SetArrayFromValue(
        xpile_data, array_type, absl::StrCat("elements[", element_index, "]"),
        absl::StrCat(object_name, ".", field_name), indent_level);
  } else if (auto* struct_type = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    return absl::StrFormat(
        "%sXLS_ASSIGN_OR_RETURN(%s.%s, %s::FromValue(elements[%d]));",
        std::string(indent_level * 2, ' '), object_name, field_name,
        type->ToString(), element_index);
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Unsupported type for transpilation: %s.", type->ToString()));
}

// Generates code for ToValue() logic for a struct scalar member.
absl::StatusOr<std::string> GenerateScalarToValue(
    const TranspileData& xpile_data, absl::string_view src_element,
    absl::string_view dst_element, BuiltinType builtin_type, int indent_level) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, GetBuiltinTypeSignedness(builtin_type));
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, GetBuiltinTypeBitCount(builtin_type));
  return absl::StrFormat("%sValue %s(%cBits(%s, /*bit_count=*/%d));",
                         std::string(indent_level * 2, ' '), dst_element,
                         is_signed ? 'S' : 'U', src_element, bit_count);
}

// Generates code for ToValue() logic for a struct array member.
absl::StatusOr<std::string> GenerateArrayToValue(
    const TranspileData& xpile_data, absl::string_view src_element,
    ArrayTypeAnnotation* array_type, int indent_level) {
  // $0: Member/base var name
  // $1: The generated setter logic
  // $2: Indentation/padding.
  constexpr absl::string_view kSetterTemplate =
      R"($2std::vector<Value> $0_elements;
$2for (int i = 0; i < ABSL_ARRAYSIZE($0); i++) {
$1
$2  $0_elements.push_back($0_element);
$2}
$2Value $0_value = Value::ArrayOrDie($0_elements);)";

  TypeAnnotation* element_type = array_type->element_type();

  std::string setter;
  XLS_ASSIGN_OR_RETURN(absl::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, element_type));
  if (as_builtin_type.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        setter,
        GenerateScalarToValue(xpile_data, absl::StrCat(src_element, "[i]"),
                              absl::StrCat(src_element, "_element"),
                              as_builtin_type.value(), indent_level + 1));
  } else {
    return absl::UnimplementedError(
        "Only scalars are currently supported as array elements.");
  }

  return absl::Substitute(kSetterTemplate, src_element, setter,
                          std::string(indent_level * 2, ' '));
}

absl::StatusOr<std::string> StructMemberToValue(const TranspileData& xpile_data,
                                                absl::string_view member_name,
                                                TypeAnnotation* type,
                                                int indent_level) {
  // Because the input DSLX must be in decl order, the translators for any types
  // we encounter here must have already been defined, so we can reference them
  // without worry.
  // $0: Indentation.
  // $1: Setter logic.
  // $2: Member name.
  constexpr absl::string_view kToValueTemplate = R"($1
$0elements.push_back($2_value);)";

  std::string setter;
  if (auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(
        setter,
        GenerateScalarToValue(xpile_data, member_name,
                              absl::StrCat(member_name, "_value"),
                              builtin_type->builtin_type(), indent_level));
  } else if (auto* array_type = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(
        setter, GenerateArrayToValue(xpile_data, member_name, array_type,
                                     indent_level));
  } else if (auto* struct_type = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    TypeRef* type_ref = struct_type->type_ref();
    // TODO(rspringer): Handle TypeRef to TypeDef to TypeDef to ... to
    // StructDef.
    if (!absl::holds_alternative<StructDef*>(type_ref->type_definition())) {
      return absl::UnimplementedError(
          "Only direct struct type references are currently supported.");
    }

    return absl::StrFormat("%selements.push_back(%s.ToValue());",
                           std::string(indent_level * 2, ' '), member_name);
  } else {
    return absl::UnimplementedError(absl::StrFormat(
        "Unsupported type for transpilation: %s.", type->ToString()));
  }

  return absl::Substitute(kToValueTemplate, std::string(indent_level * 2, ' '),
                          setter, member_name);
}

// Generates the code for the stream output operator, i.e., operator<<.
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
absl::StatusOr<std::string> TranspileStructDefHeader(
    const TranspileData& xpile_data, const StructDef* struct_def) {
  constexpr absl::string_view kStructTemplate = R"(struct $0 {
  static absl::StatusOr<$0> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const $0& data);

$1
};
)";

  std::string struct_body;
  std::vector<std::string> member_decls;
  for (int i = 0; i < struct_def->members().size(); i++) {
    std::string member_name = struct_def->members()[i].first->identifier();
    TypeAnnotation* type = struct_def->members()[i].second;

    XLS_ASSIGN_OR_RETURN(std::string type_str,
                         TypeAnnotationToString(xpile_data, type));
    // We need to split on any brackets and add them to the end of the member
    // name. This is to transform `int[32] foo` into `int foo[32]`.
    auto first_bracket = type_str.find_first_of('[');
    if (first_bracket != type_str.npos) {
      member_name.append(type_str.substr(first_bracket));
      type_str = type_str.substr(0, first_bracket);
    }
    member_decls.push_back(absl::StrFormat("  %s %s;", type_str, member_name));
  }

  return absl::Substitute(kStructTemplate, struct_def->identifier(),
                          absl::StrJoin(member_decls, "\n"));
}

absl::StatusOr<std::string> TranspileStructDefBody(
    const TranspileData& xpile_data, const StructDef* struct_def) {
  // $0: name.
  // $1: element count.
  // $2: FromValue element setters.
  // $3: ToValue element setters.
  // $4: Stream output operator.
  constexpr absl::string_view kStructTemplate =
      R"(absl::StatusOr<$0> $0::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != $1) {
    return absl::InvalidArgumentError(
        "$0::FromValue input must be a $1-tuple.");
  }

  $0 result;
$2
  return result;
}

Value $0::ToValue() const {
  std::vector<Value> elements;
$3
  return Value::Tuple(elements);
}

$4)";

  std::string struct_body;
  std::vector<std::string> setters;
  std::vector<std::string> to_values;
  for (int i = 0; i < struct_def->members().size(); i++) {
    std::string member_name = struct_def->members()[i].first->identifier();
    TypeAnnotation* type = struct_def->members()[i].second;

    XLS_ASSIGN_OR_RETURN(
        std::string setter,
        SetStructMemberFromValue(xpile_data, /*object_name=*/"result",
                                 member_name, i, type, /*indent_level=*/1));
    setters.push_back(setter);

    XLS_ASSIGN_OR_RETURN(std::string to_value,
                         StructMemberToValue(xpile_data, member_name, type,
                                             /*indent_level=*/1));
    to_values.push_back(to_value);

    XLS_ASSIGN_OR_RETURN(std::string type_str,
                         TypeAnnotationToString(xpile_data, type));
  }

  std::string body = absl::Substitute(
      kStructTemplate, struct_def->identifier(), struct_def->members().size(),
      absl::StrJoin(setters, "\n"), absl::StrJoin(to_values, "\n"),
      GenerateOutputOperator(xpile_data, struct_def));
  return body;
}

absl::StatusOr<Sources> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition) {
  return absl::visit(
      Visitor{[&](const TypeDef* type_def) {
                return TranspileTypeDef(xpile_data, type_def);
              },
              [&](const StructDef* struct_def) -> absl::StatusOr<Sources> {
                XLS_ASSIGN_OR_RETURN(
                    std::string header,
                    TranspileStructDefHeader(xpile_data, struct_def));
                XLS_ASSIGN_OR_RETURN(
                    std::string body,
                    TranspileStructDefBody(xpile_data, struct_def));
                return Sources{header, body};
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

  return Sources{absl::StrJoin(header, "\n"), absl::StrJoin(body, "\n\n")};
}

}  // namespace xls::dslx

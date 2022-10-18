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
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "xls/common/case_converters.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/bytecode.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/bytecode_interpreter.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {
namespace {

// Just a helper to avoid long signatures.
struct TranspileData {
  Module* module;
  TypeInfo* type_info;
  ImportData* import_data;
};

absl::StatusOr<InterpValue> InterpretExpr(
    ImportData* import_data, TypeInfo* type_info, Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(import_data, type_info, expr, env,
                                      /*caller_bindings=*/absl::nullopt));
  return BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{});
}

// Camelizes the input string, unless it's a builtin like "int8_t" or a
// std::tuple (whose elements have already been Camelized as appropriate.
std::string CheckedCamelize(std::string_view input) {
  if (absl::StartsWith(input, "std::tuple")) {
    return std::string(input);
  }

  int pos = input.find_first_of('[');
  std::string suffix;
  if (pos != input.npos) {
    suffix = input.substr(pos);
    input = input.substr(0, pos);
  }

  // If this check grows much more complicated, it might make sense to RE2-ify
  // it.
  if (input == "int8_t" || input == "uint8_t" || input == "int16_t" ||
      input == "uint16_t" || input == "int32_t" || input == "uint32_t" ||
      input == "int64_t" || input == "uint64_t") {
    return std::string(input) + suffix;
  }

  return Camelize(input) + suffix;
}

absl::StatusOr<Sources> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition);

absl::StatusOr<Sources> TranspileColonRef(const TranspileData& xpile_data,
                                          const ColonRef* colon_ref) {
  return absl::UnimplementedError("TranspileColonRef not yet implemented.");
}

absl::StatusOr<Sources> TranspileEnumDef(const TranspileData& xpile_data,
                                         const EnumDef* enum_def) {
  constexpr std::string_view kTemplate =
      "enum class %s {\n%s\n};\nconstexpr int64_t k%sNumElements = %d;";
  constexpr std::string_view kMemberTemplate = "  %s = %s,";

  std::vector<std::string> members;
  for (const EnumMember& member : enum_def->values()) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        InterpretExpr(xpile_data.import_data, xpile_data.type_info,
                      member.value, /*env=*/{}));

    XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
    if (bit_count > 64) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "While transpiling enum %s: "
          "Only values up to 64b in size are currently supported. "
          "Member: %s (%d)",
          enum_def->identifier(), member.name_def->identifier(), bit_count));
    }

    std::string identifier;
    if (member.name_def->identifier()[0] != 'k') {
      identifier = absl::StrCat(std::string(1, 'k'),
                                CheckedCamelize(member.name_def->identifier()));
    } else {
      identifier = member.name_def->identifier();
    }

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

  std::string camelized_id = CheckedCamelize(enum_def->identifier());
  return Sources{
      absl::StrFormat(kTemplate, camelized_id, absl::StrJoin(members, "\n"),
                      camelized_id, members.size()),
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
  }
  if (bit_count <= 16) {
    return prefix + "int16_t";
  }
  if (bit_count <= 32) {
    return prefix + "int32_t";
  }
  if (bit_count <= 64) {
    return prefix + "int64_t";
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Only bit types up to 64b wide are currently supported: ",
                   annot->ToString()));
}

absl::StatusOr<std::string> ArrayTypeAnnotationToString(
    const TranspileData& xpile_data, const ArrayTypeAnnotation* annot) {
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, annot));
  if (as_builtin_type.has_value()) {
    XLS_ASSIGN_OR_RETURN(bool is_signed,
                         GetBuiltinTypeSignedness(as_builtin_type.value()));
    // Just make 'em 64b. Works and is easy.
    return absl::StrFormat("%sint64_t", is_signed ? "" : "u");
  }
  XLS_ASSIGN_OR_RETURN(
      std::string element_type,
      TypeAnnotationToString(xpile_data, annot->element_type()));

  uint64_t dim_int;
  if (auto* number = dynamic_cast<Number*>(annot->dim())) {
    XLS_ASSIGN_OR_RETURN(dim_int, number->GetAsUint64());
  } else {
    XLS_ASSIGN_OR_RETURN(
        InterpValue dim_value,
        InterpretExpr(xpile_data.import_data, xpile_data.type_info,
                      annot->dim(), /*env=*/{}));
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
  }
  if (const ArrayTypeAnnotation* array =
                 dynamic_cast<const ArrayTypeAnnotation*>(annot);
             array != nullptr) {
    return ArrayTypeAnnotationToString(xpile_data, array);
  }
  if (const TupleTypeAnnotation* tuple =
                 dynamic_cast<const TupleTypeAnnotation*>(annot);
             tuple != nullptr) {
    return TupleTypeAnnotationToString(xpile_data, tuple);
  }
  if (const TypeRefTypeAnnotation* type_ref =
                 dynamic_cast<const TypeRefTypeAnnotation*>(annot);
             type_ref != nullptr) {
    return type_ref->ToString();
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unknown TypeAnnotation kind: ", annot->ToString()));
}

absl::StatusOr<Sources> TranspileTypeDef(const TranspileData& xpile_data,
                                         const TypeDef* type_def) {
  XLS_ASSIGN_OR_RETURN(
      std::string annot_str,
      TypeAnnotationToString(xpile_data, type_def->type_annotation()));
  return Sources{
      absl::StrFormat("using %s = %s;", CheckedCamelize(type_def->identifier()),
                      CheckedCamelize(annot_str)),
      ""};
}

absl::StatusOr<std::string> GenerateScalarFromValue(
    std::string_view src_element, std::string_view dst_element,
    BuiltinType builtin_type, int indent_level) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, GetBuiltinTypeSignedness(builtin_type));
  return absl::StrFormat("%s%s = %s.bits().To%snt64().value();",
                         std::string(indent_level * 2, ' '), dst_element,
                         src_element, is_signed ? "I" : "Ui");
}

absl::StatusOr<std::string> GenerateEnumFromValue(
    const TranspileData& xpile_data, std::string_view src_element,
    std::string_view dst_element, std::string_view enum_name,
    TypeAnnotation* type, int indent_level) {
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, type));
  XLS_CHECK(as_builtin_type.has_value());
  XLS_ASSIGN_OR_RETURN(bool is_signed,
                       GetBuiltinTypeSignedness(as_builtin_type.value()));
  return absl::StrFormat(
      "%s%s = static_cast<%s>(%s.bits().To%snt64().value());",
      std::string(indent_level * 2, ' '), dst_element,
      CheckedCamelize(enum_name), src_element, is_signed ? "I" : "Ui");
}

absl::StatusOr<std::string> GenerateArrayFromValue(
    const TranspileData& xpile_data, ArrayTypeAnnotation* array_type,
    std::string_view src_element, std::string_view dst_element,
    int indent_level) {
  constexpr std::string_view kTemplate =
      R"(%sfor (int i = 0; i < %d; i++) {
%s
%s})";

  XLS_ASSIGN_OR_RETURN(
      InterpValue array_dim_value,
      InterpretExpr(xpile_data.import_data, xpile_data.type_info,
                    array_type->dim(), /*env=*/{}));
  if (array_dim_value.IsArray()) {
    return absl::UnimplementedError(
        "Only single-dimensional arrays are currently supported.");
  }

  TypeAnnotation* element_type = array_type->element_type();
  std::string setter;
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
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
    const TranspileData& xpile_data, std::string_view object_name,
    std::string_view field_name, int element_index, TypeAnnotation* type,
    int indent_level) {
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, type));
  if (as_builtin_type.has_value()) {
    return GenerateScalarFromValue(
        absl::StrCat("elements[", element_index, "]"),
        absl::StrCat(object_name, ".", field_name), as_builtin_type.value(),
        indent_level);
  }
  if (auto* array_type = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    // GetAsBuiltinType covers the integral case above.
    return GenerateArrayFromValue(
        xpile_data, array_type, absl::StrCat("elements[", element_index, "]"),
        absl::StrCat(object_name, ".", field_name), indent_level);
  }
  if (auto* typeref_type = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    TypeDefinition type_definition =
        typeref_type->type_ref()->type_definition();
    if (std::holds_alternative<TypeDef*>(type_definition)) {
      return SetStructMemberFromValue(
          xpile_data, object_name, field_name, element_index,
          std::get<TypeDef*>(type_definition)->type_annotation(), indent_level);
    }
    if (std::holds_alternative<EnumDef*>(type_definition)) {
      EnumDef* enum_def = std::get<EnumDef*>(type_definition);
      return GenerateEnumFromValue(
          xpile_data, absl::StrCat("elements[", element_index, "]"),
          absl::StrCat(object_name, ".", field_name), enum_def->identifier(),
          enum_def->type_annotation(), indent_level);
    }
    if (std::holds_alternative<StructDef*>(type_definition)) {
      return absl::Substitute(
          "$0auto $1_or = $2::FromValue(elements[$3]);\n"
          "$0if (!$1_or.ok()) {\n"
          "$0  return $1_or.status();\n"
          "$0}\n"
          "$0$4.$1 = $1_or.value();\n",
          std::string(indent_level * 2, ' '), field_name,
          CheckedCamelize(type->ToString()), element_index, object_name);
    }
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Unsupported type for transpilation: %s.", type->ToString()));
}

// Generates code for ToValue() logic for a struct scalar member.
absl::StatusOr<std::string> GenerateScalarToValue(
    const TranspileData& xpile_data, std::string_view src_element,
    std::string_view dst_element, BuiltinType builtin_type, int indent_level) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, GetBuiltinTypeSignedness(builtin_type));
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, GetBuiltinTypeBitCount(builtin_type));
  return absl::StrFormat("%sxls::Value %s(xls::%cBits(%s, /*bit_count=*/%d));",
                         std::string(indent_level * 2, ' '), dst_element,
                         is_signed ? 'S' : 'U', src_element, bit_count);
}

absl::StatusOr<std::string> GenerateEnumToValue(const TranspileData& xpile_data,
                                                std::string_view src_element,
                                                std::string_view dst_element,
                                                BuiltinType builtin_type,
                                                int indent_level) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, GetBuiltinTypeSignedness(builtin_type));
  XLS_ASSIGN_OR_RETURN(int bit_count, GetBuiltinTypeBitCount(builtin_type));
  return absl::StrFormat(
      "%sxls::Value %s(xls::%cBits(static_cast<%sint64_t>(%s), "
      "/*bit_count=*/%d));",
      std::string(indent_level * 2, ' '), dst_element, is_signed ? 'S' : 'U',
      is_signed ? "" : "u", src_element, bit_count);
}

// Generates code for ToValue() logic for a struct array member.
absl::StatusOr<std::string> GenerateArrayToValue(
    const TranspileData& xpile_data, std::string_view src_element,
    ArrayTypeAnnotation* array_type, int indent_level) {
  // $0: Member/base var name
  // $1: The generated setter logic
  // $2: Indentation/padding.
  constexpr std::string_view kSetterTemplate =
      R"($2std::vector<xls::Value> $0_elements;
$2for (int i = 0; i < ABSL_ARRAYSIZE($0); i++) {
$1
$2  $0_elements.push_back($0_element);
$2}
$2xls::Value $0_value = xls::Value::ArrayOrDie($0_elements);)";

  TypeAnnotation* element_type = array_type->element_type();

  std::string setter;
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, element_type));
  if (as_builtin_type.has_value()) {
    // TODO(rspringer): What if we have an array of enums?
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
                                                std::string_view member_name,
                                                TypeAnnotation* type,
                                                int indent_level) {
  // Because the input DSLX must be in decl order, the translators for any types
  // we encounter here must have already been defined, so we can reference them
  // without worry.
  // $0: Indentation.
  // $1: Setter logic.
  // $2: Member name.
  constexpr std::string_view kToValueTemplate = R"($1
$0elements.push_back($2_value);)";

  std::string setter;
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, type));
  if (as_builtin_type.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        setter, GenerateScalarToValue(xpile_data, member_name,
                                      absl::StrCat(member_name, "_value"),
                                      as_builtin_type.value(), indent_level));
  } else if (auto* array_type = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(
        setter, GenerateArrayToValue(xpile_data, member_name, array_type,
                                     indent_level));
  } else if (auto* typeref_type = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    TypeRef* type_ref = typeref_type->type_ref();
    TypeDefinition type_definition = type_ref->type_definition();
    if (std::holds_alternative<TypeDef*>(type_definition)) {
      return StructMemberToValue(
          xpile_data, member_name,
          std::get<TypeDef*>(type_definition)->type_annotation(), indent_level);
    }
    if (std::holds_alternative<EnumDef*>(type_definition)) {
      EnumDef* enum_def = std::get<EnumDef*>(type_definition);
      XLS_ASSIGN_OR_RETURN(
          std::optional<BuiltinType> enum_as_builtin_type,
          GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                           xpile_data.import_data,
                           enum_def->type_annotation()));
      XLS_CHECK(enum_as_builtin_type.has_value());
      XLS_ASSIGN_OR_RETURN(
          setter,
          GenerateEnumToValue(xpile_data, member_name,
                              absl::StrCat(member_name, "_value"),
                              enum_as_builtin_type.value(), indent_level));
    } else if (std::holds_alternative<StructDef*>(type_definition)) {
      return absl::StrFormat("%selements.push_back(%s.ToValue());",
                             std::string(indent_level * 2, ' '), member_name);
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Only direct struct type references are currently supported. ",
          "Failing reference @ ", type_ref->span().ToString(), " : ",
          type_ref->ToString()));
    }
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
  constexpr std::string_view kOutputOperatorTemplate =
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

  return absl::Substitute(kOutputOperatorTemplate,
                          CheckedCamelize(struct_def->identifier()),
                          absl::StrJoin(members, "\n"));
}

absl::StatusOr<std::optional<int64_t>> GetFieldWidth(
    const TranspileData& xpile_data, const TypeAnnotation* type) {
  XLS_ASSIGN_OR_RETURN(std::optional<BuiltinType> as_builtin_type,
                       GetAsBuiltinType(xpile_data.module, xpile_data.type_info,
                                        xpile_data.import_data, type));
  if (as_builtin_type.has_value()) {
    return GetBuiltinTypeBitCount(as_builtin_type.value());
  }
  if (auto* typeref_type =
                 dynamic_cast<const TypeRefTypeAnnotation*>(type)) {
    TypeDefinition type_definition =
        typeref_type->type_ref()->type_definition();
    if (std::holds_alternative<TypeDef*>(type_definition)) {
      return GetFieldWidth(
          xpile_data, std::get<TypeDef*>(type_definition)->type_annotation());
    }
    if (std::holds_alternative<EnumDef*>(type_definition)) {
      EnumDef* enum_def = std::get<EnumDef*>(type_definition);
      return GetFieldWidth(xpile_data, enum_def->type_annotation());
    }
  }

  return absl::nullopt;
}

// Should performance become an issue, optimizing struct layouts by reordering
// (packing?) struct members could be considered.
absl::StatusOr<std::string> TranspileStructDefHeader(
    const TranspileData& xpile_data, const StructDef* struct_def) {
  constexpr std::string_view kStructTemplate = R"(struct $0 {
  static absl::StatusOr<$0> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const $0& data);

$1$2
};)";

  std::string struct_body;
  std::vector<std::string> member_decls;
  std::vector<std::string> scalar_widths;
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
    member_decls.push_back(
        absl::StrFormat("  %s %s;", CheckedCamelize(type_str), member_name));

    XLS_ASSIGN_OR_RETURN(std::optional<int64_t> width,
                         GetFieldWidth(xpile_data, type));
    if (width.has_value()) {
      scalar_widths.push_back(
          absl::StrFormat("  static constexpr int64_t k%sWidth = %d;",
                          CheckedCamelize(member_name), width.value()));
    }
  }

  std::string width_block = absl::StrJoin(scalar_widths, "\n");
  if (!width_block.empty()) {
    width_block = "\n\n" + width_block;
  }
  return absl::Substitute(kStructTemplate,
                          CheckedCamelize(struct_def->identifier()),
                          absl::StrJoin(member_decls, "\n"), width_block);
}

absl::StatusOr<std::string> TranspileStructDefBody(
    const TranspileData& xpile_data, const StructDef* struct_def) {
  // $0: name.
  // $1: element count.
  // $2: FromValue element setters.
  // $3: ToValue element setters.
  // $4: Stream output operator.
  constexpr std::string_view kStructTemplate =
      R"(absl::StatusOr<$0> $0::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != $1) {
    return absl::InvalidArgumentError(
        "$0::FromValue input must be a $1-tuple.");
  }

  $0 result;
$2
  return result;
}

xls::Value $0::ToValue() const {
  std::vector<xls::Value> elements;
$3
  return xls::Value::Tuple(elements);
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
      kStructTemplate, CheckedCamelize(struct_def->identifier()),
      struct_def->members().size(), absl::StrJoin(setters, "\n"),
      absl::StrJoin(to_values, "\n"),
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

// Need namespaces
// Need paths
absl::StatusOr<Sources> TranspileToCpp(Module* module, ImportData* import_data,
                                       std::string_view output_header_path,
                                       std::string namespaces) {
  constexpr std::string_view kHeaderTemplate =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef $0
#define $0
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

$2$1$3

#endif  // $0
)";

  constexpr std::string_view kSourceTemplate =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#include <vector>

#include "%s"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

%s
)";
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

  std::string header_guard;
  std::filesystem::path current_path = output_header_path;
  while (!current_path.empty() && current_path != current_path.root_path()) {
    std::string chunk =
        absl::AsciiStrToUpper(std::string(current_path.filename()));
    chunk = absl::StrReplaceAll(chunk, {{".", "_"}, {"-", "_"}});
    header_guard = chunk + "_" + header_guard;
    current_path = current_path.parent_path();
  }

  std::string namespace_begin;
  std::string namespace_end;
  if (!namespaces.empty()) {
    namespace_begin = absl::StrCat("namespace ", namespaces, " {\n\n");
    namespace_end = absl::StrCat("\n\n}  // namespace ", namespaces);
  }

  return Sources{absl::Substitute(kHeaderTemplate, header_guard,
                                  absl::StrJoin(header, "\n\n"),
                                  namespace_begin, namespace_end),
                 absl::StrFormat(kSourceTemplate, output_header_path,
                                 absl::StrJoin(body, "\n\n"))};
}

}  // namespace xls::dslx

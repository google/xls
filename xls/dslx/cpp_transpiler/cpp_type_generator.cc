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

#include "xls/dslx/cpp_transpiler/cpp_type_generator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/indent.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/cpp_transpiler/cpp_emitter.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

absl::StatusOr<InterpValue> InterpretExpr(
    ImportData* import_data, TypeInfo* type_info, Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(import_data, type_info, expr, env,
                                      /*caller_bindings=*/std::nullopt));
  return BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{});
}

// A type generator for emitting a C++ enum representing a dslx::EnumDef.
class EnumCppTypeGenerator : public CppTypeGenerator {
 public:
  explicit EnumCppTypeGenerator(std::string_view cpp_type,
                                std::string_view dslx_type,
                                std::unique_ptr<CppEmitter> emitter)
      : CppTypeGenerator(cpp_type, dslx_type), emitter_(std::move(emitter)) {}
  ~EnumCppTypeGenerator() override = default;

  static absl::StatusOr<std::unique_ptr<EnumCppTypeGenerator>> Create(
      const EnumDef* enum_def, TypeInfo* type_info, ImportData* import_data) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<CppEmitter> emitter,
        CppEmitter::Create(enum_def->type_annotation(), enum_def->identifier(),
                           type_info, import_data));
    auto generator = std::make_unique<EnumCppTypeGenerator>(
        DslxTypeNameToCpp(enum_def->identifier()), enum_def->identifier(),
        std::move(emitter));
    XLS_ASSIGN_OR_RETURN(generator->enum_values_,
                         BuildEnumValues(enum_def, type_info, import_data));
    return std::move(generator);
  }

  absl::StatusOr<CppSource> GetCppSource() const override {
    std::string enum_decl = absl::StrFormat(
        "enum class %s : %s {\n%s\n};", cpp_type(), emitter_->cpp_type(),
        absl::StrJoin(
            enum_values_, "\n", [](std::string* out, const EnumValue& ev) {
              absl::StrAppendFormat(out, "  %s = %s,", ev.name, ev.value_str);
            }));
    std::string num_elements_def =
        absl::StrFormat("constexpr int64_t k%sNumElements = %d;", cpp_type(),
                        enum_values_.size());
    std::string width_def = absl::StrFormat("constexpr int64_t k%sWidth = %d;",
                                            cpp_type(), dslx_bit_count());

    CppSource to_string = ToStringFunction();
    CppSource to_dslx_string = ToDslxStringFunction();
    CppSource to_value = ToValueFunction();
    CppSource from_value = FromValueFunction();
    CppSource verify = VerifyFunction();

    return CppSource{.header = absl::StrJoin(
                         {enum_decl, num_elements_def, width_def,
                          to_string.header, to_dslx_string.header,
                          to_value.header, from_value.header, verify.header},
                         "\n"),
                     .source = absl::StrJoin(
                         {to_string.source, to_dslx_string.source,
                          to_value.source, from_value.source, verify.source},
                         "\n\n")};
  }

  int64_t dslx_bit_count() const {
    return emitter_->GetBitCountIfBitVector().value();
  }

 protected:
  struct EnumValue {
    std::string name;
    std::string value_str;
  };

  // Returns the C++ cast of the given variable to the C== base type for this
  // enum (e.g., uint8_t_).
  std::string CastToCppBaseType(std::string_view name) const {
    return absl::StrFormat("static_cast<%s>(%s)", emitter_->cpp_type(), name);
  }

  static absl::StatusOr<std::vector<EnumValue>> BuildEnumValues(
      const EnumDef* enum_def, TypeInfo* type_info, ImportData* import_data) {
    std::vector<EnumValue> members;
    for (const EnumMember& member : enum_def->values()) {
      XLS_ASSIGN_OR_RETURN(
          InterpValue value,
          InterpretExpr(import_data, type_info, member.value, /*env=*/{}));

      std::string identifier;
      if (member.name_def->identifier()[0] != 'k') {
        identifier =
            absl::StrCat(std::string(1, 'k'),
                         DslxTypeNameToCpp(member.name_def->identifier()));
      } else {
        identifier = member.name_def->identifier();
      }

      std::string val_str;
      if (value.IsSigned()) {
        XLS_ASSIGN_OR_RETURN(int64_t int_val, value.GetBitValueSigned());
        val_str = absl::StrCat(int_val);
      } else {
        XLS_ASSIGN_OR_RETURN(uint64_t int_val, value.GetBitValueUnsigned());
        val_str = absl::StrCat(int_val);
      }
      members.push_back(EnumValue{.name = identifier, .value_str = val_str});
    }
    return members;
  }

  std::string EmitToStringBody(std::string_view type_name) const {
    std::vector<std::string> pieces;
    pieces.push_back("switch (value) {");
    for (const EnumValue& ev : enum_values_) {
      pieces.push_back(absl::StrFormat("  case %s::%s: return \"%s::%s (%s)\";",
                                       cpp_type(), ev.name, type_name, ev.name,
                                       ev.value_str));
    }
    pieces.push_back(
        absl::StrFormat("  default: return absl::StrFormat(\"<unknown> "
                        "(%%v)\", %s);",
                        CastToCppBaseType("value")));
    pieces.push_back("}");
    return absl::StrJoin(pieces, "\n");
  }

  CppSource ToStringFunction() const {
    return CppSource{
        .header = absl::StrFormat(
            "std::string %sToString(%s value, int64_t indent = 0);", cpp_type(),
            cpp_type()),
        .source = absl::StrFormat(
            "std::string %sToString(%s value, int64_t indent) {\n%s\n}",
            cpp_type(), cpp_type(), Indent(EmitToStringBody(cpp_type()), 2))};
  }

  CppSource ToDslxStringFunction() const {
    return CppSource{
        .header = absl::StrFormat(
            "std::string %sToDslxString(%s value, int64_t indent = 0);",
            cpp_type(), cpp_type()),
        .source = absl::StrFormat(
            "std::string %sToDslxString(%s value, int64_t indent) {\n%s\n}",
            cpp_type(), cpp_type(), Indent(EmitToStringBody(dslx_type()), 2))};
  }

  CppSource VerifyFunction() const {
    std::string signature = absl::StrFormat("absl::Status Verify%s(%s value)",
                                            cpp_type(), cpp_type());
    std::vector<std::string> pieces;

    // Verify the value fits in the width of the DSLX type.
    std::string bitvector_verification =
        emitter_->Verify(CastToCppBaseType("value"), cpp_type(),
                         /*nesting=*/0);
    pieces.push_back(bitvector_verification);

    // Verify the value is one of the defined enum values.
    pieces.push_back("switch (value) {");
    for (const EnumValue& ev : enum_values_) {
      pieces.push_back(absl::StrFormat("  case %s::%s:", cpp_type(), ev.name));
    }
    pieces.push_back("    break;");
    pieces.push_back("  default:");
    pieces.push_back(absl::StrFormat(
        "    return absl::InvalidArgumentError(absl::StrCat(\"Invalid value "
        "for %s enum: \", value));",
        cpp_type()));
    pieces.push_back("}");
    pieces.push_back("return absl::OkStatus();");
    std::string body = absl::StrJoin(pieces, "\n");
    return CppSource{
        .header = absl::StrCat(signature, ";"),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2))};
  }

  CppSource ToValueFunction() const {
    std::string signature =
        absl::StrFormat("absl::StatusOr<::xls::Value> %sToValue(%s input)",
                        cpp_type(), cpp_type());
    std::vector<std::string> pieces;
    pieces.push_back(
        absl::StrFormat("XLS_RETURN_IF_ERROR(Verify%s(input));", cpp_type()));
    pieces.push_back("::xls::Value result;");
    pieces.push_back(emitter_->AssignToValue(
        "result", CastToCppBaseType("input"), /*nesting=*/0));
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");
    return CppSource{
        .header = absl::StrCat(signature, ";"),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2))};
  }

  CppSource FromValueFunction() const {
    std::string signature = absl::StrFormat(
        "absl::StatusOr<%s> %sFromValue(const ::xls::Value& value)", cpp_type(),
        cpp_type());
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("%s result_base;", emitter_->cpp_type()));
    pieces.push_back(
        emitter_->AssignFromValue("result_base", "value", /*nesting=*/0));
    pieces.push_back(absl::StrFormat(
        "%s result = static_cast<%s>(result_base);", cpp_type(), cpp_type()));
    pieces.push_back(
        absl::StrFormat("XLS_RETURN_IF_ERROR(Verify%s(result));", cpp_type()));
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");
    return CppSource{
        .header = absl::StrCat(signature, ";"),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2))};
  }

  std::vector<EnumValue> enum_values_;
  std::unique_ptr<CppEmitter> emitter_;
};

// A type generator for emitting a C++ `using` statement and associated support
// functions for representing a dslx::TypeAlias (e.g., `type Foo = u32`).
class TypeAliasCppTypeGenerator : public CppTypeGenerator {
 public:
  explicit TypeAliasCppTypeGenerator(std::string_view cpp_type,
                                     std::string_view dslx_type,
                                     std::unique_ptr<CppEmitter> emitter)
      : CppTypeGenerator(cpp_type, dslx_type), emitter_(std::move(emitter)) {}
  ~TypeAliasCppTypeGenerator() override = default;

  static absl::StatusOr<std::unique_ptr<TypeAliasCppTypeGenerator>> Create(
      const TypeAlias* type_alias, TypeInfo* type_info,
      ImportData* import_data) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<CppEmitter> emitter,
        CppEmitter::Create(&type_alias->type_annotation(),
                           type_alias->identifier(), type_info, import_data));
    return std::make_unique<TypeAliasCppTypeGenerator>(
        DslxTypeNameToCpp(type_alias->identifier()), type_alias->identifier(),
        std::move(emitter));
  }

  absl::StatusOr<CppSource> GetCppSource() const override {
    CppSource verify_src = VerifyFunction();
    CppSource to_string_src = ToStringFunction();
    CppSource to_dslx_string_src = ToDslxStringFunction();
    CppSource to_value_src = ToValueFunction();
    CppSource from_value_src = FromValueFunction();

    std::vector<std::string> hdr_pieces;
    hdr_pieces.push_back(
        absl::StrFormat("using %s = %s;", cpp_type(), emitter_->cpp_type()));
    std::optional<int64_t> width = emitter_->GetBitCountIfBitVector();
    if (width.has_value()) {
      hdr_pieces.push_back(absl::StrFormat("constexpr int64_t k%sWidth = %d;",
                                           cpp_type(), width.value()));
    }
    hdr_pieces.push_back(verify_src.header);
    hdr_pieces.push_back(to_string_src.header);
    hdr_pieces.push_back(to_dslx_string_src.header);
    hdr_pieces.push_back(to_value_src.header);
    hdr_pieces.push_back(from_value_src.header);
    return CppSource{
        .header = absl::StrJoin(hdr_pieces, "\n"),
        .source = absl::StrJoin(
            {verify_src.source, to_string_src.source, to_dslx_string_src.source,
             to_value_src.source, from_value_src.source},
            "\n\n")};
  }

 protected:
  // Returns the type and name to use for the parameter of functions. For
  // example, if this type name is `Foo` this function might return: `Foo
  // arg_name` or `const Foo& arg_name` depending on whether the C++ type is
  // primitive or not.
  std::string GetValueParameter(std::string_view name) const {
    // Primitive types are passed by value. All others (structs, arrays, tuples)
    // are passed by reference.
    if (emitter_->IsCppPrimitiveType()) {
      return absl::StrFormat("%s %s", cpp_type(), name);
    }
    return absl::StrFormat("const %s& %s", cpp_type(), name);
  }

  CppSource VerifyFunction() const {
    std::string signature = absl::StrFormat(
        "absl::Status Verify%s(%s)", cpp_type(), GetValueParameter("value"));
    std::vector<std::string> pieces;
    std::string verification = emitter_->Verify("value", cpp_type(),
                                                /*nesting=*/0);
    pieces.push_back(verification);
    pieces.push_back("return absl::OkStatus();");
    std::string body = absl::StrJoin(pieces, "\n");
    return CppSource{
        .header = absl::StrCat(signature, ";"),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2))};
  }

  CppSource ToValueFunction() const {
    std::string signature =
        absl::StrFormat("absl::StatusOr<::xls::Value> %sToValue(%s)",
                        cpp_type(), GetValueParameter("input"));
    std::vector<std::string> pieces;
    pieces.push_back("::xls::Value value;");
    std::string assignment =
        emitter_->AssignToValue("value", "input", /*nesting=*/0);
    pieces.push_back(assignment);
    pieces.push_back("return value;");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = absl::StrCat(signature, ";"),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2))};
  }

  CppSource ToStringFunction() const {
    std::vector<std::string> pieces;
    pieces.push_back("std::string result;");
    std::string to_string =
        emitter_->ToString("result", "indent", "value", /*nesting=*/0);
    pieces.push_back(to_string);
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");
    return CppSource{
        .header =
            absl::StrFormat("std::string %sToString(%s, int64_t indent = 0);",
                            cpp_type(), GetValueParameter("value")),
        .source = absl::StrFormat(
            "std::string %sToString(%s, int64_t indent) {\n%s\n}", cpp_type(),
            GetValueParameter("value"), Indent(body, 2))};
  }

  CppSource ToDslxStringFunction() const {
    std::vector<std::string> pieces;
    pieces.push_back("std::string result;");
    std::string to_string =
        emitter_->ToDslxString("result", "indent", "value", /*nesting=*/0);
    pieces.push_back(to_string);
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");
    return CppSource{
        .header = absl::StrFormat(
            "std::string %sToDslxString(%s, int64_t indent = 0);", cpp_type(),
            GetValueParameter("value")),
        .source = absl::StrFormat(
            "std::string %sToDslxString(%s, int64_t indent) {\n%s\n}",
            cpp_type(), GetValueParameter("value"), Indent(body, 2))};
  }

  CppSource FromValueFunction() const {
    std::string signature = absl::StrFormat(
        "absl::StatusOr<%s> %sFromValue(const ::xls::Value& value)", cpp_type(),
        cpp_type());
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("%s result;", cpp_type()));

    std::string assignment =
        emitter_->AssignFromValue("result", "value", /*nesting=*/0);
    pieces.push_back(assignment);
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = absl::StrCat(signature, ";"),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2))};
  }

  std::unique_ptr<CppEmitter> emitter_;
};

// A type generator for emitting a C++ struct for representing a DSLX struct.
class StructCppTypeGenerator : public CppTypeGenerator {
 public:
  explicit StructCppTypeGenerator(
      std::string_view cpp_type, std::string_view dslx_type,
      const StructDef* struct_def,
      std::vector<std::unique_ptr<CppEmitter>> member_emitters)
      : CppTypeGenerator(cpp_type, dslx_type),
        struct_def_(struct_def),
        member_emitters_(std::move(member_emitters)),
        cpp_member_names_([struct_def]() {
          std::vector<std::string> names;
          names.reserve(struct_def->size());
          for (int64_t i = 0; i < struct_def->size(); ++i) {
            names.push_back(SanitizeCppName(struct_def->GetMemberName(i)));
          }
          return names;
        }()) {}
  ~StructCppTypeGenerator() override = default;

  static absl::StatusOr<std::unique_ptr<StructCppTypeGenerator>> Create(
      const StructDef* struct_def, TypeInfo* type_info,
      ImportData* import_data) {
    std::vector<std::unique_ptr<CppEmitter>> member_emitters;
    for (const auto& i : struct_def->members()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<CppEmitter> emitter,
                           CppEmitter::Create(i.type, i.type->ToString(),
                                              type_info, import_data));
      member_emitters.push_back(std::move(emitter));
    }
    return std::make_unique<StructCppTypeGenerator>(
        DslxTypeNameToCpp(struct_def->identifier()), struct_def->identifier(),
        struct_def, std::move(member_emitters));
  }

  absl::StatusOr<CppSource> GetCppSource() const override {
    std::vector<std::string> member_decls;
    std::vector<std::string> scalar_widths;
    for (int64_t i = 0; i < struct_def_->size(); ++i) {
      std::string member_name = cpp_member_names_[i];
      member_decls.push_back(absl::StrFormat(
          "%s %s;", member_emitters_[i]->cpp_type(), member_name));
      std::optional<int64_t> width =
          member_emitters_[i]->GetBitCountIfBitVector();
      if (width.has_value()) {
        scalar_widths.push_back(
            absl::StrFormat("static constexpr int64_t k%sWidth = %d;",
                            DslxTypeNameToCpp(member_name), width.value()));
      }
    }

    CppSource from_value_method = FromValueMethod();
    CppSource to_value_method = ToValueMethod();
    CppSource to_string_method = ToStringMethod();
    CppSource to_dslx_string_method = ToDslxStringMethod();
    CppSource verify_method = VerifyMethod();
    CppSource operator_eq_method = OperatorEqMethod();
    CppSource operator_stream_method = OperatorStreamMethod();

    std::vector<std::string> hdr_pieces;
    hdr_pieces.insert(hdr_pieces.end(), member_decls.begin(),
                      member_decls.end());
    hdr_pieces.push_back("");
    if (!scalar_widths.empty()) {
      hdr_pieces.insert(hdr_pieces.end(), scalar_widths.begin(),
                        scalar_widths.end());
      hdr_pieces.push_back("");
    }
    hdr_pieces.push_back(from_value_method.header);
    hdr_pieces.push_back(to_value_method.header);
    hdr_pieces.push_back(to_string_method.header);
    hdr_pieces.push_back(to_dslx_string_method.header);
    hdr_pieces.push_back(verify_method.header);
    hdr_pieces.push_back(operator_eq_method.header);
    hdr_pieces.push_back(absl::StrFormat(
        "bool operator!=(const %s& other) const { return !(*this == other); }",
        cpp_type()));
    hdr_pieces.push_back(operator_stream_method.header);

    std::string members = absl::StrJoin(hdr_pieces, "\n");

    std::string header =
        absl::StrFormat("struct %s {\n%s\n};", cpp_type(), Indent(members, 2));
    std::string source =
        absl::StrJoin({from_value_method.source, to_value_method.source,
                       to_string_method.source, to_dslx_string_method.source,
                       verify_method.source, operator_eq_method.source,
                       operator_stream_method.source},
                      "\n\n");
    return CppSource{.header = header, .source = source};
  }

 protected:
  CppSource FromValueMethod() const {
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat(
        "if (!value.IsTuple() || value.size() != %d) {", struct_def_->size()));
    pieces.push_back(
        absl::StrFormat("  return absl::InvalidArgumentError(\"Value is not a "
                        "tuple of %d elements.\");",
                        struct_def_->size()));
    pieces.push_back("}");
    pieces.push_back(absl::StrFormat("%s result;", cpp_type()));
    for (int i = 0; i < struct_def_->members().size(); i++) {
      std::string assignment = member_emitters_[i]->AssignFromValue(
          /*lhs=*/absl::StrFormat("result.%s", cpp_member_names_[i]),
          /*rhs=*/absl::StrFormat("value.element(%d)", i), /*nesting=*/0);
      pieces.push_back(assignment);
    }
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = absl::StrFormat(
            "static absl::StatusOr<%s> FromValue(const ::xls::Value& value);",
            cpp_type()),
        .source = absl::StrFormat(
            "absl::StatusOr<%s> %s::FromValue(const ::xls::Value& value) "
            "{\n%s\n}",
            cpp_type(), cpp_type(), Indent(body, 2))};
  }

  CppSource ToValueMethod() const {
    std::vector<std::string> pieces;
    pieces.push_back("std::vector<::xls::Value> members;");
    pieces.push_back(
        absl::StrFormat("members.resize(%d);", struct_def_->members().size()));
    for (int i = 0; i < struct_def_->members().size(); i++) {
      std::string assignment = member_emitters_[i]->AssignToValue(
          /*lhs=*/absl::StrFormat("members[%d]", i),
          /*rhs=*/cpp_member_names_[i], /*nesting=*/0);
      pieces.push_back(assignment);
    }
    pieces.push_back("return ::xls::Value::Tuple(members);");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = "absl::StatusOr<::xls::Value> ToValue() const;",
        .source = absl::StrFormat(
            "absl::StatusOr<::xls::Value> %s::ToValue() const {\n%s\n}",
            cpp_type(), Indent(body, 2))};
  }

  CppSource VerifyMethod() const {
    std::vector<std::string> pieces;
    for (int i = 0; i < struct_def_->members().size(); i++) {
      std::string verification = member_emitters_[i]->Verify(
          cpp_member_names_[i],
          absl::StrFormat("%s.%s", cpp_type(), cpp_member_names_[i]),
          /*nesting=*/0);
      pieces.push_back(verification);
    }
    pieces.push_back("return absl::OkStatus();");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = "absl::Status Verify() const;",
        .source = absl::StrFormat("absl::Status %s::Verify() const {\n%s\n}",
                                  cpp_type(), Indent(body, 2))};
  }

  CppSource ToStringMethod() const {
    std::vector<std::string> pieces;
    pieces.push_back(
        absl::StrFormat("std::string result = \"%s {\\n\";", cpp_type()));
    for (int i = 0; i < struct_def_->members().size(); i++) {
      pieces.push_back(absl::StrFormat(
          "result += __indent(indent + 1) + \"%s: \";", cpp_member_names_[i]));
      std::string to_string = member_emitters_[i]->ToString(
          "result", "indent + 2", cpp_member_names_[i], /*nesting=*/0);
      pieces.push_back(to_string);
      pieces.push_back("result += \",\\n\";");
    }
    pieces.push_back("result += __indent(indent) + \"}\";");
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{.header = "std::string ToString(int indent = 0) const;",
                     .source = absl::StrFormat(
                         "std::string %s::ToString(int indent) const {\n%s\n}",
                         cpp_type(), Indent(body, 2))};
  }

  CppSource ToDslxStringMethod() const {
    std::vector<std::string> pieces;
    pieces.push_back(
        absl::StrFormat("std::string result = \"%s {\\n\";", dslx_type()));
    for (int i = 0; i < struct_def_->members().size(); i++) {
      pieces.push_back(absl::StrFormat(
          "result += __indent(indent + 1) + \"%s: \";", cpp_member_names_[i]));
      std::string to_string = member_emitters_[i]->ToDslxString(
          "result", "indent + 2", cpp_member_names_[i], /*nesting=*/0);
      pieces.push_back(to_string);
      pieces.push_back("result += \",\\n\";");
    }
    pieces.push_back("result += __indent(indent) + \"}\";");
    pieces.push_back("return result;");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = "std::string ToDslxString(int indent = 0) const;",
        .source = absl::StrFormat(
            "std::string %s::ToDslxString(int indent) const {\n%s\n}",
            cpp_type(), Indent(body, 2))};
  }

  CppSource OperatorEqMethod() const {
    std::vector<std::string> pieces;
    if (struct_def_->members().empty()) {
      pieces.push_back("// Empty struct.");
      pieces.push_back("return true;");
    } else {
      std::vector<std::string> member_comparisons;
      for (int i = 0; i < struct_def_->members().size(); i++) {
        const std::string& member_name = cpp_member_names_[i];
        member_comparisons.push_back(
            absl::StrFormat("%s == other.%s", member_name, member_name));
      }
      pieces.push_back(absl::StrFormat(
          "  return %s;", absl::StrJoin(member_comparisons, " && ")));
    }
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{.header = absl::StrFormat(
                         "bool operator==(const %s& other) const;", cpp_type()),
                     .source = absl::StrFormat(
                         "bool %s::operator==(const %s& other) const {\n%s\n}",
                         cpp_type(), cpp_type(), Indent(body, 2))};
  }

  CppSource OperatorStreamMethod() const {
    std::string signature = absl::StrFormat(
        "std::ostream& operator<<(std::ostream& os, const %s& data)",
        cpp_type());
    std::vector<std::string> pieces;
    pieces.push_back("os << data.ToString();");
    pieces.push_back("return os;");
    std::string body = absl::StrJoin(pieces, "\n");

    return CppSource{
        .header = absl::StrFormat("friend %s;", signature),
        .source = absl::StrFormat("%s {\n%s\n}", signature, Indent(body, 2)),
    };
  }

  const StructDef* struct_def_;
  std::vector<std::unique_ptr<CppEmitter>> member_emitters_;
  std::vector<std::string> cpp_member_names_;
};

}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<CppTypeGenerator>>
CppTypeGenerator::Create(const TypeDefinition& type_definition,
                         TypeInfo* type_info, ImportData* import_data) {
  return absl::visit(
      Visitor{[&](const TypeAlias* type_alias)
                  -> absl::StatusOr<std::unique_ptr<CppTypeGenerator>> {
                return TypeAliasCppTypeGenerator::Create(type_alias, type_info,
                                                         import_data);
              },
              [&](const StructDef* struct_def)
                  -> absl::StatusOr<std::unique_ptr<CppTypeGenerator>> {
                return StructCppTypeGenerator::Create(struct_def, type_info,
                                                      import_data);
              },
              [&](const EnumDef* enum_def)
                  -> absl::StatusOr<std::unique_ptr<CppTypeGenerator>> {
                return EnumCppTypeGenerator::Create(enum_def, type_info,
                                                    import_data);
              },
              [&](const ColonRef* colon_ref)
                  -> absl::StatusOr<std::unique_ptr<CppTypeGenerator>> {
                return absl::UnimplementedError(absl::StrFormat(
                    "Unsupported type: %s", colon_ref->ToString()));
              }},
      type_definition);
}

}  // namespace xls::dslx

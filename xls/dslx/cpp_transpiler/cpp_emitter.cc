// Copyright 2023 The XLS Authors
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

#include "xls/dslx/cpp_transpiler/cpp_emitter.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/case_converters.h"
#include "xls/common/indent.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

absl::StatusOr<int64_t> InterpretInt(Expr* expr, TypeInfo* type_info,
                                     ImportData* import_data) {
  absl::flat_hash_map<std::string, InterpValue> env;
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(import_data, type_info, expr, env,
                                      /*caller_bindings=*/std::nullopt));
  XLS_ASSIGN_OR_RETURN(
      InterpValue value,
      BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{}));
  return value.GetBitValueViaSign();
}

absl::StatusOr<int64_t> GetBitCountFromBitVectorMetadata(
    const BitVectorMetadata& bit_vector_metadata, TypeInfo* type_info,
    ImportData* import_data) {
  if (std::holds_alternative<int64_t>(bit_vector_metadata.bit_count)) {
    return std::get<int64_t>(bit_vector_metadata.bit_count);
  }
  return InterpretInt(std::get<Expr*>(bit_vector_metadata.bit_count), type_info,
                      import_data);
}

// Returns the C++ type for representing a DSLX bit vector type with the given
// bit count and signedness.
absl::StatusOr<std::string> GetBitVectorCppType(int64_t bit_count,
                                                bool is_signed) {
  std::string prefix;
  if (bit_count == 1 && is_signed) {
    // Signed one-bit numbers are represented with one-bit values in the JIT
    // which treats all values as unsigned, so the c++ representation must be
    // one-bit to conform with the JIT representation. However, there is no
    // signed one-bit value in C++.
    return absl::UnimplementedError("Signed one-bit numbers are not supported");
  }
  if (bit_count == 1) {
    return "bool";
  }
  if (!is_signed) {
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

  // Type is wider that 64-bits
  //
  // This is wrong! Bit-vectors wider than 64-bits are being represented using
  // (u)int64_t. Fortunately this is no less wrong than the previous version of
  // the cpp transpiler. However, bad implementation does enable conversion of
  // files which contain (unused) >64-bit types though those types are
  // non-functional.
  // TODO(https://github.com/google/xls/issues/1135): Fix this.
  return prefix + "int64_t";
}

// Returns the number of elements in the array defined by `type_annotation`.
absl::StatusOr<int64_t> ArraySize(const ArrayTypeAnnotation* type_annotation,
                                  TypeInfo* type_info,
                                  ImportData* import_data) {
  return InterpretInt(type_annotation->dim(), type_info, import_data);
}

// An emitter for bit-vector types which are represented using C++ primitive
// types (bool, int8_t, uint16_t, etc).
class BitVectorCppEmitter : public CppEmitter {
 public:
  explicit BitVectorCppEmitter(std::string_view cpp_type,
                               std::string_view dslx_type,
                               int64_t dslx_bit_count, bool is_signed)
      : CppEmitter(cpp_type, dslx_type),
        dslx_bit_count_(dslx_bit_count),
        is_signed_(is_signed) {}
  ~BitVectorCppEmitter() override = default;

  static absl::StatusOr<std::unique_ptr<BitVectorCppEmitter>> Create(
      std::string_view dslx_type, int64_t dslx_bit_count, bool is_signed) {
    XLS_ASSIGN_OR_RETURN(std::string cpp_type,
                         GetBitVectorCppType(dslx_bit_count, is_signed));
    return std::make_unique<BitVectorCppEmitter>(cpp_type, dslx_type,
                                                 dslx_bit_count, is_signed);
  }

  std::string AssignToValue(std::string_view lhs, std::string_view rhs,
                            int64_t nesting) const override {
    std::vector<std::string> pieces;
    if (is_signed()) {
      pieces.push_back(absl::StrFormat("if (!FitsInNBitsSigned(%s, %d)) {", rhs,
                                       dslx_bit_count()));
      pieces.push_back(absl::StrFormat(
          "  return absl::InvalidArgumentError(absl::StrFormat(\"Signed value "
          "%%#x does not fit in %d bits\", %s));",
          dslx_bit_count(), rhs));
      pieces.push_back("}");
      pieces.push_back(
          absl::StrFormat("%s = ::xls::Value(::xls::SBits(%s, %d));", lhs, rhs,
                          dslx_bit_count()));
    } else {
      pieces.push_back(absl::StrFormat("if (!FitsInNBitsUnsigned(%s, %d)) {",
                                       rhs, dslx_bit_count()));
      pieces.push_back(absl::StrFormat(
          "  return absl::InvalidArgumentError(absl::StrFormat(\"Unsigned "
          "value %%#x does not fit in %d bits\", %s));",
          dslx_bit_count(), rhs));
      pieces.push_back("}");
      pieces.push_back(
          absl::StrFormat("%s = ::xls::Value(::xls::UBits(%s, %d));", lhs, rhs,
                          dslx_bit_count()));
    }
    return absl::StrJoin(pieces, "\n");
  }

  std::string AssignFromValue(std::string_view lhs, std::string_view rhs,
                              int64_t nesting) const override {
    std::vector<std::string> pieces;
    pieces.push_back(
        absl::StrFormat("if (!%s.IsBits() || %s.bits().bit_count() != %d) {",
                        rhs, rhs, dslx_bit_count()));
    pieces.push_back(
        absl::StrFormat("  return absl::InvalidArgumentError(\"Value is not a "
                        "bits type of %d bits.\");",
                        dslx_bit_count()));
    pieces.push_back("}");
    if (is_signed()) {
      pieces.push_back(
          absl::StrFormat("%s = %s.bits().ToInt64().value();", lhs, rhs));
    } else {
      pieces.push_back(
          absl::StrFormat("%s = %s.bits().ToUint64().value();", lhs, rhs));
    }
    return absl::StrJoin(pieces, "\n");
  }

  std::string Verify(std::string_view identifier, std::string_view name,
                     int64_t nesting) const override {
    std::vector<std::string> pieces;
    if (is_signed()) {
      pieces.push_back(absl::StrFormat("if (!FitsInNBitsSigned(%s, %d)) {",
                                       identifier, dslx_bit_count()));
    } else {
      pieces.push_back(absl::StrFormat("if (!FitsInNBitsUnsigned(%s, %d)) {",
                                       identifier, dslx_bit_count()));
    }
    pieces.push_back(absl::StrCat(
        "  return absl::InvalidArgumentError(", "absl::StrCat(\"", name,
        " value does not fit in ", is_signed() ? "signed " : "",
        dslx_bit_count(), " bits: \", ", ValueAsString(identifier), "));"));
    pieces.push_back("}");
    return absl::StrJoin(pieces, "\n");
  }

  std::string ToString(std::string_view str_to_append,
                       std::string_view indent_amount,
                       std::string_view identifier,
                       int64_t nesting) const override {
    return absl::StrCat(str_to_append, " += \"bits[", dslx_bit_count(),
                        "]:\" + ", ValueAsString(identifier), ";");
  }

  std::string ToDslxString(std::string_view str_to_append,
                           std::string_view indent_amount,
                           std::string_view identifier,
                           int64_t nesting) const override {
    return absl::StrCat(str_to_append, " += \"", dslx_type(), ":\" + ",
                        ValueAsDslxString(identifier), ";");
  }

  std::optional<int64_t> GetBitCountIfBitVector() const override {
    return dslx_bit_count_;
  }
  int64_t dslx_bit_count() const { return dslx_bit_count_; }
  bool is_signed() const { return is_signed_; }

 protected:
  std::string ValueAsString(std::string_view identifier) const {
    return absl::StrCat("absl::StrFormat(\"0x%x\", ", identifier, ")");
  }

  std::string ValueAsDslxString(std::string_view identifier) const {
    if (dslx_type() == "bool") {
      return absl::StrFormat("std::string{%s ? \"true\" : \"false\"}",
                             identifier);
    }
    if (is_signed()) {
      return absl::StrCat("absl::StrFormat(\"%d\", ", identifier, ")");
    }
    return ValueAsString(identifier);
  }

  // Bit-count of the underlying DSLX type.
  int64_t dslx_bit_count_;
  bool is_signed_;
};

// An emitter for DSLX type refs. The emitted C++ code delegates all
// functionality (ToString, etc) to code generated by other emitters.
class TypeRefCppEmitter : public CppEmitter {
 public:
  // `dslx_bit_count` contains the bit count of the underlying DSLX type if it
  // is a bit-vector or std::nullopt otherwise.
  explicit TypeRefCppEmitter(const TypeRefTypeAnnotation* type_annotation,
                             std::string_view cpp_type,
                             std::string_view dslx_type,
                             std::optional<int64_t> dslx_bit_count)
      : CppEmitter(cpp_type, dslx_type),
        typeref_type_annotation_(type_annotation),
        dslx_bit_count_(dslx_bit_count) {}
  ~TypeRefCppEmitter() override = default;

  static absl::StatusOr<std::unique_ptr<TypeRefCppEmitter>> Create(
      const TypeRefTypeAnnotation* type_annotation, std::string_view dslx_type,
      TypeInfo* type_info, ImportData* import_data) {
    std::string cpp_type =
        DslxTypeNameToCpp(type_annotation->type_ref()->ToString());
    std::optional<BitVectorMetadata> bit_vector_metadata =
        ExtractBitVectorMetadata(type_annotation);
    std::optional<int64_t> dslx_bit_count;
    if (bit_vector_metadata.has_value()) {
      XLS_ASSIGN_OR_RETURN(dslx_bit_count,
                           GetBitCountFromBitVectorMetadata(
                               *bit_vector_metadata, type_info, import_data));
    }
    return std::make_unique<TypeRefCppEmitter>(type_annotation, cpp_type,
                                               dslx_type, dslx_bit_count);
  }

  std::string AssignToValue(std::string_view lhs, std::string_view rhs,
                            int64_t nesting) const override {
    return absl::StrFormat(
        "XLS_ASSIGN_OR_RETURN(%s, %s);", lhs,
        TypeHasMethods() ? absl::StrFormat("%s.ToValue()", rhs)
                         : absl::StrFormat("%sToValue(%s)", cpp_type(), rhs));
  }

  std::string AssignFromValue(std::string_view lhs, std::string_view rhs,
                              int64_t nesting) const override {
    return absl::StrFormat(
        "XLS_ASSIGN_OR_RETURN(%s, %s);", lhs,
        TypeHasMethods() ? absl::StrFormat("%s::FromValue(%s)", cpp_type(), rhs)
                         : absl::StrFormat("%sFromValue(%s)", cpp_type(), rhs));
  }

  std::string Verify(std::string_view identifier, std::string_view name,
                     int64_t nesting) const override {
    return absl::StrFormat(
        "XLS_RETURN_IF_ERROR(%s);",
        TypeHasMethods()
            ? absl::StrFormat("%s.Verify()", identifier)
            : absl::StrFormat("Verify%s(%s)", cpp_type(), identifier));
  }

  std::string ToString(std::string_view str_to_append,
                       std::string_view indent_amount,
                       std::string_view identifier,
                       int64_t nesting) const override {
    return absl::StrFormat(
        "%s += %s;", str_to_append,
        TypeHasMethods()
            ? absl::StrFormat("%s.ToString(%s)", identifier, indent_amount)
            : absl::StrFormat("%sToString(%s, %s)", cpp_type(), identifier,
                              indent_amount));
  }

  std::string ToDslxString(std::string_view str_to_append,
                           std::string_view indent_amount,
                           std::string_view identifier,
                           int64_t nesting) const override {
    return absl::StrFormat(
        "%s += %s;", str_to_append,
        TypeHasMethods()
            ? absl::StrFormat("%s.ToDslxString(%s)", identifier, indent_amount)
            : absl::StrFormat("%sToDslxString(%s, %s)", cpp_type(), identifier,
                              indent_amount));
  }

  bool TypeHasMethods() const {
    return std::holds_alternative<StructDef*>(
        typeref_type_annotation_->type_ref()->type_definition());
  }

  std::optional<int64_t> GetBitCountIfBitVector() const override {
    return dslx_bit_count_;
  }

 protected:
  const TypeRefTypeAnnotation* typeref_type_annotation_;
  // Bit-count of the underlying DSLX type if it is a bitvector.
  std::optional<int64_t> dslx_bit_count_;
};

// An emitter for DSLX array types which are represented in C++ using
// std::array.
class ArrayCppEmitter : public CppEmitter {
 public:
  explicit ArrayCppEmitter(std::string_view cpp_type,
                           std::string_view dslx_type, int64_t array_size,
                           std::unique_ptr<CppEmitter> element_emitter)
      : CppEmitter(cpp_type, dslx_type),
        array_size_(array_size),
        element_emitter_(std::move(element_emitter)) {}
  ~ArrayCppEmitter() override = default;

  static absl::StatusOr<std::unique_ptr<ArrayCppEmitter>> Create(
      const ArrayTypeAnnotation* type_annotation, TypeInfo* type_info,
      ImportData* import_data) {
    XLS_ASSIGN_OR_RETURN(int64_t array_size,
                         ArraySize(type_annotation, type_info, import_data));

    // Verify that this is *not* a bitvector type which is represented with an
    // ArrayTypeAnnotation. For example: uN[...], sN[...],, and bits[...].
    XLS_RET_CHECK(!ExtractBitVectorMetadata(type_annotation).has_value());

    XLS_ASSIGN_OR_RETURN(int64_t dim,
                         ArraySize(type_annotation, type_info, import_data));
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<CppEmitter> element_emitter,
        CppEmitter::Create(type_annotation->element_type(),
                           type_annotation->element_type()->ToString(),
                           type_info, import_data));
    std::string cpp_type =
        absl::StrFormat("std::array<%s, %d>", element_emitter->cpp_type(), dim);

    return std::make_unique<ArrayCppEmitter>(
        cpp_type, type_annotation->ToString(), array_size,
        std::move(element_emitter));
  }

  std::string AssignToValue(std::string_view lhs, std::string_view rhs,
                            int64_t nesting) const override {
    std::string ind_var = absl::StrCat("i", nesting);

    std::vector<std::string> loop_body;
    loop_body.push_back("::xls::Value element;");
    std::string element_assignment = element_emitter_->AssignToValue(
        "element", absl::StrFormat("%s[%s]", rhs, ind_var), nesting + 1);
    loop_body.push_back(element_assignment);
    loop_body.push_back("elements.push_back(element);");

    std::vector<std::string> pieces;
    pieces.push_back("std::vector<::xls::Value> elements;");
    pieces.push_back(absl::StrFormat("for (int64_t %s = 0; %s < %d; ++%s) {",
                                     ind_var, ind_var, array_size(), ind_var));
    pieces.push_back(Indent(absl::StrJoin(loop_body, "\n"), 2));
    pieces.push_back("}");
    pieces.push_back(
        absl::StrFormat("%s = ::xls::Value::ArrayOrDie(elements);", lhs));
    std::string lines = absl::StrJoin(pieces, "\n");

    return absl::StrCat("{\n", Indent(lines, 2), "\n}");
  }

  std::string AssignFromValue(std::string_view lhs, std::string_view rhs,
                              int64_t nesting) const override {
    std::string ind_var = absl::StrCat("i", nesting);
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("if (!%s.IsArray() || %s.size() != %d) {",
                                     rhs, rhs, array_size()));
    pieces.push_back(
        absl::StrFormat("  return absl::InvalidArgumentError(\"Value is not a "
                        "array of %d elements.\");",
                        array_size()));
    pieces.push_back("}");
    pieces.push_back(absl::StrFormat("for (int64_t %s = 0; %s < %d; ++%s) {",
                                     ind_var, ind_var, array_size(), ind_var));
    std::string element_assignment = element_emitter_->AssignFromValue(
        absl::StrFormat("%s[%s]", lhs, ind_var),
        absl::StrFormat("%s.element(%s)", rhs, ind_var), nesting + 1);
    pieces.push_back(Indent(element_assignment, 2));
    pieces.push_back("}");
    return absl::StrJoin(pieces, "\n");
  }

  std::string Verify(std::string_view identifier, std::string_view name,
                     int64_t nesting) const override {
    std::string ind_var = absl::StrCat("i", nesting);
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("for (int64_t %s = 0; %s < %d; ++%s) {",
                                     ind_var, ind_var, array_size(), ind_var));
    std::string verification = element_emitter_->Verify(
        absl::StrFormat("%s[%s]", identifier, ind_var), name, nesting + 1);
    pieces.push_back(Indent(verification, 2));
    pieces.push_back("}");
    return absl::StrJoin(pieces, "\n");
  }

  std::string ToString(std::string_view str_to_append,
                       std::string_view indent_amount,
                       std::string_view identifier,
                       int64_t nesting) const override {
    return EmitToString(str_to_append, indent_amount, identifier, nesting,
                        [&](std::string_view str, std::string_view indent,
                            std::string_view id, int64_t nest) {
                          return element_emitter_->ToString(str, indent, id,
                                                            nest);
                        });
  }

  std::string ToDslxString(std::string_view str_to_append,
                           std::string_view indent_amount,
                           std::string_view identifier,
                           int64_t nesting) const override {
    return EmitToString(str_to_append, indent_amount, identifier, nesting,
                        [&](std::string_view str, std::string_view indent,
                            std::string_view id, int64_t nest) {
                          return element_emitter_->ToDslxString(str, indent, id,
                                                                nest);
                        });
  }

  int64_t array_size() const { return array_size_; }

 protected:
  // Emits the C++ code for printing the array using the specified emitter
  // function.
  std::string EmitToString(
      std::string_view str_to_append, std::string_view indent_amount,
      std::string_view identifier, int64_t nesting,
      const std::function<std::string(std::string_view, std::string_view,
                                      std::string_view, int64_t)>&
          emitter_function) const {
    std::string ind_var = absl::StrCat("i", nesting);
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("%s += \"[\\n\";", str_to_append));
    pieces.push_back(absl::StrFormat("for (int64_t %s = 0; %s < %d; ++%s) {",
                                     ind_var, ind_var, array_size(), ind_var));
    std::vector<std::string> loop_body;
    loop_body.push_back(absl::StrFormat("%s += __indent(%s + 1);",
                                        str_to_append, indent_amount));
    std::string to_string = emitter_function(
        str_to_append, absl::StrCat(indent_amount, " + 1"),
        absl::StrFormat("%s[%s]", identifier, ind_var), nesting + 1);
    loop_body.push_back(Indent(to_string, 2));

    pieces.push_back(Indent(absl::StrJoin(loop_body, "\n")));
    pieces.push_back(absl::StrFormat("%s += \",\\n\";", str_to_append));
    pieces.push_back("}");
    pieces.push_back(absl::StrFormat("%s += __indent(%s) + \"]\";",
                                     str_to_append, indent_amount));
    return absl::StrJoin(pieces, "\n");
  }

  int64_t array_size_;
  std::unique_ptr<CppEmitter> element_emitter_;
};

// An emitter for DSLX tuple types which are represented in C++ using
// std::tuple.
class TupleCppEmitter : public CppEmitter {
 public:
  explicit TupleCppEmitter(
      std::string_view cpp_type, std::string_view dslx_type,
      std::vector<std::unique_ptr<CppEmitter>> element_emitters)
      : CppEmitter(cpp_type, dslx_type),
        element_emitters_(std::move(element_emitters)) {}
  ~TupleCppEmitter() override = default;

  static absl::StatusOr<std::unique_ptr<TupleCppEmitter>> Create(
      const TupleTypeAnnotation* type_annotation, TypeInfo* type_info,
      ImportData* import_data) {
    std::vector<std::unique_ptr<CppEmitter>> element_emitters;
    std::vector<std::string> element_cpp_types;
    for (TypeAnnotation* element_type : type_annotation->members()) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<CppEmitter> element_emitter,
          CppEmitter::Create(element_type, element_type->ToString(), type_info,
                             import_data));
      element_cpp_types.push_back(std::string{element_emitter->cpp_type()});
      element_emitters.push_back(std::move(element_emitter));
    }
    std::string cpp_type = absl::StrFormat(
        "std::tuple<%s>", absl::StrJoin(element_cpp_types, ", "));
    return std::make_unique<TupleCppEmitter>(
        cpp_type, type_annotation->ToString(), std::move(element_emitters));
  }

  std::string AssignToValue(std::string_view lhs, std::string_view rhs,
                            int64_t nesting) const override {
    std::vector<std::string> pieces;
    pieces.push_back("std::vector<::xls::Value> members;");
    pieces.push_back(absl::StrFormat("members.resize(%d);", size()));
    for (int64_t i = 0; i < size(); ++i) {
      std::string assignment = element_emitters_[i]->AssignToValue(
          absl::StrFormat("members[%d]", i),
          absl::StrFormat("std::get<%d>(%s)", i, rhs), nesting + 1);
      pieces.push_back(assignment);
    }
    pieces.push_back(
        absl::StrFormat("%s = ::xls::Value::Tuple(members);", lhs));
    return absl::StrFormat("{\n%s\n}", Indent(absl::StrJoin(pieces, "\n"), 2));
  }

  std::string AssignFromValue(std::string_view lhs, std::string_view rhs,
                              int64_t nesting) const override {
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("if (!%s.IsTuple() || %s.size() != %d) {",
                                     rhs, rhs, size()));
    pieces.push_back(
        absl::StrFormat("  return absl::InvalidArgumentError(\"Value is not a "
                        "tuple of %d elements.\");",
                        size()));
    pieces.push_back("}");
    for (int64_t i = 0; i < size(); ++i) {
      std::string assignment = element_emitters_[i]->AssignFromValue(
          absl::StrFormat("std::get<%d>(%s)", i, lhs),
          absl::StrFormat("%s.element(%d)", rhs, i), nesting + 1);
      pieces.push_back(assignment);
    }
    return absl::StrJoin(pieces, "\n");
  }

  std::string Verify(std::string_view identifier, std::string_view name,
                     int64_t nesting) const override {
    std::vector<std::string> pieces;
    for (int64_t i = 0; i < size(); ++i) {
      std::string verification = element_emitters_[i]->Verify(
          absl::StrFormat("std::get<%d>(%s)", i, identifier),
          absl::StrFormat("%s.%d", name, i), nesting + 1);
      pieces.push_back(verification);
    }
    return absl::StrJoin(pieces, "\n");
  }

  std::string ToString(std::string_view str_to_append,
                       std::string_view indent_amount,
                       std::string_view identifier,
                       int64_t nesting) const override {
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("%s += \"(\\n\";", str_to_append));
    for (int64_t i = 0; i < size(); ++i) {
      pieces.push_back(absl::StrFormat("%s += __indent(%s + 1);", str_to_append,
                                       indent_amount));
      std::string to_string = element_emitters_[i]->ToString(
          str_to_append, absl::StrCat(indent_amount, " + 1"),
          absl::StrFormat("std::get<%d>(%s)", i, identifier), nesting + 1);
      pieces.push_back(to_string);
      pieces.push_back(absl::StrFormat("%s += \",\\n\";", str_to_append));
    }
    pieces.push_back(absl::StrFormat("%s += __indent(%s) + \")\";",
                                     str_to_append, indent_amount));
    return absl::StrJoin(pieces, "\n");
  }

  std::string ToDslxString(std::string_view str_to_append,
                           std::string_view indent_amount,
                           std::string_view identifier,
                           int64_t nesting) const override {
    std::vector<std::string> pieces;
    pieces.push_back(absl::StrFormat("%s += \"(\\n\";", str_to_append));
    for (int64_t i = 0; i < size(); ++i) {
      pieces.push_back(absl::StrFormat("%s += __indent(%s + 1);", str_to_append,
                                       indent_amount));
      std::string to_string = element_emitters_[i]->ToDslxString(
          str_to_append, absl::StrCat(indent_amount, " + 1"),
          absl::StrFormat("std::get<%d>(%s)", i, identifier), nesting + 1);
      pieces.push_back(to_string);
      pieces.push_back(absl::StrFormat("%s += \",\\n\";", str_to_append));
    }
    pieces.push_back(absl::StrFormat("%s += __indent(%s) + \")\";",
                                     str_to_append, indent_amount));
    return absl::StrJoin(pieces, "\n");
  }

  int64_t size() const { return element_emitters_.size(); }

 protected:
  std::vector<std::unique_ptr<CppEmitter>> element_emitters_;
};

}  // namespace

std::string SanitizeCppName(std::string_view name) {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>>
      kCppKeywords({"alignas",
                    "alignof",
                    "and",
                    "and_eq",
                    "asm",
                    "atomic_cancel",
                    "atomic_commit",
                    "atomic_noexcept",
                    "auto",
                    "bitand",
                    "bitor",
                    "bool",
                    "break",
                    "case",
                    "catch",
                    "char",
                    "char8_t",
                    "char16_t",
                    "char32_t",
                    "class",
                    "compl",
                    "concept",
                    "const",
                    "consteval",
                    "constexpr",
                    "constinit",
                    "const_cast",
                    "continue",
                    "co_await",
                    "co_return",
                    "co_yield",
                    "decltype",
                    "default",
                    "delete",
                    "do",
                    "double",
                    "dynamic_cast",
                    "else",
                    "enum",
                    "explicit",
                    "export",
                    "extern",
                    "false",
                    "float",
                    "for",
                    "friend",
                    "goto",
                    "if",
                    "inline",
                    "int",
                    "long",
                    "mutable",
                    "namespace",
                    "new",
                    "noexcept",
                    "not",
                    "not_eq",
                    "nullptr",
                    "operator",
                    "or",
                    "or_eq",
                    "private",
                    "protected",
                    "public",
                    "reflexpr",
                    "register",
                    "reinterpret_cast",
                    "requires",
                    "return",
                    "short",
                    "signed",
                    "sizeof",
                    "static",
                    "static_assert",
                    "static_cast",
                    "struct",
                    "switch",
                    "synchronized",
                    "template",
                    "this",
                    "thread_local",
                    "throw",
                    "true",
                    "try",
                    "typedef",
                    "typeid",
                    "typename",
                    "union",
                    "unsigned",
                    "using",
                    "virtual",
                    "void",
                    "volatile",
                    "wchar_t",
                    "while",
                    "xor",
                    "xor_eq"});
  if (kCppKeywords->contains(name)) {
    return absl::StrCat("_", name);
  }
  return std::string{name};
}

std::string DslxTypeNameToCpp(std::string_view dslx_type) {
  return SanitizeCppName(Camelize(dslx_type));
}

/* static */ absl::StatusOr<std::unique_ptr<CppEmitter>> CppEmitter::Create(
    const TypeAnnotation* type_annotation, std::string_view dslx_type,
    TypeInfo* type_info, ImportData* import_data) {
  // Both builtin (e.g., `u32`) and array types (e.g, `sU[22]`) can represent
  // bit-vector types so call IsBitVectorType to identify them.
  std::optional<BitVectorMetadata> bit_vector_metadata =
      ExtractBitVectorMetadata(type_annotation);

  if (bit_vector_metadata.has_value() &&
      bit_vector_metadata->kind == BitVectorKind::kBitType) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                         GetBitCountFromBitVectorMetadata(
                             *bit_vector_metadata, type_info, import_data));
    return BitVectorCppEmitter::Create(dslx_type, bit_count,
                                       bit_vector_metadata->is_signed);
  }
  if (const ArrayTypeAnnotation* array_type =
          dynamic_cast<const ArrayTypeAnnotation*>(type_annotation);
      array_type != nullptr) {
    return ArrayCppEmitter::Create(array_type, type_info, import_data);
  }
  if (const TupleTypeAnnotation* tuple_type =
          dynamic_cast<const TupleTypeAnnotation*>(type_annotation);
      tuple_type != nullptr) {
    return TupleCppEmitter::Create(tuple_type, type_info, import_data);
  }
  if (const TypeRefTypeAnnotation* type_ref =
          dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation);
      type_ref != nullptr) {
    return TypeRefCppEmitter::Create(type_ref, dslx_type, type_info,
                                     import_data);
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported TypeAnnotation kind: ", type_annotation->ToString()));
}

}  // namespace xls::dslx

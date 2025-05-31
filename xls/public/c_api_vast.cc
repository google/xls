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

#include "xls/public/c_api_vast.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"
#include "xls/public/c_api_format_preference.h"
#include "xls/public/c_api_impl_helpers.h"

namespace {
xls::verilog::OperatorKind ToCppOperatorKindUnary(xls_vast_operator_kind op) {
  switch (op) {
    case xls_vast_operator_kind_negate:
      return xls::verilog::OperatorKind::kNegate;
    case xls_vast_operator_kind_bitwise_not:
      return xls::verilog::OperatorKind::kBitwiseNot;
    case xls_vast_operator_kind_logical_not:
      return xls::verilog::OperatorKind::kLogicalNot;
    case xls_vast_operator_kind_and_reduce:
      return xls::verilog::OperatorKind::kAndReduce;
    case xls_vast_operator_kind_or_reduce:
      return xls::verilog::OperatorKind::kOrReduce;
    case xls_vast_operator_kind_xor_reduce:
      return xls::verilog::OperatorKind::kXorReduce;
    default:
      LOG(FATAL) << "C VAST API got invalid unary operator kind: " << op;
  }
}

xls::verilog::OperatorKind ToCppOperatorKindBinary(xls_vast_operator_kind op) {
  switch (op) {
    case xls_vast_operator_kind_add:
      return xls::verilog::OperatorKind::kAdd;
    case xls_vast_operator_kind_logical_and:
      return xls::verilog::OperatorKind::kLogicalAnd;
    case xls_vast_operator_kind_bitwise_and:
      return xls::verilog::OperatorKind::kBitwiseAnd;
    case xls_vast_operator_kind_ne:
      return xls::verilog::OperatorKind::kNe;
    case xls_vast_operator_kind_case_ne:
      return xls::verilog::OperatorKind::kCaseNe;
    case xls_vast_operator_kind_eq:
      return xls::verilog::OperatorKind::kEq;
    case xls_vast_operator_kind_case_eq:
      return xls::verilog::OperatorKind::kCaseEq;
    case xls_vast_operator_kind_ge:
      return xls::verilog::OperatorKind::kGe;
    case xls_vast_operator_kind_gt:
      return xls::verilog::OperatorKind::kGt;
    case xls_vast_operator_kind_le:
      return xls::verilog::OperatorKind::kLe;
    case xls_vast_operator_kind_lt:
      return xls::verilog::OperatorKind::kLt;
    case xls_vast_operator_kind_div:
      return xls::verilog::OperatorKind::kDiv;
    case xls_vast_operator_kind_mod:
      return xls::verilog::OperatorKind::kMod;
    case xls_vast_operator_kind_mul:
      return xls::verilog::OperatorKind::kMul;
    case xls_vast_operator_kind_power:
      return xls::verilog::OperatorKind::kPower;
    case xls_vast_operator_kind_bitwise_or:
      return xls::verilog::OperatorKind::kBitwiseOr;
    case xls_vast_operator_kind_logical_or:
      return xls::verilog::OperatorKind::kLogicalOr;
    case xls_vast_operator_kind_bitwise_xor:
      return xls::verilog::OperatorKind::kBitwiseXor;
    case xls_vast_operator_kind_shll:
      return xls::verilog::OperatorKind::kShll;
    case xls_vast_operator_kind_shra:
      return xls::verilog::OperatorKind::kShra;
    case xls_vast_operator_kind_shrl:
      return xls::verilog::OperatorKind::kShrl;
    case xls_vast_operator_kind_sub:
      return xls::verilog::OperatorKind::kSub;
    case xls_vast_operator_kind_ne_x:
      return xls::verilog::OperatorKind::kNeX;
    case xls_vast_operator_kind_eq_x:
      return xls::verilog::OperatorKind::kEqX;
    default:
      LOG(FATAL) << "C VAST API got invalid binary operator kind: " << op;
  }
}

}  // namespace

extern "C" {

struct xls_vast_verilog_file* xls_vast_make_verilog_file(
    xls_vast_file_type file_type) {
  auto* value = new xls::verilog::VerilogFile(
      static_cast<xls::verilog::FileType>(file_type));
  return reinterpret_cast<xls_vast_verilog_file*>(value);
}

void xls_vast_verilog_file_free(struct xls_vast_verilog_file* f) {
  delete reinterpret_cast<xls::verilog::VerilogFile*>(f);
}

struct xls_vast_verilog_module* xls_vast_verilog_file_add_module(
    struct xls_vast_verilog_file* f, const char* name) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::Module* cpp_module =
      cpp_file->AddModule(name, xls::SourceInfo());
  return reinterpret_cast<xls_vast_verilog_module*>(cpp_module);
}

void xls_vast_verilog_file_add_include(struct xls_vast_verilog_file* f,
                                       const char* path) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  cpp_file->AddInclude(path, xls::SourceInfo());
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_input(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> logic_ref =
      cpp_module->AddInput(name, cpp_type, xls::SourceInfo());
  CHECK_OK(logic_ref.status());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref.value());
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_output(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> logic_ref =
      cpp_module->AddOutput(name, cpp_type, xls::SourceInfo());
  CHECK_OK(logic_ref.status());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref.value());
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_wire(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> logic_ref =
      cpp_module->AddWire(name, cpp_type, xls::SourceInfo());
  CHECK_OK(logic_ref.status());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref.value());
}

char* xls_vast_verilog_file_emit(const struct xls_vast_verilog_file* f) {
  const auto* cpp_file = reinterpret_cast<const xls::verilog::VerilogFile*>(f);
  std::string result = cpp_file->Emit();
  return xls::ToOwnedCString(result);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_scalar_type(
    struct xls_vast_verilog_file* f) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type = cpp_file->ScalarType(xls::SourceInfo());
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_bit_vector_type(
    struct xls_vast_verilog_file* f, int64_t bit_count, bool is_signed) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->BitVectorType(bit_count, xls::SourceInfo(), is_signed);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_extern_package_type(
    struct xls_vast_verilog_file* f, const char* package_name,
    const char* entity_name) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::ExternPackageType>(
          xls::SourceInfo(), package_name, entity_name);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_packed_array_type(
    struct xls_vast_verilog_file* f, struct xls_vast_data_type* element_type,
    const int64_t* packed_dims, size_t packed_dims_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_element_type =
      reinterpret_cast<xls::verilog::DataType*>(element_type);
  absl::Span<const int64_t> dims(packed_dims, packed_dims_count);
  xls::verilog::DataType* type = cpp_file->Make<xls::verilog::PackedArrayType>(
      xls::SourceInfo(), cpp_element_type, dims, /*dims_are_max=*/false);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

void xls_vast_verilog_module_add_member_instantiation(
    struct xls_vast_verilog_module* m, struct xls_vast_instantiation* member) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_instantiation =
      reinterpret_cast<xls::verilog::Instantiation*>(member);
  cpp_module->AddModuleMember(cpp_instantiation);
}

void xls_vast_verilog_module_add_member_continuous_assignment(
    struct xls_vast_verilog_module* m,
    struct xls_vast_continuous_assignment* member) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_member =
      reinterpret_cast<xls::verilog::ContinuousAssignment*>(member);
  cpp_module->AddModuleMember(cpp_member);
}

void xls_vast_verilog_module_add_member_comment(
    struct xls_vast_verilog_module* m, struct xls_vast_comment* comment) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_comment = reinterpret_cast<xls::verilog::Comment*>(comment);
  cpp_module->AddModuleMember(cpp_comment);
}

struct xls_vast_literal* xls_vast_verilog_file_make_plain_literal(
    struct xls_vast_verilog_file* f, int32_t value) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::Literal* cpp_literal =
      cpp_file->PlainLiteral(value, xls::SourceInfo());
  return reinterpret_cast<xls_vast_literal*>(cpp_literal);
}

bool xls_vast_verilog_file_make_literal(struct xls_vast_verilog_file* f,
                                        struct xls_bits* bits,
                                        xls_format_preference format_preference,
                                        bool emit_bit_count, char** error_out,
                                        struct xls_vast_literal** literal_out) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_bits = reinterpret_cast<xls::Bits*>(bits);
  xls::FormatPreference cpp_pref;
  if (!xls::FormatPreferenceFromC(format_preference, &cpp_pref, error_out)) {
    return false;
  }
  xls::verilog::Literal* cpp_literal = cpp_file->Make<xls::verilog::Literal>(
      xls::SourceInfo(), *cpp_bits, cpp_pref, emit_bit_count);
  *error_out = nullptr;
  *literal_out = reinterpret_cast<xls_vast_literal*>(cpp_literal);
  return true;
}

struct xls_vast_continuous_assignment*
xls_vast_verilog_file_make_continuous_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  auto* cpp_assignment = cpp_file->Make<xls::verilog::ContinuousAssignment>(
      xls::SourceInfo(), cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_continuous_assignment*>(cpp_assignment);
}

struct xls_vast_comment* xls_vast_verilog_file_make_comment(
    struct xls_vast_verilog_file* f, const char* text) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::Comment* cpp_comment =
      cpp_file->Make<xls::verilog::Comment>(xls::SourceInfo(), text);
  return reinterpret_cast<xls_vast_comment*>(cpp_comment);
}

struct xls_vast_instantiation* xls_vast_verilog_file_make_instantiation(
    struct xls_vast_verilog_file* f, const char* module_name,
    const char* instance_name, const char** parameter_port_names,
    struct xls_vast_expression** parameter_expressions, size_t parameter_count,
    const char** connection_port_names,
    struct xls_vast_expression** connection_expressions,
    size_t connection_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);

  std::vector<xls::verilog::Connection> parameters;
  parameters.reserve(parameter_count);
  for (size_t i = 0; i < parameter_count; ++i) {
    auto* cpp_expression =
        reinterpret_cast<xls::verilog::Expression*>(parameter_expressions[i]);
    parameters.push_back(
        xls::verilog::Connection{parameter_port_names[i], cpp_expression});
  }

  std::vector<xls::verilog::Connection> connections;
  connections.reserve(connection_count);
  for (size_t i = 0; i < connection_count; ++i) {
    auto* cpp_expression =
        reinterpret_cast<xls::verilog::Expression*>(connection_expressions[i]);
    connections.push_back(
        xls::verilog::Connection{connection_port_names[i], cpp_expression});
  }

  auto* cpp_instantiation = cpp_file->Make<xls::verilog::Instantiation>(
      xls::SourceInfo(), module_name, instance_name,
      absl::MakeConstSpan(parameters), absl::MakeConstSpan(connections));
  return reinterpret_cast<xls_vast_instantiation*>(cpp_instantiation);
}

struct xls_vast_expression* xls_vast_literal_as_expression(
    struct xls_vast_literal* v) {
  auto* cpp_literal = reinterpret_cast<xls::verilog::Literal*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_literal);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_expression* xls_vast_logic_ref_as_expression(
    struct xls_vast_logic_ref* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::LogicRef*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_v);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_expression* xls_vast_slice_as_expression(
    struct xls_vast_slice* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::Slice*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_v);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_expression* xls_vast_concat_as_expression(
    struct xls_vast_concat* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::Concat*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_v);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_expression* xls_vast_index_as_expression(
    struct xls_vast_index* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::Index*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_v);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_indexable_expression*
xls_vast_logic_ref_as_indexable_expression(
    struct xls_vast_logic_ref* logic_ref) {
  auto* cpp_logic_ref = reinterpret_cast<xls::verilog::LogicRef*>(logic_ref);
  auto* cpp_indexable_expression =
      static_cast<xls::verilog::IndexableExpression*>(cpp_logic_ref);
  return reinterpret_cast<xls_vast_indexable_expression*>(
      cpp_indexable_expression);
}

struct xls_vast_indexable_expression* xls_vast_index_as_indexable_expression(
    struct xls_vast_index* index) {
  auto* cpp_index = reinterpret_cast<xls::verilog::Index*>(index);
  auto* cpp_indexable_expression =
      static_cast<xls::verilog::IndexableExpression*>(cpp_index);
  return reinterpret_cast<xls_vast_indexable_expression*>(
      cpp_indexable_expression);
}

struct xls_vast_slice* xls_vast_verilog_file_make_slice_i64(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject, int64_t hi, int64_t lo) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_subject =
      reinterpret_cast<xls::verilog::IndexableExpression*>(subject);
  xls::verilog::Slice* cpp_slice =
      cpp_file->Slice(cpp_subject, hi, lo, xls::SourceInfo());
  return reinterpret_cast<xls_vast_slice*>(cpp_slice);
}

struct xls_vast_slice* xls_vast_verilog_file_make_slice(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject,
    struct xls_vast_expression* hi, struct xls_vast_expression* lo) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_subject =
      reinterpret_cast<xls::verilog::IndexableExpression*>(subject);
  auto* cpp_hi = reinterpret_cast<xls::verilog::Expression*>(hi);
  auto* cpp_lo = reinterpret_cast<xls::verilog::Expression*>(lo);
  xls::verilog::Slice* cpp_slice =
      cpp_file->Slice(cpp_subject, cpp_hi, cpp_lo, xls::SourceInfo());
  return reinterpret_cast<xls_vast_slice*>(cpp_slice);
}

struct xls_vast_expression* xls_vast_verilog_file_make_unary(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* arg,
    xls_vast_operator_kind op) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_arg = reinterpret_cast<xls::verilog::Expression*>(arg);
  xls::verilog::Expression* result = cpp_file->Make<xls::verilog::Unary>(
      xls::SourceInfo(), cpp_arg, ToCppOperatorKindUnary(op));
  return reinterpret_cast<xls_vast_expression*>(result);
}

struct xls_vast_expression* xls_vast_verilog_file_make_binary(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs, xls_vast_operator_kind op) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::Expression* result = cpp_file->Make<xls::verilog::BinaryInfix>(
      xls::SourceInfo(), cpp_lhs, cpp_rhs, ToCppOperatorKindBinary(op));
  return reinterpret_cast<xls_vast_expression*>(result);
}

struct xls_vast_expression* xls_vast_verilog_file_make_ternary(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* cond,
    struct xls_vast_expression* consequent,
    struct xls_vast_expression* alternate) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_cond = reinterpret_cast<xls::verilog::Expression*>(cond);
  auto* cpp_consequent =
      reinterpret_cast<xls::verilog::Expression*>(consequent);
  auto* cpp_alternate = reinterpret_cast<xls::verilog::Expression*>(alternate);
  xls::verilog::Expression* result = cpp_file->Make<xls::verilog::Ternary>(
      xls::SourceInfo(), cpp_cond, cpp_consequent, cpp_alternate);
  return reinterpret_cast<xls_vast_expression*>(result);
}

struct xls_vast_concat* xls_vast_verilog_file_make_concat(
    struct xls_vast_verilog_file* f, struct xls_vast_expression** elements,
    size_t element_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  std::vector<xls::verilog::Expression*> cpp_elements;
  cpp_elements.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    auto* cpp_element =
        reinterpret_cast<xls::verilog::Expression*>(elements[i]);
    cpp_elements.push_back(cpp_element);
  }
  xls::verilog::Concat* cpp_concat =
      cpp_file->Make<xls::verilog::Concat>(xls::SourceInfo(), cpp_elements);
  return reinterpret_cast<xls_vast_concat*>(cpp_concat);
}

struct xls_vast_index* xls_vast_verilog_file_make_index_i64(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject, int64_t index) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_subject =
      reinterpret_cast<xls::verilog::IndexableExpression*>(subject);
  xls::verilog::Index* cpp_index =
      cpp_file->Index(cpp_subject, index, xls::SourceInfo());
  return reinterpret_cast<xls_vast_index*>(cpp_index);
}

struct xls_vast_index* xls_vast_verilog_file_make_index(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject,
    struct xls_vast_expression* index) {
  // Add a soundness check just in case users confuse this API with
  // xls_vast_verilog_file_make_index_i64 and pass a literal zero in the place
  // of the pointer.
  CHECK(index != nullptr)
      << "xls_vast_verilog_file_make_index: index is nullptr";
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_subject =
      reinterpret_cast<xls::verilog::IndexableExpression*>(subject);
  auto* cpp_index = reinterpret_cast<xls::verilog::Expression*>(index);
  xls::verilog::Index* result =
      cpp_file->Index(cpp_subject, cpp_index, xls::SourceInfo());
  return reinterpret_cast<xls_vast_index*>(result);
}

bool xls_vast_verilog_module_add_always_ff(
    struct xls_vast_verilog_module* m,
    struct xls_vast_expression** sensitivity_list_elements,
    size_t sensitivity_list_count, struct xls_vast_always_base** out_always_ff,
    char** error_out) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  std::vector<xls::verilog::SensitivityListElement> cpp_elements;
  cpp_elements.reserve(sensitivity_list_count);
  for (size_t i = 0; i < sensitivity_list_count; ++i) {
    auto* expr = reinterpret_cast<xls::verilog::Expression*>(sensitivity_list_elements[i]);
    if (auto* pe = dynamic_cast<xls::verilog::PosEdge*>(expr)) {
      cpp_elements.push_back(pe);
    } else if (auto* ne = dynamic_cast<xls::verilog::NegEdge*>(expr)) {
      cpp_elements.push_back(ne);
    } else if (auto* lr = dynamic_cast<xls::verilog::LogicRef*>(expr)){
      cpp_elements.push_back(lr);
    } else {
      std::string err_msg = absl::StrCat(
          "Unsupported expression type passed to sensitivity list for always_ff at index ", i,
          ". Only Posedge, Negedge, or LogicRef expressions are supported through this C API path.");
      *error_out = xls::ToOwnedCString(err_msg);
      *out_always_ff = nullptr;
      return false;
    }
  }
  xls::verilog::AlwaysFf* cpp_always_ff = cpp_module->Add<xls::verilog::AlwaysFf>(
      xls::SourceInfo(), cpp_elements);
  *out_always_ff = reinterpret_cast<xls_vast_always_base*>(cpp_always_ff);
  *error_out = nullptr;
  return true;
}

bool xls_vast_verilog_module_add_reg(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type, struct xls_vast_logic_ref** out_reg_ref,
    char** error_out) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_data_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> cpp_logic_ref_status =
      cpp_module->AddReg(name, cpp_data_type, xls::SourceInfo());
  if (!cpp_logic_ref_status.ok()) {
    *error_out = xls::ToOwnedCString(cpp_logic_ref_status.status().ToString());
    *out_reg_ref = nullptr;
    return false;
  }
  *out_reg_ref = reinterpret_cast<xls_vast_logic_ref*>(cpp_logic_ref_status.value());
  *error_out = nullptr;
  return true;
}

struct xls_vast_expression* xls_vast_verilog_file_make_pos_edge(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* signal_expr) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_signal_expr =
      reinterpret_cast<xls::verilog::Expression*>(signal_expr);
  xls::verilog::PosEdge* cpp_pos_edge =
      cpp_file->Make<xls::verilog::PosEdge>(xls::SourceInfo(), cpp_signal_expr);
  return reinterpret_cast<xls_vast_expression*>(cpp_pos_edge);
}

struct xls_vast_statement*
xls_vast_verilog_file_make_nonblocking_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::NonblockingAssignment* cpp_assignment =
      cpp_file->Make<xls::verilog::NonblockingAssignment>(
          xls::SourceInfo(), cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

struct xls_vast_statement_block* xls_vast_always_base_get_statement_block(
    struct xls_vast_always_base* always_base) {
  auto* cpp_always_base =
      reinterpret_cast<xls::verilog::AlwaysBase*>(always_base);
  xls::verilog::StatementBlock* cpp_block = cpp_always_base->statements();
  return reinterpret_cast<xls_vast_statement_block*>(cpp_block);
}

struct xls_vast_statement*
xls_vast_statement_block_add_nonblocking_assignment(
    struct xls_vast_statement_block* block,
    struct xls_vast_expression* lhs, struct xls_vast_expression* rhs) {
  auto* cpp_block =
      reinterpret_cast<xls::verilog::StatementBlock*>(block);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  // Use the StatementBlock's Add<T> method to create and add the statement.
  // This ensures the statement is owned by the VerilogFile associated with the StatementBlock.
  xls::verilog::NonblockingAssignment* cpp_assignment =
      cpp_block->Add<xls::verilog::NonblockingAssignment>(
          xls::SourceInfo(), cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

}  // extern "C"

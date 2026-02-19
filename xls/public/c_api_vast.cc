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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
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

void xls_vast_verilog_file_add_blank_line(struct xls_vast_verilog_file* f) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* blank = cpp_file->Make<xls::verilog::BlankLine>(xls::SourceInfo());
  cpp_file->Add(blank);
}

void xls_vast_verilog_file_add_comment(struct xls_vast_verilog_file* f,
                                       const char* text) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* comment =
      cpp_file->Make<xls::verilog::Comment>(xls::SourceInfo(), text);
  cpp_file->Add(comment);
}

char* xls_vast_verilog_module_get_name(struct xls_vast_verilog_module* m) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  return xls::ToOwnedCString(cpp_module->name());
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

struct xls_vast_logic_ref* xls_vast_verilog_module_add_logic_input(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> logic_ref = cpp_module->AddInput(
      name, cpp_type, xls::SourceInfo(), xls::verilog::DataKind::kLogic);
  CHECK_OK(logic_ref.status());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref.value());
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_logic_output(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> logic_ref = cpp_module->AddOutput(
      name, cpp_type, xls::SourceInfo(), xls::verilog::DataKind::kLogic);
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

struct xls_vast_generate_loop* xls_vast_verilog_module_add_generate_loop(
    struct xls_vast_verilog_module* m, const char* genvar_name,
    struct xls_vast_expression* init, struct xls_vast_expression* limit,
    const char* label) {
  CHECK_NE(genvar_name, nullptr);
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_init = reinterpret_cast<xls::verilog::Expression*>(init);
  auto* cpp_limit = reinterpret_cast<xls::verilog::Expression*>(limit);
  std::optional<std::string> cpp_label =
      label == nullptr ? std::nullopt : std::optional<std::string>(label);
  xls::verilog::GenerateLoop* cpp_loop =
      cpp_module->Add<xls::verilog::GenerateLoop>(
          xls::SourceInfo(), std::string_view(genvar_name), cpp_init, cpp_limit,
          cpp_label);
  return reinterpret_cast<xls_vast_generate_loop*>(cpp_loop);
}

struct xls_vast_expression* xls_vast_verilog_module_add_parameter_port(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_expression* rhs) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::ParameterRef* parameter_ref =
      cpp_module->AddParameterPort(name, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_expression*>(parameter_ref);
}

struct xls_vast_expression* xls_vast_verilog_module_add_typed_parameter_port(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type, struct xls_vast_expression* rhs) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  auto* cpp_def = cpp_module->file()->Make<xls::verilog::Def>(
      xls::SourceInfo(), name, xls::verilog::DataKind::kLogic, cpp_type);
  xls::verilog::ParameterRef* parameter_ref =
      cpp_module->AddParameterPort(cpp_def, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_expression*>(parameter_ref);
}

struct xls_vast_parameter_ref* xls_vast_verilog_module_add_parameter(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_expression* rhs) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::ParameterRef* param_ref =
      cpp_module->AddParameter(name, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_parameter_ref*>(param_ref);
}

struct xls_vast_localparam_ref* xls_vast_verilog_module_add_localparam(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_expression* rhs) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::LocalParam* lp =
      cpp_module->Add<xls::verilog::LocalParam>(xls::SourceInfo());
  xls::verilog::LocalParamItemRef* item_ref =
      lp->AddItem(name, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_localparam_ref*>(item_ref);
}

struct xls_vast_parameter_ref* xls_vast_verilog_module_add_parameter_with_def(
    struct xls_vast_verilog_module* m, struct xls_vast_def* def,
    struct xls_vast_expression* rhs) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_def = reinterpret_cast<xls::verilog::Def*>(def);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::ParameterRef* param_ref =
      cpp_module->AddParameter(cpp_def, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_parameter_ref*>(param_ref);
}

struct xls_vast_localparam_ref* xls_vast_verilog_module_add_localparam_with_def(
    struct xls_vast_verilog_module* m, struct xls_vast_def* def,
    struct xls_vast_expression* rhs) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_def = reinterpret_cast<xls::verilog::Def*>(def);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::LocalParam* lp =
      cpp_module->Add<xls::verilog::LocalParam>(xls::SourceInfo());
  xls::verilog::LocalParamItemRef* item_ref =
      lp->AddItem(cpp_def, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_localparam_ref*>(item_ref);
}

char* xls_vast_verilog_file_emit(const struct xls_vast_verilog_file* f) {
  const auto* cpp_file = reinterpret_cast<const xls::verilog::VerilogFile*>(f);
  std::string result = cpp_file->Emit();
  return xls::ToOwnedCString(result);
}

char* xls_vast_expression_emit(struct xls_vast_expression* expr) {
  CHECK_NE(expr, nullptr);
  auto* cpp_expr = reinterpret_cast<xls::verilog::Expression*>(expr);
  std::string result = cpp_expr->Emit(/*line_info=*/nullptr);
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

struct xls_vast_data_type*
xls_vast_verilog_file_make_bit_vector_type_with_expression(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* expression,
    bool is_signed) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_expression =
      reinterpret_cast<xls::verilog::Expression*>(expression);
  xls::verilog::DataType* type = cpp_file->Make<xls::verilog::BitVectorType>(
      xls::SourceInfo(), cpp_expression, is_signed);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_integer_type(
    struct xls_vast_verilog_file* f, bool is_signed) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::IntegerType>(xls::SourceInfo(), is_signed);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_int_type(
    struct xls_vast_verilog_file* f, bool is_signed) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::IntType>(xls::SourceInfo(), is_signed);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_def* xls_vast_verilog_file_make_integer_def(
    struct xls_vast_verilog_file* f, const char* name, bool is_signed) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::IntegerType>(xls::SourceInfo(), is_signed);
  xls::verilog::Def* def = cpp_file->Make<xls::verilog::Def>(
      xls::SourceInfo(), name, xls::verilog::DataKind::kInteger, type);
  return reinterpret_cast<xls_vast_def*>(def);
}

struct xls_vast_def* xls_vast_verilog_file_make_int_def(
    struct xls_vast_verilog_file* f, const char* name, bool is_signed) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::IntType>(xls::SourceInfo(), is_signed);
  xls::verilog::Def* def = cpp_file->Make<xls::verilog::Def>(
      xls::SourceInfo(), name, xls::verilog::DataKind::kInt, type);
  return reinterpret_cast<xls_vast_def*>(def);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_extern_package_type(
    struct xls_vast_verilog_file* f, const char* package_name,
    const char* entity_name) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type = cpp_file->Make<xls::verilog::ExternType>(
      xls::SourceInfo(), package_name, entity_name);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_data_type* xls_vast_verilog_file_make_extern_type(
    struct xls_vast_verilog_file* f, const char* entity_name) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::ExternType>(xls::SourceInfo(), entity_name);
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

struct xls_vast_data_type* xls_vast_verilog_file_make_unpacked_array_type(
    struct xls_vast_verilog_file* f, struct xls_vast_data_type* element_type,
    const int64_t* unpacked_dims, size_t unpacked_dims_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_element_type =
      reinterpret_cast<xls::verilog::DataType*>(element_type);
  absl::Span<const int64_t> dims(unpacked_dims, unpacked_dims_count);
  xls::verilog::DataType* type =
      cpp_file->Make<xls::verilog::UnpackedArrayType>(xls::SourceInfo(),
                                                      cpp_element_type, dims);
  return reinterpret_cast<xls_vast_data_type*>(type);
}

struct xls_vast_def* xls_vast_verilog_file_make_def(
    struct xls_vast_verilog_file* f, const char* name, xls_vast_data_kind kind,
    struct xls_vast_data_type* type) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  auto cpp_kind = static_cast<xls::verilog::DataKind>(kind);
  xls::verilog::Def* def = cpp_file->Make<xls::verilog::Def>(
      xls::SourceInfo(), name, cpp_kind, cpp_type);
  return reinterpret_cast<xls_vast_def*>(def);
}

struct xls_vast_expression* xls_vast_verilog_file_make_array_assignment_pattern(
    struct xls_vast_verilog_file* f, struct xls_vast_expression** elements,
    size_t element_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  std::vector<xls::verilog::Expression*> cpp_elements;
  cpp_elements.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    cpp_elements.push_back(
        reinterpret_cast<xls::verilog::Expression*>(elements[i]));
  }
  xls::verilog::Expression* expr =
      cpp_file->ArrayAssignmentPattern(cpp_elements, xls::SourceInfo());
  return reinterpret_cast<xls_vast_expression*>(expr);
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
void xls_vast_verilog_module_add_member_blank_line(
    struct xls_vast_verilog_module* m, struct xls_vast_blank_line* blank) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_blank = reinterpret_cast<xls::verilog::BlankLine*>(blank);
  cpp_module->AddModuleMember(cpp_blank);
}
void xls_vast_verilog_module_add_member_inline_statement(
    struct xls_vast_verilog_module* m,
    struct xls_vast_inline_verilog_statement* stmt) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_stmt =
      reinterpret_cast<xls::verilog::InlineVerilogStatement*>(stmt);
  cpp_module->AddModuleMember(cpp_stmt);
}

void xls_vast_verilog_module_add_member_macro_statement(
    struct xls_vast_verilog_module* m,
    struct xls_vast_macro_statement* statement) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_stmt = reinterpret_cast<xls::verilog::MacroStatement*>(statement);
  cpp_module->AddModuleMember(cpp_stmt);
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

struct xls_vast_expression* xls_vast_verilog_file_make_unsized_one_literal(
    struct xls_vast_verilog_file* f) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_expr =
      cpp_file->Make<xls::verilog::UnsizedOneLiteral>(xls::SourceInfo());
  return reinterpret_cast<xls_vast_expression*>(cpp_expr);
}

struct xls_vast_expression* xls_vast_verilog_file_make_unsized_zero_literal(
    struct xls_vast_verilog_file* f) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_expr =
      cpp_file->Make<xls::verilog::UnsizedZeroLiteral>(xls::SourceInfo());
  return reinterpret_cast<xls_vast_expression*>(cpp_expr);
}

struct xls_vast_expression* xls_vast_verilog_file_make_unsized_x_literal(
    struct xls_vast_verilog_file* f) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_expr =
      cpp_file->Make<xls::verilog::UnsizedXLiteral>(xls::SourceInfo());
  return reinterpret_cast<xls_vast_expression*>(cpp_expr);
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
struct xls_vast_blank_line* xls_vast_verilog_file_make_blank_line(
    struct xls_vast_verilog_file* f) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::BlankLine* cpp_blank =
      cpp_file->Make<xls::verilog::BlankLine>(xls::SourceInfo());
  return reinterpret_cast<xls_vast_blank_line*>(cpp_blank);
}
struct xls_vast_inline_verilog_statement*
xls_vast_verilog_file_make_inline_verilog_statement(
    struct xls_vast_verilog_file* f, const char* text) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::InlineVerilogStatement* cpp_stmt =
      cpp_file->Make<xls::verilog::InlineVerilogStatement>(xls::SourceInfo(),
                                                           text);
  return reinterpret_cast<xls_vast_inline_verilog_statement*>(cpp_stmt);
}

struct xls_vast_macro_ref* xls_vast_verilog_file_make_macro_ref(
    struct xls_vast_verilog_file* f, const char* name) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_ref =
      cpp_file->Make<xls::verilog::MacroRef>(xls::SourceInfo(), name);
  return reinterpret_cast<xls_vast_macro_ref*>(cpp_ref);
}

struct xls_vast_macro_ref* xls_vast_verilog_file_make_macro_ref_with_args(
    struct xls_vast_verilog_file* f, const char* name,
    struct xls_vast_expression** args, size_t arg_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  if (arg_count == 0) {
    auto* cpp_ref = cpp_file->Make<xls::verilog::MacroRef>(
        xls::SourceInfo(), name, std::vector<xls::verilog::Expression*>{});
    return reinterpret_cast<xls_vast_macro_ref*>(cpp_ref);
  }
  std::vector<xls::verilog::Expression*> cpp_args;
  cpp_args.reserve(arg_count);
  for (size_t i = 0; i < arg_count; ++i) {
    cpp_args.push_back(reinterpret_cast<xls::verilog::Expression*>(args[i]));
  }
  auto* cpp_ref = cpp_file->Make<xls::verilog::MacroRef>(
      xls::SourceInfo(), name, absl::MakeConstSpan(cpp_args));
  return reinterpret_cast<xls_vast_macro_ref*>(cpp_ref);
}

struct xls_vast_expression* xls_vast_macro_ref_as_expression(
    struct xls_vast_macro_ref* ref) {
  auto* cpp_ref = reinterpret_cast<xls::verilog::MacroRef*>(ref);
  auto* cpp_expr = static_cast<xls::verilog::Expression*>(cpp_ref);
  return reinterpret_cast<xls_vast_expression*>(cpp_expr);
}

struct xls_vast_macro_statement* xls_vast_verilog_file_make_macro_statement(
    struct xls_vast_verilog_file* f, struct xls_vast_macro_ref* ref,
    bool emit_semicolon) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_ref = reinterpret_cast<xls::verilog::MacroRef*>(ref);
  auto* cpp_stmt = cpp_file->Make<xls::verilog::MacroStatement>(
      xls::SourceInfo(), cpp_ref, emit_semicolon);
  return reinterpret_cast<xls_vast_macro_statement*>(cpp_stmt);
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

struct xls_vast_expression* xls_vast_parameter_ref_as_expression(
    struct xls_vast_parameter_ref* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::ParameterRef*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_v);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_expression* xls_vast_localparam_ref_as_expression(
    struct xls_vast_localparam_ref* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::LocalParamItemRef*>(v);
  auto* cpp_expression = static_cast<xls::verilog::Expression*>(cpp_v);
  return reinterpret_cast<xls_vast_expression*>(cpp_expression);
}

struct xls_vast_expression* xls_vast_indexable_expression_as_expression(
    struct xls_vast_indexable_expression* v) {
  auto* cpp_v = reinterpret_cast<xls::verilog::IndexableExpression*>(v);
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

struct xls_vast_indexable_expression*
xls_vast_parameter_ref_as_indexable_expression(
    struct xls_vast_parameter_ref* parameter_ref) {
  auto* cpp_parameter_ref =
      reinterpret_cast<xls::verilog::ParameterRef*>(parameter_ref);
  auto* cpp_indexable_expression =
      static_cast<xls::verilog::IndexableExpression*>(cpp_parameter_ref);
  return reinterpret_cast<xls_vast_indexable_expression*>(
      cpp_indexable_expression);
}

struct xls_vast_logic_ref* xls_vast_generate_loop_get_genvar(
    struct xls_vast_generate_loop* loop) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  return reinterpret_cast<xls_vast_logic_ref*>(cpp_loop->genvar());
}

struct xls_vast_generate_loop* xls_vast_generate_loop_add_generate_loop(
    struct xls_vast_generate_loop* loop, const char* genvar_name,
    struct xls_vast_expression* init, struct xls_vast_expression* limit,
    const char* label) {
  CHECK_NE(genvar_name, nullptr);
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_init = reinterpret_cast<xls::verilog::Expression*>(init);
  auto* cpp_limit = reinterpret_cast<xls::verilog::Expression*>(limit);
  std::optional<std::string> cpp_label =
      label == nullptr ? std::nullopt : std::optional<std::string>(label);
  xls::verilog::GenerateLoop* cpp_inner_loop =
      cpp_loop->Add<xls::verilog::GenerateLoop>(xls::SourceInfo(),
                                                std::string_view(genvar_name),
                                                cpp_init, cpp_limit, cpp_label);
  return reinterpret_cast<xls_vast_generate_loop*>(cpp_inner_loop);
}

void xls_vast_generate_loop_add_blank_line(
    struct xls_vast_generate_loop* loop) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  cpp_loop->Add<xls::verilog::BlankLine>(xls::SourceInfo());
}

void xls_vast_generate_loop_add_comment(struct xls_vast_generate_loop* loop,
                                        struct xls_vast_comment* comment) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_comment = reinterpret_cast<xls::verilog::Comment*>(comment);
  cpp_loop->AddMember(cpp_comment);
}

void xls_vast_generate_loop_add_instantiation(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_instantiation* instantiation) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_inst =
      reinterpret_cast<xls::verilog::Instantiation*>(instantiation);
  cpp_loop->AddMember(cpp_inst);
}

void xls_vast_generate_loop_add_inline_verilog_statement(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_inline_verilog_statement* stmt) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_stmt =
      reinterpret_cast<xls::verilog::InlineVerilogStatement*>(stmt);
  cpp_loop->AddMember(cpp_stmt);
}

bool xls_vast_generate_loop_add_always_comb(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_always_base** out_always_comb, char** error_out) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  xls::verilog::AlwaysComb* cpp_always_comb =
      cpp_loop->Add<xls::verilog::AlwaysComb>(xls::SourceInfo());
  if (cpp_always_comb == nullptr) {
    *error_out = xls::ToOwnedCString(
        "Failed to create always_comb block in generate loop.");
    *out_always_comb = nullptr;
    return false;
  }
  *out_always_comb = reinterpret_cast<xls_vast_always_base*>(cpp_always_comb);
  *error_out = nullptr;
  return true;
}

bool xls_vast_generate_loop_add_always_ff(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_expression** sensitivity_list_elements,
    size_t sensitivity_list_count, struct xls_vast_always_base** out_always_ff,
    char** error_out) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  std::vector<xls::verilog::SensitivityListElement> cpp_elements;
  cpp_elements.reserve(sensitivity_list_count);
  for (size_t i = 0; i < sensitivity_list_count; ++i) {
    auto* expr = reinterpret_cast<xls::verilog::Expression*>(
        sensitivity_list_elements[i]);
    if (auto* pe = dynamic_cast<xls::verilog::PosEdge*>(expr)) {
      cpp_elements.push_back(pe);
    } else if (auto* ne = dynamic_cast<xls::verilog::NegEdge*>(expr)) {
      cpp_elements.push_back(ne);
    } else if (auto* lr = dynamic_cast<xls::verilog::LogicRef*>(expr)) {
      cpp_elements.push_back(lr);
    } else {
      std::string err_msg = absl::StrCat(
          "Unsupported expression type passed to sensitivity list for "
          "always_ff at index ",
          i,
          ". Only Posedge, Negedge, or LogicRef expressions are supported "
          "through this C API path.");
      *error_out = xls::ToOwnedCString(err_msg);
      *out_always_ff = nullptr;
      return false;
    }
  }
  xls::verilog::AlwaysFf* cpp_always_ff =
      cpp_loop->Add<xls::verilog::AlwaysFf>(xls::SourceInfo(), cpp_elements);
  *out_always_ff = reinterpret_cast<xls_vast_always_base*>(cpp_always_ff);
  *error_out = nullptr;
  return true;
}

struct xls_vast_localparam_ref* xls_vast_generate_loop_add_localparam(
    struct xls_vast_generate_loop* loop, const char* name,
    struct xls_vast_expression* rhs) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::LocalParam* lp =
      cpp_loop->Add<xls::verilog::LocalParam>(xls::SourceInfo());
  xls::verilog::LocalParamItemRef* item_ref =
      lp->AddItem(name, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_localparam_ref*>(item_ref);
}

struct xls_vast_localparam_ref* xls_vast_generate_loop_add_localparam_with_def(
    struct xls_vast_generate_loop* loop, struct xls_vast_def* def,
    struct xls_vast_expression* rhs) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_def = reinterpret_cast<xls::verilog::Def*>(def);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::LocalParam* lp =
      cpp_loop->Add<xls::verilog::LocalParam>(xls::SourceInfo());
  xls::verilog::LocalParamItemRef* item_ref =
      lp->AddItem(cpp_def, cpp_rhs, xls::SourceInfo());
  return reinterpret_cast<xls_vast_localparam_ref*>(item_ref);
}

struct xls_vast_statement* xls_vast_generate_loop_add_continuous_assignment(
    struct xls_vast_generate_loop* loop, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::ContinuousAssignment* cpp_assignment =
      cpp_loop->Add<xls::verilog::ContinuousAssignment>(xls::SourceInfo(),
                                                        cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

struct xls_vast_macro_statement* xls_vast_generate_loop_add_macro_statement(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_macro_statement* statement) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_statement =
      reinterpret_cast<xls::verilog::MacroStatement*>(statement);
  cpp_loop->AddMember(cpp_statement);
  return reinterpret_cast<xls_vast_macro_statement*>(cpp_statement);
}

struct xls_vast_indexable_expression* xls_vast_index_as_indexable_expression(
    struct xls_vast_index* index) {
  auto* cpp_index = reinterpret_cast<xls::verilog::Index*>(index);
  auto* cpp_indexable_expression =
      static_cast<xls::verilog::IndexableExpression*>(cpp_index);
  return reinterpret_cast<xls_vast_indexable_expression*>(
      cpp_indexable_expression);
}

char* xls_vast_logic_ref_get_name(struct xls_vast_logic_ref* logic_ref) {
  auto* cpp_logic_ref = reinterpret_cast<xls::verilog::LogicRef*>(logic_ref);
  return xls::ToOwnedCString(cpp_logic_ref->GetName());
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

struct xls_vast_expression* xls_vast_verilog_file_make_width_cast(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* width,
    struct xls_vast_expression* value) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_width = reinterpret_cast<xls::verilog::Expression*>(width);
  auto* cpp_value = reinterpret_cast<xls::verilog::Expression*>(value);
  xls::verilog::Expression* result = cpp_file->Make<xls::verilog::WidthCast>(
      xls::SourceInfo(), cpp_width, cpp_value);
  return reinterpret_cast<xls_vast_expression*>(result);
}

struct xls_vast_expression* xls_vast_verilog_file_make_type_cast(
    struct xls_vast_verilog_file* f, struct xls_vast_data_type* type,
    struct xls_vast_expression* value) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  auto* cpp_value = reinterpret_cast<xls::verilog::Expression*>(value);
  xls::verilog::Expression* result = cpp_file->Make<xls::verilog::TypeCast>(
      xls::SourceInfo(), cpp_type, cpp_value);
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

struct xls_vast_concat* xls_vast_verilog_file_make_replicated_concat(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* replication,
    struct xls_vast_expression** elements, size_t element_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_rep = reinterpret_cast<xls::verilog::Expression*>(replication);
  std::vector<xls::verilog::Expression*> cpp_elements;
  cpp_elements.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    cpp_elements.push_back(
        reinterpret_cast<xls::verilog::Expression*>(elements[i]));
  }
  xls::verilog::Concat* cpp_concat = cpp_file->Make<xls::verilog::Concat>(
      xls::SourceInfo(), cpp_rep, absl::MakeConstSpan(cpp_elements));
  return reinterpret_cast<xls_vast_concat*>(cpp_concat);
}

struct xls_vast_concat* xls_vast_verilog_file_make_replicated_concat_i64(
    struct xls_vast_verilog_file* f, int64_t replication_count,
    struct xls_vast_expression** elements, size_t element_count) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::Expression* cpp_rep =
      cpp_file->PlainLiteral(replication_count, xls::SourceInfo());
  std::vector<xls::verilog::Expression*> cpp_elements;
  cpp_elements.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    cpp_elements.push_back(
        reinterpret_cast<xls::verilog::Expression*>(elements[i]));
  }
  xls::verilog::Concat* cpp_concat = cpp_file->Make<xls::verilog::Concat>(
      xls::SourceInfo(), cpp_rep, absl::MakeConstSpan(cpp_elements));
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

bool xls_vast_verilog_module_add_always_at(
    struct xls_vast_verilog_module* m,
    struct xls_vast_expression** sensitivity_list_elements,
    size_t sensitivity_list_count, struct xls_vast_always_base** out_always_at,
    char** error_out) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  std::vector<xls::verilog::SensitivityListElement> cpp_elements;
  cpp_elements.reserve(sensitivity_list_count);
  for (size_t i = 0; i < sensitivity_list_count; ++i) {
    auto* expr = reinterpret_cast<xls::verilog::Expression*>(
        sensitivity_list_elements[i]);
    if (auto* pe = dynamic_cast<xls::verilog::PosEdge*>(expr)) {
      cpp_elements.push_back(pe);
    } else if (auto* ne = dynamic_cast<xls::verilog::NegEdge*>(expr)) {
      cpp_elements.push_back(ne);
    } else if (auto* lr = dynamic_cast<xls::verilog::LogicRef*>(expr)) {
      cpp_elements.push_back(lr);
    } else if (expr == nullptr && sensitivity_list_count == 1 &&
               sensitivity_list_elements[0] == nullptr) {
      cpp_elements.push_back(xls::verilog::ImplicitEventExpression{});
    } else {
      *error_out = xls::ToOwnedCString(absl::StrFormat(
          "Unsupported sensitivity list element type at index %d for always @.",
          i));
      *out_always_at = nullptr;
      return false;
    }
  }

  xls::verilog::Always* cpp_always_at =
      cpp_module->Add<xls::verilog::Always>(xls::SourceInfo(), cpp_elements);
  if (cpp_always_at == nullptr) {
    *error_out = xls::ToOwnedCString(
        "Failed to create always @ block in Verilog module.");
    *out_always_at = nullptr;
    return false;
  }
  *out_always_at = reinterpret_cast<xls_vast_always_base*>(cpp_always_at);
  *error_out = nullptr;
  return true;
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
    auto* expr = reinterpret_cast<xls::verilog::Expression*>(
        sensitivity_list_elements[i]);
    if (auto* pe = dynamic_cast<xls::verilog::PosEdge*>(expr)) {
      cpp_elements.push_back(pe);
    } else if (auto* ne = dynamic_cast<xls::verilog::NegEdge*>(expr)) {
      cpp_elements.push_back(ne);
    } else if (auto* lr = dynamic_cast<xls::verilog::LogicRef*>(expr)) {
      cpp_elements.push_back(lr);
    } else {
      std::string err_msg = absl::StrCat(
          "Unsupported expression type passed to sensitivity list for ",
          "always_ff at index ", i,
          ". Only Posedge, Negedge, or LogicRef expressions are supported ",
          "through this C API path.");
      *error_out = xls::ToOwnedCString(err_msg);
      *out_always_ff = nullptr;
      return false;
    }
  }
  xls::verilog::AlwaysFf* cpp_always_ff =
      cpp_module->Add<xls::verilog::AlwaysFf>(xls::SourceInfo(), cpp_elements);
  *out_always_ff = reinterpret_cast<xls_vast_always_base*>(cpp_always_ff);
  *error_out = nullptr;
  return true;
}

bool xls_vast_verilog_module_add_always_comb(
    struct xls_vast_verilog_module* m,
    struct xls_vast_always_base** out_always_comb, char** error_out) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  xls::verilog::AlwaysComb* cpp_always_comb =
      cpp_module->Add<xls::verilog::AlwaysComb>(xls::SourceInfo());
  if (cpp_always_comb == nullptr) {
    *error_out = xls::ToOwnedCString(
        "Failed to create always_comb block in Verilog module.");
    *out_always_comb = nullptr;
    return false;
  }
  *out_always_comb = reinterpret_cast<xls_vast_always_base*>(cpp_always_comb);
  *error_out = nullptr;
  return true;
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_inout(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> logic_ref =
      cpp_module->AddInOut(name, cpp_type, xls::SourceInfo());
  CHECK_OK(logic_ref.status());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref.value());
}

bool xls_vast_verilog_module_add_reg(struct xls_vast_verilog_module* m,
                                     const char* name,
                                     struct xls_vast_data_type* type,
                                     struct xls_vast_logic_ref** out_reg_ref,
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
  *out_reg_ref =
      reinterpret_cast<xls_vast_logic_ref*>(cpp_logic_ref_status.value());
  *error_out = nullptr;
  return true;
}

bool xls_vast_verilog_module_add_logic(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type, struct xls_vast_logic_ref** out_logic_ref,
    char** error_out) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_data_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<xls::verilog::LogicRef*> cpp_logic_ref_status =
      cpp_module->AddLogic(name, cpp_data_type, xls::SourceInfo());
  if (!cpp_logic_ref_status.ok()) {
    *error_out = xls::ToOwnedCString(cpp_logic_ref_status.status().ToString());
    *out_logic_ref = nullptr;
    return false;
  }
  *out_logic_ref =
      reinterpret_cast<xls_vast_logic_ref*>(cpp_logic_ref_status.value());
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

struct xls_vast_statement* xls_vast_verilog_file_make_nonblocking_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::NonblockingAssignment* cpp_assignment =
      cpp_file->Make<xls::verilog::NonblockingAssignment>(xls::SourceInfo(),
                                                          cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

struct xls_vast_statement* xls_vast_verilog_file_make_blocking_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::BlockingAssignment* cpp_assignment =
      cpp_file->Make<xls::verilog::BlockingAssignment>(xls::SourceInfo(),
                                                       cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

struct xls_vast_statement_block* xls_vast_always_base_get_statement_block(
    struct xls_vast_always_base* always_base) {
  auto* cpp_always_base =
      reinterpret_cast<xls::verilog::AlwaysBase*>(always_base);
  xls::verilog::StatementBlock* cpp_block = cpp_always_base->statements();
  return reinterpret_cast<xls_vast_statement_block*>(cpp_block);
}

struct xls_vast_statement* xls_vast_statement_block_add_nonblocking_assignment(
    struct xls_vast_statement_block* block, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  // Use the StatementBlock's Add<T> method to create and add the statement.
  // This ensures the statement is owned by the VerilogFile associated with the
  // StatementBlock.
  xls::verilog::NonblockingAssignment* cpp_assignment =
      cpp_block->Add<xls::verilog::NonblockingAssignment>(xls::SourceInfo(),
                                                          cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}
struct xls_vast_statement* xls_vast_statement_block_add_comment_text(
    struct xls_vast_statement_block* block, const char* text) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  xls::verilog::Comment* cpp_comment =
      cpp_block->Add<xls::verilog::Comment>(xls::SourceInfo(), text);
  return reinterpret_cast<xls_vast_statement*>(cpp_comment);
}
struct xls_vast_statement* xls_vast_statement_block_add_blank_line(
    struct xls_vast_statement_block* block) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  xls::verilog::BlankLine* cpp_blank =
      cpp_block->Add<xls::verilog::BlankLine>(xls::SourceInfo());
  return reinterpret_cast<xls_vast_statement*>(cpp_blank);
}
struct xls_vast_statement* xls_vast_statement_block_add_inline_text(
    struct xls_vast_statement_block* block, const char* text) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  xls::verilog::InlineVerilogStatement* cpp_stmt =
      cpp_block->Add<xls::verilog::InlineVerilogStatement>(xls::SourceInfo(),
                                                           text);
  return reinterpret_cast<xls_vast_statement*>(cpp_stmt);
}

struct xls_vast_statement* xls_vast_statement_block_add_blocking_assignment(
    struct xls_vast_statement_block* block, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::BlockingAssignment* cpp_assignment =
      cpp_block->Add<xls::verilog::BlockingAssignment>(xls::SourceInfo(),
                                                       cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

struct xls_vast_statement* xls_vast_statement_block_add_continuous_assignment(
    struct xls_vast_statement_block* block, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  auto* cpp_lhs = reinterpret_cast<xls::verilog::Expression*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::verilog::Expression*>(rhs);
  xls::verilog::ContinuousAssignment* cpp_assignment =
      cpp_block->Add<xls::verilog::ContinuousAssignment>(xls::SourceInfo(),
                                                         cpp_lhs, cpp_rhs);
  return reinterpret_cast<xls_vast_statement*>(cpp_assignment);
}

struct xls_vast_conditional* xls_vast_statement_block_add_conditional(
    struct xls_vast_statement_block* block, struct xls_vast_expression* cond) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  auto* cpp_cond_expr = reinterpret_cast<xls::verilog::Expression*>(cond);
  xls::verilog::Conditional* cpp_cond =
      cpp_block->Add<xls::verilog::Conditional>(xls::SourceInfo(),
                                                cpp_cond_expr);
  return reinterpret_cast<xls_vast_conditional*>(cpp_cond);
}

struct xls_vast_statement_block* xls_vast_conditional_get_then_block(
    struct xls_vast_conditional* cond) {
  auto* cpp_cond = reinterpret_cast<xls::verilog::Conditional*>(cond);
  return reinterpret_cast<xls_vast_statement_block*>(cpp_cond->consequent());
}

struct xls_vast_statement_block* xls_vast_conditional_add_else_if(
    struct xls_vast_conditional* cond, struct xls_vast_expression* expr_cond) {
  auto* cpp_cond = reinterpret_cast<xls::verilog::Conditional*>(cond);
  auto* cpp_cond_expr = reinterpret_cast<xls::verilog::Expression*>(expr_cond);
  xls::verilog::StatementBlock* block = cpp_cond->AddAlternate(cpp_cond_expr);
  return reinterpret_cast<xls_vast_statement_block*>(block);
}

struct xls_vast_statement_block* xls_vast_conditional_add_else(
    struct xls_vast_conditional* cond) {
  auto* cpp_cond = reinterpret_cast<xls::verilog::Conditional*>(cond);
  xls::verilog::StatementBlock* block = cpp_cond->AddAlternate(nullptr);
  return reinterpret_cast<xls_vast_statement_block*>(block);
}

struct xls_vast_conditional* xls_vast_verilog_module_add_conditional(
    struct xls_vast_verilog_module* m, struct xls_vast_expression* cond) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_cond = reinterpret_cast<xls::verilog::Expression*>(cond);
  xls::verilog::Conditional* cpp_if =
      cpp_module->Add<xls::verilog::Conditional>(xls::SourceInfo(), cpp_cond);
  return reinterpret_cast<xls_vast_conditional*>(cpp_if);
}

struct xls_vast_conditional* xls_vast_generate_loop_add_conditional(
    struct xls_vast_generate_loop* loop, struct xls_vast_expression* cond) {
  auto* cpp_loop = reinterpret_cast<xls::verilog::GenerateLoop*>(loop);
  auto* cpp_cond = reinterpret_cast<xls::verilog::Expression*>(cond);
  xls::verilog::Conditional* cpp_if =
      cpp_loop->Add<xls::verilog::Conditional>(xls::SourceInfo(), cpp_cond);
  return reinterpret_cast<xls_vast_conditional*>(cpp_if);
}

struct xls_vast_case_statement* xls_vast_statement_block_add_case(
    struct xls_vast_statement_block* block,
    struct xls_vast_expression* selector) {
  auto* cpp_block = reinterpret_cast<xls::verilog::StatementBlock*>(block);
  auto* cpp_selector = reinterpret_cast<xls::verilog::Expression*>(selector);
  xls::verilog::Case* cpp_case =
      cpp_block->Add<xls::verilog::Case>(xls::SourceInfo(), cpp_selector);
  return reinterpret_cast<xls_vast_case_statement*>(cpp_case);
}

struct xls_vast_statement_block* xls_vast_case_statement_add_item(
    struct xls_vast_case_statement* case_stmt,
    struct xls_vast_expression* match_expr) {
  auto* cpp_case = reinterpret_cast<xls::verilog::Case*>(case_stmt);
  auto* cpp_match = reinterpret_cast<xls::verilog::Expression*>(match_expr);
  xls::verilog::StatementBlock* block = cpp_case->AddCaseArm(cpp_match);
  return reinterpret_cast<xls_vast_statement_block*>(block);
}

struct xls_vast_statement_block* xls_vast_case_statement_add_default(
    struct xls_vast_case_statement* case_stmt) {
  auto* cpp_case = reinterpret_cast<xls::verilog::Case*>(case_stmt);
  xls::verilog::StatementBlock* block =
      cpp_case->AddCaseArm(xls::verilog::DefaultSentinel{});
  return reinterpret_cast<xls_vast_statement_block*>(block);
}

struct xls_vast_module_port** xls_vast_verilog_module_get_ports(
    struct xls_vast_verilog_module* m, size_t* out_count) {
  CHECK(out_count != nullptr);
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  absl::Span<const xls::verilog::ModulePort> ports = cpp_module->ports();
  *out_count = ports.size();
  if (ports.empty()) {
    return nullptr;
  }
  // Allocate an array owned by the caller (to be freed by
  // xls_vast_verilog_module_free_ports).
  auto** result = new xls_vast_module_port*[ports.size()];
  for (size_t i = 0; i < ports.size(); ++i) {
    // Cast away const to satisfy the C API – callers must treat the objects as
    // read-only.
    result[i] = reinterpret_cast<xls_vast_module_port*>(
        const_cast<xls::verilog::ModulePort*>(&ports[i]));
  }
  return result;
}

void xls_vast_verilog_module_free_ports(struct xls_vast_module_port** ports,
                                        size_t /*count*/) {
  // Only the outer array was allocated – the pointed-to ModulePort objects are
  // owned by the parent Module.
  delete[] ports;
}

xls_vast_module_port_direction xls_vast_verilog_module_port_get_direction(
    struct xls_vast_module_port* port) {
  auto* cpp_port = reinterpret_cast<xls::verilog::ModulePort*>(port);
  switch (cpp_port->direction) {
    case xls::verilog::ModulePortDirection::kInput:
      return xls_vast_module_port_direction_input;
    case xls::verilog::ModulePortDirection::kOutput:
      return xls_vast_module_port_direction_output;
    case xls::verilog::ModulePortDirection::kInOut:
      return xls_vast_module_port_direction_inout;
  }
  LOG(FATAL) << "Invalid ModulePortDirection encountered.";
}

xls_vast_def* xls_vast_verilog_module_port_get_def(
    struct xls_vast_module_port* port) {
  auto* cpp_port = reinterpret_cast<xls::verilog::ModulePort*>(port);
  return reinterpret_cast<xls_vast_def*>(cpp_port->wire);
}

char* xls_vast_def_get_name(struct xls_vast_def* def) {
  auto* cpp_def = reinterpret_cast<xls::verilog::Def*>(def);
  return xls::ToOwnedCString(cpp_def->GetName());
}

xls_vast_data_type* xls_vast_def_get_data_type(struct xls_vast_def* def) {
  auto* cpp_def = reinterpret_cast<xls::verilog::Def*>(def);
  return reinterpret_cast<xls_vast_data_type*>(cpp_def->data_type());
}

bool xls_vast_data_type_width_as_int64(struct xls_vast_data_type* type,
                                       int64_t* out_width, char** error_out) {
  CHECK(out_width != nullptr);
  CHECK(error_out != nullptr);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<int64_t> result = cpp_type->WidthAsInt64();
  if (!result.ok()) {
    *error_out = xls::ToOwnedCString(result.status().ToString());
    return false;
  }
  *out_width = result.value();
  *error_out = nullptr;
  return true;
}

bool xls_vast_data_type_flat_bit_count_as_int64(struct xls_vast_data_type* type,
                                                int64_t* out_flat_bit_count,
                                                char** error_out) {
  CHECK(out_flat_bit_count != nullptr);
  CHECK(error_out != nullptr);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  absl::StatusOr<int64_t> result = cpp_type->FlatBitCountAsInt64();
  if (!result.ok()) {
    *error_out = xls::ToOwnedCString(result.status().ToString());
    return false;
  }
  *out_flat_bit_count = result.value();
  *error_out = nullptr;
  return true;
}

struct xls_vast_expression* xls_vast_data_type_width(
    struct xls_vast_data_type* type) {
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  std::optional<xls::verilog::Expression*> expr_opt = cpp_type->width();
  if (!expr_opt.has_value()) {
    return nullptr;
  }
  return reinterpret_cast<xls_vast_expression*>(*expr_opt);
}

bool xls_vast_data_type_is_signed(struct xls_vast_data_type* type) {
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  return cpp_type->is_signed();
}

}  // extern "C"

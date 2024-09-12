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

#include "xls/public/c_api.h"

#include <string.h>  // NOLINT(modernize-deprecated-headers)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/init_xls.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/public/runtime_build_actions.h"

namespace {

std::vector<std::filesystem::path> ToCpp(const char* additional_search_paths[],
                                         size_t additional_search_paths_count) {
  std::vector<std::filesystem::path> additional_search_paths_cpp;
  additional_search_paths_cpp.reserve(additional_search_paths_count);
  for (size_t i = 0; i < additional_search_paths_count; ++i) {
    const char* additional_search_path = additional_search_paths[i];
    CHECK(additional_search_path != nullptr);
    additional_search_paths_cpp.push_back(additional_search_path);
  }
  return additional_search_paths_cpp;
}

char* ToOwnedCString(const std::string& s) { return strdup(s.c_str()); }

// Helper function that we can use to adapt to the common C API pattern when
// we're returning an `absl::StatusOr<std::string>` value.
bool ReturnStringHelper(absl::StatusOr<std::string>& to_return,
                        char** error_out, char** value_out) {
  if (to_return.ok()) {
    *value_out = ToOwnedCString(to_return.value());
    *error_out = nullptr;
    return true;
  }

  *value_out = nullptr;
  *error_out = ToOwnedCString(to_return.status().ToString());
  return false;
}

bool FormatPreferenceFromC(xls_format_preference c_pref,
                           xls::FormatPreference* cpp_pref, char** error_out) {
  switch (c_pref) {
    case xls_format_preference_default:
      *cpp_pref = xls::FormatPreference::kDefault;
      break;
    case xls_format_preference_binary:
      *cpp_pref = xls::FormatPreference::kBinary;
      break;
    case xls_format_preference_signed_decimal:
      *cpp_pref = xls::FormatPreference::kSignedDecimal;
      break;
    case xls_format_preference_unsigned_decimal:
      *cpp_pref = xls::FormatPreference::kUnsignedDecimal;
      break;
    case xls_format_preference_hex:
      *cpp_pref = xls::FormatPreference::kHex;
      break;
    case xls_format_preference_plain_binary:
      *cpp_pref = xls::FormatPreference::kPlainBinary;
      break;
    case xls_format_preference_plain_hex:
      *cpp_pref = xls::FormatPreference::kPlainHex;
      break;
    default:
      *error_out = ToOwnedCString(
          absl::StrFormat("Invalid format preference value: %d", c_pref));
      return false;
  }
  return true;
}

}  // namespace

extern "C" {

void xls_init_xls(const char* usage, int argc, char* argv[]) {
  (void)(xls::InitXls(usage, argc, argv));
}

bool xls_convert_dslx_to_ir(const char* dslx, const char* path,
                            const char* module_name,
                            const char* dslx_stdlib_path,
                            const char* additional_search_paths[],
                            size_t additional_search_paths_count,
                            char** error_out, char** ir_out) {
  CHECK(dslx != nullptr);
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxToIr(
      dslx, path, module_name, dslx_stdlib_path, additional_search_paths_cpp);
  return ReturnStringHelper(result, error_out, ir_out);
}

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out) {
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxPathToIr(
      path, dslx_stdlib_path, additional_search_paths_cpp);
  return ReturnStringHelper(result, error_out, ir_out);
}

bool xls_optimize_ir(const char* ir, const char* top, char** error_out,
                     char** ir_out) {
  CHECK(ir != nullptr);
  CHECK(top != nullptr);

  absl::StatusOr<std::string> result = xls::OptimizeIr(ir, top);
  return ReturnStringHelper(result, error_out, ir_out);
}

bool xls_mangle_dslx_name(const char* module_name, const char* function_name,
                          char** error_out, char** mangled_out) {
  CHECK(module_name != nullptr);
  CHECK(function_name != nullptr);

  absl::StatusOr<std::string> result =
      xls::MangleDslxName(module_name, function_name);
  return ReturnStringHelper(result, error_out, mangled_out);
}

bool xls_parse_typed_value(const char* input, char** error_out,
                           xls_value** xls_value_out) {
  CHECK(input != nullptr);
  CHECK(error_out != nullptr);
  CHECK(xls_value_out != nullptr);

  absl::StatusOr<xls::Value> value_or = xls::Parser::ParseTypedValue(input);
  if (value_or.ok()) {
    *xls_value_out = reinterpret_cast<xls_value*>(
        new xls::Value(std::move(value_or).value()));
    return true;
  }

  *xls_value_out = nullptr;
  *error_out = ToOwnedCString(value_or.status().ToString());
  return false;
}

struct xls_value* xls_value_make_token() {
  auto* value = new xls::Value(xls::Value::Token());
  return reinterpret_cast<xls_value*>(value);
}

struct xls_value* xls_value_make_true() {
  auto* value = new xls::Value(xls::Value::Bool(true));
  return reinterpret_cast<xls_value*>(value);
}

struct xls_value* xls_value_make_false() {
  auto* value = new xls::Value(xls::Value::Bool(false));
  return reinterpret_cast<xls_value*>(value);
}

void xls_value_free(xls_value* v) {
  delete reinterpret_cast<xls::Value*>(v);
}

void xls_package_free(struct xls_package* p) {
  delete reinterpret_cast<xls::Package*>(p);
}

void xls_c_str_free(char* c_str) {
  free(c_str);
}

bool xls_value_to_string(const struct xls_value* v, char** string_out) {
  CHECK(v != nullptr);
  CHECK(string_out != nullptr);
  std::string s = reinterpret_cast<const xls::Value*>(v)->ToString();
  *string_out = strdup(s.c_str());
  return *string_out != nullptr;
}

bool xls_value_to_string_format_preference(
    const struct xls_value* v, xls_format_preference format_preference,
    char** error_out, char** result_out) {
  CHECK(v != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);

  xls::FormatPreference cpp_pref;
  if (!FormatPreferenceFromC(format_preference, &cpp_pref, error_out)) {
    return false;
  }

  std::string s = reinterpret_cast<const xls::Value*>(v)->ToString(cpp_pref);
  *result_out = ToOwnedCString(s);
  return true;
}

bool xls_value_eq(const struct xls_value* v, const struct xls_value* w) {
  CHECK(v != nullptr);
  CHECK(w != nullptr);

  const auto* lhs = reinterpret_cast<const xls::Value*>(v);
  const auto* rhs = reinterpret_cast<const xls::Value*>(w);
  return *lhs == *rhs;
}

bool xls_format_preference_from_string(const char* s, char** error_out,
                                       xls_format_preference* result_out) {
  CHECK(s != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);

  std::string_view got(s);
  if (got == "default") {
    *result_out = xls_format_preference_default;
  } else if (got == "binary") {
    *result_out = xls_format_preference_binary;
  } else if (got == "signed_decimal") {
    *result_out = xls_format_preference_signed_decimal;
  } else if (got == "unsigned_decimal") {
    *result_out = xls_format_preference_unsigned_decimal;
  } else if (got == "hex") {
    *result_out = xls_format_preference_hex;
  } else if (got == "plain_binary") {
    *result_out = xls_format_preference_plain_binary;
  } else if (got == "plain_hex") {
    *result_out = xls_format_preference_plain_hex;
  } else {
    absl::Status error = absl::InvalidArgumentError(absl::StrFormat(
        "Invalid value for conversion to XLS format preference: `%s`", s));
    *error_out = ToOwnedCString(error.ToString());
    return false;
  }

  *error_out = nullptr;
  return true;
}

bool xls_package_to_string(const struct xls_package* p, char** string_out) {
  CHECK(p != nullptr);
  CHECK(string_out != nullptr);
  std::string s = reinterpret_cast<const xls::Package*>(p)->DumpIr();
  *string_out = strdup(s.c_str());
  return *string_out != nullptr;
}

bool xls_parse_ir_package(const char* ir, const char* filename,
                          char** error_out,
                          struct xls_package** xls_package_out) {
  CHECK(ir != nullptr);
  CHECK(error_out != nullptr);
  CHECK(xls_package_out != nullptr);

  std::optional<std::string_view> cpp_filename;
  if (filename != nullptr) {
    cpp_filename.emplace(filename);
  }
  absl::StatusOr<std::unique_ptr<xls::Package>> package_or =
      xls::Parser::ParsePackage(ir, cpp_filename);
  if (package_or.ok()) {
    *xls_package_out =
        reinterpret_cast<xls_package*>(package_or.value().release());
    *error_out = nullptr;
    return true;
  }

  *xls_package_out = nullptr;
  *error_out = ToOwnedCString(package_or.status().ToString());
  return false;
}

bool xls_package_get_function(struct xls_package* package,
                              const char* function_name, char** error_out,
                              struct xls_function** result_out) {
  xls::Package* xls_package = reinterpret_cast<xls::Package*>(package);
  absl::StatusOr<xls::Function*> function_or =
      xls_package->GetFunction(function_name);
  if (function_or.ok()) {
    *result_out = reinterpret_cast<struct xls_function*>(function_or.value());
    return true;
  }

  *result_out = nullptr;
  *error_out = ToOwnedCString(function_or.status().ToString());
  return false;
}

bool xls_package_get_type_for_value(struct xls_package* package,
                                    struct xls_value* value, char** error_out,
                                    struct xls_type** result_out) {
  CHECK(package != nullptr);
  CHECK(value != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  xls::Package* xls_package = reinterpret_cast<xls::Package*>(package);
  xls::Value* xls_value = reinterpret_cast<xls::Value*>(value);
  xls::Type* type = xls_package->GetTypeForValue(*xls_value);
  *result_out = reinterpret_cast<struct xls_type*>(type);
  return true;
}

bool xls_type_to_string(struct xls_type* type, char** error_out,
                        char** result_out) {
  CHECK(type != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  xls::Type* xls_type = reinterpret_cast<xls::Type*>(type);
  *error_out = nullptr;
  *result_out = ToOwnedCString(xls_type->ToString());
  return true;
}

bool xls_function_get_name(struct xls_function* function, char** error_out,
                           char** string_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(string_out != nullptr);
  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);

  *error_out = nullptr;
  *string_out = ToOwnedCString(xls_function->name());
  return true;
}

bool xls_function_get_type(struct xls_function* function, char** error_out,
                           xls_function_type** result_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);
  xls::FunctionType* type = xls_function->GetType();

  *error_out = nullptr;
  *result_out = reinterpret_cast<xls_function_type*>(type);
  return true;
}

bool xls_function_type_to_string(struct xls_function_type* type,
                                 char** error_out, char** string_out) {
  CHECK(type != nullptr);
  CHECK(error_out != nullptr);
  CHECK(string_out != nullptr);
  xls::FunctionType* ft = reinterpret_cast<xls::FunctionType*>(type);
  *error_out = nullptr;
  *string_out = ToOwnedCString(ft->ToString());
  return true;
}

bool xls_interpret_function(struct xls_function* function, size_t argc,
                            const struct xls_value** args, char** error_out,
                            struct xls_value** result_out) {
  CHECK(function != nullptr);
  CHECK(args != nullptr);

  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);

  std::vector<xls::Value> xls_args;
  xls_args.reserve(argc);
  for (size_t i = 0; i < argc; ++i) {
    CHECK(args[i] != nullptr);
    xls_args.push_back(*reinterpret_cast<const xls::Value*>(args[i]));
  }

  absl::StatusOr<xls::InterpreterResult<xls::Value>> result_or =
      xls::InterpretFunction(xls_function, xls_args);

  auto return_status = [result_out, error_out](const absl::Status& status) {
    *result_out = nullptr;
    *error_out = ToOwnedCString(status.ToString());
    return false;
  };

  if (!result_or.ok()) {
    return return_status(result_or.status());
  }

  // TODO(cdleary): 2024-05-30 We should pass back interpreter events through
  // this API instead of dropping them.
  xls::InterpreterResult<xls::Value> result = std::move(result_or).value();

  // Note that DropInterpreterEvents reifies any assertions into an error
  // status.
  absl::StatusOr<xls::Value> result_value =
      xls::InterpreterResultToStatusOrValue(result);

  if (!result_value.ok()) {
    return return_status(result_value.status());
  }

  *result_out = reinterpret_cast<struct xls_value*>(
      new xls::Value(std::move(result_value.value())));
  *error_out = nullptr;
  return true;
}

// -- VAST

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
  xls::verilog::LogicRef* logic_ref =
      cpp_module->AddInput(name, cpp_type, xls::SourceInfo());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref);
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_output(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  xls::verilog::LogicRef* logic_ref =
      cpp_module->AddOutput(name, cpp_type, xls::SourceInfo());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref);
}

struct xls_vast_logic_ref* xls_vast_verilog_module_add_wire(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type) {
  auto* cpp_module = reinterpret_cast<xls::verilog::Module*>(m);
  auto* cpp_type = reinterpret_cast<xls::verilog::DataType*>(type);
  xls::verilog::LogicRef* logic_ref =
      cpp_module->AddWire(name, cpp_type, xls::SourceInfo());
  return reinterpret_cast<xls_vast_logic_ref*>(logic_ref);
}

char* xls_vast_verilog_file_emit(const struct xls_vast_verilog_file* f) {
  const auto* cpp_file = reinterpret_cast<const xls::verilog::VerilogFile*>(f);
  std::string result = cpp_file->Emit();
  return ToOwnedCString(result);
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

struct xls_vast_literal* xls_vast_verilog_file_make_plain_literal(
    struct xls_vast_verilog_file* f, int32_t value) {
  auto* cpp_file = reinterpret_cast<xls::verilog::VerilogFile*>(f);
  xls::verilog::Literal* cpp_literal =
      cpp_file->PlainLiteral(value, xls::SourceInfo());
  return reinterpret_cast<xls_vast_literal*>(cpp_literal);
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

struct xls_vast_indexable_expression*
xls_vast_logic_ref_as_indexable_expression(
    struct xls_vast_logic_ref* logic_ref) {
  auto* cpp_logic_ref = reinterpret_cast<xls::verilog::LogicRef*>(logic_ref);
  auto* cpp_indexable_expression =
      static_cast<xls::verilog::IndexableExpression*>(cpp_logic_ref);
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

}  // extern "C"

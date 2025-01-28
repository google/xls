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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bit_push_buffer.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/public/c_api_format_preference.h"
#include "xls/public/c_api_impl_helpers.h"
#include "xls/public/c_api_vast.h"
#include "xls/public/runtime_build_actions.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

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
      xls::ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxToIr(
      dslx, path, module_name, dslx_stdlib_path, additional_search_paths_cpp);
  return xls::ReturnStringHelper(result, error_out, ir_out);
}

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out) {
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      xls::ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxPathToIr(
      path, dslx_stdlib_path, additional_search_paths_cpp);
  return xls::ReturnStringHelper(result, error_out, ir_out);
}

bool xls_optimize_ir(const char* ir, const char* top, char** error_out,
                     char** ir_out) {
  CHECK(ir != nullptr);
  CHECK(top != nullptr);

  absl::StatusOr<std::string> result = xls::OptimizeIr(ir, top);
  return xls::ReturnStringHelper(result, error_out, ir_out);
}

bool xls_mangle_dslx_name(const char* module_name, const char* function_name,
                          char** error_out, char** mangled_out) {
  CHECK(module_name != nullptr);
  CHECK(function_name != nullptr);

  absl::StatusOr<std::string> result =
      xls::MangleDslxName(module_name, function_name);
  return xls::ReturnStringHelper(result, error_out, mangled_out);
}

bool xls_schedule_and_codegen_package(
    xls_package* p, const char* scheduling_options_flags_proto,
    const char* codegen_flags_proto, bool with_delay_model, char** error_out,
    struct xls_schedule_and_codegen_result** result_out) {
  CHECK(p != nullptr);
  CHECK(scheduling_options_flags_proto != nullptr);
  CHECK(codegen_flags_proto != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);

  xls::Package* cpp_package = reinterpret_cast<xls::Package*>(p);

  // Get the proto objects by parsing the textprotos given.
  xls::SchedulingOptionsFlagsProto scheduling_options_flags;
  xls::CodegenFlagsProto codegen_flags;

  if (absl::Status parse_status =
          xls::ParseTextProto(scheduling_options_flags_proto, /*file_name=*/"",
                              &scheduling_options_flags);
      !parse_status.ok()) {
    *error_out = xls::ToOwnedCString(parse_status.ToString());
    return false;
  }
  if (absl::Status parse_status = xls::ParseTextProto(
          codegen_flags_proto, /*file_name=*/"", &codegen_flags);
      !parse_status.ok()) {
    *error_out = xls::ToOwnedCString(parse_status.ToString());
    return false;
  }

  absl::StatusOr<xls::ScheduleAndCodegenResult> result =
      xls::ScheduleAndCodegenPackage(cpp_package, scheduling_options_flags,
                                     codegen_flags, with_delay_model);
  if (!result.ok()) {
    *error_out = xls::ToOwnedCString(result.status().ToString());
    return false;
  }

  *result_out = reinterpret_cast<xls_schedule_and_codegen_result*>(
      new xls::ScheduleAndCodegenResult(std::move(result.value())));
  *error_out = nullptr;
  return true;
}

char* xls_schedule_and_codegen_result_get_verilog_text(
    const struct xls_schedule_and_codegen_result* result) {
  CHECK(result != nullptr);
  auto* cpp_result =
      reinterpret_cast<const xls::ScheduleAndCodegenResult*>(result);
  return xls::ToOwnedCString(cpp_result->module_generator_result.verilog_text);
}

void xls_schedule_and_codegen_result_free(
    struct xls_schedule_and_codegen_result* result) {
  delete reinterpret_cast<xls::ScheduleAndCodegenResult*>(result);
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
  *error_out = xls::ToOwnedCString(value_or.status().ToString());
  return false;
}

bool xls_value_get_bits(const struct xls_value* value, char** error_out,
                        struct xls_bits** bits_out) {
  CHECK(value != nullptr);
  CHECK(error_out != nullptr);
  CHECK(bits_out != nullptr);

  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  absl::StatusOr<xls::Bits> bits_or = cpp_value->GetBitsWithStatus();
  if (bits_or.ok()) {
    xls::Bits* heap_bits = new xls::Bits(std::move(bits_or).value());
    *bits_out = reinterpret_cast<xls_bits*>(heap_bits);
    return true;
  }

  *bits_out = nullptr;
  *error_out = xls::ToOwnedCString(bits_or.status().ToString());
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

bool xls_bits_make_ubits(int64_t bit_count, uint64_t value, char** error_out,
                         struct xls_bits** bits_out) {
  CHECK(error_out != nullptr);
  CHECK(bits_out != nullptr);
  absl::StatusOr<xls::Bits> bits = xls::UBitsWithStatus(value, bit_count);
  if (!bits.ok()) {
    *error_out = xls::ToOwnedCString(bits.status().ToString());
    return false;
  }
  *bits_out =
      reinterpret_cast<xls_bits*>(new xls::Bits(std::move(bits).value()));
  *error_out = nullptr;
  return true;
}

bool xls_bits_make_sbits(int64_t bit_count, int64_t value, char** error_out,
                         struct xls_bits** bits_out) {
  CHECK(error_out != nullptr);
  CHECK(bits_out != nullptr);
  absl::StatusOr<xls::Bits> bits = xls::SBitsWithStatus(value, bit_count);
  if (!bits.ok()) {
    *error_out = xls::ToOwnedCString(bits.status().ToString());
    return false;
  }
  *bits_out =
      reinterpret_cast<xls_bits*>(new xls::Bits(std::move(bits).value()));
  *error_out = nullptr;
  return true;
}

bool xls_bits_get_bit(const struct xls_bits* bits, int64_t index) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  return cpp_bits->Get(index);
}

struct xls_bits* xls_bits_width_slice(const struct xls_bits* bits,
                                      int64_t start, int64_t width) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  return reinterpret_cast<xls_bits*>(
      new xls::Bits(cpp_bits->Slice(start, width)));
}

struct xls_bits* xls_bits_shift_left_logical(const struct xls_bits* bits,
                                             int64_t shift_amount) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  xls::Bits result = xls::bits_ops::ShiftLeftLogical(*cpp_bits, shift_amount);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_shift_right_logical(const struct xls_bits* bits,
                                              int64_t shift_amount) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  xls::Bits result = xls::bits_ops::ShiftRightLogical(*cpp_bits, shift_amount);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_shift_right_arithmetic(const struct xls_bits* bits,
                                                 int64_t shift_amount) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  xls::Bits result = xls::bits_ops::ShiftRightArith(*cpp_bits, shift_amount);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_negate(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  xls::Bits result = xls::bits_ops::Negate(*cpp_bits);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_abs(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  xls::Bits result = xls::bits_ops::Abs(*cpp_bits);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_not(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  xls::Bits result = xls::bits_ops::Not(*cpp_bits);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_add(const struct xls_bits* lhs,
                              const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::Add(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_sub(const struct xls_bits* lhs,
                              const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::Sub(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_and(const struct xls_bits* lhs,
                              const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::And(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_or(const struct xls_bits* lhs,
                             const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::Or(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_xor(const struct xls_bits* lhs,
                              const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::Xor(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_umul(const struct xls_bits* lhs,
                               const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::UMul(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

struct xls_bits* xls_bits_smul(const struct xls_bits* lhs,
                               const struct xls_bits* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::Bits*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::Bits*>(rhs);
  xls::Bits result = xls::bits_ops::SMul(*cpp_lhs, *cpp_rhs);
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(result)));
}

char* xls_bits_to_debug_string(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  std::string s = cpp_bits->ToDebugString();
  return xls::ToOwnedCString(s);
}

struct xls_value* xls_value_make_tuple(size_t element_count,
                                       struct xls_value** elements) {
  CHECK(elements != nullptr);
  std::vector<xls::Value> cpp_elements;
  cpp_elements.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    cpp_elements.push_back(*reinterpret_cast<xls::Value*>(elements[i]));
  }
  auto* value = new xls::Value(xls::Value::TupleOwned(std::move(cpp_elements)));
  return reinterpret_cast<xls_value*>(value);
}

struct xls_bits* xls_value_flatten_to_bits(const struct xls_value* value) {
  CHECK(value != nullptr);
  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  xls::BitPushBuffer push_buffer;
  cpp_value->FlattenTo(&push_buffer);
  LOG(INFO) << "Flattened " << cpp_value->ToString() << " to "
            << push_buffer.size_in_bits() << " bits";
  xls::Bits bits = xls::Bits::FromBitmap(push_buffer.ToBitmap());
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(bits)));
}

bool xls_bits_eq(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return *cpp_a == *cpp_b;
}

int64_t xls_bits_get_bit_count(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  return cpp_bits->bit_count();
}

void xls_bits_free(xls_bits* b) { delete reinterpret_cast<xls::Bits*>(b); }

void xls_value_free(xls_value* v) { delete reinterpret_cast<xls::Value*>(v); }

struct xls_value* xls_value_from_bits(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  return reinterpret_cast<xls_value*>(new xls::Value(std::move(*cpp_bits)));
}

struct xls_function_base* xls_package_get_top(struct xls_package* p) {
  CHECK(p != nullptr);
  xls::Package* cpp_package = reinterpret_cast<xls::Package*>(p);
  std::optional<xls::FunctionBase*> top = cpp_package->GetTop();
  if (!top.has_value()) {
    return nullptr;
  }
  return reinterpret_cast<xls_function_base*>(top.value());
}

bool xls_package_set_top_by_name(struct xls_package* p, const char* name,
                                 char** error_out) {
  CHECK(p != nullptr);
  xls::Package* cpp_package = reinterpret_cast<xls::Package*>(p);
  absl::Status status = cpp_package->SetTopByName(name);
  if (!status.ok()) {
    *error_out = xls::ToOwnedCString(status.ToString());
    return false;
  }
  *error_out = nullptr;
  return true;
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
  *result_out = xls::ToOwnedCString(s);
  return true;
}

bool xls_bits_to_string(const struct xls_bits* bits,
                        xls_format_preference format_preference,
                        bool include_bit_count, char** error_out,
                        char** result_out) {
  CHECK(bits != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);

  xls::FormatPreference cpp_format_preference;
  if (!FormatPreferenceFromC(format_preference, &cpp_format_preference,
                             error_out)) {
    return false;
  }

  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  std::string s =
      xls::BitsToString(*cpp_bits, cpp_format_preference, include_bit_count);
  *result_out = xls::ToOwnedCString(s);
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
  } else if (got == "zero_padded_binary") {
    *result_out = xls_format_preference_zero_padded_binary;
  } else if (got == "zero_padded_hex") {
    *result_out = xls_format_preference_zero_padded_hex;
  } else {
    absl::Status error = absl::InvalidArgumentError(absl::StrFormat(
        "Invalid value for conversion to XLS format preference: `%s`", s));
    *error_out = xls::ToOwnedCString(error.ToString());
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
  *error_out = xls::ToOwnedCString(package_or.status().ToString());
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
  *error_out = xls::ToOwnedCString(function_or.status().ToString());
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
  *result_out = xls::ToOwnedCString(xls_type->ToString());
  return true;
}

bool xls_function_get_name(struct xls_function* function, char** error_out,
                           char** string_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(string_out != nullptr);
  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);

  *error_out = nullptr;
  *string_out = xls::ToOwnedCString(xls_function->name());
  return true;
}

bool xls_function_get_type(struct xls_function* function, char** error_out,
                           xls_function_type** xls_fn_type_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(xls_fn_type_out != nullptr);
  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);
  xls::FunctionType* type = xls_function->GetType();

  *error_out = nullptr;
  *xls_fn_type_out = reinterpret_cast<xls_function_type*>(type);
  return true;
}

bool xls_function_type_to_string(struct xls_function_type* type,
                                 char** error_out, char** string_out) {
  CHECK(type != nullptr);
  CHECK(error_out != nullptr);
  CHECK(string_out != nullptr);
  xls::FunctionType* ft = reinterpret_cast<xls::FunctionType*>(type);
  *error_out = nullptr;
  *string_out = xls::ToOwnedCString(ft->ToString());
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
    *error_out = xls::ToOwnedCString(status.ToString());
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

}  // extern "C"

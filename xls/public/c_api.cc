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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bit_push_buffer.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/jit/function_jit.h"
#include "xls/public/c_api_format_preference.h"
#include "xls/public/c_api_impl_helpers.h"
#include "xls/public/c_api_vast.h"
#include "xls/public/runtime_build_actions.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

extern "C" {

void xls_init_xls(const char* usage, int argc, char* argv[]) {
  (void)(xls::InitXls(usage, argc, argv));
}

bool xls_convert_dslx_to_ir_with_warnings(
    const char* dslx, const char* path, const char* module_name,
    const char* dslx_stdlib_path, const char* additional_search_paths[],
    size_t additional_search_paths_count, const char* enable_warnings[],
    size_t enable_warnings_count, const char* disable_warnings[],
    size_t disable_warnings_count, bool warnings_as_errors,
    char*** warnings_out, size_t* warnings_out_count, char** error_out,
    char** ir_out) {
  CHECK(dslx != nullptr);
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      xls::ToCppPaths(additional_search_paths, additional_search_paths_count);
  std::vector<std::string_view> enable_warnings_cpp =
      xls::ToCppStringViews(enable_warnings, enable_warnings_count);
  std::vector<std::string_view> disable_warnings_cpp =
      xls::ToCppStringViews(disable_warnings, disable_warnings_count);

  std::vector<std::string> warnings_out_cpp;

  const xls::ConvertDslxToIrOptions options = {
      .dslx_stdlib_path = dslx_stdlib_path,
      .additional_search_paths = additional_search_paths_cpp,
      .enable_warnings = enable_warnings_cpp,
      .disable_warnings = disable_warnings_cpp,
      .warnings_as_errors = warnings_as_errors,
      .warnings_out = &warnings_out_cpp,
  };

  absl::StatusOr<std::string> result =
      xls::ConvertDslxToIr(dslx, path, module_name, options);

  if (warnings_out != nullptr) {
    xls::ToOwnedCStrings(warnings_out_cpp, warnings_out, warnings_out_count);
  } else if (warnings_out_count != nullptr) {
    *warnings_out_count = warnings_out_cpp.size();
  }

  return xls::ReturnStringHelper(result, error_out, ir_out);
}

bool xls_convert_dslx_to_ir(const char* dslx, const char* path,
                            const char* module_name,
                            const char* dslx_stdlib_path,
                            const char* additional_search_paths[],
                            size_t additional_search_paths_count,
                            char** error_out, char** ir_out) {
  const char* enable_warnings[] = {};
  const char* disable_warnings[] = {};
  return xls_convert_dslx_to_ir_with_warnings(
      dslx, path, module_name, dslx_stdlib_path, additional_search_paths,
      additional_search_paths_count, enable_warnings, 0, disable_warnings, 0,
      /*warnings_as_errors=*/false,
      /*warnings_out=*/nullptr,
      /*warnings_out_count=*/nullptr, error_out, ir_out);
}

bool xls_convert_dslx_path_to_ir_with_warnings(
    const char* path, const char* dslx_stdlib_path,
    const char* additional_search_paths[], size_t additional_search_paths_count,
    const char* enable_warnings[], size_t enable_warnings_count,
    const char* disable_warnings[], size_t disable_warnings_count,
    bool warnings_as_errors, char*** warnings_out, size_t* warnings_out_count,
    char** error_out, char** ir_out) {
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);
  if (warnings_out != nullptr) {
    CHECK(warnings_out_count != nullptr);
  }

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      xls::ToCppPaths(additional_search_paths, additional_search_paths_count);
  std::vector<std::string_view> enable_warnings_cpp =
      xls::ToCppStringViews(enable_warnings, enable_warnings_count);
  std::vector<std::string_view> disable_warnings_cpp =
      xls::ToCppStringViews(disable_warnings, disable_warnings_count);

  std::vector<std::string> warnings_out_cpp;

  const xls::ConvertDslxToIrOptions options = {
      .dslx_stdlib_path = dslx_stdlib_path,
      .additional_search_paths = additional_search_paths_cpp,
      .enable_warnings = enable_warnings_cpp,
      .disable_warnings = disable_warnings_cpp,
      .warnings_as_errors = warnings_as_errors,
      .warnings_out = &warnings_out_cpp,
  };
  absl::StatusOr<std::string> result = xls::ConvertDslxPathToIr(path, options);

  if (warnings_out != nullptr) {
    xls::ToOwnedCStrings(warnings_out_cpp, warnings_out, warnings_out_count);
  } else if (warnings_out_count != nullptr) {
    *warnings_out_count = warnings_out_cpp.size();
  }

  return xls::ReturnStringHelper(result, error_out, ir_out);
}

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out) {
  const char* enable_warnings[] = {};
  const char* disable_warnings[] = {};
  return xls_convert_dslx_path_to_ir_with_warnings(
      path, dslx_stdlib_path, additional_search_paths,
      additional_search_paths_count, enable_warnings, 0, disable_warnings, 0,
      /*warnings_as_errors=*/false,
      /*warnings_out=*/nullptr,
      /*warnings_out_count=*/nullptr, error_out, ir_out);
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

static bool CallingConventionFromC(xls_calling_convention cc,
                                   xls::dslx::CallingConvention* out,
                                   char** error_out) {
  switch (cc) {
    case xls_calling_convention_typical:
      *out = xls::dslx::CallingConvention::kTypical;
      return true;
    case xls_calling_convention_implicit_token:
      *out = xls::dslx::CallingConvention::kImplicitToken;
      return true;
    case xls_calling_convention_proc_next:
      *out = xls::dslx::CallingConvention::kProcNext;
      return true;
    default: {
      absl::Status st =
          absl::InvalidArgumentError("Invalid calling convention enum");
      *error_out = xls::ToOwnedCString(st.ToString());
      return false;
    }
  }
}

bool xls_mangle_dslx_name_full(
    const char* module_name, const char* function_name,
    xls_calling_convention convention, const char* const free_keys[],
    size_t free_keys_count, const struct xls_dslx_parametric_env* param_env,
    const char* scope, char** error_out, char** mangled_out) {
  CHECK(module_name != nullptr);
  CHECK(function_name != nullptr);
  CHECK(error_out != nullptr);
  CHECK(mangled_out != nullptr);

  xls::dslx::CallingConvention cc_cpp;
  if (!CallingConventionFromC(convention, &cc_cpp, error_out)) {
    return false;
  }

  absl::btree_set<std::string> free_key_set;
  for (size_t i = 0; i < free_keys_count; ++i) {
    CHECK(free_keys[i] != nullptr);
    free_key_set.insert(free_keys[i]);
  }

  const xls::dslx::ParametricEnv* env =
      reinterpret_cast<const xls::dslx::ParametricEnv*>(param_env);

  std::string_view scope_sv;
  if (scope != nullptr) {
    scope_sv = scope;
  }

  absl::StatusOr<std::string> result = xls::dslx::MangleDslxName(
      module_name, function_name, cc_cpp, free_key_set, env, scope_sv);
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
  return xls::ToOwnedCString(cpp_result->codegen_result.verilog_text);
}

void xls_schedule_and_codegen_result_free(
    struct xls_schedule_and_codegen_result* result) {
  delete reinterpret_cast<xls::ScheduleAndCodegenResult*>(result);
}

bool xls_verify_package(struct xls_package* p, char** error_out) {
  CHECK(p != nullptr);
  CHECK(error_out != nullptr);

  xls::Package* cpp_package = reinterpret_cast<xls::Package*>(p);
  absl::Status st = xls::VerifyPackage(cpp_package);
  if (!st.ok()) {
    *error_out = xls::ToOwnedCString(st.ToString());
    return false;
  }
  *error_out = nullptr;
  return true;
}

bool xls_parse_typed_value(const char* input, char** error_out,
                           xls_value** xls_value_out) {
  CHECK(input != nullptr);
  CHECK(error_out != nullptr);
  CHECK(xls_value_out != nullptr);

  absl::StatusOr<xls::Value> value = xls::Parser::ParseTypedValue(input);
  if (value.ok()) {
    *xls_value_out =
        reinterpret_cast<xls_value*>(new xls::Value(*std::move(value)));
    return true;
  }

  *xls_value_out = nullptr;
  *error_out = xls::ToOwnedCString(value.status().ToString());
  return false;
}

bool xls_value_get_bits(const struct xls_value* value, char** error_out,
                        struct xls_bits** bits_out) {
  CHECK(value != nullptr);
  CHECK(error_out != nullptr);
  CHECK(bits_out != nullptr);

  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  absl::StatusOr<xls::Bits> bits = cpp_value->GetBitsWithStatus();
  if (bits.ok()) {
    xls::Bits* heap_bits = new xls::Bits(*std::move(bits));
    *bits_out = reinterpret_cast<xls_bits*>(heap_bits);
    return true;
  }

  *bits_out = nullptr;
  *error_out = xls::ToOwnedCString(bits.status().ToString());
  return false;
}

bool xls_value_get_element_count(const struct xls_value* value,
                                 char** error_out, int64_t* count_out) {
  CHECK(value != nullptr);
  CHECK(error_out != nullptr);
  CHECK(count_out != nullptr);
  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  switch (cpp_value->kind()) {
    case xls::ValueKind::kTuple:
    case xls::ValueKind::kArray:
      *count_out = cpp_value->size();
      return true;
    default:
      *error_out = xls::ToOwnedCString(
          absl::InvalidArgumentError(
              absl::StrFormat("Value is not a tuple or array, and so has no "
                              "element count; kind: %s value: %s",
                              xls::ValueKindToString(cpp_value->kind()),
                              cpp_value->ToString()))
              .ToString());
      return false;
  }
}

bool xls_value_get_element(const struct xls_value* value, size_t index,
                           char** error_out, struct xls_value** element_out) {
  CHECK(value != nullptr);
  CHECK(error_out != nullptr);
  CHECK(element_out != nullptr);
  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  switch (cpp_value->kind()) {
    case xls::ValueKind::kTuple:
    case xls::ValueKind::kArray:
      *element_out = reinterpret_cast<xls_value*>(
          new xls::Value(cpp_value->element(index)));
      break;
    case xls::ValueKind::kBits:
    case xls::ValueKind::kToken:
      *error_out = xls::ToOwnedCString(
          absl::InvalidArgumentError(
              absl::StrFormat(
                  "Value does not hold elements; kind: %s value: %s",
                  xls::ValueKindToString(cpp_value->kind()),
                  cpp_value->ToString()))
              .ToString());
      return false;
    case xls::ValueKind::kInvalid:
      *error_out = xls::ToOwnedCString(
          absl::InvalidArgumentError("Value is invalid").ToString());
      return false;
  }
  return true;
}

struct xls_value* xls_value_clone(const struct xls_value* value) {
  CHECK(value != nullptr);
  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  return reinterpret_cast<xls_value*>(new xls::Value(*cpp_value));
}

bool xls_value_get_kind(const struct xls_value* value, char** error_out,
                        xls_value_kind* kind_out) {
  CHECK(value != nullptr);
  const auto* cpp_value = reinterpret_cast<const xls::Value*>(value);
  *error_out = nullptr;
  switch (cpp_value->kind()) {
    case xls::ValueKind::kBits:
      *kind_out = xls_value_kind_bits;
      break;
    case xls::ValueKind::kToken:
      *kind_out = xls_value_kind_token;
      break;
    case xls::ValueKind::kArray:
      *kind_out = xls_value_kind_array;
      break;
    case xls::ValueKind::kTuple:
      *kind_out = xls_value_kind_tuple;
      break;
    case xls::ValueKind::kInvalid:
      *error_out = xls::ToOwnedCString(
          absl::InvalidArgumentError("Value is invalid").ToString());
      return false;
  }
  return true;
}

bool xls_value_make_ubits(int64_t bit_count, uint64_t value, char** error_out,
                          struct xls_value** xls_value_out) {
  CHECK(error_out != nullptr);
  CHECK(xls_value_out != nullptr);
  absl::StatusOr<xls::Bits> bits = xls::UBitsWithStatus(value, bit_count);
  if (!bits.ok()) {
    *error_out = xls::ToOwnedCString(bits.status().ToString());
    return false;
  }
  *xls_value_out =
      reinterpret_cast<xls_value*>(new xls::Value(std::move(bits).value()));
  *error_out = nullptr;
  return true;
}

bool xls_value_make_sbits(int64_t bit_count, int64_t value, char** error_out,
                          struct xls_value** xls_value_out) {
  CHECK(error_out != nullptr);
  CHECK(xls_value_out != nullptr);
  absl::StatusOr<xls::Bits> bits = xls::SBitsWithStatus(value, bit_count);
  if (!bits.ok()) {
    *error_out = xls::ToOwnedCString(bits.status().ToString());
    return false;
  }
  *xls_value_out =
      reinterpret_cast<xls_value*>(new xls::Value(std::move(bits).value()));
  *error_out = nullptr;
  return true;
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

struct xls_bits_rope* xls_create_bits_rope(int64_t bit_count) {
  return reinterpret_cast<xls_bits_rope*>(new xls::BitsRope(bit_count));
}

void xls_bits_rope_append_bits(struct xls_bits_rope* bits_rope,
                               const struct xls_bits* bits) {
  CHECK(bits_rope != nullptr);
  CHECK(bits != nullptr);
  auto* cpp_bits_rope = reinterpret_cast<xls::BitsRope*>(bits_rope);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  cpp_bits_rope->push_back(*cpp_bits);
}

struct xls_bits* xls_bits_rope_get_bits(struct xls_bits_rope* bits_rope) {
  CHECK(bits_rope != nullptr);
  auto* cpp_bits_rope = reinterpret_cast<xls::BitsRope*>(bits_rope);
  xls::Bits bits = cpp_bits_rope->Build();
  return reinterpret_cast<xls_bits*>(new xls::Bits(std::move(bits)));
}

bool xls_bits_make_bits_from_bytes(size_t bit_count, const uint8_t* bytes,
                                   size_t byte_count, char** error_out,
                                   struct xls_bits** bits_out) {
  CHECK(bits_out != nullptr);
  CHECK(error_out != nullptr);

  *bits_out = nullptr;

  if (bytes == nullptr) {
    *error_out = xls::ToOwnedCString("bytes is null");
    return false;
  }

  if (byte_count == 0) {
    *error_out = xls::ToOwnedCString("byte_count is 0");
    return false;
  }

  if (bit_count == 0) {
    *error_out = xls::ToOwnedCString("bit_count is 0");
    return false;
  }

  if (byte_count * 8 < byte_count) {
    // detect int overflow
    *error_out = xls::ToOwnedCString("byte_count is too large");
    return false;
  }

  if (byte_count * 8 < bit_count) {
    *error_out = xls::ToOwnedCString("byte_count*8 < bit_count");
    return false;
  }

  xls::Bits* cpp_bits = new xls::Bits(
      xls::Bits::FromBytes(absl::MakeSpan(bytes, byte_count), bit_count));

  *bits_out = reinterpret_cast<struct xls_bits*>(cpp_bits);

  return true;
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

bool xls_bits_to_bytes(const struct xls_bits* bits, char** error_out,
                       uint8_t** bytes_out, size_t* byte_count_out) {
  CHECK(bits != nullptr);
  CHECK(bytes_out != nullptr);
  CHECK(error_out != nullptr);
  CHECK(byte_count_out != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  std::vector<uint8_t> bytes = cpp_bits->ToBytes();
  *byte_count_out = bytes.size();
  *bytes_out = new uint8_t[bytes.size()];
  std::copy(bytes.begin(), bytes.end(), *bytes_out);
  return true;
}

bool xls_bits_to_uint64(const struct xls_bits* bits, char** error_out,
                        uint64_t* value_out) {
  CHECK(bits != nullptr);
  CHECK(value_out != nullptr);
  CHECK(error_out != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  absl::StatusOr<uint64_t> result = cpp_bits->ToUint64();
  if (!result.ok()) {
    *error_out = xls::ToOwnedCString(result.status().ToString());
    return false;
  }
  *value_out = result.value();
  return true;
}

bool xls_bits_to_int64(const struct xls_bits* bits, char** error_out,
                       int64_t* value_out) {
  CHECK(bits != nullptr);
  CHECK(value_out != nullptr);
  CHECK(error_out != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  absl::StatusOr<int64_t> result = cpp_bits->ToInt64();
  if (!result.ok()) {
    *error_out = xls::ToOwnedCString(result.status().ToString());
    return false;
  }
  *value_out = result.value();
  return true;
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

bool xls_value_make_array(size_t element_count, struct xls_value** elements,
                          char** error_out, struct xls_value** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  std::vector<xls::Value> cpp_elements;
  cpp_elements.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    cpp_elements.push_back(*reinterpret_cast<xls::Value*>(elements[i]));
  }
  absl::StatusOr<xls::Value> value = xls::Value::Array(std::move(cpp_elements));
  if (!value.ok()) {
    *error_out = xls::ToOwnedCString(value.status().ToString());
    return false;
  }
  *error_out = nullptr;
  *result_out = reinterpret_cast<xls_value*>(new xls::Value(std::move(*value)));
  return true;
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

bool xls_bits_ne(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  return !xls_bits_eq(a, b);
}

bool xls_bits_ult(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::ULessThan(*cpp_a, *cpp_b);
}

bool xls_bits_ule(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::ULessThanOrEqual(*cpp_a, *cpp_b);
}

bool xls_bits_ugt(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::UGreaterThan(*cpp_a, *cpp_b);
}

bool xls_bits_uge(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::UGreaterThanOrEqual(*cpp_a, *cpp_b);
}

bool xls_bits_slt(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::SLessThan(*cpp_a, *cpp_b);
}

bool xls_bits_sle(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::SLessThanOrEqual(*cpp_a, *cpp_b);
}

bool xls_bits_sgt(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::SGreaterThan(*cpp_a, *cpp_b);
}

bool xls_bits_sge(const struct xls_bits* a, const struct xls_bits* b) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const auto* cpp_a = reinterpret_cast<const xls::Bits*>(a);
  const auto* cpp_b = reinterpret_cast<const xls::Bits*>(b);
  return xls::bits_ops::SGreaterThanOrEqual(*cpp_a, *cpp_b);
}

int64_t xls_bits_get_bit_count(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  return cpp_bits->bit_count();
}

void xls_bits_free(xls_bits* b) { delete reinterpret_cast<xls::Bits*>(b); }

void xls_bits_rope_free(xls_bits_rope* b) {
  delete reinterpret_cast<xls::BitsRope*>(b);
}

void xls_value_free(xls_value* v) { delete reinterpret_cast<xls::Value*>(v); }

struct xls_value* xls_value_from_bits(const struct xls_bits* bits) {
  CHECK(bits != nullptr);
  const auto* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  return reinterpret_cast<xls_value*>(new xls::Value(*cpp_bits));
}

struct xls_value* xls_value_from_bits_owned(struct xls_bits* bits) {
  CHECK(bits != nullptr);
  auto* cpp_bits = reinterpret_cast<xls::Bits*>(bits);
  auto* result =
      reinterpret_cast<xls_value*>(new xls::Value(std::move(*cpp_bits)));
  delete cpp_bits;
  return result;
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

void xls_c_str_free(char* c_str) { free(c_str); }

void xls_c_strs_free(char** c_strs, size_t count) {
  CHECK_EQ(count == 0, c_strs == nullptr) << absl::StreamFormat(
      "xls_c_strs_free expected consistency between zero-size and "
      "array being nullptr; count = %d, c_strs = %p",
      count, c_strs);

  if (c_strs == nullptr) {
    return;
  }

  for (size_t i = 0; i < count; ++i) {
    free(c_strs[i]);
  }
  delete[] c_strs;
}

void xls_bytes_free(uint8_t* bytes) { delete[] bytes; }

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
  absl::StatusOr<std::unique_ptr<xls::Package>> package =
      xls::Parser::ParsePackage(ir, cpp_filename);
  if (package.ok()) {
    *xls_package_out = reinterpret_cast<xls_package*>(package->release());
    *error_out = nullptr;
    return true;
  }

  *xls_package_out = nullptr;
  *error_out = xls::ToOwnedCString(package.status().ToString());
  return false;
}

bool xls_package_get_function(struct xls_package* package,
                              const char* function_name, char** error_out,
                              struct xls_function** result_out) {
  xls::Package* xls_package = reinterpret_cast<xls::Package*>(package);
  absl::StatusOr<xls::Function*> function =
      xls_package->GetFunction(function_name);
  if (function.ok()) {
    *result_out = reinterpret_cast<struct xls_function*>(*function);
    return true;
  }

  *result_out = nullptr;
  *error_out = xls::ToOwnedCString(function.status().ToString());
  return false;
}

bool xls_package_get_functions(struct xls_package* package, char** error_out,
                               struct xls_function*** result_out,
                               size_t* count_out) {
  CHECK(package != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  CHECK(count_out != nullptr);
  xls::Package* xls_package = reinterpret_cast<xls::Package*>(package);
  absl::Span<const std::unique_ptr<xls::Function>> functions =
      xls_package->functions();
  *count_out = functions.size();

  *error_out = nullptr;

  if (*count_out == 0) {
    *result_out = nullptr;
    return true;
  }

  *result_out = new struct xls_function*[*count_out];

  for (size_t i = 0; i < *count_out; ++i) {
    (*result_out)[i] = reinterpret_cast<struct xls_function*>(
        const_cast<xls::Function*>(functions[i].get()));
  }

  return true;
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

bool xls_type_get_kind(struct xls_type* type, char** error_out,
                       xls_value_kind* kind_out) {
  CHECK(type != nullptr);
  CHECK(error_out != nullptr);
  CHECK(kind_out != nullptr);
  xls::Type* xls_type = reinterpret_cast<xls::Type*>(type);
  *error_out = nullptr;
  switch (xls_type->kind()) {
    case xls::TypeKind::kBits:
      *kind_out = xls_value_kind_bits;
      break;
    case xls::TypeKind::kTuple:
      *kind_out = xls_value_kind_tuple;
      break;
    case xls::TypeKind::kArray:
      *kind_out = xls_value_kind_array;
      break;
    case xls::TypeKind::kToken:
      *kind_out = xls_value_kind_token;
      break;
    default:
      *error_out = xls::ToOwnedCString(
          absl::StrFormat("Unknown type kind: %d", xls_type->kind()));
      *kind_out = xls_value_kind_invalid;
      return false;
  }
  return true;
}

int64_t xls_type_get_leaf_count(struct xls_type* type) {
  CHECK(type != nullptr);
  xls::Type* xls_type = reinterpret_cast<xls::Type*>(type);
  return xls_type->leaf_count();
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

int64_t xls_type_get_flat_bit_count(struct xls_type* type) {
  CHECK(type != nullptr);
  xls::Type* xls_type = reinterpret_cast<xls::Type*>(type);
  return xls_type->GetFlatBitCount();
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

int64_t xls_function_type_get_param_count(struct xls_function_type* type) {
  CHECK(type != nullptr);
  const auto* ft = reinterpret_cast<const xls::FunctionType*>(type);
  return ft->parameter_count();
}

bool xls_function_type_get_param_type(struct xls_function_type* type,
                                      size_t index, char** error_out,
                                      struct xls_type** param_type_out) {
  CHECK(type != nullptr);
  CHECK(error_out != nullptr);
  CHECK(param_type_out != nullptr);
  const auto* ft = reinterpret_cast<const xls::FunctionType*>(type);
  if (index >= static_cast<size_t>(ft->parameter_count())) {
    *error_out = xls::ToOwnedCString(
        absl::InvalidArgumentError(
            absl::StrFormat("Parameter index %d out of range; "
                            "parameter_count: %d",
                            index, ft->parameter_count()))
            .ToString());
    *param_type_out = nullptr;
    return false;
  }
  *error_out = nullptr;
  *param_type_out = reinterpret_cast<xls_type*>(ft->parameter_type(index));
  return true;
}

struct xls_type* xls_function_type_get_return_type(
    struct xls_function_type* type) {
  CHECK(type != nullptr);
  const auto* ft = reinterpret_cast<const xls::FunctionType*>(type);
  return reinterpret_cast<xls_type*>(ft->return_type());
}

bool xls_function_get_param_name(struct xls_function* function, size_t index,
                                 char** error_out, char** name_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(name_out != nullptr);

  xls::Function* cpp_fn = reinterpret_cast<xls::Function*>(function);
  int64_t param_count = cpp_fn->params().size();
  if (index >= static_cast<size_t>(param_count)) {
    *name_out = nullptr;
    *error_out = xls::ToOwnedCString(
        absl::InvalidArgumentError(
            absl::StrFormat("Parameter index %d out of range; "
                            "parameter_count: %d",
                            index, param_count))
            .ToString());
    return false;
  }

  xls::Param* param = cpp_fn->param(index);
  *name_out = xls::ToOwnedCString(param->name());
  *error_out = nullptr;
  return true;
}

bool xls_make_function_jit(struct xls_function* function, char** error_out,
                           struct xls_function_jit** result_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);

  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);
  absl::StatusOr<std::unique_ptr<xls::FunctionJit>> jit =
      xls::FunctionJit::Create(xls_function);
  if (!jit.ok()) {
    *error_out = xls::ToOwnedCString(jit.status().ToString());
    return false;
  }
  *result_out = reinterpret_cast<struct xls_function_jit*>(jit->release());
  *error_out = nullptr;
  return true;
}

void xls_function_jit_free(struct xls_function_jit* jit) {
  delete reinterpret_cast<xls::FunctionJit*>(jit);
}

void xls_function_ptr_array_free(struct xls_function** function_pointer_array) {
  delete[] function_pointer_array;
}

bool xls_function_jit_run(struct xls_function_jit* jit, size_t argc,
                          const struct xls_value* const* args, char** error_out,
                          struct xls_trace_message** trace_messages_out,
                          size_t* trace_messages_count_out,
                          char*** assert_messages_out,
                          size_t* assert_messages_count_out,
                          struct xls_value** result_out) {
  CHECK(jit != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);

  std::vector<xls::Value> cpp_args;
  cpp_args.reserve(argc);
  for (size_t i = 0; i < argc; ++i) {
    CHECK(args[i] != nullptr);
    cpp_args.push_back(*reinterpret_cast<const xls::Value*>(args[i]));
  }

  xls::FunctionJit* xls_jit = reinterpret_cast<xls::FunctionJit*>(jit);
  absl::StatusOr<xls::InterpreterResult<xls::Value>> result =
      xls_jit->Run(absl::MakeConstSpan(cpp_args));
  if (!result.ok()) {
    *error_out = xls::ToOwnedCString(result.status().ToString());
    return false;
  }

  std::vector<std::string> trace_msgs = result->events.GetTraceMessageStrings();
  std::unique_ptr<xls_trace_message[]> trace_messages =
      std::make_unique<xls_trace_message[]>(trace_msgs.size());
  for (int i = 0; i < trace_msgs.size(); ++i) {
    trace_messages[i].message = xls::ToOwnedCString(trace_msgs[i]);
    trace_messages[i].verbosity = 0;
  }
  *trace_messages_out = trace_messages.release();
  *trace_messages_count_out = trace_msgs.size();

  std::vector<std::string> assert_msgs = result->events.GetAssertMessages();
  xls::ToOwnedCStrings(assert_msgs, assert_messages_out,
                       assert_messages_count_out);

  *result_out = reinterpret_cast<struct xls_value*>(
      new xls::Value(std::move(result->value)));
  *error_out = nullptr;
  return true;
}

void xls_trace_messages_free(struct xls_trace_message* trace_messages,
                             size_t count) {
  if (trace_messages == nullptr) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    xls_c_str_free(trace_messages[i].message);
  }
  delete[] trace_messages;
}

bool xls_interpret_function(struct xls_function* function, size_t argc,
                            const struct xls_value* const* args,
                            char** error_out, struct xls_value** result_out) {
  CHECK(function != nullptr);
  CHECK(args != nullptr);

  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);

  std::vector<xls::Value> xls_args;
  xls_args.reserve(argc);
  for (size_t i = 0; i < argc; ++i) {
    CHECK(args[i] != nullptr);
    xls_args.push_back(*reinterpret_cast<const xls::Value*>(args[i]));
  }

  absl::StatusOr<xls::InterpreterResult<xls::Value>> result =
      xls::InterpretFunction(xls_function, xls_args);

  auto return_status = [result_out, error_out](const absl::Status& status) {
    *result_out = nullptr;
    *error_out = xls::ToOwnedCString(status.ToString());
    return false;
  };

  if (!result.ok()) {
    return return_status(result.status());
  }

  // TODO(cdleary): 2024-05-30 We should pass back interpreter events through
  // this API instead of dropping them.
  // Note that DropInterpreterEvents reifies any assertions into an error
  // status.
  absl::StatusOr<xls::Value> result_value =
      xls::InterpreterResultToStatusOrValue(*std::move(result));

  if (!result_value.ok()) {
    return return_status(result_value.status());
  }

  *result_out = reinterpret_cast<struct xls_value*>(
      new xls::Value(*std::move(result_value)));
  *error_out = nullptr;
  return true;
}

bool xls_function_to_z3_smtlib(struct xls_function* function, char** error_out,
                               char** string_out) {
  CHECK(function != nullptr);
  CHECK(error_out != nullptr);
  CHECK(string_out != nullptr);

  xls::Function* cpp_function = reinterpret_cast<xls::Function*>(function);
  absl::StatusOr<std::string> smtlib =
      xls::solvers::z3::EmitFunctionAsSmtLib(cpp_function);
  return xls::ReturnStringHelper(smtlib, error_out, string_out);
}

}  // extern "C"

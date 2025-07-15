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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/macros.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/public/c_api_dslx.h"
#include "xls/public/c_api_format_preference.h"
#include "xls/public/c_api_ir_builder.h"

namespace {

using ::testing::HasSubstr;

// Smoke test for `xls_convert_dslx_to_ir` C API.
TEST(XlsCApiTest, ConvertDslxToIrSimple) {
  const std::string kProgram = "fn id(x: u32) -> u32 { x }";
  const char* additional_search_paths[] = {};
  char* error_out = nullptr;
  char* ir_out = nullptr;
  std::string dslx_stdlib_path = std::string(xls::kDefaultDslxStdlibPath);
  bool ok =
      xls_convert_dslx_to_ir(kProgram.c_str(), "my_module.x", "my_module",
                             /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(),
                             additional_search_paths, 0, &error_out, &ir_out);

  absl::Cleanup free_cstrs([&] {
    xls_c_str_free(error_out);
    xls_c_str_free(ir_out);
  });

  // We should get IR and no error.
  ASSERT_TRUE(ok);
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(ir_out, nullptr);

  EXPECT_THAT(ir_out, HasSubstr("fn __my_module__id"));
}

TEST(XlsCApiTest, ConvertDslxToIrWithWarningsSet) {
  const std::string kProgram = "fn id() { let x = u32:1; }";
  const char* additional_search_paths[] = {};
  const std::string dslx_stdlib_path = std::string(xls::kDefaultDslxStdlibPath);

  {
    char* error_out = nullptr;
    char* ir_out = nullptr;

    absl::Cleanup free_cstrs([&] {
      xls_c_str_free(error_out);
      xls_c_str_free(ir_out);
    });

    LOG(INFO)
        << "converting with warnings in default state, should see warning...";
    char** warnings = nullptr;
    size_t warnings_count = 0;
    absl::Cleanup free_warnings(
        [&] { xls_c_strs_free(warnings, warnings_count); });

    bool ok = xls_convert_dslx_to_ir_with_warnings(
        kProgram.c_str(), "my_module.x", "my_module",
        /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(), additional_search_paths,
        0,
        /*enable_warnings=*/nullptr, 0, /*disable_warnings=*/nullptr, 0,
        /*warnings_as_errors=*/true, &warnings, &warnings_count, &error_out,
        &ir_out);

    // Check we got the warning data even though the return code is non-ok.
    ASSERT_EQ(warnings_count, 1);
    ASSERT_NE(warnings, nullptr);
    ASSERT_NE(warnings[0], nullptr);
    EXPECT_THAT(warnings[0], HasSubstr("is not used in function"));

    // Since we set warnings-as-errors to true, we should have gotten "not ok"
    // back.
    ASSERT_FALSE(ok);
    ASSERT_EQ(ir_out, nullptr);
    EXPECT_THAT(error_out, HasSubstr("Conversion of DSLX to IR failed due to "
                                     "warnings during parsing/typechecking."));
  }

  // Now try with the warning disabled.
  {
    char* error_out = nullptr;
    char* ir_out = nullptr;

    absl::Cleanup free_cstrs([&] {
      xls_c_str_free(error_out);
      xls_c_str_free(ir_out);
    });
    const char* enable_warnings[] = {};
    const char* disable_warnings[] = {"unused_definition"};
    LOG(INFO) << "converting with warning disabled, should not see warning...";
    bool ok = xls_convert_dslx_to_ir_with_warnings(
        kProgram.c_str(), "my_module.x", "my_module",
        /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(), additional_search_paths,
        0, enable_warnings, 0, disable_warnings, 1, /*warnings_as_errors=*/true,
        /*warnings_out=*/nullptr, /*warnings_out_count=*/nullptr, &error_out,
        &ir_out);
    ASSERT_TRUE(ok);
    ASSERT_EQ(error_out, nullptr);
    ASSERT_NE(ir_out, nullptr);
  }
}

TEST(XlsCApiTest, ConvertWithNoWarnings) {
  const std::string kProgram = "fn id(x: u32) -> u32 { x }";
  const std::string dslx_stdlib_path = std::string(xls::kDefaultDslxStdlibPath);
  const char* additional_search_paths[] = {};
  char* error_out = nullptr;
  char* ir_out = nullptr;

  absl::Cleanup free_cstrs([&] {
    xls_c_str_free(error_out);
    xls_c_str_free(ir_out);
  });

  char** warnings = nullptr;
  size_t warnings_count = 0;
  absl::Cleanup free_warnings(
      [&] { xls_c_strs_free(warnings, warnings_count); });

  bool ok = xls_convert_dslx_to_ir_with_warnings(
      kProgram.c_str(), "my_module.x", "my_module",
      /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(), additional_search_paths, 0,
      /*enable_warnings=*/nullptr, 0, /*disable_warnings=*/nullptr, 0,
      /*warnings_as_errors=*/true, &warnings, &warnings_count, &error_out,
      &ir_out);
  ASSERT_TRUE(ok);
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(ir_out, nullptr);
  EXPECT_THAT(ir_out, HasSubstr("fn __my_module__id"));
  // Validate that in the no-warnings case we get a zero count and also a
  // nullptr value populating our warnings ptr.
  ASSERT_EQ(warnings_count, 0);
  ASSERT_EQ(warnings, nullptr);
}

TEST(XlsCApiTest, ConvertDslxToIrError) {
  const std::string kInvalidProgram = "@!";
  const char* additional_search_paths[] = {};
  char* error_out = nullptr;
  char* ir_out = nullptr;

  absl::Cleanup free_cstrs([&] {
    xls_c_str_free(error_out);
    xls_c_str_free(ir_out);
  });

  const std::string dslx_stdlib_path = std::string(xls::kDefaultDslxStdlibPath);
  bool ok = xls_convert_dslx_to_ir(
      kInvalidProgram.c_str(), "my_module.x", "my_module",
      /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(), additional_search_paths, 0,
      &error_out, &ir_out);
  ASSERT_FALSE(ok);

  // We should get an error and not get IR.
  ASSERT_NE(error_out, nullptr);
  ASSERT_EQ(ir_out, nullptr);

  EXPECT_THAT(error_out, HasSubstr("Unrecognized character: '@'"));
}

// Smoke test for `xls_convert_dslx_path_to_ir` C API.
TEST(XlsCApiTest, ConvertDslxPathToIr) {
  const std::string kProgram = "fn id(x: u32) -> u32 { x }";

  XLS_ASSERT_OK_AND_ASSIGN(xls::TempDirectory tempdir,
                           xls::TempDirectory::Create());
  const std::filesystem::path module_path = tempdir.path() / "my_module.x";
  XLS_ASSERT_OK(xls::SetFileContents(module_path, kProgram));

  const char* additional_search_paths[] = {};
  char* error_out = nullptr;
  char* ir_out = nullptr;
  const std::string dslx_stdlib_path = std::string(xls::kDefaultDslxStdlibPath);
  bool ok = xls_convert_dslx_path_to_ir(
      module_path.c_str(),
      /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(), additional_search_paths, 0,
      &error_out, &ir_out);

  absl::Cleanup free_cstrs([&] {
    xls_c_str_free(error_out);
    xls_c_str_free(ir_out);
  });

  // We should get IR and no error.
  ASSERT_TRUE(ok);
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(ir_out, nullptr);

  EXPECT_THAT(ir_out, HasSubstr("fn __my_module__id"));

  // Now we take the IR and schedule/codegen it.
  struct xls_package* package = nullptr;
  ASSERT_TRUE(
      xls_parse_ir_package(ir_out, "my_module.ir", &error_out, &package));
  absl::Cleanup free_package([package] { xls_package_free(package); });

  EXPECT_EQ(xls_package_get_top(package), nullptr);
  ASSERT_TRUE(
      xls_package_set_top_by_name(package, "__my_module__id", &error_out));
  EXPECT_NE(xls_package_get_top(package), nullptr);

  const char* kSchedulingOptionsFlagsProto = R"(
pipeline_stages: 1
delay_model: "unit"
)";
  const char* kCodegenFlagsProto = R"(
register_merge_strategy: STRATEGY_DONT_MERGE
generator: GENERATOR_KIND_PIPELINE
)";

  struct xls_schedule_and_codegen_result* result = nullptr;
  ASSERT_TRUE(xls_schedule_and_codegen_package(
      package, /*scheduling_options_flags_proto=*/kSchedulingOptionsFlagsProto,
      /*codegen_flags_proto=*/kCodegenFlagsProto, /*with_delay_model=*/false,
      &error_out, &result))
      << "xls_schedule_and_codegen_package error: " << error_out;
  absl::Cleanup free_result(
      [result] { xls_schedule_and_codegen_result_free(result); });

  char* verilog_out = xls_schedule_and_codegen_result_get_verilog_text(result);
  ASSERT_NE(verilog_out, nullptr);
  absl::Cleanup free_verilog([verilog_out] { xls_c_str_free(verilog_out); });

  LOG(INFO) << "== Verilog";
  XLS_LOG_LINES(INFO, verilog_out);

  EXPECT_THAT(verilog_out, HasSubstr("module __my_module__id"));
}

TEST(XlsCApiTest, ParseTypedValueAndFreeIt) {
  char* error = nullptr;
  struct xls_value* value = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[32]:0x42", &error, &value));

  char* string_out = nullptr;
  ASSERT_TRUE(xls_value_to_string(value, &string_out));
  EXPECT_EQ(std::string{string_out}, "bits[32]:66");
  xls_c_str_free(string_out);
  string_out = nullptr;

  // Also to-string it via a format preference.
  xls_format_preference fmt_pref;
  ASSERT_TRUE(xls_format_preference_from_string("hex", &error, &fmt_pref));
  ASSERT_TRUE(xls_value_to_string_format_preference(value, fmt_pref, &error,
                                                    &string_out));
  EXPECT_EQ(std::string{string_out}, "bits[32]:0x42");
  xls_c_str_free(string_out);

  xls_value_free(value);
}

// Takes a bits-based value and flattens it to a bits buffer and checks it's the
// same as the bits inside the value.
TEST(XlsCApiTest, FlattenBitsValueToBits) {
  char* error_out = nullptr;
  xls_value* value = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[32]:0x42", &error_out, &value));
  absl::Cleanup free_value([value] { xls_value_free(value); });

  xls_value_kind kind = xls_value_kind_invalid;
  ASSERT_TRUE(xls_value_get_kind(value, &error_out, &kind));
  EXPECT_EQ(kind, xls_value_kind_bits);
  ASSERT_EQ(error_out, nullptr);

  // Get the bits from within the value. Note that it's owned by the caller.
  xls_bits* value_bits = nullptr;
  ASSERT_TRUE(xls_value_get_bits(value, &error_out, &value_bits));
  absl::Cleanup free_value_bits([value_bits] { xls_bits_free(value_bits); });

  // Flatten the value to a bits buffer.
  xls_bits* flattened = xls_value_flatten_to_bits(value);
  absl::Cleanup free_flattened([flattened] { xls_bits_free(flattened); });

  // The flattened bits should be the same as the original bits.
  char* value_bits_str = xls_bits_to_debug_string(value_bits);
  absl::Cleanup free_value_bits_str(
      [=] { xls_c_str_free(value_bits_str); });  // Ensure free
  char* flattened_str = xls_bits_to_debug_string(flattened);
  absl::Cleanup free_flattened_str(
      [=] { xls_c_str_free(flattened_str); });  // Ensure free
  EXPECT_TRUE(xls_bits_eq(value_bits, flattened))
      << "value_bits: " << value_bits_str << "\nflattened:  " << flattened_str;
}

TEST(XlsCApiTest, FlattenTupleValueToBits) {
  char* error_out = nullptr;
  xls_value* u3_7 = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[3]:0x7", &error_out, &u3_7));
  absl::Cleanup free_u3_7([u3_7] { xls_value_free(u3_7); });

  xls_value* u2_0 = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[2]:0x0", &error_out, &u2_0));
  absl::Cleanup free_u2_0([u2_0] { xls_value_free(u2_0); });

  // Make them into a tuple; i.e. (bits[3]:0x7, bits[2]:0x0).
  xls_value* elements[] = {u3_7, u2_0};
  xls_value* tuple = xls_value_make_tuple(/*element_count=*/2, elements);
  absl::Cleanup free_tuple([tuple] { xls_value_free(tuple); });

  xls_value_kind kind = xls_value_kind_invalid;
  ASSERT_TRUE(xls_value_get_kind(tuple, &error_out, &kind));
  EXPECT_EQ(kind, xls_value_kind_tuple);

  // Get the elements and check they are equal to the originals.
  xls_value* u3_7_extracted = nullptr;
  ASSERT_TRUE(xls_value_get_element(tuple, 0, &error_out, &u3_7_extracted));
  absl::Cleanup free_u3_7_extracted(
      [u3_7_extracted] { xls_value_free(u3_7_extracted); });
  EXPECT_TRUE(xls_value_eq(u3_7, u3_7_extracted));
  // White-box check that the pointers are not the same as the extracted value
  // is an independent value owned by the caller.
  EXPECT_NE(u3_7, u3_7_extracted);

  xls_value* u2_0_extracted = nullptr;
  ASSERT_TRUE(xls_value_get_element(tuple, 1, &error_out, &u2_0_extracted));
  absl::Cleanup free_u2_0_extracted(
      [u2_0_extracted] { xls_value_free(u2_0_extracted); });
  EXPECT_TRUE(xls_value_eq(u2_0, u2_0_extracted));
  // White-box check that the pointers are not the same as the extracted value
  // is an independent value owned by the caller.
  EXPECT_NE(u2_0, u2_0_extracted);

  // Flatten the tuple to a bits value.
  xls_bits* flattened = xls_value_flatten_to_bits(tuple);
  absl::Cleanup free_flattened([flattened] { xls_bits_free(flattened); });

  // Make the desired value.
  xls_bits* want_bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(5, 0b11100, &error_out, &want_bits));
  absl::Cleanup free_want_bits([want_bits] { xls_bits_free(want_bits); });

  // Check they're equivalent.
  char* flattened_str = xls_bits_to_debug_string(flattened);
  absl::Cleanup free_flattened_str(
      [=] { xls_c_str_free(flattened_str); });  // Ensure free
  char* want_bits_str = xls_bits_to_debug_string(want_bits);
  absl::Cleanup free_want_bits_str(
      [=] { xls_c_str_free(want_bits_str); });  // Ensure free
  EXPECT_TRUE(xls_bits_eq(flattened, want_bits))
      << "flattened: " << flattened_str << "\nwant_bits: " << want_bits_str;
}

TEST(XlsCApiTest, MakeArrayValue) {
  char* error_out = nullptr;

  xls_value* u3_7 = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[3]:0x7", &error_out, &u3_7));
  absl::Cleanup free_u3_7([u3_7] { xls_value_free(u3_7); });

  xls_value* u2_0 = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[2]:0x0", &error_out, &u2_0));
  absl::Cleanup free_u2_0([u2_0] { xls_value_free(u2_0); });

  // Make a valid array of two elements.
  {
    xls_value* elements[] = {u3_7, u3_7};
    xls_value* array = nullptr;
    ASSERT_TRUE(xls_value_make_array(
        /*element_count=*/2, elements, &error_out, &array));
    absl::Cleanup free_array([array] { xls_value_free(array); });

    xls_value_kind kind = xls_value_kind_invalid;
    ASSERT_TRUE(xls_value_get_kind(array, &error_out, &kind));
    EXPECT_EQ(kind, xls_value_kind_array);

    char* value_str = nullptr;
    ASSERT_TRUE(xls_value_to_string(array, &value_str));
    absl::Cleanup free_value_str([value_str] { xls_c_str_free(value_str); });
    EXPECT_EQ(std::string_view{value_str}, "[bits[3]:7, bits[3]:7]");
  }

  // Make an invalid array of two elements.
  {
    xls_value* elements[] = {u3_7, u2_0};
    xls_value* array = nullptr;
    ASSERT_FALSE(xls_value_make_array(
        /*element_count=*/2, elements, &error_out, &array));
    absl::Cleanup free_error([error_out] { xls_c_str_free(error_out); });
    EXPECT_THAT(std::string_view{error_out}, HasSubstr("SameTypeAs"));
  }
}

TEST(XlsCApiTest, MakeBitsFromUint8DataWithMsbPadding) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(6, 0b11'0000, &error_out, &bits));
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });

  EXPECT_EQ(xls_bits_get_bit_count(bits), 6);
  EXPECT_EQ(xls_bits_get_bit(bits, 0), 0);
  EXPECT_EQ(xls_bits_get_bit(bits, 1), 0);
  EXPECT_EQ(xls_bits_get_bit(bits, 2), 0);
  EXPECT_EQ(xls_bits_get_bit(bits, 3), 0);
  EXPECT_EQ(xls_bits_get_bit(bits, 4), 1);
  EXPECT_EQ(xls_bits_get_bit(bits, 5), 1);
}

TEST(XlsCApiTest, BitsToBytes) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(32, 0x0A0B0C0D, &error_out, &bits));
  uint8_t* bytes = nullptr;
  size_t byte_count = 0;

  absl::Cleanup free_memory([&error_out, &bits, &bytes] {
    xls_c_str_free(error_out);
    xls_bits_free(bits);
    xls_bytes_free(bytes);
  });

  ASSERT_TRUE(xls_bits_to_bytes(bits, &error_out, &bytes, &byte_count));
  EXPECT_EQ(byte_count, 4);
  EXPECT_EQ(bytes[0], 0x0D);
  EXPECT_EQ(bytes[1], 0x0C);
  EXPECT_EQ(bytes[2], 0x0B);
  EXPECT_EQ(bytes[3], 0x0A);
}

TEST(XlsCApiTest, BitsToBytesWithPadding) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(
      34, 0b11'0000'0000'0000'0000'0000'0000'0000'0000, &error_out, &bits));
  uint8_t* bytes = nullptr;
  size_t byte_count = 0;

  absl::Cleanup free_memory([&error_out, &bits, &bytes] {
    xls_c_str_free(error_out);
    xls_bits_free(bits);
    xls_bytes_free(bytes);
  });

  ASSERT_TRUE(xls_bits_to_bytes(bits, &error_out, &bytes, &byte_count));
  EXPECT_EQ(byte_count, 5);
  EXPECT_EQ(bytes[0], 0x00);
  EXPECT_EQ(bytes[1], 0x00);
  EXPECT_EQ(bytes[2], 0x00);
  EXPECT_EQ(bytes[3], 0x00);
  EXPECT_EQ(bytes[4], 0b11);
}

TEST(XlsCApiTest, BitsToUint64Fit) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(64, 0x0A0B0C0D, &error_out, &bits));
  absl::Cleanup free_memory([&error_out, &bits] {
    xls_c_str_free(error_out);
    xls_bits_free(bits);
  });

  uint64_t value = 0;
  ASSERT_TRUE(xls_bits_to_uint64(bits, &error_out, &value));
  EXPECT_EQ(value, 0x0A0B0C0D);
}

TEST(XlsCApiTest, BitsToInt64Fit) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(64, 0x0A0B0C0D, &error_out, &bits));
  absl::Cleanup free_memory([&error_out, &bits] {
    xls_c_str_free(error_out);
    xls_bits_free(bits);
  });

  int64_t value = 0;
  ASSERT_TRUE(xls_bits_to_int64(bits, &error_out, &value));
  EXPECT_EQ(value, 0x0A0B0C0D);
}

TEST(XlsCApiTest, MakeSbitsDoesNotFit) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_FALSE(xls_bits_make_sbits(1, -2, &error_out, &bits));
  absl::Cleanup free_error([error_out] { xls_c_str_free(error_out); });
  EXPECT_THAT(std::string_view{error_out},
              HasSubstr("Value 0xfffffffffffffffe requires 2 bits to fit in an "
                        "signed datatype"));

  ASSERT_TRUE(xls_bits_make_sbits(2, -2, &error_out, &bits));
  char* bits_str = xls_bits_to_debug_string(bits);
  absl::Cleanup free_bits_str(
      [=] { xls_c_str_free(bits_str); });  // Ensure free
  EXPECT_EQ(std::string(bits_str), "0b10");
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });
}

TEST(XlsCApiTest, FlattenArrayValueToBits) {
  char* error_out = nullptr;
  xls_value* array = nullptr;
  ASSERT_TRUE(
      xls_parse_typed_value("[bits[3]:0x7, bits[3]:0x0]", &error_out, &array));
  absl::Cleanup free_array([array] { xls_value_free(array); });

  xls_bits* flattened = xls_value_flatten_to_bits(array);
  absl::Cleanup free_flattened([flattened] { xls_bits_free(flattened); });

  xls_bits* want_bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(6, 0b111000, &error_out, &want_bits));
  absl::Cleanup free_want_bits([want_bits] { xls_bits_free(want_bits); });

  char* flattened_str = xls_bits_to_debug_string(flattened);
  absl::Cleanup free_flattened_str(
      [=] { xls_c_str_free(flattened_str); });  // Ensure free
  char* want_bits_str = xls_bits_to_debug_string(want_bits);
  absl::Cleanup free_want_bits_str(
      [=] { xls_c_str_free(want_bits_str); });  // Ensure free
  EXPECT_TRUE(xls_bits_eq(flattened, want_bits))
      << "flattened: " << flattened_str << "\nwant_bits: " << want_bits_str;
}

TEST(XlsCApiTest, MakeSignedBits) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_sbits(5, -1, &error_out, &bits));
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });

  EXPECT_EQ(xls_bits_get_bit_count(bits), 5);
  EXPECT_EQ(xls_bits_get_bit(bits, 0), 1);

  xls_value* value = nullptr;
  ASSERT_TRUE(xls_value_make_sbits(5, -1, &error_out, &value));
  absl::Cleanup free_value([value] { xls_value_free(value); });

  xls_bits* value_bits = nullptr;
  ASSERT_TRUE(xls_value_get_bits(value, &error_out, &value_bits));
  absl::Cleanup free_value_bits([value_bits] { xls_bits_free(value_bits); });

  EXPECT_EQ(xls_bits_get_bit_count(value_bits), 5);
  EXPECT_EQ(xls_bits_get_bit(value_bits, 0), 1);
  // Check that they are equal even though they were created different ways, and
  // that their pointer are different because one is held inside a value (white
  // box knowledge but just a gut check).
  EXPECT_TRUE(xls_bits_eq(bits, value_bits));
  EXPECT_NE(bits, value_bits);
}

TEST(XlsCApiTest, MakeUnsignedBits) {
  // First create via the xls_bits_make_ubits API.
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(5, 0b10101, &error_out, &bits));
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });

  // Now create via the xls_value_make_ubits API.
  xls_value* value = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(5, 0b10101, &error_out, &value));
  absl::Cleanup free_value([value] { xls_value_free(value); });

  // Check that the bits are the same.
  xls_bits* value_bits = nullptr;
  ASSERT_TRUE(xls_value_get_bits(value, &error_out, &value_bits));
  absl::Cleanup free_value_bits([value_bits] { xls_bits_free(value_bits); });

  EXPECT_TRUE(xls_bits_eq(bits, value_bits));
  EXPECT_NE(bits, value_bits);
}

TEST(XlsCApiTest, ParsePackageAndGetFunctions) {
  const std::string kPackage0 = R"(package p)";
  const std::string kPackage1 = R"(package p
fn f(x: bits[32] id=3) -> bits[32] {
  ret y: bits[32] = identity(x, id=2)
}
)";
  const std::string kPackage2 = R"(package p
fn f(x: bits[32] id=3) -> bits[32] {
  ret y: bits[32] = identity(x, id=2)
}
fn g(x: bits[32] id=4) -> bits[32] {
  ret y: bits[32] = identity(x, id=5)
}
)";

  char* error = nullptr;
  struct xls_package* package0 = nullptr;
  ASSERT_TRUE(
      xls_parse_ir_package(kPackage0.c_str(), "p.ir", &error, &package0))
      << "xls_parse_ir_package error: " << error;
  ASSERT_TRUE(error == nullptr);
  absl::Cleanup free_package0([&package0] { xls_package_free(package0); });

  struct xls_function** functions0 = nullptr;
  size_t function_count = 0;
  ASSERT_TRUE(xls_package_get_functions(package0, &error, &functions0,
                                        &function_count));
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(package0, nullptr);

  ASSERT_EQ(function_count, 0);
  ASSERT_EQ(functions0, nullptr);

  ASSERT_EQ(error, nullptr);

  struct xls_package* package1 = nullptr;
  struct xls_function** functions1 = nullptr;
  ASSERT_TRUE(
      xls_parse_ir_package(kPackage1.c_str(), "p.ir", &error, &package1))
      << "xls_parse_ir_package error: " << error;
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(package1, nullptr);
  ASSERT_TRUE(xls_package_get_functions(package1, &error, &functions1,
                                        &function_count));
  ASSERT_EQ(error, nullptr);
  absl::Cleanup free_package_and_function_array1([&package1, &functions1] {
    xls_function_ptr_array_free(functions1);
    xls_package_free(package1);
  });
  ASSERT_EQ(function_count, 1);
  ASSERT_NE(functions1, nullptr);
  ASSERT_NE(functions1[0], nullptr);

  struct xls_package* package2 = nullptr;
  struct xls_function** functions2 = nullptr;
  ASSERT_TRUE(
      xls_parse_ir_package(kPackage2.c_str(), "p.ir", &error, &package2))
      << "xls_parse_ir_package error: " << error;
  ASSERT_NE(package2, nullptr);
  ASSERT_EQ(error, nullptr);
  ASSERT_TRUE(xls_package_get_functions(package2, &error, &functions2,
                                        &function_count));
  ASSERT_EQ(error, nullptr);
  absl::Cleanup free__package_and_function_array2([&package2, &functions2] {
    xls_function_ptr_array_free(functions2);
    xls_package_free(package2);
  });
  ASSERT_EQ(error, nullptr);
  ASSERT_EQ(function_count, 2);
  ASSERT_NE(functions2, nullptr);
  ASSERT_NE(functions2[0], nullptr);
  ASSERT_NE(functions2[1], nullptr);
}

TEST(XlsCApiTest, ParsePackageAndInterpretFunctionInIt) {
  const std::string kPackage = R"(package p

fn f(x: bits[32] id=3) -> bits[32] {
  ret y: bits[32] = identity(x, id=2)
}
)";

  char* error = nullptr;
  struct xls_package* package = nullptr;
  ASSERT_TRUE(xls_parse_ir_package(kPackage.c_str(), "p.ir", &error, &package))
      << "xls_parse_ir_package error: " << error;
  absl::Cleanup free_package([package] { xls_package_free(package); });

  struct xls_function** functions = nullptr;
  size_t function_count = 0;
  char* function_name = nullptr;
  ASSERT_TRUE(
      xls_package_get_functions(package, &error, &functions, &function_count));
  absl::Cleanup free_function_array_and_function_name(
      [&functions, &function_name] {
        xls_function_ptr_array_free(functions);
        xls_c_str_free(function_name);
      });
  ASSERT_EQ(function_count, 1);
  ASSERT_TRUE(xls_function_get_name(functions[0], &error, &function_name));
  EXPECT_EQ(std::string_view(function_name), "f");

  char* dumped = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &dumped));
  absl::Cleanup free_dumped([dumped] { xls_c_str_free(dumped); });
  EXPECT_EQ(std::string_view(dumped), kPackage);

  struct xls_function* function = nullptr;
  ASSERT_TRUE(xls_package_get_function(package, "f", &error, &function));

  // Test out the get_name functionality on the function.
  char* name = nullptr;
  ASSERT_TRUE(xls_function_get_name(function, &error, &name));
  absl::Cleanup free_name([name] { xls_c_str_free(name); });
  EXPECT_EQ(std::string_view(name), "f");

  // Test out the get_type functionality on the function.
  struct xls_function_type* f_type = nullptr;
  ASSERT_TRUE(xls_function_get_type(function, &error, &f_type));

  char* type_str = nullptr;
  ASSERT_TRUE(xls_function_type_to_string(f_type, &error, &type_str));
  absl::Cleanup free_type_str([type_str] { xls_c_str_free(type_str); });
  EXPECT_EQ(std::string_view(type_str), "(bits[32]) -> bits[32]");

  int64_t param_count = xls_function_type_get_param_count(f_type);
  EXPECT_EQ(param_count, 1);

  struct xls_type* param_type = nullptr;
  ASSERT_TRUE(xls_function_type_get_param_type(f_type, /*index=*/0, &error,
                                               &param_type));
  ASSERT_NE(param_type, nullptr);

  xls_value_kind kind;
  char* kind_error = nullptr;
  ASSERT_TRUE(xls_type_get_kind(param_type, &kind_error, &kind));
  absl::Cleanup free_kind_error([kind_error] { xls_c_str_free(kind_error); });
  EXPECT_EQ(kind_error, nullptr);
  ASSERT_EQ(kind, xls_value_kind_bits);

  int64_t bit_count = xls_type_get_flat_bit_count(param_type);
  EXPECT_EQ(bit_count, 32);

  int64_t leaf_count = xls_type_get_leaf_count(param_type);
  EXPECT_EQ(leaf_count, 1);

  EXPECT_EQ(kind, xls_value_kind_bits);
  char* param_type_str = nullptr;
  ASSERT_TRUE(xls_type_to_string(param_type, &error, &param_type_str));
  absl::Cleanup free_param_type_str(
      [param_type_str] { xls_c_str_free(param_type_str); });
  EXPECT_EQ(std::string_view(param_type_str), "bits[32]");

  struct xls_type* ret_type = xls_function_type_get_return_type(f_type);
  ASSERT_NE(ret_type, nullptr);
  char* ret_type_str = nullptr;
  ASSERT_TRUE(xls_type_to_string(ret_type, &error, &ret_type_str));
  absl::Cleanup free_ret_type_str(
      [ret_type_str] { xls_c_str_free(ret_type_str); });
  EXPECT_EQ(std::string_view(ret_type_str), "bits[32]");

  struct xls_value* ft = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[32]:0x42", &error, &ft));
  absl::Cleanup free_ft([ft] { xls_value_free(ft); });

  // Check that we can get the type of the value.
  struct xls_type* ft_type = nullptr;
  ASSERT_TRUE(xls_package_get_type_for_value(package, ft, &error, &ft_type));

  // Now convert that type to a string so we can observe it.
  char* ft_type_str = nullptr;
  ASSERT_TRUE(xls_type_to_string(ft_type, &error, &ft_type_str));
  absl::Cleanup free_ft_type_str(
      [ft_type_str] { xls_c_str_free(ft_type_str); });
  EXPECT_EQ(std::string_view(ft_type_str), "bits[32]");

  const struct xls_value* args[] = {ft};

  struct xls_value* result = nullptr;
  ASSERT_TRUE(
      xls_interpret_function(function, /*argc=*/1, args, &error, &result));
  absl::Cleanup free_result([result] { xls_value_free(result); });

  ASSERT_TRUE(xls_value_eq(ft, result));
}

TEST(XlsCApiTest, ParsePackageAndOptimizeFunctionInIt) {
  const std::string kPackage = R"(
package p

fn f() -> bits[32] {
  one: bits[32] = literal(value=1)
  ret result: bits[32] = add(one, one)
}
)";

  char* error = nullptr;
  char* opt_ir = nullptr;
  ASSERT_TRUE(xls_optimize_ir(kPackage.c_str(), "f", &error, &opt_ir));
  absl::Cleanup free_opt_ir([opt_ir] { xls_c_str_free(opt_ir); });

  ASSERT_NE(opt_ir, nullptr);

  const std::string kWant = R"(package p

top fn f() -> bits[32] {
  ret result: bits[32] = literal(value=2, id=5)
}
)";

  EXPECT_EQ(std::string_view(opt_ir), kWant);
}

TEST(XlsCApiTest, MangleDslxName) {
  std::string module_name = "foo_bar";
  std::string function_name = "baz_bat";

  char* error = nullptr;
  char* mangled = nullptr;
  ASSERT_TRUE(xls_mangle_dslx_name(module_name.c_str(), function_name.c_str(),
                                   &error, &mangled));
  absl::Cleanup free_mangled([mangled] { xls_c_str_free(mangled); });

  EXPECT_EQ(std::string_view(mangled), "__foo_bar__baz_bat");
}

TEST(XlsCApiTest, ValueToStringFormatPreferences) {
  char* error = nullptr;
  struct xls_value* value = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[32]:0x42", &error, &value));
  absl::Cleanup free_value([value] { xls_value_free(value); });

  struct TestCase {
    std::string name;
    std::string want;
  } kTestCases[] = {
      {.name = "default", .want = "bits[32]:66"},
      {.name = "binary", .want = "bits[32]:0b100_0010"},
      {.name = "signed_decimal", .want = "bits[32]:66"},
      {.name = "unsigned_decimal", .want = "bits[32]:66"},
      {.name = "hex", .want = "bits[32]:0x42"},
      {.name = "plain_binary", .want = "bits[32]:1000010"},
      {.name = "plain_hex", .want = "bits[32]:42"},
  };

  for (const auto& [name, want] : kTestCases) {
    xls_format_preference fmt_pref;
    ASSERT_TRUE(
        xls_format_preference_from_string(name.c_str(), &error, &fmt_pref));

    char* string_out = nullptr;
    ASSERT_TRUE(xls_value_to_string_format_preference(value, fmt_pref, &error,
                                                      &string_out));
    absl::Cleanup free_string_out([string_out] { xls_c_str_free(string_out); });
    EXPECT_EQ(std::string{string_out}, want);
  }
}

TEST(XlsCApiTest, InterpretDslxFailFunction) {
  // Convert the DSLX function to IR.
  const std::string kDslxModule = R"(fn just_fail() {
  fail!("only_failure_here", ())
})";

  const char* additional_search_paths[] = {};
  char* error = nullptr;
  char* ir = nullptr;
  const std::string dslx_stdlib_path = std::string(xls::kDefaultDslxStdlibPath);
  ASSERT_TRUE(
      xls_convert_dslx_to_ir(kDslxModule.c_str(), "my_module.x", "my_module",
                             /*dslx_stdlib_path=*/dslx_stdlib_path.c_str(),
                             additional_search_paths, 0, &error, &ir))
      << error;

  absl::Cleanup free_cstrs([&] {
    xls_c_str_free(error);
    xls_c_str_free(ir);
  });

  struct xls_package* package = nullptr;
  ASSERT_TRUE(xls_parse_ir_package(ir, "p.ir", &error, &package))
      << "xls_parse_ir_package error: " << error;
  absl::Cleanup free_package([package] { xls_package_free(package); });

  // Get the function.
  struct xls_function* function = nullptr;
  ASSERT_TRUE(xls_package_get_function(package, "__itok__my_module__just_fail",
                                       &error, &function));

  struct xls_value* token = xls_value_make_token();
  struct xls_value* activated = xls_value_make_true();
  absl::Cleanup free_values([&] {
    xls_value_free(token);
    xls_value_free(activated);
  });

  const struct xls_value* args[] = {token, activated};
  struct xls_value* result = nullptr;
  ASSERT_FALSE(
      xls_interpret_function(function, /*argc=*/2, args, &error, &result));
  EXPECT_EQ(std::string{error},
            "ABORTED: Assertion failure via fail! @ my_module.x:2:8-2:33");
}

TEST(XlsCApiTest, DslxInspectTypeDefinitions) {
  const char kProgram[] = R"(const EIGHT = u5:8;

struct MyStruct {
    some_field: s42,
    other_field: u64,
}

enum MyEnum : u5 {
    A = u5:2,
    B = u5:4,
    C = EIGHT,
}
)";
  const char* additional_search_paths[] = {};

  xls_dslx_import_data* import_data = xls_dslx_import_data_create(
      std::string{xls::kDefaultDslxStdlibPath}.c_str(), additional_search_paths,
      0);
  ASSERT_NE(import_data, nullptr);
  absl::Cleanup free_import_data(
      [=] { xls_dslx_import_data_free(import_data); });

  xls_dslx_typechecked_module* tm = nullptr;
  char* error = nullptr;
  bool ok = xls_dslx_parse_and_typecheck(kProgram, "foo.x", "foo", import_data,
                                         &error, &tm);
  absl::Cleanup free_tm([=] { xls_dslx_typechecked_module_free(tm); });
  ASSERT_TRUE(ok) << "got not-ok result from parse-and-typecheck; error: "
                  << error;
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(tm, nullptr);

  xls_dslx_module* module = xls_dslx_typechecked_module_get_module(tm);
  xls_dslx_type_info* type_info = xls_dslx_typechecked_module_get_type_info(tm);

  char* module_name = xls_dslx_module_get_name(module);
  absl::Cleanup free_module_name([=] { xls_c_str_free(module_name); });
  EXPECT_EQ(std::string_view{module_name}, std::string_view{"foo"});

  int64_t type_definition_count =
      xls_dslx_module_get_type_definition_count(module);
  ASSERT_EQ(type_definition_count, 2);

  xls_dslx_type_definition_kind kind0 =
      xls_dslx_module_get_type_definition_kind(module, 0);
  xls_dslx_type_definition_kind kind1 =
      xls_dslx_module_get_type_definition_kind(module, 1);
  EXPECT_EQ(kind0, xls_dslx_type_definition_kind_struct_def);
  EXPECT_EQ(kind1, xls_dslx_type_definition_kind_enum_def);

  {
    xls_dslx_struct_def* struct_def =
        xls_dslx_module_get_type_definition_as_struct_def(module, 0);
    char* identifier = xls_dslx_struct_def_get_identifier(struct_def);
    absl::Cleanup free_identifier([=] { xls_c_str_free(identifier); });
    EXPECT_EQ(std::string_view{identifier}, std::string_view{"MyStruct"});

    EXPECT_FALSE(xls_dslx_struct_def_is_parametric(struct_def));
    EXPECT_EQ(xls_dslx_struct_def_get_member_count(struct_def), 2);

    // Get the concrete type that this resolves to.
    const xls_dslx_type* struct_def_type =
        xls_dslx_type_info_get_type_struct_def(type_info, struct_def);
    int64_t total_bit_count = 0;
    ASSERT_TRUE(xls_dslx_type_get_total_bit_count(struct_def_type, &error,
                                                  &total_bit_count))
        << "got not-ok result from get-total-bit-count; error: " << error;
    ASSERT_EQ(error, nullptr);
    EXPECT_EQ(total_bit_count, 42 + 64);

    // Get the two members and see how many bits they resolve to.
    xls_dslx_struct_member* member0 =
        xls_dslx_struct_def_get_member(struct_def, 0);
    xls_dslx_struct_member* member1 =
        xls_dslx_struct_def_get_member(struct_def, 1);

    char* member0_name = xls_dslx_struct_member_get_name(member0);
    absl::Cleanup free_member0_name([=] { xls_c_str_free(member0_name); });
    ASSERT_NE(member0_name, nullptr);
    EXPECT_EQ(std::string_view{member0_name}, "some_field");

    xls_dslx_type_annotation* member0_type_annotation =
        xls_dslx_struct_member_get_type(member0);
    xls_dslx_type_annotation* member1_type_annotation =
        xls_dslx_struct_member_get_type(member1);

    const xls_dslx_type* member0_type =
        xls_dslx_type_info_get_type_type_annotation(type_info,
                                                    member0_type_annotation);
    const xls_dslx_type* member1_type =
        xls_dslx_type_info_get_type_type_annotation(type_info,
                                                    member1_type_annotation);

    bool is_signed;
    ASSERT_TRUE(xls_dslx_type_is_signed_bits(member0_type, &error, &is_signed));
    EXPECT_TRUE(is_signed);
    ASSERT_TRUE(xls_dslx_type_is_signed_bits(member1_type, &error, &is_signed));
    EXPECT_FALSE(is_signed);

    ASSERT_TRUE(xls_dslx_type_get_total_bit_count(member0_type, &error,
                                                  &total_bit_count));
    EXPECT_EQ(total_bit_count, 42);

    ASSERT_TRUE(xls_dslx_type_get_total_bit_count(member1_type, &error,
                                                  &total_bit_count));
    EXPECT_EQ(total_bit_count, 64);
  }

  {
    xls_dslx_enum_def* enum_def =
        xls_dslx_module_get_type_definition_as_enum_def(module, 1);
    char* enum_identifier = xls_dslx_enum_def_get_identifier(enum_def);
    absl::Cleanup free_enum_identifier(
        [=] { xls_c_str_free(enum_identifier); });
    EXPECT_EQ(std::string_view{enum_identifier}, std::string_view{"MyEnum"});

    int64_t enum_member_count = xls_dslx_enum_def_get_member_count(enum_def);
    EXPECT_EQ(enum_member_count, 3);

    xls_dslx_enum_member* member2 = xls_dslx_enum_def_get_member(enum_def, 2);
    xls_dslx_expr* member2_expr = xls_dslx_enum_member_get_value(member2);

    char* member2_name = xls_dslx_enum_member_get_name(member2);
    absl::Cleanup free_member2_name([=] { xls_c_str_free(member2_name); });
    ASSERT_NE(member2_name, nullptr);
    EXPECT_EQ(std::string_view{member2_name}, "C");

    xls_dslx_interp_value* member2_value = nullptr;
    ASSERT_TRUE(xls_dslx_type_info_get_const_expr(type_info, member2_expr,
                                                  &error, &member2_value));
    absl::Cleanup free_member2_value(
        [=] { xls_dslx_interp_value_free(member2_value); });
    ASSERT_NE(member2_value, nullptr);

    xls_value* member2_ir_value = nullptr;
    ASSERT_TRUE(xls_dslx_interp_value_convert_to_ir(member2_value, &error,
                                                    &member2_ir_value));
    absl::Cleanup free_member2_ir_value(
        [=] { xls_value_free(member2_ir_value); });

    char* value_str = nullptr;
    ASSERT_TRUE(xls_value_to_string(member2_ir_value, &value_str));
    absl::Cleanup free_value_str([=] { xls_c_str_free(value_str); });
    EXPECT_EQ(std::string_view{value_str}, "bits[5]:8");

    const xls_dslx_type* enum_def_type =
        xls_dslx_type_info_get_type_enum_def(type_info, enum_def);
    int64_t total_bit_count = 0;
    ASSERT_TRUE(xls_dslx_type_get_total_bit_count(enum_def_type, &error,
                                                  &total_bit_count))
        << "got not-ok result from get-total-bit-count; error: " << error;
    ASSERT_EQ(error, nullptr);
    EXPECT_EQ(total_bit_count, 5);

    // Check the signedness of the underlying type.
    bool is_signed = true;
    ASSERT_TRUE(
        xls_dslx_type_is_signed_bits(enum_def_type, &error, &is_signed));
    ASSERT_EQ(error, nullptr);
    EXPECT_FALSE(is_signed);
  }
}

TEST(XlsCApiTest, DslxInspectTypeRefTypeAnnotation) {
  const char kImported[] = "pub type SomeType = u32;";
  XLS_ASSERT_OK_AND_ASSIGN(xls::TempDirectory tempdir,
                           xls::TempDirectory::Create());
  const std::filesystem::path& tempdir_path = tempdir.path();
  const std::filesystem::path module_path =
      tempdir_path / "my_imported_module.x";
  XLS_ASSERT_OK(xls::SetFileContents(module_path, kImported));

  const char kProgram[] = R"(import my_imported_module;

type MyTypeAlias = my_imported_module::SomeType;
type MyOtherTypeAlias = MyTypeAlias;
)";
  const char* additional_search_paths[] = {tempdir_path.c_str()};

  xls_dslx_import_data* import_data = xls_dslx_import_data_create(
      std::string{xls::kDefaultDslxStdlibPath}.c_str(), additional_search_paths,
      ABSL_ARRAYSIZE(additional_search_paths));
  ASSERT_NE(import_data, nullptr);
  absl::Cleanup free_import_data(
      [=] { xls_dslx_import_data_free(import_data); });

  xls_dslx_typechecked_module* tm = nullptr;
  char* error = nullptr;
  bool ok = xls_dslx_parse_and_typecheck(kProgram, "foo.x", "foo", import_data,
                                         &error, &tm);
  absl::Cleanup free_tm([=] { xls_dslx_typechecked_module_free(tm); });
  ASSERT_TRUE(ok) << "got not-ok result from parse-and-typecheck; error: "
                  << error;
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(tm, nullptr);

  xls_dslx_module* module = xls_dslx_typechecked_module_get_module(tm);
  xls_dslx_type_alias* my_type_alias =
      xls_dslx_module_get_type_definition_as_type_alias(module, 0);

  // Validate that the name is "MyTypeAlias".
  {
    char* identifier = xls_dslx_type_alias_get_identifier(my_type_alias);
    absl::Cleanup free_identifier([=] { xls_c_str_free(identifier); });
    EXPECT_EQ(std::string_view{identifier}, std::string_view{"MyTypeAlias"});
  }

  // Get the type definition for the right hand side -- it should be a
  // TypeRefTypeAnnotation, which we traverse to a TypeRef where we can resolve
  // its subject as an import.
  {
    xls_dslx_type_annotation* type =
        xls_dslx_type_alias_get_type_annotation(my_type_alias);
    xls_dslx_type_ref_type_annotation* type_ref_type_annotation =
        xls_dslx_type_annotation_get_type_ref_type_annotation(type);
    xls_dslx_type_ref* type_ref =
        xls_dslx_type_ref_type_annotation_get_type_ref(
            type_ref_type_annotation);
    xls_dslx_type_definition* type_definition =
        xls_dslx_type_ref_get_type_definition(type_ref);
    xls_dslx_colon_ref* colon_ref =
        xls_dslx_type_definition_get_colon_ref(type_definition);
    xls_dslx_import* import_subject =
        xls_dslx_colon_ref_resolve_import_subject(colon_ref);
    EXPECT_NE(import_subject, nullptr);

    char* attr = xls_dslx_colon_ref_get_attr(colon_ref);
    absl::Cleanup free_attr([=] { xls_c_str_free(attr); });
    EXPECT_EQ(std::string_view{attr}, std::string_view{"SomeType"});
  }

  // Validate that we can get the type definition for `MyOtherTypeAlias`.
  {
    xls_dslx_type_alias* other_type_alias =
        xls_dslx_module_get_type_definition_as_type_alias(module, 1);
    // Check it's the alias we were expecting via its identifier.
    char* other_type_alias_identifier =
        xls_dslx_type_alias_get_identifier(other_type_alias);
    absl::Cleanup free_other_type_alias_identifier(
        [=] { xls_c_str_free(other_type_alias_identifier); });
    EXPECT_EQ(std::string_view{other_type_alias_identifier},
              std::string_view{"MyOtherTypeAlias"});

    // Get the right hand side and understand it is referencing the other type
    // alias.
    xls_dslx_type_annotation* rhs =
        xls_dslx_type_alias_get_type_annotation(other_type_alias);
    xls_dslx_type_ref_type_annotation* rhs_type_ref_type_annotation =
        xls_dslx_type_annotation_get_type_ref_type_annotation(rhs);
    xls_dslx_type_ref* rhs_type_ref =
        xls_dslx_type_ref_type_annotation_get_type_ref(
            rhs_type_ref_type_annotation);
    xls_dslx_type_definition* rhs_type_definition =
        xls_dslx_type_ref_get_type_definition(rhs_type_ref);
    xls_dslx_type_alias* rhs_type_alias =
        xls_dslx_type_definition_get_type_alias(rhs_type_definition);
    EXPECT_EQ(rhs_type_alias, my_type_alias);
  }
}

TEST(XlsCApiTest, DslxModuleMembers) {
  const std::string_view kProgram = R"(
    struct MyStruct {}
    enum MyEnum: u32 {}
    type MyTypeAlias = ();
    const MY_CONSTANT: u32 = u32:42;
  )";

  const char* additional_search_paths[] = {};
  xls_dslx_import_data* import_data = xls_dslx_import_data_create(
      std::string{xls::kDefaultDslxStdlibPath}.c_str(), additional_search_paths,
      0);
  ASSERT_NE(import_data, nullptr);
  absl::Cleanup free_import_data(
      [=] { xls_dslx_import_data_free(import_data); });

  char* error = nullptr;
  xls_dslx_typechecked_module* tm = nullptr;
  bool ok = xls_dslx_parse_and_typecheck(kProgram.data(), "<test>", "top",
                                         import_data, &error, &tm);
  ASSERT_TRUE(ok) << "error: " << error;
  absl::Cleanup free_tm([=] { xls_dslx_typechecked_module_free(tm); });

  xls_dslx_module* module = xls_dslx_typechecked_module_get_module(tm);
  xls_dslx_type_info* type_info = xls_dslx_typechecked_module_get_type_info(tm);

  int64_t member_count = xls_dslx_module_get_member_count(module);
  EXPECT_EQ(member_count, 4);

  // module member 0: `MyStruct`
  {
    xls_dslx_module_member* struct_def_member =
        xls_dslx_module_get_member(module, 0);
    xls_dslx_struct_def* struct_def =
        xls_dslx_module_member_get_struct_def(struct_def_member);
    char* struct_def_identifier =
        xls_dslx_struct_def_get_identifier(struct_def);
    absl::Cleanup free_struct_def_identifier(
        [&] { xls_c_str_free(struct_def_identifier); });
    EXPECT_EQ(std::string_view{struct_def_identifier}, "MyStruct");
  }

  // module member 1: `MyEnum`
  {
    xls_dslx_module_member* enum_def_member =
        xls_dslx_module_get_member(module, 1);
    xls_dslx_enum_def* enum_def =
        xls_dslx_module_member_get_enum_def(enum_def_member);
    char* enum_def_identifier = xls_dslx_enum_def_get_identifier(enum_def);
    absl::Cleanup free_enum_def_identifier(
        [&] { xls_c_str_free(enum_def_identifier); });
    EXPECT_EQ(std::string_view{enum_def_identifier}, "MyEnum");
  }

  // module member 2: `MyTypeAlias`
  {
    xls_dslx_module_member* type_alias_member =
        xls_dslx_module_get_member(module, 2);
    xls_dslx_type_alias* type_alias =
        xls_dslx_module_member_get_type_alias(type_alias_member);
    char* type_alias_identifier =
        xls_dslx_type_alias_get_identifier(type_alias);
    absl::Cleanup free_type_alias_identifier(
        [&] { xls_c_str_free(type_alias_identifier); });
    EXPECT_EQ(std::string_view{type_alias_identifier}, "MyTypeAlias");
  }

  // module member 3: `MY_CONSTANT`
  {
    xls_dslx_module_member* constant_def_member =
        xls_dslx_module_get_member(module, 3);
    xls_dslx_constant_def* constant_def =
        xls_dslx_module_member_get_constant_def(constant_def_member);
    char* constant_def_name = xls_dslx_constant_def_get_name(constant_def);
    absl::Cleanup free_constant_def_name(
        [&] { xls_c_str_free(constant_def_name); });
    EXPECT_EQ(std::string_view{constant_def_name}, "MY_CONSTANT");

    xls_dslx_expr* interp_value = xls_dslx_constant_def_get_value(constant_def);
    // Get the constexpr value via the type information.
    char* error = nullptr;
    xls_dslx_interp_value* result = nullptr;
    ASSERT_TRUE(xls_dslx_type_info_get_const_expr(type_info, interp_value,
                                                  &error, &result));
    absl::Cleanup free_result([&] { xls_dslx_interp_value_free(result); });

    // Spot check the interpreter value we got from constexpr evaluation.
    char* interp_value_str = xls_dslx_interp_value_to_string(result);
    absl::Cleanup free_interp_value_str(
        [&] { xls_c_str_free(interp_value_str); });
    EXPECT_EQ(std::string_view{interp_value_str}, "u32:42");
  }
}

TEST(XlsCApiTest, ValueGetElementCount) {
  const std::initializer_list<
      std::pair<const char*, std::variant<int64_t, std::string_view>>>
      kTestCases = {
          {"())", 0},
          {"(bits[32]:42)", 1},
          {"(bits[32]:42, bits[32]:43)", 2},
          // Arrays
          {"[bits[32]:42]", 1},
          {"[bits[32]:42, bits[32]:43]", 2},
          // Errors
          {"bits[32]:42", "no element count"},
      };
  for (const auto& [input, expected] : kTestCases) {
    xls_value* value = nullptr;
    char* error = nullptr;
    absl::Cleanup free_error([&] { xls_c_str_free(error); });
    ASSERT_TRUE(xls_parse_typed_value(input, &error, &value));
    absl::Cleanup free_value([&] { xls_value_free(value); });

    int64_t element_count = 0;
    bool success = xls_value_get_element_count(value, &error, &element_count);
    ASSERT_EQ(success, std::holds_alternative<int64_t>(expected));
    if (std::holds_alternative<int64_t>(expected)) {
      EXPECT_EQ(element_count, std::get<int64_t>(expected));
    } else {
      EXPECT_THAT(error, HasSubstr(std::get<std::string_view>(expected)));
    }
  }
}

// In the `_owned` variation of the API we don't need to free the bits value.
TEST(XlsCApiTest, ValueFromBitsOwned) {
  xls_bits* bits = nullptr;
  char* error = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(32, 42, &error, &bits));

  {
    char* bits_str = xls_bits_to_debug_string(bits);
    absl::Cleanup free_bits_str([=] { xls_c_str_free(bits_str); });
    EXPECT_EQ(std::string_view{bits_str}, "0b00000000000000000000000000101010");
  }

  xls_value* value = xls_value_from_bits_owned(bits);
  absl::Cleanup free_value([=] { xls_value_free(value); });

  {
    char* value_str = nullptr;
    ASSERT_TRUE(xls_value_to_string(value, &value_str));
    absl::Cleanup free_value_str([=] { xls_c_str_free(value_str); });
    EXPECT_EQ(std::string_view{value_str}, "bits[32]:42");
  }
}

TEST(XlsCApiTest, ValueFromBitsUnowned) {
  xls_bits* bits = nullptr;
  char* error = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(32, 42, &error, &bits));
  absl::Cleanup free_bits([=] { xls_bits_free(bits); });

  // We'll create two values from this one bits object to try to show its guts
  // are not moved or corrupted in some way.
  xls_value* value1 = xls_value_from_bits(bits);
  absl::Cleanup free_value1([=] { xls_value_free(value1); });
  xls_value* value2 = xls_value_from_bits(bits);
  absl::Cleanup free_value2([=] { xls_value_free(value2); });

  EXPECT_TRUE(xls_value_eq(value1, value2));
  EXPECT_TRUE(xls_value_eq(value2, value1));

  // Check that the bits object can be turned to string still.
  char* bits_str = xls_bits_to_debug_string(bits);
  absl::Cleanup free_bits_str([=] { xls_c_str_free(bits_str); });
  EXPECT_EQ(std::string_view{bits_str}, "0b00000000000000000000000000101010");
}

TEST(XlsCApiTest, FunctionJit) {
  const std::string_view kIr = R"(package my_package

top fn add_one(tok: token, x: bits[32]) -> bits[32] {
  one: bits[32] = literal(value=1)
  add: bits[32] = add(x, one)
  always_on: bits[1] = literal(value=1)
  trace: token = trace(tok, always_on, format="result: {}", data_operands=[add])
  ret result: bits[32] =identity(add)
}
)";
  char* error = nullptr;
  xls_package* package = nullptr;
  ASSERT_TRUE(xls_parse_ir_package(kIr.data(), "test.ir", &error, &package))
      << "error: " << error;
  ASSERT_NE(package, nullptr);
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_function* function = nullptr;
  ASSERT_TRUE(xls_package_get_function(package, "add_one", &error, &function));
  ASSERT_NE(function, nullptr);

  xls_function_jit* fn_jit = nullptr;
  ASSERT_TRUE(xls_make_function_jit(function, &error, &fn_jit));
  ASSERT_NE(fn_jit, nullptr);
  absl::Cleanup free_fn_jit([=] { xls_function_jit_free(fn_jit); });

  xls_bits* mol_bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(32, 42, &error, &mol_bits));
  xls_value* mol = xls_value_from_bits_owned(mol_bits);
  absl::Cleanup free_mol([=] { xls_value_free(mol); });

  xls_value* tok = xls_value_make_token();
  absl::Cleanup free_tok([=] { xls_value_free(tok); });

  std::vector<xls_value*> args = {tok, mol};
  xls_value* result = nullptr;
  xls_trace_message* trace_messages = nullptr;
  size_t trace_messages_count = 0;
  char** assert_messages = nullptr;
  size_t assert_messages_count = 0;
  ASSERT_TRUE(xls_function_jit_run(fn_jit, args.size(), args.data(), &error,
                                   &trace_messages, &trace_messages_count,
                                   &assert_messages, &assert_messages_count,
                                   &result));
  absl::Cleanup free_result([=] { xls_value_free(result); });
  absl::Cleanup free_trace_messages(
      [=] { xls_trace_messages_free(trace_messages, trace_messages_count); });
  absl::Cleanup free_assert_messages(
      [=] { xls_c_strs_free(assert_messages, assert_messages_count); });

  ASSERT_EQ(trace_messages_count, 1);
  ASSERT_EQ(assert_messages_count, 0);
  EXPECT_EQ(std::string_view{trace_messages[0].message}, "result: 43");
  EXPECT_EQ(trace_messages[0].verbosity, 0);

  char* result_str = nullptr;
  ASSERT_TRUE(xls_value_to_string(result, &result_str));
  absl::Cleanup free_result_str([=] { xls_c_str_free(result_str); });
  EXPECT_EQ(std::string_view{result_str}, "bits[32]:43");
}

// Tests that we can build a simple sample function. For fun we make one that
// corresponds to an AOI21 gate.
//
// AOI21 formula is `fn aoi21(a, b, c) { !((a & b) | c) }`
//
// Just to test we can also handle tuple types we replicate the bit to be a
// member of the result tuple 2x.
TEST(XlsCApiTest, FnBuilder) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  // Note: this is tied to the package lifetime.
  xls_type* u1 = xls_package_get_bits_type(package, 1);
  xls_type* tuple_members[] = {u1, u1, u1};
  xls_type* tuple_u1_u1_u1 =
      xls_package_get_tuple_type(package, tuple_members, 3);

  const char kFunctionName[] = "aoi21";
  xls_function_builder* fn_builder = xls_function_builder_create(
      kFunctionName, package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  // This value aliases the `fn_builder` so it does not need to be freed.
  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  xls_bvalue* t =
      xls_function_builder_add_parameter(fn_builder, "inputs", tuple_u1_u1_u1);
  bvalues_to_free.push_back(t);

  xls_bvalue* a = xls_builder_base_add_tuple_index(fn_builder_base, t, 0, "a");
  bvalues_to_free.push_back(a);

  xls_bvalue* b = xls_builder_base_add_tuple_index(fn_builder_base, t, 1, "b");
  bvalues_to_free.push_back(b);

  xls_bvalue* c = xls_builder_base_add_tuple_index(fn_builder_base, t, 2, "c");
  bvalues_to_free.push_back(c);

  // Show passing nullptr for the name.
  xls_bvalue* a_and_b =
      xls_builder_base_add_and(fn_builder_base, a, b, /*name=*/nullptr);
  bvalues_to_free.push_back(a_and_b);

  xls_bvalue* a_and_b_or_c =
      xls_builder_base_add_or(fn_builder_base, a_and_b, c, "a_and_b_or_c");
  bvalues_to_free.push_back(a_and_b_or_c);

  xls_bvalue* not_a_and_b_or_c = xls_builder_base_add_not(
      fn_builder_base, a_and_b_or_c, "not_a_and_b_or_c");
  bvalues_to_free.push_back(not_a_and_b_or_c);

  xls_bvalue* tuple_operands[] = {not_a_and_b_or_c, not_a_and_b_or_c};
  xls_bvalue* result =
      xls_builder_base_add_tuple(fn_builder_base, tuple_operands, 2, "result");
  bvalues_to_free.push_back(result);

  xls_function* function = nullptr;
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, result,
                                                             &error, &function))
        << "error: " << error;
    ASSERT_NE(function, nullptr);
    // Note: the built function is placed in the package's lifetime and so there
    // is no need to free it.
  }

  // Mark this function we built as the package top.
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_package_set_top_by_name(package, kFunctionName, &error));
    ASSERT_EQ(error, nullptr);
  }

  // Convert the package to string and make sure it's what we expect the
  // contents are.
  char* package_str = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &package_str));
  absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
  const std::string_view kWant = R"(package my_package

top fn aoi21(inputs: (bits[1], bits[1], bits[1]) id=1) -> (bits[1], bits[1]) {
  a: bits[1] = tuple_index(inputs, index=0, id=2)
  b: bits[1] = tuple_index(inputs, index=1, id=3)
  and.5: bits[1] = and(a, b, id=5)
  c: bits[1] = tuple_index(inputs, index=2, id=4)
  a_and_b_or_c: bits[1] = or(and.5, c, id=6)
  not_a_and_b_or_c: bits[1] = not(a_and_b_or_c, id=7)
  ret result: (bits[1], bits[1]) = tuple(not_a_and_b_or_c, not_a_and_b_or_c, id=8)
}
)";
  EXPECT_EQ(std::string_view{package_str}, kWant);
}

TEST(XlsCApiTest, FnBuilderConcatAndSlice) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_function_builder* fn_builder = xls_function_builder_create(
      "concat_and_slice", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  // Concat two 16 bit values and slice out the last 8 bits statically and some
  // other 8 bits dynamically.
  xls_type* u16 = xls_package_get_bits_type(package, 16);

  xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u16);
  absl::Cleanup free_x([=] { xls_bvalue_free(x); });
  xls_bvalue* y = xls_function_builder_add_parameter(fn_builder, "y", u16);
  absl::Cleanup free_y([=] { xls_bvalue_free(y); });

  xls_bvalue* concat_operands[] = {x, y};
  xls_bvalue* concat = xls_builder_base_add_concat(
      fn_builder_base, concat_operands, 2, "concat");
  absl::Cleanup free_concat([=] { xls_bvalue_free(concat); });

  xls_bvalue* last_8b = xls_builder_base_add_bit_slice(fn_builder_base, concat,
                                                       32 - 8, 8, "last_8b");
  absl::Cleanup free_last_8b([=] { xls_bvalue_free(last_8b); });

  xls_value* dynamic_start_value = nullptr;
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_value_make_ubits(32, 0, &error, &dynamic_start_value));
  }
  absl::Cleanup free_dynamic_start_value(
      [=] { xls_value_free(dynamic_start_value); });

  xls_bvalue* dynamic_start = xls_builder_base_add_literal(
      fn_builder_base, dynamic_start_value, "dynamic_start");
  absl::Cleanup free_dynamic_start([=] { xls_bvalue_free(dynamic_start); });

  xls_bvalue* dynamic_slice = xls_builder_base_add_dynamic_bit_slice(
      fn_builder_base, concat, dynamic_start, 8, "dynamic_slice");
  absl::Cleanup free_dynamic_slice([=] { xls_bvalue_free(dynamic_slice); });

  xls_bvalue* slices_members[] = {last_8b, dynamic_slice};
  xls_bvalue* slices =
      xls_builder_base_add_tuple(fn_builder_base, slices_members, 2, "slices");
  absl::Cleanup free_slices([=] { xls_bvalue_free(slices); });

  xls_function* function = nullptr;
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_function_builder_build_with_return_value(
        fn_builder, slices, &error, &function));
    ASSERT_NE(function, nullptr);
  }

  // Convert the package to string and make sure it's what we expect the
  // contents are.
  char* package_str = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &package_str));
  absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
  const std::string_view kWant = R"(package my_package

fn concat_and_slice(x: bits[16] id=1, y: bits[16] id=2) -> (bits[8], bits[8]) {
  concat: bits[32] = concat(x, y, id=3)
  dynamic_start: bits[32] = literal(value=0, id=5)
  last_8b: bits[8] = bit_slice(concat, start=24, width=8, id=4)
  dynamic_slice: bits[8] = dynamic_bit_slice(concat, dynamic_start, width=8, id=6)
  ret slices: (bits[8], bits[8]) = tuple(last_8b, dynamic_slice, id=7)
}
)";
  EXPECT_EQ(std::string_view{package_str}, kWant);
}

TEST(XlsCApiTest, FnBuilderBinops) {
  struct TestCase {
    std::string_view op_name;
    bool is_comparison;
    std::function<xls_bvalue*(xls_builder_base*, xls_bvalue*, xls_bvalue*,
                              const char*)>
        add_op;
  };
  const std::vector<TestCase> kBinops = {
      {"add", false, xls_builder_base_add_add},    // +
      {"umul", false, xls_builder_base_add_umul},  // *
      {"smul", false, xls_builder_base_add_smul},  // *
      {"sub", false, xls_builder_base_add_sub},    // -
      {"and", false, xls_builder_base_add_and},    // &
      {"nand", false, xls_builder_base_add_nand},  // !&
      {"or", false, xls_builder_base_add_or},      // |
      {"xor", false, xls_builder_base_add_xor},    // ^
      {"eq", true, xls_builder_base_add_eq},       // ==
      {"ne", true, xls_builder_base_add_ne},       // !=
      {"ult", true, xls_builder_base_add_ult},     // unsigned <
      {"ule", true, xls_builder_base_add_ule},     // unsigned <=
      {"ugt", true, xls_builder_base_add_ugt},     // unsigned >
      {"uge", true, xls_builder_base_add_uge},     // unsigned >=
      {"slt", true, xls_builder_base_add_slt},     // signed <
      {"sle", true, xls_builder_base_add_sle},     // signed <=
      {"sgt", true, xls_builder_base_add_sgt},     // signed >
      {"sge", true, xls_builder_base_add_sge},     // signed >=
      {"udiv", false, xls_builder_base_add_udiv},  // unsigned /
      {"sdiv", false, xls_builder_base_add_sdiv},  // signed /
      {"umod", false, xls_builder_base_add_umod},  // unsigned %
      {"smod", false, xls_builder_base_add_smod},  // signed %
  };

  for (const TestCase& test_case : kBinops) {
    xls_package* package = xls_package_create("my_package");
    absl::Cleanup free_package([=] { xls_package_free(package); });

    xls_type* u8 = xls_package_get_bits_type(package, 8);

    xls_function_builder* fn_builder =
        xls_function_builder_create("binop", package, /*should_verify=*/true);
    absl::Cleanup free_fn_builder(
        [=] { xls_function_builder_free(fn_builder); });

    xls_builder_base* fn_builder_base =
        xls_function_builder_as_builder_base(fn_builder);

    xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u8);
    absl::Cleanup free_x([=] { xls_bvalue_free(x); });
    xls_bvalue* y = xls_function_builder_add_parameter(fn_builder, "y", u8);
    absl::Cleanup free_y([=] { xls_bvalue_free(y); });

    xls_bvalue* result = test_case.add_op(fn_builder_base, x, y, "result");
    absl::Cleanup free_result([=] { xls_bvalue_free(result); });

    xls_function* function = nullptr;
    {
      char* error = nullptr;
      ASSERT_TRUE(xls_function_builder_build_with_return_value(
          fn_builder, result, &error, &function))
          << "error: " << error;
      ASSERT_NE(function, nullptr);
    }

    // Convert to string and extract the one node from the body.
    char* package_str = nullptr;
    ASSERT_TRUE(xls_package_to_string(package, &package_str));
    absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
    const std::string_view kWantTmpl =
        R"(package my_package

fn binop(x: bits[8] id=1, y: bits[8] id=2) -> bits[%d] {
  ret result: bits[%d] = %s(x, y, id=3)
}
)";

    int64_t result_bits = test_case.is_comparison ? 1 : 8;
    EXPECT_THAT(std::string_view{package_str},
                HasSubstr(absl::StrFormat(kWantTmpl, result_bits, result_bits,
                                          test_case.op_name)));
  }
}

TEST(XlsCApiTest, FnBuilderUnaryOps) {
  struct TestCase {
    // Operation name to expect in the IR text.
    std::string_view op_name;
    // The number of output result bits we expect for the op.
    int64_t result_bits;
    // Builder operation to add the unary operation.
    std::function<xls_bvalue*(xls_builder_base*, xls_bvalue*, const char*)>
        add_op;
    // Any extra attributes we expect in the unary operation output IR text.
    std::string extra_attributes;
  };
  const std::vector<TestCase> kUnaryOps = {
      TestCase{"not", 8, xls_builder_base_add_not},
      TestCase{"neg", 8, xls_builder_base_add_negate},
      TestCase{"reverse", 8, xls_builder_base_add_reverse},
      TestCase{"and_reduce", 1, xls_builder_base_add_and_reduce},
      TestCase{"or_reduce", 1, xls_builder_base_add_or_reduce},
      TestCase{"xor_reduce", 1, xls_builder_base_add_xor_reduce},
      TestCase{"one_hot", 9,
               [](xls_builder_base* builder, xls_bvalue* x, const char* name) {
                 return xls_builder_base_add_one_hot(
                     builder, x, /*lsb_is_priority=*/true, name);
               },
               ", lsb_prio=true"},
      TestCase{"one_hot", 9,
               [](xls_builder_base* builder, xls_bvalue* x, const char* name) {
                 return xls_builder_base_add_one_hot(
                     builder, x, /*lsb_is_priority=*/false, name);
               },
               ", lsb_prio=false"},
      TestCase{"sign_ext", 16,
               [](xls_builder_base* builder, xls_bvalue* x, const char* name) {
                 return xls_builder_base_add_sign_extend(builder, x, 16, name);
               },
               ", new_bit_count=16"},
      TestCase{"zero_ext", 16,
               [](xls_builder_base* builder, xls_bvalue* x, const char* name) {
                 return xls_builder_base_add_zero_extend(builder, x, 16, name);
               },
               ", new_bit_count=16"},
  };

  for (const TestCase& test_case : kUnaryOps) {
    xls_package* package = xls_package_create("my_package");
    absl::Cleanup free_package([=] { xls_package_free(package); });

    xls_type* u8 = xls_package_get_bits_type(package, 8);

    xls_function_builder* fn_builder =
        xls_function_builder_create("unaryop", package, /*should_verify=*/true);
    absl::Cleanup free_fn_builder(
        [=] { xls_function_builder_free(fn_builder); });

    xls_builder_base* fn_builder_base =
        xls_function_builder_as_builder_base(fn_builder);

    xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u8);
    absl::Cleanup free_x([=] { xls_bvalue_free(x); });

    xls_bvalue* result = test_case.add_op(fn_builder_base, x, "result");
    absl::Cleanup free_result([=] { xls_bvalue_free(result); });

    xls_function* function = nullptr;
    {
      char* error = nullptr;
      ASSERT_TRUE(xls_function_builder_build_with_return_value(
          fn_builder, result, &error, &function))
          << "error: " << error;
      ASSERT_NE(function, nullptr);
    }

    // Convert to string and extract the one node from the body.
    char* package_str = nullptr;
    ASSERT_TRUE(xls_package_to_string(package, &package_str));
    absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
    const std::string_view kWantTmpl =
        R"(package my_package

fn unaryop(x: bits[8] id=1) -> bits[%d] {
  ret result: bits[%d] = %s(x%s, id=2)
}
)";
    EXPECT_THAT(std::string_view{package_str},
                HasSubstr(absl::StrFormat(
                    kWantTmpl, test_case.result_bits, test_case.result_bits,
                    test_case.op_name, test_case.extra_attributes)));
  }
}

TEST(XlsCApiTest, FnBuilderArrayOps) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_type* u8 = xls_package_get_bits_type(package, 8);
  xls_type* u32 = xls_package_get_bits_type(package, 32);
  xls_type* u8_arr3 = xls_package_get_array_type(package, u8, 3);

  xls_function_builder* fn_builder =
      xls_function_builder_create("array_ops", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u8_arr3);
  bvalues_to_free.push_back(x);
  xls_bvalue* y = xls_function_builder_add_parameter(fn_builder, "y", u8_arr3);
  bvalues_to_free.push_back(y);
  xls_bvalue* idx = xls_function_builder_add_parameter(fn_builder, "idx", u32);
  bvalues_to_free.push_back(idx);
  xls_bvalue* update_val =
      xls_function_builder_add_parameter(fn_builder, "update_val", u8);
  bvalues_to_free.push_back(update_val);

  // Array literal: lit = u8[3]:[1, 2, 3]
  xls_bvalue* lit = nullptr;
  {
    char* error = nullptr;
    xls_value* v1 = nullptr;
    ASSERT_TRUE(xls_value_make_ubits(8, 1, &error, &v1));
    absl::Cleanup free_v1([=] { xls_value_free(v1); });
    xls_value* v2 = nullptr;
    ASSERT_TRUE(xls_value_make_ubits(8, 2, &error, &v2));
    absl::Cleanup free_v2([=] { xls_value_free(v2); });
    xls_value* v3 = nullptr;
    ASSERT_TRUE(xls_value_make_ubits(8, 3, &error, &v3));
    absl::Cleanup free_v3([=] { xls_value_free(v3); });

    xls_value* elements[] = {v1, v2, v3};
    xls_value* arr_val = nullptr;
    ASSERT_TRUE(
        xls_value_make_array(/*element_count=*/3, elements, &error, &arr_val));
    absl::Cleanup free_arr_val([=] { xls_value_free(arr_val); });

    lit = xls_builder_base_add_literal(fn_builder_base, arr_val, "lit");
    bvalues_to_free.push_back(lit);
  }

  // Index into x: x_idx = x[idx]
  xls_bvalue* indices_idx[] = {idx};
  xls_bvalue* x_idx = xls_builder_base_add_array_index(
      fn_builder_base, x, indices_idx, 1, /*assumed_in_bounds=*/true, "x_idx");
  bvalues_to_free.push_back(x_idx);

  // Slice x: x_slice = x[idx: width 2]
  xls_bvalue* x_slice =
      xls_builder_base_add_array_slice(fn_builder_base, x, idx, 2, "x_slice");
  bvalues_to_free.push_back(x_slice);

  // Update x: x_upd = x[idx: update_val]
  xls_bvalue* indices_upd[] = {idx};
  xls_bvalue* x_upd = xls_builder_base_add_array_update(
      fn_builder_base, x, update_val, indices_upd, 1,
      /*assumed_in_bounds=*/true, "x_upd");
  bvalues_to_free.push_back(x_upd);

  // Concat x and y: concat = x ++ y
  xls_bvalue* concat_ops[] = {x, y};
  xls_bvalue* concat_arr = xls_builder_base_add_array_concat(
      fn_builder_base, concat_ops, 2, "concat_arr");
  bvalues_to_free.push_back(concat_arr);

  xls_bvalue* return_elts[] = {lit, x_idx, x_slice, x_upd, concat_arr};
  xls_bvalue* result =
      xls_builder_base_add_tuple(fn_builder_base, return_elts, 5, "result");
  bvalues_to_free.push_back(result);

  xls_function* function = nullptr;
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, result,
                                                             &error, &function))
        << "error: " << error;
    ASSERT_NE(function, nullptr);
  }

  char* package_str = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &package_str));
  absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
  const std::string_view kWant = R"(package my_package

fn array_ops(x: bits[8][3] id=1, y: bits[8][3] id=2, idx: bits[32] id=3, update_val: bits[8] id=4) -> (bits[8][3], bits[8], bits[8][2], bits[8][3], bits[8][6]) {
  lit: bits[8][3] = literal(value=[1, 2, 3], id=5)
  x_idx: bits[8] = array_index(x, indices=[idx], assumed_in_bounds=true, id=6)
  x_slice: bits[8][2] = array_slice(x, idx, width=2, id=7)
  x_upd: bits[8][3] = array_update(x, update_val, indices=[idx], assumed_in_bounds=true, id=8)
  concat_arr: bits[8][6] = array_concat(x, y, id=9)
  ret result: (bits[8][3], bits[8], bits[8][2], bits[8][3], bits[8][6]) = tuple(lit, x_idx, x_slice, x_upd, concat_arr, id=10)
}
)";
  EXPECT_EQ(std::string_view{package_str}, kWant);
}

TEST(XlsCApiTest, FnBuilderShiftOps) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_type* u8 = xls_package_get_bits_type(package, 8);
  xls_type* u3 = xls_package_get_bits_type(package, 3);

  xls_function_builder* fn_builder =
      xls_function_builder_create("shift_ops", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u8);
  bvalues_to_free.push_back(x);
  xls_bvalue* amt = xls_function_builder_add_parameter(fn_builder, "amt", u3);
  bvalues_to_free.push_back(amt);

  xls_bvalue* shra_op =
      xls_builder_base_add_shra(fn_builder_base, x, amt, "shra_op");
  bvalues_to_free.push_back(shra_op);
  xls_bvalue* shrl_op =
      xls_builder_base_add_shrl(fn_builder_base, x, amt, "shrl_op");
  bvalues_to_free.push_back(shrl_op);
  xls_bvalue* shll_op =
      xls_builder_base_add_shll(fn_builder_base, x, amt, "shll_op");
  bvalues_to_free.push_back(shll_op);

  xls_bvalue* return_elts[] = {shra_op, shrl_op, shll_op};
  xls_bvalue* result =
      xls_builder_base_add_tuple(fn_builder_base, return_elts, 3, "result");
  bvalues_to_free.push_back(result);

  xls_function* function = nullptr;
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, result,
                                                             &error, &function))
        << "error: " << error;
    ASSERT_NE(function, nullptr);
  }

  char* package_str = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &package_str));
  absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
  const std::string_view kWant = R"(package my_package

fn shift_ops(x: bits[8] id=1, amt: bits[3] id=2) -> (bits[8], bits[8], bits[8]) {
  shra_op: bits[8] = shra(x, amt, id=3)
  shrl_op: bits[8] = shrl(x, amt, id=4)
  shll_op: bits[8] = shll(x, amt, id=5)
  ret result: (bits[8], bits[8], bits[8]) = tuple(shra_op, shrl_op, shll_op, id=6)
}
)";
  EXPECT_EQ(std::string_view{package_str}, kWant);
}

TEST(XlsCApiTest, FnBuilderBitwiseUpdateAndNor) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_type* u16 = xls_package_get_bits_type(package, 16);
  xls_type* u8 = xls_package_get_bits_type(package, 8);
  xls_type* u4 = xls_package_get_bits_type(package, 4);

  xls_function_builder* fn_builder = xls_function_builder_create(
      "bitwise_update_nor", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u16);
  bvalues_to_free.push_back(x);
  xls_bvalue* start =
      xls_function_builder_add_parameter(fn_builder, "start", u4);
  bvalues_to_free.push_back(start);
  xls_bvalue* update =
      xls_function_builder_add_parameter(fn_builder, "update", u8);
  bvalues_to_free.push_back(update);
  xls_bvalue* y = xls_function_builder_add_parameter(fn_builder, "y", u16);
  bvalues_to_free.push_back(y);

  xls_bvalue* bsu = xls_builder_base_add_bit_slice_update(fn_builder_base, x,
                                                          start, update, "bsu");
  bvalues_to_free.push_back(bsu);

  xls_bvalue* nor_op =
      xls_builder_base_add_nor(fn_builder_base, x, y, "nor_op");
  bvalues_to_free.push_back(nor_op);

  xls_bvalue* return_elts[] = {bsu, nor_op};
  xls_bvalue* result =
      xls_builder_base_add_tuple(fn_builder_base, return_elts, 2, "result");
  bvalues_to_free.push_back(result);

  xls_function* function = nullptr;
  char* error = nullptr;
  ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, result,
                                                           &error, &function))
      << "error: " << error;
  ASSERT_NE(function, nullptr);

  // Prepare inputs
  // x = 0b1111_0000_1111_0000 = 0xF0F0
  // start = 4
  // update = 0b_1010_1010 = 0xAA
  // y = 0b0000_1111_0000_1111 = 0x0F0F
  xls_value* x_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 0xF0F0, &error, &x_v));
  absl::Cleanup c_x([=] { xls_value_free(x_v); });
  xls_value* start_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(4, 4, &error, &start_v));
  absl::Cleanup c_start([=] { xls_value_free(start_v); });
  xls_value* update_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 0xAA, &error, &update_v));
  absl::Cleanup c_update([=] { xls_value_free(update_v); });
  xls_value* y_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 0x0F0F, &error, &y_v));
  absl::Cleanup c_y([=] { xls_value_free(y_v); });

  std::vector<xls_value*> args = {x_v, start_v, update_v, y_v};

  // Run Interpreter
  xls_value* actual_result = nullptr;
  ASSERT_TRUE(xls_interpret_function(function, args.size(), args.data(), &error,
                                     &actual_result))
      << "error: " << error;
  absl::Cleanup free_actual_result([=] { xls_value_free(actual_result); });

  // Prepare expected output
  // bsu = x with bits[11:4] replaced by update
  // x     = 1111_0000_1111_0000
  // update=       1010_1010
  // result= 1111_1010_1010_0000 = 0xFAAF
  xls_value* exp_bsu = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 0xFAA0, &error, &exp_bsu));
  absl::Cleanup c_exp_bsu([=] { xls_value_free(exp_bsu); });
  // nor = !(x | y)
  // x | y = 0xF0F0 | 0x0F0F = 0xFFF
  // !(0xFFF) = 0x0000
  xls_value* exp_nor = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 0x0000, &error, &exp_nor));
  absl::Cleanup c_exp_nor([=] { xls_value_free(exp_nor); });

  xls_value* expected_elts[] = {exp_bsu, exp_nor};
  xls_value* expected_result = xls_value_make_tuple(2, expected_elts);
  absl::Cleanup free_expected_result([=] { xls_value_free(expected_result); });

  // Compare results
  char* actual_str = nullptr;
  char* expected_str = nullptr;
  ASSERT_TRUE(xls_value_to_string(actual_result, &actual_str));
  absl::Cleanup free_actual_str([=] { xls_c_str_free(actual_str); });
  ASSERT_TRUE(xls_value_to_string(expected_result, &expected_str));
  absl::Cleanup free_expected_str([=] { xls_c_str_free(expected_str); });
  EXPECT_TRUE(xls_value_eq(actual_result, expected_result))
      << "Actual: " << actual_str << "\\nExpected: " << expected_str;
}

TEST(XlsCApiTest, FnBuilderMiscOps) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_type* u2 = xls_package_get_bits_type(package, 2);
  xls_type* u4 = xls_package_get_bits_type(package, 4);
  xls_type* u8 = xls_package_get_bits_type(package, 8);
  xls_type* u16 = xls_package_get_bits_type(package, 16);

  xls_function_builder* fn_builder =
      xls_function_builder_create("misc_ops", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  // Parameters
  xls_bvalue* sel = xls_function_builder_add_parameter(fn_builder, "sel", u2);
  bvalues_to_free.push_back(sel);
  xls_bvalue* c0 = xls_function_builder_add_parameter(fn_builder, "c0", u8);
  bvalues_to_free.push_back(c0);
  xls_bvalue* c1 = xls_function_builder_add_parameter(fn_builder, "c1", u8);
  bvalues_to_free.push_back(c1);
  xls_bvalue* c2 = xls_function_builder_add_parameter(fn_builder, "c2", u8);
  bvalues_to_free.push_back(c2);
  xls_bvalue* c3 = xls_function_builder_add_parameter(fn_builder, "c3", u8);
  bvalues_to_free.push_back(c3);
  xls_bvalue* z_arg =
      xls_function_builder_add_parameter(fn_builder, "z_arg", u16);
  bvalues_to_free.push_back(z_arg);
  xls_bvalue* enc_arg =
      xls_function_builder_add_parameter(fn_builder, "enc_arg", u4);
  bvalues_to_free.push_back(enc_arg);
  xls_bvalue* dec_arg =
      xls_function_builder_add_parameter(fn_builder, "dec_arg", u4);
  bvalues_to_free.push_back(dec_arg);
  xls_bvalue* id_arg =
      xls_function_builder_add_parameter(fn_builder, "id_arg", u8);
  bvalues_to_free.push_back(id_arg);

  // Operations
  xls_bvalue* cases[] = {c0, c1, c2, c3};
  xls_bvalue* select_op = xls_builder_base_add_select(
      fn_builder_base, sel, cases, 4, /*default_value=*/nullptr, "select_op");
  bvalues_to_free.push_back(select_op);

  xls_bvalue* clz_op =
      xls_builder_base_add_clz(fn_builder_base, z_arg, "clz_op");
  bvalues_to_free.push_back(clz_op);
  xls_bvalue* ctz_op =
      xls_builder_base_add_ctz(fn_builder_base, z_arg, "ctz_op");
  bvalues_to_free.push_back(ctz_op);

  xls_bvalue* encode_op =
      xls_builder_base_add_encode(fn_builder_base, enc_arg, "encode_op");
  bvalues_to_free.push_back(encode_op);

  xls_bvalue* decode_op = xls_builder_base_add_decode(fn_builder_base, dec_arg,
                                                      nullptr, "decode_op");
  bvalues_to_free.push_back(decode_op);
  int64_t decode_width = 8;
  xls_bvalue* decode_op_wide = xls_builder_base_add_decode(
      fn_builder_base, dec_arg, &decode_width, "decode_op_wide");
  bvalues_to_free.push_back(decode_op_wide);

  xls_bvalue* id_op =
      xls_builder_base_add_identity(fn_builder_base, id_arg, "id_op");
  bvalues_to_free.push_back(id_op);

  // Result tuple
  xls_bvalue* return_elts[] = {select_op, clz_op,         ctz_op, encode_op,
                               decode_op, decode_op_wide, id_op};
  xls_bvalue* result =
      xls_builder_base_add_tuple(fn_builder_base, return_elts, 7, "result");
  bvalues_to_free.push_back(result);

  // Build function
  xls_function* function = nullptr;
  char* error = nullptr;
  ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, result,
                                                           &error, &function))
      << "error: " << error;
  ASSERT_NE(function, nullptr);

  // Prepare inputs
  xls_value* sel_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(2, 1, &error, &sel_v));
  absl::Cleanup c_sel([=] { xls_value_free(sel_v); });
  xls_value* c0_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 10, &error, &c0_v));
  absl::Cleanup c_c0([=] { xls_value_free(c0_v); });
  xls_value* c1_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 20, &error, &c1_v));
  absl::Cleanup c_c1([=] { xls_value_free(c1_v); });
  xls_value* c2_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 30, &error, &c2_v));
  absl::Cleanup c_c2([=] { xls_value_free(c2_v); });
  xls_value* c3_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 40, &error, &c3_v));
  absl::Cleanup c_c3([=] { xls_value_free(c3_v); });
  xls_value* z_arg_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 0b0000101100001000, &error, &z_arg_v));
  absl::Cleanup c_z_arg([=] { xls_value_free(z_arg_v); });
  xls_value* enc_arg_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(4, 0b1000, &error, &enc_arg_v));
  absl::Cleanup c_enc_arg([=] { xls_value_free(enc_arg_v); });
  xls_value* dec_arg_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(4, 0b0100, &error, &dec_arg_v));
  absl::Cleanup c_dec_arg([=] { xls_value_free(dec_arg_v); });
  xls_value* id_arg_v = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 55, &error, &id_arg_v));
  absl::Cleanup c_id_arg([=] { xls_value_free(id_arg_v); });

  std::vector<xls_value*> args = {sel_v,   c0_v,      c1_v,      c2_v,    c3_v,
                                  z_arg_v, enc_arg_v, dec_arg_v, id_arg_v};

  // Run Interpreter
  xls_value* actual_result = nullptr;
  ASSERT_TRUE(xls_interpret_function(function, args.size(), args.data(), &error,
                                     &actual_result))
      << "error: " << error;
  absl::Cleanup free_actual_result([=] { xls_value_free(actual_result); });

  // Prepare expected output
  xls_value* exp_sel = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 20, &error, &exp_sel));
  absl::Cleanup c_exp_sel([=] { xls_value_free(exp_sel); });  // sel=1 -> c1_v
  xls_value* exp_clz = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 4, &error, &exp_clz));
  absl::Cleanup c_exp_clz([=] {
    xls_value_free(exp_clz);
  });  // clz(0b00001...) = 4. Actual type seems to be input width.
  xls_value* exp_ctz = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 3, &error, &exp_ctz));
  absl::Cleanup c_exp_ctz([=] {
    xls_value_free(exp_ctz);
  });  // ctz(...01000) = 3. Actual type seems to be input width.
  xls_value* exp_enc = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(2, 3, &error, &exp_enc));
  absl::Cleanup c_exp_enc(
      [=] { xls_value_free(exp_enc); });  // encode(0b1000) = 3. ceil(log2(4))=2
  xls_value* exp_dec = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(16, 16, &error, &exp_dec));
  absl::Cleanup c_exp_dec([=] {
    xls_value_free(exp_dec);
  });  // decode(bits[4]:4) = 16 (0b10000). default width 2^4=16
  xls_value* exp_dec_w = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 16, &error, &exp_dec_w));
  absl::Cleanup c_exp_dec_w([=] {
    xls_value_free(exp_dec_w);
  });  // decode(bits[4]:4, width=8) = 16 (0b10000)
  xls_value* exp_id = nullptr;
  ASSERT_TRUE(xls_value_make_ubits(8, 55, &error, &exp_id));
  absl::Cleanup c_exp_id([=] { xls_value_free(exp_id); });  // identity(55) = 55

  xls_value* expected_elts[] = {exp_sel, exp_clz,   exp_ctz, exp_enc,
                                exp_dec, exp_dec_w, exp_id};
  xls_value* expected_result = xls_value_make_tuple(7, expected_elts);
  absl::Cleanup free_expected_result([=] { xls_value_free(expected_result); });

  // Compare results
  char* actual_str = nullptr;
  char* expected_str = nullptr;
  ASSERT_TRUE(xls_value_to_string(actual_result, &actual_str));
  absl::Cleanup free_actual_str(
      [=] { xls_c_str_free(actual_str); });  // Ensure free
  ASSERT_TRUE(xls_value_to_string(expected_result, &expected_str));
  absl::Cleanup free_expected_str(
      [=] { xls_c_str_free(expected_str); });  // Ensure free
  EXPECT_TRUE(xls_value_eq(actual_result, expected_result))
      << "Actual: " << actual_str          // Use directly now
      << "\\nExpected: " << expected_str;  // Use directly now
}

TEST(XlsCApiTest, FnBuilderTokenOps) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_type* token_type = xls_package_get_token_type(package);

  xls_function_builder* fn_builder =
      xls_function_builder_create("token_ops", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  xls_bvalue* tok1 =
      xls_function_builder_add_parameter(fn_builder, "tok1", token_type);
  bvalues_to_free.push_back(tok1);
  xls_bvalue* tok2 =
      xls_function_builder_add_parameter(fn_builder, "tok2", token_type);
  bvalues_to_free.push_back(tok2);

  xls_bvalue* deps[] = {tok1, tok2};
  xls_bvalue* after_all_op =
      xls_builder_base_add_after_all(fn_builder_base, deps, 2, "after_all_op");
  bvalues_to_free.push_back(after_all_op);

  xls_function* function = nullptr;
  {
    char* error = nullptr;
    ASSERT_TRUE(xls_function_builder_build_with_return_value(
        fn_builder, after_all_op, &error, &function))
        << "error: " << error;
    ASSERT_NE(function, nullptr);
  }

  char* package_str = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &package_str));
  absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
  const std::string_view kWant = R"(package my_package

fn token_ops(tok1: token id=1, tok2: token id=2) -> token {
  ret after_all_op: token = after_all(tok1, tok2, id=3)
}
)";
  EXPECT_EQ(std::string_view{package_str}, kWant);
}

TEST(XlsCApiTest, FnBuilderGetTypeAndLastValue) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  xls_type* u32 = xls_package_get_bits_type(package, 32);

  xls_function_builder* fn_builder = xls_function_builder_create(
      "get_type_last_val", package, /*should_verify=*/true);
  absl::Cleanup free_fn_builder([=] { xls_function_builder_free(fn_builder); });

  xls_builder_base* fn_builder_base =
      xls_function_builder_as_builder_base(fn_builder);

  std::vector<xls_bvalue*> bvalues_to_free;
  absl::Cleanup free_bvalues([&] {
    for (xls_bvalue* b : bvalues_to_free) {
      xls_bvalue_free(b);
    }
  });

  xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u32);
  bvalues_to_free.push_back(x);

  xls_bvalue* y = xls_builder_base_add_add(fn_builder_base, x, x, "y");
  bvalues_to_free.push_back(y);  // Add y to cleanup list

  // Test GetLastValue
  xls_bvalue* last_val = nullptr;
  char* error = nullptr;
  ASSERT_TRUE(
      xls_builder_base_get_last_value(fn_builder_base, &error, &last_val))
      << "error: " << error;
  absl::Cleanup free_last_val([=] { xls_bvalue_free(last_val); });
  ASSERT_NE(last_val, nullptr);
  // Note: Cannot directly compare BValue pointers reliably. The IR string check
  // implicitly verifies GetLastValue returned the correct BValue `y`.

  // Test GetType
  xls_type* last_val_type =
      xls_builder_base_get_type(fn_builder_base, last_val);
  ASSERT_NE(last_val_type, nullptr);
  EXPECT_EQ(last_val_type, u32);  // Check if the type pointer matches u32

  // Test GetType on an earlier value
  xls_type* x_type = xls_builder_base_get_type(fn_builder_base, x);
  ASSERT_NE(x_type, nullptr);
  EXPECT_EQ(x_type, u32);  // Check if the type pointer matches u32

  xls_function* function = nullptr;
  ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, last_val,
                                                           &error, &function))
      << "error: " << error;
  ASSERT_NE(function, nullptr);

  char* package_str = nullptr;
  ASSERT_TRUE(xls_package_to_string(package, &package_str));
  absl::Cleanup free_package_str([=] { xls_c_str_free(package_str); });
  const std::string_view kWant = R"(package my_package

fn get_type_last_val(x: bits[32] id=1) -> bits[32] {
  ret y: bits[32] = add(x, x, id=2)
}
)";
  EXPECT_EQ(std::string_view{package_str}, kWant);

  char* param_name = nullptr;
  ASSERT_TRUE(
      xls_function_get_param_name(function, /*index=*/0, &error, &param_name));
  absl::Cleanup free_param_name([param_name] { xls_c_str_free(param_name); });
  EXPECT_EQ(std::string_view(param_name), "x");

  // Out-of-bounds param name.
  char* bogus_param_name = nullptr;
  ASSERT_FALSE(xls_function_get_param_name(function, /*index=*/1, &error,
                                           &bogus_param_name));
  EXPECT_THAT(std::string_view{error}, HasSubstr("out of range"));
  xls_c_str_free(error);
  error = nullptr;
}

TEST(XlsCApiTest, TypeGetFlatBitCount) {
  xls_package* package = xls_package_create("my_package");
  absl::Cleanup free_package([=] { xls_package_free(package); });

  // Simple bits type
  xls_type* u32_type = xls_package_get_bits_type(package, 32);
  EXPECT_EQ(xls_type_get_flat_bit_count(u32_type), 32);

  // Token type
  xls_type* token_type = xls_package_get_token_type(package);
  EXPECT_EQ(xls_type_get_flat_bit_count(token_type), 0);

  // Tuple type
  xls_type* u8_type = xls_package_get_bits_type(package, 8);
  xls_type* u16_type = xls_package_get_bits_type(package, 16);
  xls_type* tuple_members[] = {u8_type, u16_type};
  xls_type* tuple_type = xls_package_get_tuple_type(package, tuple_members, 2);
  EXPECT_EQ(xls_type_get_flat_bit_count(tuple_type), 24);  // 8 + 16

  // Array type
  xls_type* u4_type = xls_package_get_bits_type(package, 4);
  xls_type* array_type = xls_package_get_array_type(package, u4_type, 3);
  EXPECT_EQ(xls_type_get_flat_bit_count(array_type), 12);  // 4 * 3

  // Nested tuple and array
  // ((bits[2], bits[3]), bits[1][5])
  xls_type* u2_type = xls_package_get_bits_type(package, 2);
  xls_type* u3_type = xls_package_get_bits_type(package, 3);
  xls_type* inner_tuple_members[] = {u2_type, u3_type};
  xls_type* inner_tuple_type =
      xls_package_get_tuple_type(package, inner_tuple_members, 2);  // 2 + 3 = 5

  xls_type* u1_type = xls_package_get_bits_type(package, 1);
  xls_type* inner_array_type =
      xls_package_get_array_type(package, u1_type, 5);  // 1 * 5 = 5

  xls_type* outer_tuple_members[] = {inner_tuple_type, inner_array_type};
  xls_type* nested_type =
      xls_package_get_tuple_type(package, outer_tuple_members, 2);
  EXPECT_EQ(xls_type_get_flat_bit_count(nested_type), 10);  // 5 + 5
}

TEST(XlsCApiTest, BitsRopeCreateFree) {
  xls_bits_rope* rope = xls_create_bits_rope(10);
  ASSERT_NE(rope, nullptr);
  xls_bits_rope_free(rope);
}

TEST(XlsCApiTest, BitsRopePushBack) {
  char* error_out = nullptr;
  xls_bits_rope* rope = xls_create_bits_rope(10);
  ASSERT_NE(rope, nullptr);
  absl::Cleanup free_rope([rope] { xls_bits_rope_free(rope); });

  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(3, 0b101, &error_out, &bits));
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });

  xls_bits_rope_append_bits(rope, bits);
  ASSERT_EQ(error_out, nullptr);
}

TEST(XlsCApiTest, BitsRopeBuild) {
  char* error_out = nullptr;
  xls_bits_rope* rope = xls_create_bits_rope(3);
  ASSERT_NE(rope, nullptr);
  absl::Cleanup free_rope([rope] { xls_bits_rope_free(rope); });

  xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(3, 0b101, &error_out, &bits));
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });

  xls_bits_rope_append_bits(rope, bits);
  ASSERT_EQ(error_out, nullptr);

  xls_bits* result = xls_bits_rope_get_bits(rope);
  ASSERT_NE(result, nullptr);
  absl::Cleanup free_result([result] { xls_bits_free(result); });

  char* result_str = xls_bits_to_debug_string(result);
  absl::Cleanup free_result_str([result_str] { xls_c_str_free(result_str); });
  EXPECT_EQ(std::string(result_str), "0b101");
}

TEST(XlsCApiTest, BitsRopePushMultipleAndBuild) {
  char* error_out = nullptr;
  xls_bits_rope* rope = xls_create_bits_rope(5);
  ASSERT_NE(rope, nullptr);
  absl::Cleanup free_rope([rope] { xls_bits_rope_free(rope); });

  xls_bits* bits1 = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(3, 0b101, &error_out, &bits1));
  absl::Cleanup free_bits1([bits1] { xls_bits_free(bits1); });

  xls_bits* bits2 = nullptr;
  ASSERT_TRUE(xls_bits_make_ubits(2, 0b10, &error_out, &bits2));
  absl::Cleanup free_bits2([bits2] { xls_bits_free(bits2); });

  xls_bits_rope_append_bits(rope, bits1);
  ASSERT_EQ(error_out, nullptr);
  xls_bits_rope_append_bits(rope, bits2);
  ASSERT_EQ(error_out, nullptr);

  xls_bits* result = xls_bits_rope_get_bits(rope);
  ASSERT_NE(result, nullptr);
  absl::Cleanup free_result([result] { xls_bits_free(result); });

  char* result_str = xls_bits_to_debug_string(result);
  absl::Cleanup free_result_str([result_str] { xls_c_str_free(result_str); });
  EXPECT_EQ(std::string(result_str), "0b10101");
}

TEST(XlsCApiTest, FnBuilderPartialProductOps) {
  struct TestCase {
    std::string_view op_name;
    std::function<xls_bvalue*(xls_builder_base*, xls_bvalue*, xls_bvalue*,
                              const char*)>
        add_op;
  };
  const std::vector<TestCase> kCases = {
      {"umulp", xls_builder_base_add_umulp},
      {"smulp", xls_builder_base_add_smulp},
  };

  for (const auto& tc : kCases) {
    xls_package* package = xls_package_create("my_package");
    absl::Cleanup free_package([=] { xls_package_free(package); });

    xls_type* u8 = xls_package_get_bits_type(package, 8);
    xls_function_builder* fn_builder =
        xls_function_builder_create("pp", package, /*should_verify=*/true);
    absl::Cleanup free_fn_builder(
        [=] { xls_function_builder_free(fn_builder); });
    xls_builder_base* base = xls_function_builder_as_builder_base(fn_builder);

    xls_bvalue* x = xls_function_builder_add_parameter(fn_builder, "x", u8);
    absl::Cleanup free_x([=] { xls_bvalue_free(x); });
    xls_bvalue* y = xls_function_builder_add_parameter(fn_builder, "y", u8);
    absl::Cleanup free_y([=] { xls_bvalue_free(y); });

    xls_bvalue* result = tc.add_op(base, x, y, "result");
    absl::Cleanup free_result([=] { xls_bvalue_free(result); });

    xls_function* function = nullptr;
    char* error = nullptr;
    ASSERT_TRUE(xls_function_builder_build_with_return_value(fn_builder, result,
                                                             &error, &function))
        << error;

    // Check that the IR contains the operation.
    char* pkg_str = nullptr;
    ASSERT_TRUE(xls_package_to_string(package, &pkg_str));
    absl::Cleanup free_pkg_str([=] { xls_c_str_free(pkg_str); });
    std::string_view text(pkg_str);
    EXPECT_THAT(text, HasSubstr(absl::StrFormat(" = %s(", tc.op_name)));
  }
}

TEST(XlsCApiTest, FunctionToZ3Smtlib) {
  const std::string kPackage = R"(package p

top fn add(x: bits[32], y: bits[32]) -> bits[32] {
  ret result: bits[32] = add(x, y)
}
)";

  char* error = nullptr;
  xls_package* package = nullptr;
  ASSERT_TRUE(xls_parse_ir_package(kPackage.c_str(), "p.ir", &error, &package))
      << "xls_parse_ir_package error: " << error;
  absl::Cleanup free_package([package] { xls_package_free(package); });

  xls_function* function = nullptr;
  ASSERT_TRUE(xls_package_get_function(package, "add", &error, &function));

  char* smtlib = nullptr;
  ASSERT_TRUE(xls_function_to_z3_smtlib(function, &error, &smtlib))
      << "xls_function_to_z3_smtlib error: " << error;
  absl::Cleanup free_smtlib([smtlib] { xls_c_str_free(smtlib); });

  EXPECT_EQ(std::string_view{smtlib},
            "(lambda ((x (_ BitVec 32)) (y (_ BitVec 32))) (bvadd x y))");
}

// Tests that QuickCheck module members can be accessed and their properties
// inspected via the C API.
TEST(XlsCApiTest, DslxQuickCheckIntrospection) {
  // A simple property test with an explicit test_count.
  const std::string_view kProgram = R"(
#[quickcheck(test_count=123)]
fn prop(x: u8) -> bool {
  x == x
}
)";

  const char* additional_search_paths[] = {};
  xls_dslx_import_data* import_data = xls_dslx_import_data_create(
      std::string{xls::kDefaultDslxStdlibPath}.c_str(), additional_search_paths,
      0);
  ASSERT_NE(import_data, nullptr);
  absl::Cleanup free_import_data(
      [&] { xls_dslx_import_data_free(import_data); });

  char* error = nullptr;
  xls_dslx_typechecked_module* tm = nullptr;
  bool ok = xls_dslx_parse_and_typecheck(kProgram.data(), "<test>", "top",
                                         import_data, &error, &tm);
  ASSERT_TRUE(ok) << "error: " << error;
  absl::Cleanup free_tm([&] { xls_dslx_typechecked_module_free(tm); });

  xls_dslx_module* module = xls_dslx_typechecked_module_get_module(tm);
  ASSERT_NE(module, nullptr);

  int64_t member_count = xls_dslx_module_get_member_count(module);
  ASSERT_EQ(member_count, 1);  // Only the QuickCheck member is present.

  xls_dslx_module_member* member = xls_dslx_module_get_member(module, 0);
  ASSERT_NE(member, nullptr);

  // Retrieve the QuickCheck node.
  xls_dslx_quickcheck* qc = xls_dslx_module_member_get_quickcheck(member);
  ASSERT_NE(qc, nullptr);

  // Inspect the associated function.
  xls_dslx_function* fn = xls_dslx_quickcheck_get_function(qc);
  EXPECT_NE(fn, nullptr);

  // Inspect the test-cases specifier.
  EXPECT_FALSE(xls_dslx_quickcheck_is_exhaustive(qc));
  int64_t count = 0;
  ASSERT_TRUE(xls_dslx_quickcheck_get_count(qc, &count));
  EXPECT_EQ(count, 123);
}

}  // namespace

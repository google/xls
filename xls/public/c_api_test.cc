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

#include <variant>
#include <utility>
#include <functional>
#include <initializer_list>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/macros.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
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

// Helper for invoking APIs where we want to immediately free the C-API-provided
// string and convert it into a C++ type.
//
// This is useful for temporaries like for display in error messages we we want
// convenience instead of explicitly showing how the lifetimes are used.
static std::string ToOwnedCppString(char* cstr) {
  std::string result = std::string(cstr);
  xls_c_str_free(cstr);
  return result;
}

// Takes a bits-based value and flattens it to a bits buffer and checks it's the
// same as the bits inside the value.
TEST(XlsCApiTest, FlattenBitsValueToBits) {
  char* error_out = nullptr;
  xls_value* value = nullptr;
  ASSERT_TRUE(xls_parse_typed_value("bits[32]:0x42", &error_out, &value));
  absl::Cleanup free_value([value] { xls_value_free(value); });

  // Get the bits from within the value. Note that it's owned by the caller.
  xls_bits* value_bits = nullptr;
  ASSERT_TRUE(xls_value_get_bits(value, &error_out, &value_bits));
  absl::Cleanup free_value_bits([value_bits] { xls_bits_free(value_bits); });

  // Flatten the value to a bits buffer.
  xls_bits* flattened = xls_value_flatten_to_bits(value);
  absl::Cleanup free_flattened([flattened] { xls_bits_free(flattened); });

  // The flattened bits should be the same as the original bits.
  EXPECT_TRUE(xls_bits_eq(value_bits, flattened))
      << "value_bits: "
      << ToOwnedCppString(xls_bits_to_debug_string(value_bits))
      << "\nflattened:  "
      << ToOwnedCppString(xls_bits_to_debug_string(flattened));
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
  EXPECT_TRUE(xls_bits_eq(flattened, want_bits))
      << "flattened: " << ToOwnedCppString(xls_bits_to_debug_string(flattened))
      << "\nwant_bits: "
      << ToOwnedCppString(xls_bits_to_debug_string(want_bits));
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

TEST(XlsCApiTest, MakeSbitsDoesNotFit) {
  char* error_out = nullptr;
  xls_bits* bits = nullptr;
  ASSERT_FALSE(xls_bits_make_sbits(1, -2, &error_out, &bits));
  absl::Cleanup free_error([error_out] { xls_c_str_free(error_out); });
  EXPECT_THAT(std::string_view{error_out},
              HasSubstr("Value 0xfffffffffffffffe requires 2 bits to fit in an "
                        "signed datatype"));

  ASSERT_TRUE(xls_bits_make_sbits(2, -2, &error_out, &bits));
  EXPECT_EQ(ToOwnedCppString(xls_bits_to_debug_string(bits)), "0b10");
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

  EXPECT_TRUE(xls_bits_eq(flattened, want_bits))
      << "flattened: " << ToOwnedCppString(xls_bits_to_debug_string(flattened))
      << "\nwant_bits: "
      << ToOwnedCppString(xls_bits_to_debug_string(want_bits));
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

}  // namespace

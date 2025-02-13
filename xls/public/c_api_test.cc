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
#include "xls/public/c_api_vast.h"

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

TEST(XlsCApiTest, VastAddIncludesAndEmit) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_file_add_include(f, "one_include.v");
  xls_vast_verilog_file_add_include(f, "another_include.v");

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "my_empty_module");
  ASSERT_NE(m, nullptr);

  // Add input/output and a wire.
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, false);
  xls_vast_verilog_module_add_input(m, "my_input", u8);
  xls_vast_verilog_module_add_output(m, "my_output", scalar);
  xls_vast_verilog_module_add_wire(m, "my_wire", scalar);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(`include "one_include.v"
`include "another_include.v"
module my_empty_module(
  input wire [7:0] my_input,
  output wire my_output
);
  wire my_wire;
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

// Test that instantiates a module with the ports tied to literal zero.
TEST(XlsCApiTest, VastInstantiate) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_literal* zero = xls_vast_verilog_file_make_plain_literal(f, 0);
  xls_vast_expression* zero_expr = xls_vast_literal_as_expression(zero);

  // Within the module we'll instantiate `mod_def_name` with `mod_inst_name`.
  const char* connection_port_names[] = {"portA", "portB",
                                         "portEmptyConnection"};
  xls_vast_expression* connection_expressions[] = {zero_expr, zero_expr,
                                                   nullptr};
  xls_vast_instantiation* instantiation =
      xls_vast_verilog_file_make_instantiation(
          f, "mod_def_name", "mod_inst_name",
          /*parameter_port_names=*/{},
          /*parameter_expressions=*/{},
          /*parameter_count=*/0, connection_port_names, connection_expressions,
          /*connection_count=*/3);

  xls_vast_verilog_module_add_member_instantiation(m, instantiation);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module;
  mod_def_name mod_inst_name (
    .portA(0),
    .portB(0),
    .portEmptyConnection()
  );
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

// Test that creates a module definition that continuous-assigns a slice of the
// input to the output.
TEST(XlsCApiTest, ContinuousAssignmentOfSlice) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, false);
  xls_vast_data_type* u4 =
      xls_vast_verilog_file_make_bit_vector_type(f, 4, false);
  xls_vast_logic_ref* input_ref =
      xls_vast_verilog_module_add_input(m, "my_input", u8);
  xls_vast_logic_ref* output_ref =
      xls_vast_verilog_module_add_output(m, "my_output", u4);

  xls_vast_slice* input_slice = xls_vast_verilog_file_make_slice_i64(
      f, xls_vast_logic_ref_as_indexable_expression(input_ref), 3, 0);

  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(output_ref),
          xls_vast_slice_as_expression(input_slice));

  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module(
  input wire [7:0] my_input,
  output wire [3:0] my_output
);
  assign my_output = my_input[3:0];
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

// Tests that we can assign a 128-bit output wire using a 128-bit literal
// value.
TEST(XlsCApiTest, ContinuousAssignmentOf128BitLiteral) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u128 =
      xls_vast_verilog_file_make_bit_vector_type(f, 128, false);
  xls_vast_logic_ref* output_ref =
      xls_vast_verilog_module_add_output(m, "my_output", u128);

  char* error_out = nullptr;
  struct xls_value* value = nullptr;
  ASSERT_TRUE(xls_parse_typed_value(
      "bits[128]:"
      "0b1001001000110100010101100111100010010000101010111100110111101111000100"
      "1000110100010101100111100010010000101010111100110111101111",
      &error_out, &value));
  ASSERT_EQ(error_out, nullptr);
  absl::Cleanup free_value([value] { xls_value_free(value); });

  struct xls_bits* bits = nullptr;
  ASSERT_TRUE(xls_value_get_bits(value, &error_out, &bits));
  ASSERT_EQ(error_out, nullptr);
  absl::Cleanup free_bits([bits] { xls_bits_free(bits); });

  xls_vast_literal* literal_128b = nullptr;
  ASSERT_TRUE(xls_vast_verilog_file_make_literal(
      f, bits, xls_format_preference_binary, /*emit_bit_count=*/true,
      &error_out, &literal_128b));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(literal_128b, nullptr);

  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(output_ref),
          xls_vast_literal_as_expression(literal_128b));

  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module(
  output wire [127:0] my_output
);
  assign my_output = 128'b1001_0010_0011_0100_0101_0110_0111_1000_1001_0000_1010_1011_1100_1101_1110_1111_0001_0010_0011_0100_0101_0110_0111_1000_1001_0000_1010_1011_1100_1101_1110_1111;
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
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

TEST(XlsCApiTest, VastExternPackageTypePackedArrayPort) {
  const std::string_view kWantEmitted = R"(module top(
  input mypack::mystruct_t [1:0][2:0][3:0] my_input
);

endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* my_struct =
      xls_vast_verilog_file_make_extern_package_type(f, "mypack", "mystruct_t");
  const std::vector<int64_t> packed_dims = {2, 3, 4};
  xls_vast_data_type* my_input_type =
      xls_vast_verilog_file_make_packed_array_type(
          f, my_struct, packed_dims.data(), packed_dims.size());

  xls_vast_verilog_module_add_input(m, "my_input", my_input_type);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

// Tests that we can reference a slice of a multidimensional packed array on
// the LHS or RHS of an assign statement; e.g.
// ```
// assign a[1][2][3:4] = b[1:0];
// assign a[3:4] = c[2:1];
TEST(XlsCApiTest, VastPackedArraySliceAssignment) {
  const std::string_view kWantEmitted = R"(module top;
  wire [1:0][2:0][4:0] a;
  wire [1:0] b;
  wire [2:0] c;
  assign a[1][2][3:4] = b[1:0];
  assign a[3:4] = c[2:1];
endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u2 =
      xls_vast_verilog_file_make_bit_vector_type(f, 2, /*is_signed=*/false);
  xls_vast_data_type* u3 =
      xls_vast_verilog_file_make_bit_vector_type(f, 3, /*is_signed=*/false);

  const std::vector<int64_t> packed_dims = {3, 5};
  xls_vast_data_type* a_type = xls_vast_verilog_file_make_packed_array_type(
      f, u2, packed_dims.data(), packed_dims.size());
  auto* b_type = u2;
  auto* c_type = u3;

  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_wire(m, "a", a_type);
  xls_vast_logic_ref* b_ref = xls_vast_verilog_module_add_wire(m, "b", b_type);
  xls_vast_logic_ref* c_ref = xls_vast_verilog_module_add_wire(m, "c", c_type);

  xls_vast_literal* literal_0 = xls_vast_verilog_file_make_plain_literal(f, 0);
  xls_vast_literal* literal_1 = xls_vast_verilog_file_make_plain_literal(f, 1);
  xls_vast_literal* literal_2 = xls_vast_verilog_file_make_plain_literal(f, 2);
  xls_vast_literal* literal_3 = xls_vast_verilog_file_make_plain_literal(f, 3);
  xls_vast_literal* literal_4 = xls_vast_verilog_file_make_plain_literal(f, 4);

  // Build the first assign statement.
  {
    // The lhs is `a[1][2][3:4]`. We have to build this up via a nested
    // index/index/slice.
    xls_vast_index* a_index_0 = xls_vast_verilog_file_make_index(
        f, xls_vast_logic_ref_as_indexable_expression(a_ref),
        xls_vast_literal_as_expression(literal_1));
    xls_vast_index* a_index_1 = xls_vast_verilog_file_make_index(
        f, xls_vast_index_as_indexable_expression(a_index_0),
        xls_vast_literal_as_expression(literal_2));
    xls_vast_slice* a_slice = xls_vast_verilog_file_make_slice(
        f, xls_vast_index_as_indexable_expression(a_index_1),
        xls_vast_literal_as_expression(literal_3),
        xls_vast_literal_as_expression(literal_4));

    // The rhs is `b[1:0]`.
    xls_vast_slice* b_slice = xls_vast_verilog_file_make_slice(
        f, xls_vast_logic_ref_as_indexable_expression(b_ref),
        xls_vast_literal_as_expression(literal_1),
        xls_vast_literal_as_expression(literal_0));

    xls_vast_continuous_assignment* assignment =
        xls_vast_verilog_file_make_continuous_assignment(
            f, xls_vast_slice_as_expression(a_slice),
            xls_vast_slice_as_expression(b_slice));
    xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);
  }

  // Build the second assign statement.
  {
    xls_vast_slice* lhs = xls_vast_verilog_file_make_slice(
        f, xls_vast_logic_ref_as_indexable_expression(a_ref),
        xls_vast_literal_as_expression(literal_3),
        xls_vast_literal_as_expression(literal_4));
    xls_vast_slice* rhs = xls_vast_verilog_file_make_slice(
        f, xls_vast_logic_ref_as_indexable_expression(c_ref),
        xls_vast_literal_as_expression(literal_2),
        xls_vast_literal_as_expression(literal_1));
    xls_vast_continuous_assignment* assignment =
        xls_vast_verilog_file_make_continuous_assignment(
            f, xls_vast_slice_as_expression(lhs),
            xls_vast_slice_as_expression(rhs));
    xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);
  }

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastSimpleConcat) {
  const std::string_view kWantEmitted = R"(module top;
  wire [7:0] a;
  wire [3:0] b;
  wire [11:0] c;
  assign c = {a, b};
endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  xls_vast_data_type* u4 =
      xls_vast_verilog_file_make_bit_vector_type(f, 4, /*is_signed=*/false);
  xls_vast_data_type* u12 =
      xls_vast_verilog_file_make_bit_vector_type(f, 12, /*is_signed=*/false);

  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_wire(m, "a", u8);
  xls_vast_logic_ref* b_ref = xls_vast_verilog_module_add_wire(m, "b", u4);
  xls_vast_logic_ref* c_ref = xls_vast_verilog_module_add_wire(m, "c", u12);

  std::vector<xls_vast_expression*> concat_members = {
      xls_vast_logic_ref_as_expression(a_ref),
      xls_vast_logic_ref_as_expression(b_ref)};
  xls_vast_concat* concat = xls_vast_verilog_file_make_concat(
      f, concat_members.data(), concat_members.size());
  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(c_ref),
          xls_vast_concat_as_expression(concat));
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
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

}  // namespace

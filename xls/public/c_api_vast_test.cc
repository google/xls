// Copyright 2025 The XLS Authors
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

// Tests for the "protected" VAST portion of the XLS C API.

#include "xls/public/c_api_vast.h"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "xls/public/c_api.h"
#include "xls/public/c_api_format_preference.h"

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

TEST(XlsCApiTest, VastUnaryOps) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  xls_vast_logic_ref* input_ref =
      xls_vast_verilog_module_add_input(m, "arg", u8);

  for (auto [op, name] :
       std::vector<std::pair<xls_vast_operator_kind, const char*>>{
           {xls_vast_operator_kind_negate, "my_negate"},
           {xls_vast_operator_kind_bitwise_not, "my_bitwise_not"},
           {xls_vast_operator_kind_logical_not, "my_logical_not"},
           {xls_vast_operator_kind_and_reduce, "my_and_reduce"},
           {xls_vast_operator_kind_or_reduce, "my_or_reduce"},
           {xls_vast_operator_kind_xor_reduce, "my_xor_reduce"},
       }) {
    xls_vast_logic_ref* logic_ref =
        xls_vast_verilog_module_add_wire(m, name, u8);
    xls_vast_expression* result = xls_vast_verilog_file_make_unary(
        f, xls_vast_logic_ref_as_expression(input_ref), op);
    xls_vast_continuous_assignment* assignment =
        xls_vast_verilog_file_make_continuous_assignment(
            f, xls_vast_logic_ref_as_expression(logic_ref), result);
    xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);
  }

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module(
  input wire [7:0] arg
);
  wire [7:0] my_negate;
  assign my_negate = -arg;
  wire [7:0] my_bitwise_not;
  assign my_bitwise_not = ~arg;
  wire [7:0] my_logical_not;
  assign my_logical_not = !arg;
  wire [7:0] my_and_reduce;
  assign my_and_reduce = &arg;
  wire [7:0] my_or_reduce;
  assign my_or_reduce = |arg;
  wire [7:0] my_xor_reduce;
  assign my_xor_reduce = ^arg;
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

TEST(XlsCApiTest, VastBinaryOps) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);

  // Add lhs and rhs as inputs.
  xls_vast_logic_ref* lhs_ref = xls_vast_verilog_module_add_input(m, "lhs", u8);
  ASSERT_NE(lhs_ref, nullptr);

  xls_vast_logic_ref* rhs_ref = xls_vast_verilog_module_add_input(m, "rhs", u8);
  ASSERT_NE(rhs_ref, nullptr);

  for (auto [op, name] :
       std::vector<std::pair<xls_vast_operator_kind, const char*>>{
           {xls_vast_operator_kind_add, "my_add"},
           {xls_vast_operator_kind_sub, "my_sub"},
           {xls_vast_operator_kind_mul, "my_mul"},
           {xls_vast_operator_kind_div, "my_div"},
           {xls_vast_operator_kind_mod, "my_mod"},
           {xls_vast_operator_kind_shll, "my_shll"},
           {xls_vast_operator_kind_shra, "my_shra"},
           {xls_vast_operator_kind_shrl, "my_shrl"},
           {xls_vast_operator_kind_bitwise_and, "my_bitwise_and"},
           {xls_vast_operator_kind_bitwise_or, "my_bitwise_or"},
           {xls_vast_operator_kind_bitwise_xor, "my_bitwise_xor"},
           {xls_vast_operator_kind_logical_and, "my_logical_and"},
           {xls_vast_operator_kind_logical_or, "my_logical_or"},
           {xls_vast_operator_kind_ne, "my_ne"},
           {xls_vast_operator_kind_eq, "my_eq"},
           {xls_vast_operator_kind_ge, "my_ge"},
           {xls_vast_operator_kind_gt, "my_gt"},
           {xls_vast_operator_kind_le, "my_le"},
           {xls_vast_operator_kind_lt, "my_lt"},
           {xls_vast_operator_kind_ne_x, "my_ne_x"},
           {xls_vast_operator_kind_eq_x, "my_eq_x"},
       }) {
    xls_vast_logic_ref* logic_ref =
        xls_vast_verilog_module_add_wire(m, name, u8);
    xls_vast_expression* result = xls_vast_verilog_file_make_binary(
        f, xls_vast_logic_ref_as_expression(lhs_ref),
        xls_vast_logic_ref_as_expression(rhs_ref), op);
    xls_vast_continuous_assignment* assignment =
        xls_vast_verilog_file_make_continuous_assignment(
            f, xls_vast_logic_ref_as_expression(logic_ref), result);
    xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);
  }

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module(
  input wire [7:0] lhs,
  input wire [7:0] rhs
);
  wire [7:0] my_add;
  assign my_add = lhs + rhs;
  wire [7:0] my_sub;
  assign my_sub = lhs - rhs;
  wire [7:0] my_mul;
  assign my_mul = lhs * rhs;
  wire [7:0] my_div;
  assign my_div = lhs / rhs;
  wire [7:0] my_mod;
  assign my_mod = lhs % rhs;
  wire [7:0] my_shll;
  assign my_shll = lhs << rhs;
  wire [7:0] my_shra;
  assign my_shra = lhs >>> rhs;
  wire [7:0] my_shrl;
  assign my_shrl = lhs >> rhs;
  wire [7:0] my_bitwise_and;
  assign my_bitwise_and = lhs & rhs;
  wire [7:0] my_bitwise_or;
  assign my_bitwise_or = lhs | rhs;
  wire [7:0] my_bitwise_xor;
  assign my_bitwise_xor = lhs ^ rhs;
  wire [7:0] my_logical_and;
  assign my_logical_and = lhs && rhs;
  wire [7:0] my_logical_or;
  assign my_logical_or = lhs || rhs;
  wire [7:0] my_ne;
  assign my_ne = lhs != rhs;
  wire [7:0] my_eq;
  assign my_eq = lhs == rhs;
  wire [7:0] my_ge;
  assign my_ge = lhs >= rhs;
  wire [7:0] my_gt;
  assign my_gt = lhs > rhs;
  wire [7:0] my_le;
  assign my_le = lhs <= rhs;
  wire [7:0] my_lt;
  assign my_lt = lhs < rhs;
  wire [7:0] my_ne_x;
  assign my_ne_x = lhs !== rhs;
  wire [7:0] my_eq_x;
  assign my_eq_x = lhs === rhs;
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

TEST(XlsCApiTest, VastTernary) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u1 =
      xls_vast_verilog_file_make_bit_vector_type(f, 1, /*is_signed=*/false);
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);

  xls_vast_logic_ref* cond_ref =
      xls_vast_verilog_module_add_input(m, "cond", u1);
  ASSERT_NE(cond_ref, nullptr);

  xls_vast_logic_ref* consequent_ref =
      xls_vast_verilog_module_add_input(m, "consequent", u8);
  ASSERT_NE(consequent_ref, nullptr);

  xls_vast_logic_ref* alternate_ref =
      xls_vast_verilog_module_add_input(m, "alternate", u8);
  ASSERT_NE(alternate_ref, nullptr);

  xls_vast_logic_ref* output_ref =
      xls_vast_verilog_module_add_output(m, "output", u8);
  ASSERT_NE(output_ref, nullptr);

  xls_vast_expression* result = xls_vast_verilog_file_make_ternary(
      f, xls_vast_logic_ref_as_expression(cond_ref),
      xls_vast_logic_ref_as_expression(consequent_ref),
      xls_vast_logic_ref_as_expression(alternate_ref));

  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(output_ref), result);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module(
  input wire cond,
  input wire [7:0] consequent,
  input wire [7:0] alternate,
  output wire [7:0] output
);
  assign output = cond ? consequent : alternate;
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

// Tests for conversion of VAST constructs to expressions, e.g. for use in a
// concatenation.
TEST(XlsCApiTest, VastExpressions) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);

  // Make a continuous assignment for a concat expression as the RHS.
  xls_vast_logic_ref* input_ref =
      xls_vast_verilog_module_add_input(m, "input", u8);
  ASSERT_NE(input_ref, nullptr);

  xls_vast_logic_ref* output_ref =
      xls_vast_verilog_module_add_output(m, "output", u8);
  ASSERT_NE(output_ref, nullptr);

  // Create a concat expression -- use an index and a slice as the concatenated
  // elements.
  xls_vast_index* index = xls_vast_verilog_file_make_index_i64(
      f, xls_vast_logic_ref_as_indexable_expression(input_ref), 0);
  ASSERT_NE(index, nullptr);

  // Now convert the index to an expression.
  xls_vast_expression* index_expr = xls_vast_index_as_expression(index);
  ASSERT_NE(index_expr, nullptr);

  xls_vast_slice* slice = xls_vast_verilog_file_make_slice(
      f, /*subject=*/xls_vast_logic_ref_as_indexable_expression(input_ref),
      /*hi=*/
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 7)),
      /*lo=*/
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 0)));
  ASSERT_NE(slice, nullptr);

  // Now convert the slice to an expression.
  xls_vast_expression* slice_expr = xls_vast_slice_as_expression(slice);
  ASSERT_NE(slice_expr, nullptr);

  xls_vast_expression* concat_elements[2] = {index_expr, slice_expr};
  xls_vast_concat* concat =
      xls_vast_verilog_file_make_concat(f, concat_elements, 2);
  ASSERT_NE(concat, nullptr);

  // Now convert the concat to an expression.
  xls_vast_expression* concat_expr = xls_vast_concat_as_expression(concat);
  ASSERT_NE(concat_expr, nullptr);

  // Create a continuous assignment.
  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(output_ref), concat_expr);
  ASSERT_NE(assignment, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module(
  input wire [7:0] input,
  output wire [7:0] output
);
  assign output = {input[0], input[7:0]};
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

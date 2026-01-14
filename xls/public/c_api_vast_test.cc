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

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "gtest/gtest.h"
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

  char* name = xls_vast_verilog_module_get_name(m);
  ASSERT_NE(name, nullptr);
  absl::Cleanup free_name([&] { xls_c_str_free(name); });
  EXPECT_EQ(std::string_view{name}, "my_empty_module");

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

TEST(XlsCApiTest, VastModuleWithInOutPort) {
  const std::string_view kWantEmitted = R"(module top(
  inout wire [7:0] io
);

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
  xls_vast_logic_ref* io = xls_vast_verilog_module_add_inout(m, "io", u8);
  ASSERT_NE(io, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
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

  char* input_name = xls_vast_logic_ref_get_name(input_ref);
  ASSERT_NE(input_name, nullptr);
  absl::Cleanup free_input_name([&] { xls_c_str_free(input_name); });
  EXPECT_EQ(std::string_view{input_name}, "my_input");

  char* output_name = xls_vast_logic_ref_get_name(output_ref);
  ASSERT_NE(output_name, nullptr);
  absl::Cleanup free_output_name([&] { xls_c_str_free(output_name); });
  EXPECT_EQ(std::string_view{output_name}, "my_output");

  xls_vast_slice* input_slice = xls_vast_verilog_file_make_slice_i64(
      f, xls_vast_logic_ref_as_indexable_expression(input_ref), 3, 0);

  xls_vast_comment* comment =
      xls_vast_verilog_file_make_comment(f, "This is a comment");
  xls_vast_verilog_module_add_member_comment(m, comment);

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
  // This is a comment
  assign my_output = my_input[3:0];
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

TEST(XlsCApiTest, VastExternTypesPackedArrayPort) {
  const std::string_view kWantEmitted = R"(module top(
  input mypack::mystruct_t [1:0][2:0][3:0] my_input,
  output yourstruct_t my_output
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
  xls_vast_data_type* your_struct =
      xls_vast_verilog_file_make_extern_type(f, "yourstruct_t");
  const std::vector<int64_t> packed_dims = {2, 3, 4};
  xls_vast_data_type* my_input_type =
      xls_vast_verilog_file_make_packed_array_type(
          f, my_struct, packed_dims.data(), packed_dims.size());

  xls_vast_verilog_module_add_input(m, "my_input", my_input_type);
  xls_vast_verilog_module_add_output(m, "my_output", your_struct);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastGenerateLoopElementwiseAssignment) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  xls_vast_logic_ref* in_ref = xls_vast_verilog_module_add_input(m, "in", u8);
  xls_vast_logic_ref* out_ref =
      xls_vast_verilog_module_add_output(m, "out", u8);

  xls_vast_literal* zero_lit = xls_vast_verilog_file_make_plain_literal(f, 0);
  xls_vast_literal* eight_lit = xls_vast_verilog_file_make_plain_literal(f, 8);
  xls_vast_expression* zero_expr = xls_vast_literal_as_expression(zero_lit);
  xls_vast_expression* eight_expr = xls_vast_literal_as_expression(eight_lit);

  xls_vast_generate_loop* loop = xls_vast_verilog_module_add_generate_loop(
      m, "i", zero_expr, eight_expr, "gen");
  ASSERT_NE(loop, nullptr);

  xls_vast_logic_ref* genvar_ref = xls_vast_generate_loop_get_genvar(loop);
  ASSERT_NE(genvar_ref, nullptr);
  xls_vast_expression* genvar_expr =
      xls_vast_logic_ref_as_expression(genvar_ref);

  xls_vast_index* out_index = xls_vast_verilog_file_make_index(
      f, xls_vast_logic_ref_as_indexable_expression(out_ref), genvar_expr);
  xls_vast_index* in_index = xls_vast_verilog_file_make_index(
      f, xls_vast_logic_ref_as_indexable_expression(in_ref), genvar_expr);

  xls_vast_statement* assignment_stmt =
      xls_vast_generate_loop_add_continuous_assignment(
          loop, xls_vast_index_as_expression(out_index),
          xls_vast_index_as_expression(in_index));
  ASSERT_NE(assignment_stmt, nullptr);

  // Add a blank line, a comment, an inline verilog statement, and an empty
  // instantiation inside the generate loop.
  xls_vast_generate_loop_add_blank_line(loop);
  xls_vast_comment* comment =
      xls_vast_verilog_file_make_comment(f, "This is a comment.");
  xls_vast_generate_loop_add_comment(loop, comment);
  xls_vast_inline_verilog_statement* inline_stmt =
      xls_vast_verilog_file_make_inline_verilog_statement(
          f, "inline_verilog_statement;");
  xls_vast_generate_loop_add_inline_verilog_statement(loop, inline_stmt);

  // Add macro statements with and without semicolons, with and without args.
  // `GL_MACRO1;
  {
    xls_vast_macro_ref* mr =
        xls_vast_verilog_file_make_macro_ref(f, "GL_MACRO1");
    ASSERT_NE(mr, nullptr);
    xls_vast_macro_statement* ms = xls_vast_verilog_file_make_macro_statement(
        f, mr, /*emit_semicolon=*/true);
    ASSERT_NE(ms, nullptr);
    ASSERT_NE(xls_vast_generate_loop_add_macro_statement(loop, ms), nullptr);
  }
  // `GL_MACRO2(i)
  {
    xls_vast_expression* args[] = {genvar_expr};
    xls_vast_macro_ref* mr =
        xls_vast_verilog_file_make_macro_ref_with_args(f, "GL_MACRO2", args, 1);
    ASSERT_NE(mr, nullptr);
    xls_vast_macro_statement* ms = xls_vast_verilog_file_make_macro_statement(
        f, mr, /*emit_semicolon=*/false);
    ASSERT_NE(ms, nullptr);
    ASSERT_NE(xls_vast_generate_loop_add_macro_statement(loop, ms), nullptr);
  }
  // `GL_MACRO3(out[i], in[i]);
  {
    xls_vast_expression* args[] = {xls_vast_index_as_expression(out_index),
                                   xls_vast_index_as_expression(in_index)};
    xls_vast_macro_ref* mr =
        xls_vast_verilog_file_make_macro_ref_with_args(f, "GL_MACRO3", args, 2);
    ASSERT_NE(mr, nullptr);
    xls_vast_macro_statement* ms = xls_vast_verilog_file_make_macro_statement(
        f, mr, /*emit_semicolon=*/true);
    ASSERT_NE(ms, nullptr);
    ASSERT_NE(xls_vast_generate_loop_add_macro_statement(loop, ms), nullptr);
  }

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });

  const std::string_view kWantEmittedWithExtras = R"(module top(
  input wire [7:0] in,
  output wire [7:0] out
);
  for (genvar i = 0; i < 8; i = i + 1) begin : gen
    assign out[i] = in[i];

    // This is a comment.
    inline_verilog_statement;
    `GL_MACRO1;
    `GL_MACRO2(i)
    `GL_MACRO3(out[i], in[i]);
  end
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWantEmittedWithExtras);
}

TEST(XlsCApiTest, VastFileLevelBlankLineAndComment) {
  const std::string_view kWantEmitted = R"(// file header

module top;

endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  // Add a file-level comment and a blank line.
  xls_vast_verilog_file_add_comment(f, "file header");
  xls_vast_verilog_file_add_blank_line(f);

  // Add a trivial module so the file emits something after the header.
  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastEmitExpression) {
  // Create a file and module.
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  // Expression: plain literal 7 -> "7"
  {
    xls_vast_literal* lit7 = xls_vast_verilog_file_make_plain_literal(f, 7);
    ASSERT_NE(lit7, nullptr);
    xls_vast_expression* expr = xls_vast_literal_as_expression(lit7);
    ASSERT_NE(expr, nullptr);
    char* emitted = xls_vast_expression_emit(expr);
    ASSERT_NE(emitted, nullptr);
    absl::Cleanup free_expr([&] { xls_c_str_free(emitted); });
    EXPECT_EQ(std::string_view{emitted}, "7");
  }
}

// Tests that we can reference a slice of a multidimensional packed array on
// the LHS or RHS of an assign statement; e.g.
// ```
// assign a[1][2][3:4] = b[1:0];
// assign a[3:4] = c[2:1];
TEST(XlsCApiTest, VastPackedArraySliceAssignment) {
  const std::string_view kWantEmitted = R"(module top;
  wire [2:0][4:0][1:0] a;
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

TEST(XlsCApiTest, VastReplicatedConcatSingle) {
  const std::string_view kWantEmitted = R"(module top;
  wire [3:0] a;
  wire [11:0] y;
  assign y = {3{a}};
endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u4 =
      xls_vast_verilog_file_make_bit_vector_type(f, 4, /*is_signed=*/false);
  xls_vast_data_type* u12 =
      xls_vast_verilog_file_make_bit_vector_type(f, 12, /*is_signed=*/false);

  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_wire(m, "a", u4);
  xls_vast_logic_ref* y_ref = xls_vast_verilog_module_add_wire(m, "y", u12);

  xls_vast_expression* elems[] = {xls_vast_logic_ref_as_expression(a_ref)};
  xls_vast_concat* rc =
      xls_vast_verilog_file_make_replicated_concat_i64(f, 3, elems, 1);

  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(y_ref),
          xls_vast_concat_as_expression(rc));
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastReplicatedConcatMulti) {
  const std::string_view kWantEmitted = R"(module top;
  wire [3:0] a;
  wire [1:0] b;
  wire [11:0] y;
  assign y = {2{a, b}};
endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u4 =
      xls_vast_verilog_file_make_bit_vector_type(f, 4, /*is_signed=*/false);
  xls_vast_data_type* u2 =
      xls_vast_verilog_file_make_bit_vector_type(f, 2, /*is_signed=*/false);
  xls_vast_data_type* u12 =
      xls_vast_verilog_file_make_bit_vector_type(f, 12, /*is_signed=*/false);

  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_wire(m, "a", u4);
  xls_vast_logic_ref* b_ref = xls_vast_verilog_module_add_wire(m, "b", u2);
  xls_vast_logic_ref* y_ref = xls_vast_verilog_module_add_wire(m, "y", u12);

  xls_vast_expression* elems[] = {xls_vast_logic_ref_as_expression(a_ref),
                                  xls_vast_logic_ref_as_expression(b_ref)};
  xls_vast_literal* two = xls_vast_verilog_file_make_plain_literal(f, 2);
  xls_vast_concat* rc = xls_vast_verilog_file_make_replicated_concat(
      f, xls_vast_literal_as_expression(two), elems, 2);

  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(y_ref),
          xls_vast_concat_as_expression(rc));
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastModuleCommentsAndBlankLines) {
  const std::string_view kWantEmitted = R"(module top;
  // top comment

  if (1) ;
endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_comment* c = xls_vast_verilog_file_make_comment(f, "top comment");
  xls_vast_verilog_module_add_member_comment(m, c);
  xls_vast_blank_line* b = xls_vast_verilog_file_make_blank_line(f);
  xls_vast_verilog_module_add_member_blank_line(m, b);
  xls_vast_inline_verilog_statement* inl =
      xls_vast_verilog_file_make_inline_verilog_statement(f, "if (1) ;");
  xls_vast_verilog_module_add_member_inline_statement(m, inl);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastBlockCommentsAndInline) {
  const std::string_view kWantEmitted = R"(module top;
  always @ (*) begin
    // inside

    // raw
  end
endmodule
)";
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_expression* sl[] = {nullptr};
  xls_vast_always_base* ab = nullptr;
  char* err = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_always_at(m, sl, 1, &ab, &err));
  ASSERT_EQ(err, nullptr);

  xls_vast_statement_block* block =
      xls_vast_always_base_get_statement_block(ab);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_comment_text(block, "inside"),
            nullptr);
  ASSERT_NE(xls_vast_statement_block_add_blank_line(block), nullptr);
  ASSERT_NE(xls_vast_statement_block_add_inline_text(block, "// raw"), nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

// Adds module parameters and uses them in expressions.
TEST(XlsCApiTest, VastModuleParameters) {
  const std::string_view kWantEmitted = R"(module param_module;
  parameter WIDTH = 8;
  wire [7:0] a;
  wire [7:0] b;
  assign b = a << WIDTH;
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "param_module");
  ASSERT_NE(m, nullptr);

  // parameter WIDTH = 8;
  xls_vast_literal* literal_8 = xls_vast_verilog_file_make_plain_literal(f, 8);
  ASSERT_NE(literal_8, nullptr);
  auto* width_param = xls_vast_verilog_module_add_parameter(
      m, "WIDTH", xls_vast_literal_as_expression(literal_8));
  ASSERT_NE(width_param, nullptr);

  // wire [7:0] a; wire [7:0] b;
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_wire(m, "a", u8);
  ASSERT_NE(a_ref, nullptr);
  xls_vast_logic_ref* b_ref = xls_vast_verilog_module_add_wire(m, "b", u8);
  ASSERT_NE(b_ref, nullptr);

  // assign b = a << WIDTH;
  xls_vast_expression* shift_expr = xls_vast_verilog_file_make_binary(
      f, xls_vast_logic_ref_as_expression(a_ref),
      xls_vast_parameter_ref_as_expression(width_param),
      xls_vast_operator_kind_shll);
  ASSERT_NE(shift_expr, nullptr);
  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(b_ref), shift_expr);
  ASSERT_NE(assignment, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastWidthCast) {
  const std::string_view kWantEmitted = R"(module top(
  input wire [7:0] a,
  input wire [3:0] b,
  output wire [7:0] out_literal,
  output wire [11:0] out_param,
  output wire [15:0] out_expr
);
  parameter WidthParam = 12;
  assign out_literal = 8'(a + 1);
  assign out_param = WidthParam'({a, b});
  assign out_expr = (WidthParam + 4)'(a ^ {b, 2'h3});
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
  xls_vast_data_type* u16 =
      xls_vast_verilog_file_make_bit_vector_type(f, 16, /*is_signed=*/false);

  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_input(m, "a", u8);
  ASSERT_NE(a_ref, nullptr);
  xls_vast_logic_ref* b_ref = xls_vast_verilog_module_add_input(m, "b", u4);
  ASSERT_NE(b_ref, nullptr);
  xls_vast_logic_ref* out_literal_ref =
      xls_vast_verilog_module_add_output(m, "out_literal", u8);
  ASSERT_NE(out_literal_ref, nullptr);
  xls_vast_logic_ref* out_param_ref =
      xls_vast_verilog_module_add_output(m, "out_param", u12);
  ASSERT_NE(out_param_ref, nullptr);
  xls_vast_logic_ref* out_expr_ref =
      xls_vast_verilog_module_add_output(m, "out_expr", u16);
  ASSERT_NE(out_expr_ref, nullptr);

  xls_vast_literal* literal_12 =
      xls_vast_verilog_file_make_plain_literal(f, 12);
  ASSERT_NE(literal_12, nullptr);
  xls_vast_parameter_ref* width_param = xls_vast_verilog_module_add_parameter(
      m, "WidthParam", xls_vast_literal_as_expression(literal_12));
  ASSERT_NE(width_param, nullptr);
  xls_vast_expression* width_param_expr =
      xls_vast_parameter_ref_as_expression(width_param);

  xls_vast_literal* literal_8 = xls_vast_verilog_file_make_plain_literal(f, 8);
  ASSERT_NE(literal_8, nullptr);
  xls_vast_literal* literal_1 = xls_vast_verilog_file_make_plain_literal(f, 1);
  ASSERT_NE(literal_1, nullptr);
  xls_vast_expression* a_expr = xls_vast_logic_ref_as_expression(a_ref);
  ASSERT_NE(a_expr, nullptr);
  xls_vast_expression* a_plus_one = xls_vast_verilog_file_make_binary(
      f, a_expr, xls_vast_literal_as_expression(literal_1),
      xls_vast_operator_kind_add);
  ASSERT_NE(a_plus_one, nullptr);

  xls_vast_expression* width_cast_literal =
      xls_vast_verilog_file_make_width_cast(
          f, xls_vast_literal_as_expression(literal_8), a_plus_one);
  ASSERT_NE(width_cast_literal, nullptr);
  xls_vast_continuous_assignment* assign_literal =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(out_literal_ref),
          width_cast_literal);
  ASSERT_NE(assign_literal, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assign_literal);

  std::vector<xls_vast_expression*> param_concat_args = {
      xls_vast_logic_ref_as_expression(a_ref),
      xls_vast_logic_ref_as_expression(b_ref)};
  xls_vast_concat* param_concat = xls_vast_verilog_file_make_concat(
      f, param_concat_args.data(), param_concat_args.size());
  ASSERT_NE(param_concat, nullptr);
  xls_vast_expression* width_cast_param = xls_vast_verilog_file_make_width_cast(
      f, width_param_expr, xls_vast_concat_as_expression(param_concat));
  ASSERT_NE(width_cast_param, nullptr);
  xls_vast_continuous_assignment* assign_param =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(out_param_ref), width_cast_param);
  ASSERT_NE(assign_param, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assign_param);

  xls_vast_literal* literal_4 = xls_vast_verilog_file_make_plain_literal(f, 4);
  ASSERT_NE(literal_4, nullptr);
  xls_vast_expression* complex_width = xls_vast_verilog_file_make_binary(
      f, width_param_expr, xls_vast_literal_as_expression(literal_4),
      xls_vast_operator_kind_add);
  ASSERT_NE(complex_width, nullptr);

  struct xls_bits* bits = nullptr;
  char* error_out = nullptr;
  ASSERT_TRUE(
      xls_bits_make_ubits(/*bit_count=*/2, /*value=*/3, &error_out, &bits));
  ASSERT_EQ(error_out, nullptr);
  absl::Cleanup free_bits([&] { xls_bits_free(bits); });

  xls_vast_literal* literal_2h3 = nullptr;
  ASSERT_TRUE(xls_vast_verilog_file_make_literal(
      f, bits, xls_format_preference_hex, /*emit_bit_count=*/true, &error_out,
      &literal_2h3));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(literal_2h3, nullptr);

  std::vector<xls_vast_expression*> tail_concat_args = {
      xls_vast_logic_ref_as_expression(b_ref),
      xls_vast_literal_as_expression(literal_2h3)};
  xls_vast_concat* tail_concat = xls_vast_verilog_file_make_concat(
      f, tail_concat_args.data(), tail_concat_args.size());
  ASSERT_NE(tail_concat, nullptr);

  xls_vast_expression* complex_value = xls_vast_verilog_file_make_binary(
      f, a_expr, xls_vast_concat_as_expression(tail_concat),
      xls_vast_operator_kind_bitwise_xor);
  ASSERT_NE(complex_value, nullptr);

  xls_vast_expression* width_cast_complex =
      xls_vast_verilog_file_make_width_cast(f, complex_width, complex_value);
  ASSERT_NE(width_cast_complex, nullptr);
  xls_vast_continuous_assignment* assign_complex =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(out_expr_ref),
          width_cast_complex);
  ASSERT_NE(assign_complex, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assign_complex);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastTypeCast) {
  const std::string_view kWantEmitted = R"(module top(
  input wire [7:0] a,
  output work::foobar out_foo,
  output my_pkg::my_type_t out_pkg
);
  assign out_foo = work::foobar'(a + 1);
  assign out_pkg = my_pkg::my_type_t'(a);
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
  ASSERT_NE(u8, nullptr);

  xls_vast_logic_ref* a_ref = xls_vast_verilog_module_add_input(m, "a", u8);
  ASSERT_NE(a_ref, nullptr);

  xls_vast_data_type* work_foobar =
      xls_vast_verilog_file_make_extern_package_type(f, "work", "foobar");
  ASSERT_NE(work_foobar, nullptr);
  xls_vast_logic_ref* out_foo_ref =
      xls_vast_verilog_module_add_output(m, "out_foo", work_foobar);
  ASSERT_NE(out_foo_ref, nullptr);

  xls_vast_data_type* mypkg_mytype =
      xls_vast_verilog_file_make_extern_package_type(f, "my_pkg", "my_type_t");
  ASSERT_NE(mypkg_mytype, nullptr);
  xls_vast_logic_ref* out_pkg_ref =
      xls_vast_verilog_module_add_output(m, "out_pkg", mypkg_mytype);
  ASSERT_NE(out_pkg_ref, nullptr);

  xls_vast_literal* lit1 = xls_vast_verilog_file_make_plain_literal(f, 1);
  ASSERT_NE(lit1, nullptr);
  xls_vast_expression* a_plus_one = xls_vast_verilog_file_make_binary(
      f, xls_vast_logic_ref_as_expression(a_ref),
      xls_vast_literal_as_expression(lit1), xls_vast_operator_kind_add);
  ASSERT_NE(a_plus_one, nullptr);

  xls_vast_expression* cast_foo =
      xls_vast_verilog_file_make_type_cast(f, work_foobar, a_plus_one);
  ASSERT_NE(cast_foo, nullptr);
  xls_vast_continuous_assignment* assign_foo =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(out_foo_ref), cast_foo);
  ASSERT_NE(assign_foo, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assign_foo);

  xls_vast_expression* cast_pkg = xls_vast_verilog_file_make_type_cast(
      f, mypkg_mytype, xls_vast_logic_ref_as_expression(a_ref));
  ASSERT_NE(cast_pkg, nullptr);
  xls_vast_continuous_assignment* assign_pkg =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(out_pkg_ref), cast_pkg);
  ASSERT_NE(assign_pkg, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assign_pkg);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastUnsizedLiteralsParameters) {
  const std::string_view kWantEmitted = R"(module top;
  parameter P0 = '0;
  parameter P1 = '1;
  parameter P2 = 'X;
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  // parameter P0 = '0; parameter P1 = '1; parameter P2 = 'X;
  ASSERT_NE(xls_vast_verilog_module_add_parameter(
                m, "P0", xls_vast_verilog_file_make_unsized_zero_literal(f)),
            nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter(
                m, "P1", xls_vast_verilog_file_make_unsized_one_literal(f)),
            nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter(
                m, "P2", xls_vast_verilog_file_make_unsized_x_literal(f)),
            nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

// Makes module parameters with explicit types via Def: logic bit-vector and
// integer.
TEST(XlsCApiTest, VastModuleParameterTypes) {
  const std::string_view kWantEmitted = R"(module param_types;
  parameter logic [15:0] P = 5;
  parameter integer I = -1;
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "param_types");
  ASSERT_NE(m, nullptr);

  // parameter logic [15:0] P = 5;
  xls_vast_data_type* u16 =
      xls_vast_verilog_file_make_bit_vector_type(f, 16, /*is_signed=*/false);
  xls_vast_def* p_def =
      xls_vast_verilog_file_make_def(f, "P", xls_vast_data_kind_logic, u16);
  ASSERT_NE(p_def, nullptr);
  xls_vast_literal* lit5 = xls_vast_verilog_file_make_plain_literal(f, 5);
  ASSERT_NE(lit5, nullptr);
  auto* p_ref = xls_vast_verilog_module_add_parameter_with_def(
      m, p_def, xls_vast_literal_as_expression(lit5));
  ASSERT_NE(p_ref, nullptr);

  // parameter integer I = -1;
  xls_vast_data_type* integer_type =
      xls_vast_verilog_file_make_integer_type(f, /*is_signed=*/true);
  ASSERT_NE(integer_type, nullptr);
  xls_vast_def* i_def = xls_vast_verilog_file_make_def(
      f, "I", xls_vast_data_kind_integer, integer_type);
  ASSERT_NE(i_def, nullptr);
  xls_vast_literal* lit1 = xls_vast_verilog_file_make_plain_literal(f, 1);
  ASSERT_NE(lit1, nullptr);
  xls_vast_expression* neg1 = xls_vast_verilog_file_make_unary(
      f, xls_vast_literal_as_expression(lit1), xls_vast_operator_kind_negate);
  ASSERT_NE(neg1, nullptr);
  auto* i_ref = xls_vast_verilog_module_add_parameter_with_def(m, i_def, neg1);
  ASSERT_NE(i_ref, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastArrayParametersWithDefAndAssignmentPattern) {
  const std::string_view kWantEmitted = R"(module top;
  parameter P0[2] = '{'0, '0};
  parameter int P1[3] = '{1, 2, 3};
  parameter logic [7:0] P2[2] = '{8'h42, 8'h43};
  parameter integer P3[1][4] = '{'{1, 2, 3, 4}};
  parameter int P4[2][3] = '{'{1, 2, 3}, '{4, 5, 6}};
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_system_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  auto make_assignment_pattern =
      [&](std::vector<xls_vast_expression*> elements) -> xls_vast_expression* {
    return xls_vast_verilog_file_make_array_assignment_pattern(
        f, elements.data(), elements.size());
  };

  // Helper: make an N-bit hex literal like 8'h42.
  auto make_hex_literal = [&](int64_t bit_count,
                              uint64_t value) -> xls_vast_expression* {
    xls_bits* bits = nullptr;
    char* error = nullptr;
    absl::Cleanup free_error([&] { xls_c_str_free(error); });
    EXPECT_TRUE(xls_bits_make_ubits(bit_count, value, &error, &bits))
        << (error ? error : "");
    absl::Cleanup free_bits([&] { xls_bits_free(bits); });
    xls_vast_literal* lit = nullptr;
    EXPECT_TRUE(xls_vast_verilog_file_make_literal(
        f, bits, xls_format_preference_hex,
        /*emit_bit_count=*/true, &error, &lit))
        << (error ? error : "");
    EXPECT_NE(lit, nullptr);
    return xls_vast_literal_as_expression(lit);
  };

  // P0: parameter P0[2] = '{'0, '0};
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar, nullptr);
  const int64_t p0_dims[] = {2};
  xls_vast_data_type* p0_type =
      xls_vast_verilog_file_make_unpacked_array_type(f, scalar, p0_dims, 1);
  ASSERT_NE(p0_type, nullptr);
  xls_vast_def* p0_def =
      xls_vast_verilog_file_make_def(f, "P0", xls_vast_data_kind_user, p0_type);
  ASSERT_NE(p0_def, nullptr);
  xls_vast_expression* tick0 =
      xls_vast_verilog_file_make_unsized_zero_literal(f);
  ASSERT_NE(tick0, nullptr);
  xls_vast_expression* p0_rhs = make_assignment_pattern({tick0, tick0});
  ASSERT_NE(p0_rhs, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter_with_def(m, p0_def, p0_rhs),
            nullptr);

  // P1: parameter int P1[3] = '{1, 2, 3};
  const int64_t p1_dims[] = {3};
  xls_vast_data_type* p1_type =
      xls_vast_verilog_file_make_unpacked_array_type(f, scalar, p1_dims, 1);
  ASSERT_NE(p1_type, nullptr);
  xls_vast_def* p1_def =
      xls_vast_verilog_file_make_def(f, "P1", xls_vast_data_kind_int, p1_type);
  ASSERT_NE(p1_def, nullptr);
  xls_vast_literal* one = xls_vast_verilog_file_make_plain_literal(f, 1);
  xls_vast_literal* two = xls_vast_verilog_file_make_plain_literal(f, 2);
  xls_vast_literal* three = xls_vast_verilog_file_make_plain_literal(f, 3);
  xls_vast_literal* four = xls_vast_verilog_file_make_plain_literal(f, 4);
  xls_vast_literal* five = xls_vast_verilog_file_make_plain_literal(f, 5);
  xls_vast_literal* six = xls_vast_verilog_file_make_plain_literal(f, 6);
  ASSERT_NE(one, nullptr);
  ASSERT_NE(two, nullptr);
  ASSERT_NE(three, nullptr);
  ASSERT_NE(four, nullptr);
  ASSERT_NE(five, nullptr);
  ASSERT_NE(six, nullptr);
  xls_vast_expression* p1_rhs = make_assignment_pattern(
      {xls_vast_literal_as_expression(one), xls_vast_literal_as_expression(two),
       xls_vast_literal_as_expression(three)});
  ASSERT_NE(p1_rhs, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter_with_def(m, p1_def, p1_rhs),
            nullptr);

  // P2: parameter logic [7:0] P2[2] = '{8'h42, 8'h43};
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  ASSERT_NE(u8, nullptr);
  const int64_t p2_dims[] = {2};
  xls_vast_data_type* p2_type =
      xls_vast_verilog_file_make_unpacked_array_type(f, u8, p2_dims, 1);
  ASSERT_NE(p2_type, nullptr);
  xls_vast_def* p2_def = xls_vast_verilog_file_make_def(
      f, "P2", xls_vast_data_kind_logic, p2_type);
  ASSERT_NE(p2_def, nullptr);
  xls_vast_expression* p2_rhs = make_assignment_pattern(
      {make_hex_literal(8, 0x42), make_hex_literal(8, 0x43)});
  ASSERT_NE(p2_rhs, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter_with_def(m, p2_def, p2_rhs),
            nullptr);

  // P3: parameter integer P3[1][4] = '{'{1, 2, 3, 4}};
  const int64_t p3_dims[] = {1, 4};
  xls_vast_data_type* p3_type =
      xls_vast_verilog_file_make_unpacked_array_type(f, scalar, p3_dims, 2);
  ASSERT_NE(p3_type, nullptr);
  xls_vast_def* p3_def = xls_vast_verilog_file_make_def(
      f, "P3", xls_vast_data_kind_integer, p3_type);
  ASSERT_NE(p3_def, nullptr);
  xls_vast_expression* inner_1234 = make_assignment_pattern(
      {xls_vast_literal_as_expression(one), xls_vast_literal_as_expression(two),
       xls_vast_literal_as_expression(three),
       xls_vast_literal_as_expression(four)});
  ASSERT_NE(inner_1234, nullptr);
  xls_vast_expression* p3_rhs = make_assignment_pattern({inner_1234});
  ASSERT_NE(p3_rhs, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter_with_def(m, p3_def, p3_rhs),
            nullptr);

  // P4: parameter int P4[2][3] = '{'{1, 2, 3}, '{4, 5, 6}};
  const int64_t p4_dims[] = {2, 3};
  xls_vast_data_type* p4_type =
      xls_vast_verilog_file_make_unpacked_array_type(f, scalar, p4_dims, 2);
  ASSERT_NE(p4_type, nullptr);
  xls_vast_def* p4_def =
      xls_vast_verilog_file_make_def(f, "P4", xls_vast_data_kind_int, p4_type);
  ASSERT_NE(p4_def, nullptr);
  xls_vast_expression* row_123 = make_assignment_pattern(
      {xls_vast_literal_as_expression(one), xls_vast_literal_as_expression(two),
       xls_vast_literal_as_expression(three)});
  xls_vast_expression* row_456 =
      make_assignment_pattern({xls_vast_literal_as_expression(four),
                               xls_vast_literal_as_expression(five),
                               xls_vast_literal_as_expression(six)});
  ASSERT_NE(row_123, nullptr);
  ASSERT_NE(row_456, nullptr);
  xls_vast_expression* p4_rhs = make_assignment_pattern({row_123, row_456});
  ASSERT_NE(p4_rhs, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_parameter_with_def(m, p4_def, p4_rhs),
            nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastIndexUnpackedArrayParameter) {
  const std::string_view kWantEmitted = R"(module top;
  parameter P0[2] = '{'0, '1};
  wire w;
  assign w = P0[0];
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_system_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  // parameter P0[2] = '{'0, '1};
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar, nullptr);
  const int64_t p0_dims[] = {2};
  xls_vast_data_type* p0_type =
      xls_vast_verilog_file_make_unpacked_array_type(f, scalar, p0_dims, 1);
  ASSERT_NE(p0_type, nullptr);
  xls_vast_def* p0_def =
      xls_vast_verilog_file_make_def(f, "P0", xls_vast_data_kind_user, p0_type);
  ASSERT_NE(p0_def, nullptr);
  xls_vast_expression* tick0 =
      xls_vast_verilog_file_make_unsized_zero_literal(f);
  xls_vast_expression* tick1 =
      xls_vast_verilog_file_make_unsized_one_literal(f);
  ASSERT_NE(tick0, nullptr);
  ASSERT_NE(tick1, nullptr);
  std::vector<xls_vast_expression*> p0_elems = {tick0, tick1};
  xls_vast_expression* p0_rhs =
      xls_vast_verilog_file_make_array_assignment_pattern(f, p0_elems.data(),
                                                          p0_elems.size());
  ASSERT_NE(p0_rhs, nullptr);
  xls_vast_parameter_ref* p0_ref =
      xls_vast_verilog_module_add_parameter_with_def(m, p0_def, p0_rhs);
  ASSERT_NE(p0_ref, nullptr);

  // wire w;
  xls_vast_logic_ref* w = xls_vast_verilog_module_add_wire(m, "w", scalar);
  ASSERT_NE(w, nullptr);

  // assign w = P0[0];
  xls_vast_indexable_expression* p0_indexable =
      xls_vast_parameter_ref_as_indexable_expression(p0_ref);
  ASSERT_NE(p0_indexable, nullptr);
  xls_vast_index* p0_0 =
      xls_vast_verilog_file_make_index_i64(f, p0_indexable, 0);
  ASSERT_NE(p0_0, nullptr);
  xls_vast_expression* rhs = xls_vast_index_as_expression(p0_0);
  ASSERT_NE(rhs, nullptr);
  xls_vast_continuous_assignment* assign =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(w), rhs);
  ASSERT_NE(assign, nullptr);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assign);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastLocalparamKinds) {
  const std::string_view kWantEmitted = R"(module top;
  localparam Foo = 42;
  localparam int Baz = 42;
  localparam logic [7:0] Bar = 8'h42;
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_system_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  // localparam Foo = 42;
  ASSERT_NE(xls_vast_verilog_module_add_localparam(
                m, "Foo",
                xls_vast_literal_as_expression(
                    xls_vast_verilog_file_make_plain_literal(f, 42))),
            nullptr);

  // localparam int Baz = 42;
  xls_vast_def* baz_def =
      xls_vast_verilog_file_make_int_def(f, "Baz", /*is_signed=*/true);
  ASSERT_NE(baz_def, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_localparam_with_def(
                m, baz_def,
                xls_vast_literal_as_expression(
                    xls_vast_verilog_file_make_plain_literal(f, 42))),
            nullptr);

  // localparam logic [7:0] Bar = 8'h42;
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  xls_vast_def* bar_def =
      xls_vast_verilog_file_make_def(f, "Bar", xls_vast_data_kind_logic, u8);
  ASSERT_NE(bar_def, nullptr);

  // Build 8'h42 literal via bits API to ensure hex with bit count.
  struct xls_bits* bits = nullptr;
  char* error_out = nullptr;
  ASSERT_TRUE(
      xls_bits_make_ubits(/*bit_count=*/8, /*value=*/0x42, &error_out, &bits));
  ASSERT_EQ(error_out, nullptr);
  absl::Cleanup free_bits([&] { xls_bits_free(bits); });

  xls_vast_literal* lit = nullptr;
  ASSERT_TRUE(xls_vast_verilog_file_make_literal(
      f, bits, xls_format_preference_hex, /*emit_bit_count=*/true, &error_out,
      &lit));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(lit, nullptr);

  ASSERT_NE(xls_vast_verilog_module_add_localparam_with_def(
                m, bar_def, xls_vast_literal_as_expression(lit)),
            nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastModuleScopeConditional) {
  const std::string_view kWantEmitted = R"(module top;
  parameter A = 1;
  parameter B = 2;
  wire out;
  if (A == B) begin
    assign out = 1'h1;
  end else begin
    assign out = 1'h0;
  end
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_system_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  // parameter A = 1; parameter B = 2;
  xls_vast_parameter_ref* A = xls_vast_verilog_module_add_parameter(
      m, "A",
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 1)));
  xls_vast_parameter_ref* B = xls_vast_verilog_module_add_parameter(
      m, "B",
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 2)));
  ASSERT_NE(A, nullptr);
  ASSERT_NE(B, nullptr);

  // wire out;
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar, nullptr);
  xls_vast_logic_ref* out = xls_vast_verilog_module_add_wire(m, "out", scalar);
  ASSERT_NE(out, nullptr);

  auto make_ubits_literal = [&](int64_t bit_count,
                                uint64_t value) -> xls_vast_expression* {
    xls_bits* bits = nullptr;
    char* error = nullptr;
    absl::Cleanup free_error([&] { xls_c_str_free(error); });
    EXPECT_TRUE(xls_bits_make_ubits(bit_count, value, &error, &bits))
        << (error ? error : "");
    absl::Cleanup free_bits([&] { xls_bits_free(bits); });
    xls_vast_literal* lit = nullptr;
    EXPECT_TRUE(xls_vast_verilog_file_make_literal(
        f, bits, xls_format_preference_hex, /*emit_bit_count=*/true, &error,
        &lit))
        << (error ? error : "");
    EXPECT_NE(lit, nullptr);
    return xls_vast_literal_as_expression(lit);
  };

  // if (A == B) ...
  xls_vast_expression* eq = xls_vast_verilog_file_make_binary(
      f, xls_vast_parameter_ref_as_expression(A),
      xls_vast_parameter_ref_as_expression(B), xls_vast_operator_kind_eq);
  ASSERT_NE(eq, nullptr);
  xls_vast_conditional* cond = xls_vast_verilog_module_add_conditional(m, eq);
  ASSERT_NE(cond, nullptr);

  xls_vast_statement_block* then_block =
      xls_vast_conditional_get_then_block(cond);
  ASSERT_NE(then_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_continuous_assignment(
                then_block, xls_vast_logic_ref_as_expression(out),
                make_ubits_literal(/*bit_count=*/1, /*value=*/1)),
            nullptr);

  xls_vast_statement_block* else_block = xls_vast_conditional_add_else(cond);
  ASSERT_NE(else_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_continuous_assignment(
                else_block, xls_vast_logic_ref_as_expression(out),
                make_ubits_literal(/*bit_count=*/1, /*value=*/0)),
            nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastGenerateLoopConditional) {
  const std::string_view kWantEmitted = R"(module top;
  wire [2:0] out;
  for (genvar i = 0; i < 3; i = i + 1) begin : g
    if (i == 0) begin
      assign out[i] = 1'h0;
    end else if (i == 1) begin
      assign out[i] = 1'h1;
    end else begin
      assign out[i] = 'X;
    end
  end
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_system_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* u3 =
      xls_vast_verilog_file_make_bit_vector_type(f, 3, /*is_signed=*/false);
  xls_vast_logic_ref* out = xls_vast_verilog_module_add_wire(m, "out", u3);
  ASSERT_NE(out, nullptr);

  xls_vast_generate_loop* loop = xls_vast_verilog_module_add_generate_loop(
      m, "i",
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 0)),
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 3)),
      "g");
  ASSERT_NE(loop, nullptr);

  xls_vast_logic_ref* i_lr = xls_vast_generate_loop_get_genvar(loop);
  ASSERT_NE(i_lr, nullptr);

  // Helper to form out[i] (as expression).
  xls_vast_indexable_expression* out_idxable =
      xls_vast_logic_ref_as_indexable_expression(out);
  ASSERT_NE(out_idxable, nullptr);
  xls_vast_index* out_i = xls_vast_verilog_file_make_index(
      f, out_idxable, xls_vast_logic_ref_as_expression(i_lr));
  ASSERT_NE(out_i, nullptr);
  xls_vast_expression* out_i_expr = xls_vast_index_as_expression(out_i);
  ASSERT_NE(out_i_expr, nullptr);

  auto make_ubits_literal = [&](int64_t bit_count,
                                uint64_t value) -> xls_vast_expression* {
    xls_bits* bits = nullptr;
    char* error = nullptr;
    absl::Cleanup free_error([&] { xls_c_str_free(error); });
    EXPECT_TRUE(xls_bits_make_ubits(bit_count, value, &error, &bits))
        << (error ? error : "");
    absl::Cleanup free_bits([&] { xls_bits_free(bits); });
    xls_vast_literal* lit = nullptr;
    EXPECT_TRUE(xls_vast_verilog_file_make_literal(
        f, bits, xls_format_preference_hex, /*emit_bit_count=*/true, &error,
        &lit))
        << (error ? error : "");
    EXPECT_NE(lit, nullptr);
    return xls_vast_literal_as_expression(lit);
  };

  // if (i == 0) ...
  xls_vast_expression* i_eq_0 = xls_vast_verilog_file_make_binary(
      f, xls_vast_logic_ref_as_expression(i_lr),
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 0)),
      xls_vast_operator_kind_eq);
  ASSERT_NE(i_eq_0, nullptr);
  xls_vast_conditional* cond =
      xls_vast_generate_loop_add_conditional(loop, i_eq_0);
  ASSERT_NE(cond, nullptr);

  xls_vast_statement_block* then_block =
      xls_vast_conditional_get_then_block(cond);
  ASSERT_NE(then_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_continuous_assignment(
                then_block, out_i_expr,
                make_ubits_literal(/*bit_count=*/1, /*value=*/0)),
            nullptr);

  xls_vast_expression* i_eq_1 = xls_vast_verilog_file_make_binary(
      f, xls_vast_logic_ref_as_expression(i_lr),
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 1)),
      xls_vast_operator_kind_eq);
  xls_vast_statement_block* else_if_block =
      xls_vast_conditional_add_else_if(cond, i_eq_1);
  ASSERT_NE(else_if_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_continuous_assignment(
                else_if_block, out_i_expr,
                make_ubits_literal(/*bit_count=*/1, /*value=*/1)),
            nullptr);

  xls_vast_statement_block* else_block = xls_vast_conditional_add_else(cond);
  ASSERT_NE(else_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_continuous_assignment(
                else_block, out_i_expr,
                xls_vast_verilog_file_make_unsized_x_literal(f)),
            nullptr);

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

TEST(XlsCApiTest, VastAlwaysFfReg) {
  xls_vast_verilog_file* f = xls_vast_make_verilog_file(
      xls_vast_file_type_system_verilog);  // Use SystemVerilog for always_ff
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "test_module");
  ASSERT_NE(m, nullptr);

  // Data types
  xls_vast_data_type* scalar_type = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar_type, nullptr);

  // Input ports
  xls_vast_logic_ref* clk_ref =
      xls_vast_verilog_module_add_input(m, "clk", scalar_type);
  ASSERT_NE(clk_ref, nullptr);
  xls_vast_logic_ref* pred_ref =
      xls_vast_verilog_module_add_input(m, "pred", scalar_type);
  ASSERT_NE(pred_ref, nullptr);
  xls_vast_logic_ref* x_ref =
      xls_vast_verilog_module_add_input(m, "x", scalar_type);
  ASSERT_NE(x_ref, nullptr);

  // Output port
  xls_vast_logic_ref* out_ref __attribute__((unused)) =
      xls_vast_verilog_module_add_output(m, "out", scalar_type);
  ASSERT_NE(out_ref, nullptr);

  // Registers
  xls_vast_logic_ref* p0_pred_reg_ref = nullptr;
  char* error_out = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_reg(m, "p0_pred", scalar_type,
                                              &p0_pred_reg_ref, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(p0_pred_reg_ref, nullptr);

  xls_vast_logic_ref* p0_x_reg_ref = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_reg(m, "p0_x", scalar_type,
                                              &p0_x_reg_ref, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(p0_x_reg_ref, nullptr);

  // always_ff block
  xls_vast_expression* clk_expr = xls_vast_logic_ref_as_expression(clk_ref);
  ASSERT_NE(clk_expr, nullptr);
  xls_vast_expression* posedge_clk_expr =
      xls_vast_verilog_file_make_pos_edge(f, clk_expr);
  ASSERT_NE(posedge_clk_expr, nullptr);

  xls_vast_expression* sensitivity_list[] = {posedge_clk_expr};
  xls_vast_always_base* always_ff_block = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_always_ff(
      m, sensitivity_list, 1, &always_ff_block, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(always_ff_block, nullptr);

  xls_vast_statement_block* stmt_block =
      xls_vast_always_base_get_statement_block(always_ff_block);
  ASSERT_NE(stmt_block, nullptr);

  // Non-blocking assignments
  xls_vast_expression* p0_pred_reg_expr =
      xls_vast_logic_ref_as_expression(p0_pred_reg_ref);
  ASSERT_NE(p0_pred_reg_expr, nullptr);
  xls_vast_expression* pred_expr = xls_vast_logic_ref_as_expression(pred_ref);
  ASSERT_NE(pred_expr, nullptr);
  xls_vast_statement* assign_p0_pred =
      xls_vast_statement_block_add_nonblocking_assignment(
          stmt_block, p0_pred_reg_expr, pred_expr);
  ASSERT_NE(assign_p0_pred, nullptr);

  xls_vast_expression* p0_x_reg_expr =
      xls_vast_logic_ref_as_expression(p0_x_reg_ref);
  ASSERT_NE(p0_x_reg_expr, nullptr);
  xls_vast_expression* x_expr = xls_vast_logic_ref_as_expression(x_ref);
  ASSERT_NE(x_expr, nullptr);
  xls_vast_statement* assign_p0_x =
      xls_vast_statement_block_add_nonblocking_assignment(
          stmt_block, p0_x_reg_expr, x_expr);
  ASSERT_NE(assign_p0_x, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });

  const std::string_view kWant = R"(module test_module(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  reg p0_pred;
  reg p0_x;
  always_ff @ (posedge clk) begin
    p0_pred <= pred;
    p0_x <= x;
  end
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

// Parameterized test for sequential block generation (always_ff vs always @)
class VastSequentialBlockTest : public testing::TestWithParam<bool> {
 protected:
  bool use_system_verilog() const { return GetParam(); }
};

TEST_P(VastSequentialBlockTest, GenerateSequentialLogic) {
  xls_vast_verilog_file* f = xls_vast_make_verilog_file(
      use_system_verilog() ? xls_vast_file_type_system_verilog
                           : xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "test_module");
  ASSERT_NE(m, nullptr);

  // Data types
  xls_vast_data_type* scalar_type = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar_type, nullptr);

  // Input ports
  xls_vast_logic_ref* clk_ref =
      xls_vast_verilog_module_add_input(m, "clk", scalar_type);
  ASSERT_NE(clk_ref, nullptr);
  xls_vast_logic_ref* pred_ref =
      xls_vast_verilog_module_add_input(m, "pred", scalar_type);
  ASSERT_NE(pred_ref, nullptr);
  xls_vast_logic_ref* x_ref =
      xls_vast_verilog_module_add_input(m, "x", scalar_type);
  ASSERT_NE(x_ref, nullptr);

  // Output port
  xls_vast_logic_ref* out_ref __attribute__((unused)) =
      xls_vast_verilog_module_add_output(m, "out", scalar_type);
  ASSERT_NE(out_ref, nullptr);

  // Registers
  xls_vast_logic_ref* p0_pred_reg_ref = nullptr;
  char* error_out = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_reg(m, "p0_pred", scalar_type,
                                              &p0_pred_reg_ref, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(p0_pred_reg_ref, nullptr);

  xls_vast_logic_ref* p0_x_reg_ref = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_reg(m, "p0_x", scalar_type,
                                              &p0_x_reg_ref, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(p0_x_reg_ref, nullptr);

  // always_ff block
  xls_vast_expression* clk_expr = xls_vast_logic_ref_as_expression(clk_ref);
  ASSERT_NE(clk_expr, nullptr);
  xls_vast_expression* posedge_clk_expr =
      xls_vast_verilog_file_make_pos_edge(f, clk_expr);
  ASSERT_NE(posedge_clk_expr, nullptr);

  xls_vast_expression* sensitivity_list[] = {posedge_clk_expr};
  xls_vast_always_base* always_block = nullptr;
  if (use_system_verilog()) {
    ASSERT_TRUE(xls_vast_verilog_module_add_always_ff(
        m, sensitivity_list, 1, &always_block, &error_out));
  } else {
    // Call xls_vast_verilog_module_add_always_at for Verilog-2001 style
    ASSERT_TRUE(xls_vast_verilog_module_add_always_at(
        m, sensitivity_list, 1, &always_block, &error_out));
  }
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(always_block, nullptr);

  xls_vast_statement_block* stmt_block =
      xls_vast_always_base_get_statement_block(always_block);
  ASSERT_NE(stmt_block, nullptr);

  // Non-blocking assignments
  xls_vast_expression* p0_pred_reg_expr =
      xls_vast_logic_ref_as_expression(p0_pred_reg_ref);
  ASSERT_NE(p0_pred_reg_expr, nullptr);
  xls_vast_expression* pred_expr = xls_vast_logic_ref_as_expression(pred_ref);
  ASSERT_NE(pred_expr, nullptr);
  xls_vast_statement* assign_p0_pred =
      xls_vast_statement_block_add_nonblocking_assignment(
          stmt_block, p0_pred_reg_expr, pred_expr);
  ASSERT_NE(assign_p0_pred, nullptr);

  xls_vast_expression* p0_x_reg_expr =
      xls_vast_logic_ref_as_expression(p0_x_reg_ref);
  ASSERT_NE(p0_x_reg_expr, nullptr);
  xls_vast_expression* x_expr = xls_vast_logic_ref_as_expression(x_ref);
  ASSERT_NE(x_expr, nullptr);
  xls_vast_statement* assign_p0_x =
      xls_vast_statement_block_add_nonblocking_assignment(
          stmt_block, p0_x_reg_expr, x_expr);
  ASSERT_NE(assign_p0_x, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });

  std::string kWant_string;
  if (use_system_verilog()) {
    kWant_string = R"(module test_module(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  reg p0_pred;
  reg p0_x;
  always_ff @ (posedge clk) begin
    p0_pred <= pred;
    p0_x <= x;
  end
endmodule
)";
  } else {
    kWant_string = R"(module test_module(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  reg p0_pred;
  reg p0_x;
  always @ (posedge clk) begin
    p0_pred <= pred;
    p0_x <= x;
  end
endmodule
)";
  }
  std::string_view kWant = kWant_string;
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

INSTANTIATE_TEST_SUITE_P(VastSequentialBlockTestSuite, VastSequentialBlockTest,
                         testing::Bool());

TEST(XlsCApiTest, VastAlwaysComb) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_system_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "test_module");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* scalar_type = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar_type, nullptr);
  xls_vast_logic_ref* a_ref =
      xls_vast_verilog_module_add_logic_input(m, "a", scalar_type);
  ASSERT_NE(a_ref, nullptr);
  xls_vast_logic_ref* b_ref =
      xls_vast_verilog_module_add_logic_output(m, "b", scalar_type);
  ASSERT_NE(b_ref, nullptr);

  char* error_out = nullptr;
  xls_vast_data_type* u32 =
      xls_vast_verilog_file_make_bit_vector_type(f, 32, /*is_signed=*/false);
  ASSERT_NE(u32, nullptr);
  xls_vast_logic_ref* c_ref =
      xls_vast_verilog_module_add_logic_input(m, "c", u32);
  ASSERT_NE(c_ref, nullptr);
  xls_vast_logic_ref* tmp_ref = nullptr;
  ASSERT_TRUE(
      xls_vast_verilog_module_add_logic(m, "tmp", u32, &tmp_ref, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(tmp_ref, nullptr);

  xls_vast_always_base* always_comb_block = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_always_comb(m, &always_comb_block,
                                                      &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(always_comb_block, nullptr);

  xls_vast_statement_block* stmt_block =
      xls_vast_always_base_get_statement_block(always_comb_block);
  ASSERT_NE(stmt_block, nullptr);
  xls_vast_expression* a_expr = xls_vast_logic_ref_as_expression(a_ref);
  ASSERT_NE(a_expr, nullptr);
  xls_vast_expression* b_expr = xls_vast_logic_ref_as_expression(b_ref);
  ASSERT_NE(b_expr, nullptr);
  xls_vast_expression* c_expr = xls_vast_logic_ref_as_expression(c_ref);
  ASSERT_NE(c_expr, nullptr);
  xls_vast_expression* tmp_expr = xls_vast_logic_ref_as_expression(tmp_ref);
  ASSERT_NE(tmp_expr, nullptr);
  xls_vast_statement* assign_stmt =
      xls_vast_statement_block_add_blocking_assignment(stmt_block, a_expr,
                                                       b_expr);
  ASSERT_NE(assign_stmt, nullptr);
  xls_vast_statement* assign_tmp_to_c =
      xls_vast_statement_block_add_blocking_assignment(stmt_block, tmp_expr,
                                                       c_expr);
  ASSERT_NE(assign_tmp_to_c, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });

  const std::string_view kWant = R"(module test_module(
  input logic a,
  output logic b,
  input logic [31:0] c
);
  logic [31:0] tmp;
  always_comb begin
    a = b;
    tmp = c;
  end
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

// Tests enumeration and inspection of module ports.
TEST(XlsCApiTest, VastModulePortEnumerationAndInspection) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "my_port_enum_module");
  ASSERT_NE(m, nullptr);

  // Create a scalar type once and reuse it for ports.
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);

  // Add one input and one output port.
  ASSERT_NE(xls_vast_verilog_module_add_input(m, "in_sig", scalar), nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_output(m, "out_sig", scalar), nullptr);

  // Enumerate ports.
  size_t port_count = 0;
  xls_vast_module_port** ports =
      xls_vast_verilog_module_get_ports(m, &port_count);
  ASSERT_EQ(port_count, 2);
  ASSERT_NE(ports, nullptr);
  absl::Cleanup free_ports(
      [&] { xls_vast_verilog_module_free_ports(ports, port_count); });

  // Verify the first port (input).
  {
    xls_vast_module_port* port = ports[0];
    EXPECT_EQ(xls_vast_verilog_module_port_get_direction(port),
              xls_vast_module_port_direction_input);
    xls_vast_def* def = xls_vast_verilog_module_port_get_def(port);
    ASSERT_NE(def, nullptr);
    char* def_name = xls_vast_def_get_name(def);
    ASSERT_NE(def_name, nullptr);
    absl::Cleanup free_name([&] { xls_c_str_free(def_name); });
    EXPECT_EQ(std::string_view{def_name}, "in_sig");
    EXPECT_EQ(xls_vast_def_get_data_type(def), scalar);
  }

  // Verify the second port (output).
  {
    xls_vast_module_port* port = ports[1];
    EXPECT_EQ(xls_vast_verilog_module_port_get_direction(port),
              xls_vast_module_port_direction_output);
    xls_vast_def* def = xls_vast_verilog_module_port_get_def(port);
    ASSERT_NE(def, nullptr);
    char* def_name = xls_vast_def_get_name(def);
    ASSERT_NE(def_name, nullptr);
    absl::Cleanup free_name([&] { xls_c_str_free(def_name); });
    EXPECT_EQ(std::string_view{def_name}, "out_sig");
    EXPECT_EQ(xls_vast_def_get_data_type(def), scalar);
  }
}

// Tests accessor helpers for DataType and Def.
TEST(XlsCApiTest, VastDataTypeAccessors) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  // Unsigned 8-bit vector type.
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type(f, 8, /*is_signed=*/false);
  int64_t width = 0;
  char* error_out = nullptr;
  ASSERT_TRUE(xls_vast_data_type_width_as_int64(u8, &width, &error_out));
  ASSERT_EQ(error_out, nullptr);
  EXPECT_EQ(width, 8);

  // Width expression should not be null for bit vectors created via literal
  // width.
  EXPECT_NE(xls_vast_data_type_width(u8), nullptr);
  EXPECT_FALSE(xls_vast_data_type_is_signed(u8));

  // Signed 4-bit vector.
  xls_vast_data_type* s4 =
      xls_vast_verilog_file_make_bit_vector_type(f, 4, /*is_signed=*/true);
  EXPECT_TRUE(xls_vast_data_type_is_signed(s4));

  // Packed array: u8[3][2] => flat bit count 8*3*2 = 48.
  const std::vector<int64_t> dims = {3, 2};
  xls_vast_data_type* packed_arr = xls_vast_verilog_file_make_packed_array_type(
      f, u8, dims.data(), dims.size());
  int64_t flat_bits = 0;
  ASSERT_TRUE(xls_vast_data_type_flat_bit_count_as_int64(packed_arr, &flat_bits,
                                                         &error_out));
  ASSERT_EQ(error_out, nullptr);
  EXPECT_EQ(flat_bits, 48);

  // Scalar type should have width 1 and no width expression.
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_TRUE(xls_vast_data_type_width_as_int64(scalar, &width, &error_out));
  EXPECT_EQ(width, 1);
  EXPECT_EQ(xls_vast_data_type_width(scalar), nullptr);

  // Create module with an input of the packed array type; test Def helpers.
  xls_vast_verilog_module* m =
      xls_vast_verilog_file_add_module(f, "dtype_module");
  ASSERT_NE(m, nullptr);
  ASSERT_NE(xls_vast_verilog_module_add_input(m, "arr", packed_arr), nullptr);

  size_t port_count = 0;
  xls_vast_module_port** ports =
      xls_vast_verilog_module_get_ports(m, &port_count);
  ASSERT_EQ(port_count, 1);
  ASSERT_NE(ports, nullptr);
  absl::Cleanup free_ports(
      [&] { xls_vast_verilog_module_free_ports(ports, port_count); });

  xls_vast_def* def = xls_vast_verilog_module_port_get_def(ports[0]);
  ASSERT_NE(def, nullptr);
  // Obtain width via def->data_type path.
  xls_vast_data_type* def_type = xls_vast_def_get_data_type(def);
  ASSERT_NE(def_type, nullptr);
  int64_t def_width = 0;
  ASSERT_TRUE(xls_vast_data_type_flat_bit_count_as_int64(def_type, &def_width,
                                                         &error_out));
  ASSERT_EQ(error_out, nullptr);
  EXPECT_EQ(def_width, flat_bits);  // Should match packed array flat width.

  char* def_name = xls_vast_def_get_name(def);
  ASSERT_NE(def_name, nullptr);
  absl::Cleanup free_name([&] { xls_c_str_free(def_name); });
  EXPECT_EQ(std::string_view{def_name}, "arr");
}

TEST(XlsCApiTest, VastProceduralControlAndBlocking) {
  const std::string_view kWantEmitted = R"XLS(module pc_block(
  input wire clk,
  input wire pred1,
  input wire pred2,
  input wire sel,
  input wire a_in,
  input wire b_in
);
  reg a;
  reg b;
  always @ (posedge clk) begin
    a = b;
    if (pred1) begin
      a = 1;
    end else if (pred1 & pred2) begin
      a = 0;
    end else begin
      a = b;
    end
    case (sel)
      1: begin
        a = b;
      end
      default: begin
        a = 0;
      end
    endcase
    b <= a_in;
  end
endmodule
)XLS";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "pc_block");
  ASSERT_NE(m, nullptr);

  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  ASSERT_NE(scalar, nullptr);

  xls_vast_logic_ref* clk = xls_vast_verilog_module_add_input(m, "clk", scalar);
  xls_vast_logic_ref* pred1 =
      xls_vast_verilog_module_add_input(m, "pred1", scalar);
  xls_vast_logic_ref* pred2 =
      xls_vast_verilog_module_add_input(m, "pred2", scalar);
  xls_vast_logic_ref* sel = xls_vast_verilog_module_add_input(m, "sel", scalar);
  xls_vast_logic_ref* a_in =
      xls_vast_verilog_module_add_input(m, "a_in", scalar);
  xls_vast_logic_ref* b_in =
      xls_vast_verilog_module_add_input(m, "b_in", scalar);
  ASSERT_NE(clk, nullptr);
  ASSERT_NE(pred1, nullptr);
  ASSERT_NE(pred2, nullptr);
  ASSERT_NE(sel, nullptr);
  ASSERT_NE(a_in, nullptr);
  ASSERT_NE(b_in, nullptr);

  xls_vast_logic_ref* a_reg = nullptr;
  char* error_out = nullptr;
  ASSERT_TRUE(
      xls_vast_verilog_module_add_reg(m, "a", scalar, &a_reg, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(a_reg, nullptr);
  xls_vast_logic_ref* b_reg = nullptr;
  ASSERT_TRUE(
      xls_vast_verilog_module_add_reg(m, "b", scalar, &b_reg, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(b_reg, nullptr);

  xls_vast_expression* clk_expr = xls_vast_logic_ref_as_expression(clk);
  xls_vast_expression* pos_clk =
      xls_vast_verilog_file_make_pos_edge(f, clk_expr);
  xls_vast_expression* sens_list[] = {pos_clk};
  xls_vast_always_base* always_block = nullptr;
  ASSERT_TRUE(xls_vast_verilog_module_add_always_at(m, sens_list, 1,
                                                    &always_block, &error_out));
  ASSERT_EQ(error_out, nullptr);
  ASSERT_NE(always_block, nullptr);

  xls_vast_statement_block* block =
      xls_vast_always_base_get_statement_block(always_block);
  ASSERT_NE(block, nullptr);

  // blocking: a = b;
  xls_vast_expression* a_expr = xls_vast_logic_ref_as_expression(a_reg);
  xls_vast_expression* b_expr = xls_vast_logic_ref_as_expression(b_reg);
  ASSERT_NE(a_expr, nullptr);
  ASSERT_NE(b_expr, nullptr);
  ASSERT_NE(
      xls_vast_statement_block_add_blocking_assignment(block, a_expr, b_expr),
      nullptr);

  // if (pred1) { a = 1; } else if (pred1 & pred2) { a = 0; } else { a = b; }
  xls_vast_conditional* cond = xls_vast_statement_block_add_conditional(
      block, xls_vast_logic_ref_as_expression(pred1));
  ASSERT_NE(cond, nullptr);
  xls_vast_statement_block* then_block =
      xls_vast_conditional_get_then_block(cond);
  ASSERT_NE(then_block, nullptr);
  xls_vast_statement* then_assign =
      xls_vast_statement_block_add_blocking_assignment(
          then_block, a_expr,
          xls_vast_literal_as_expression(
              xls_vast_verilog_file_make_plain_literal(f, 1)));
  ASSERT_NE(then_assign, nullptr);

  xls_vast_expression* and_expr = xls_vast_verilog_file_make_binary(
      f, xls_vast_logic_ref_as_expression(pred1),
      xls_vast_logic_ref_as_expression(pred2),
      xls_vast_operator_kind_bitwise_and);
  xls_vast_statement_block* elseif_block =
      xls_vast_conditional_add_else_if(cond, and_expr);
  ASSERT_NE(elseif_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_blocking_assignment(
                elseif_block, a_expr,
                xls_vast_literal_as_expression(
                    xls_vast_verilog_file_make_plain_literal(f, 0))),
            nullptr);

  xls_vast_statement_block* else_block = xls_vast_conditional_add_else(cond);
  ASSERT_NE(else_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_blocking_assignment(else_block, a_expr,
                                                             b_expr),
            nullptr);

  // case (sel) 1: a = b; default: a = 0; endcase
  xls_vast_case_statement* case_stmt = xls_vast_statement_block_add_case(
      block, xls_vast_logic_ref_as_expression(sel));
  ASSERT_NE(case_stmt, nullptr);
  xls_vast_statement_block* item_block = xls_vast_case_statement_add_item(
      case_stmt, xls_vast_literal_as_expression(
                     xls_vast_verilog_file_make_plain_literal(f, 1)));
  ASSERT_NE(item_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_blocking_assignment(item_block, a_expr,
                                                             b_expr),
            nullptr);
  xls_vast_statement_block* default_block =
      xls_vast_case_statement_add_default(case_stmt);
  ASSERT_NE(default_block, nullptr);
  ASSERT_NE(xls_vast_statement_block_add_blocking_assignment(
                default_block, a_expr,
                xls_vast_literal_as_expression(
                    xls_vast_verilog_file_make_plain_literal(f, 0))),
            nullptr);

  // nonblocking: b <= a_in;
  ASSERT_NE(xls_vast_statement_block_add_nonblocking_assignment(
                block, xls_vast_logic_ref_as_expression(b_reg),
                xls_vast_logic_ref_as_expression(a_in)),
            nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

// Covers the file-level factory path for blocking assignments.
// The grouped procedural test exercises only the block-level adder.
// Context/rationale: parity with the existing nonblocking factory; supports
// prebuild-and-pass of a Statement* for APIs that may accept statements; and
// validates allocation/identity when creating a statement owned by the file
// before insertion. This path is not exercised elsewhere.
TEST(XlsCApiTest, VastMakeBlockingAssignmentFactorySmoke) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_literal* lit0 = xls_vast_verilog_file_make_plain_literal(f, 0);
  xls_vast_statement* stmt = xls_vast_verilog_file_make_blocking_assignment(
      f, xls_vast_literal_as_expression(lit0),
      xls_vast_literal_as_expression(lit0));
  ASSERT_NE(stmt, nullptr);
}

TEST(XlsCApiTest, VastMacroStatements) {
  const std::string_view kWantEmitted = R"(module top(
  input wire [7:0] a,
  input wire [7:0] b
);
  `MY_MACRO1;
  `MY_MACRO2()
  `MY_MACRO3(a)
  `MY_MACRO4(a, b);
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
  xls_vast_logic_ref* a = xls_vast_verilog_module_add_input(m, "a", u8);
  xls_vast_logic_ref* b = xls_vast_verilog_module_add_input(m, "b", u8);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  // `MY_MACRO1;
  xls_vast_macro_ref* mr1 =
      xls_vast_verilog_file_make_macro_ref(f, "MY_MACRO1");
  ASSERT_NE(mr1, nullptr);
  xls_vast_macro_statement* ms1 = xls_vast_verilog_file_make_macro_statement(
      f, mr1, /*emit_semicolon=*/true);
  xls_vast_verilog_module_add_member_macro_statement(m, ms1);

  // `MY_MACRO2();
  xls_vast_expression** empty_args = nullptr;
  xls_vast_macro_ref* mr2 = xls_vast_verilog_file_make_macro_ref_with_args(
      f, "MY_MACRO2", empty_args, 0);
  ASSERT_NE(mr2, nullptr);
  xls_vast_macro_statement* ms2 = xls_vast_verilog_file_make_macro_statement(
      f, mr2, /*emit_semicolon=*/false);
  xls_vast_verilog_module_add_member_macro_statement(m, ms2);

  // `MY_MACRO3(a);
  xls_vast_expression* args3[] = {
      xls_vast_logic_ref_as_expression(a),
  };
  xls_vast_macro_ref* mr3 =
      xls_vast_verilog_file_make_macro_ref_with_args(f, "MY_MACRO3", args3, 1);
  ASSERT_NE(mr3, nullptr);
  xls_vast_macro_statement* ms3 = xls_vast_verilog_file_make_macro_statement(
      f, mr3, /*emit_semicolon=*/false);
  xls_vast_verilog_module_add_member_macro_statement(m, ms3);

  // `MY_MACRO4(a, b);
  xls_vast_expression* args4[] = {
      xls_vast_logic_ref_as_expression(a),
      xls_vast_logic_ref_as_expression(b),
  };
  xls_vast_macro_ref* mr4 =
      xls_vast_verilog_file_make_macro_ref_with_args(f, "MY_MACRO4", args4, 2);
  ASSERT_NE(mr4, nullptr);
  xls_vast_macro_statement* ms4 = xls_vast_verilog_file_make_macro_statement(
      f, mr4, /*emit_semicolon=*/true);
  xls_vast_verilog_module_add_member_macro_statement(m, ms4);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });

  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, ModuleWithParameterPort) {
  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "my_module");
  ASSERT_NE(m, nullptr);

  // Add input/output and a wire.
  xls_vast_data_type* scalar = xls_vast_verilog_file_make_scalar_type(f);
  xls_vast_expression* my_param = xls_vast_verilog_module_add_parameter_port(
      m, "my_param",
      xls_vast_literal_as_expression(
          xls_vast_verilog_file_make_plain_literal(f, 8)));
  xls_vast_data_type* u8 =
      xls_vast_verilog_file_make_bit_vector_type_with_expression(f, my_param,
                                                                 false);
  xls_vast_verilog_module_add_input(m, "my_input", u8);
  xls_vast_logic_ref* output_ref =
      xls_vast_verilog_module_add_output(m, "my_output", scalar);

  xls_vast_continuous_assignment* assignment =
      xls_vast_verilog_file_make_continuous_assignment(
          f, xls_vast_logic_ref_as_expression(output_ref), my_param);
  xls_vast_verilog_module_add_member_continuous_assignment(m, assignment);
  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  const std::string_view kWant = R"(module my_module #(
  parameter my_param = 8
) (
  input wire [my_param - 1:0] my_input,
  output wire my_output
);
  assign my_output = my_param;
endmodule
)";
  EXPECT_EQ(std::string_view{emitted}, kWant);
}

TEST(XlsCApiTest, VastGenerateLoopWithAlwaysBlocks) {
  const std::string_view kWantEmitted = R"(module top(
  input wire clk
);
  for (genvar i = 0; i < 8; i = i + 1) begin : gen
    always_comb begin end
    always_ff @ (posedge clk) begin end
  end
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  // IO
  xls_vast_data_type* clk_t = xls_vast_verilog_file_make_scalar_type(f);
  xls_vast_logic_ref* clk = xls_vast_verilog_module_add_input(m, "clk", clk_t);
  ASSERT_NE(clk, nullptr);

  // Generate loop 0..8 with label "gen".
  xls_vast_literal* zero_lit = xls_vast_verilog_file_make_plain_literal(f, 0);
  xls_vast_literal* eight_lit = xls_vast_verilog_file_make_plain_literal(f, 8);
  xls_vast_expression* zero_expr = xls_vast_literal_as_expression(zero_lit);
  xls_vast_expression* eight_expr = xls_vast_literal_as_expression(eight_lit);
  xls_vast_generate_loop* loop = xls_vast_verilog_module_add_generate_loop(
      m, "i", zero_expr, eight_expr, "gen");
  ASSERT_NE(loop, nullptr);

  // Add empty always_comb inside the generate loop.
  xls_vast_always_base* out_ac = nullptr;
  char* err = nullptr;
  ASSERT_TRUE(xls_vast_generate_loop_add_always_comb(loop, &out_ac, &err));
  ASSERT_NE(out_ac, nullptr);
  ASSERT_EQ(err, nullptr);

  // Add empty always_ff @ (posedge clk) inside the generate loop.
  xls_vast_expression* sens_list[1];
  sens_list[0] = xls_vast_verilog_file_make_pos_edge(
      f, xls_vast_logic_ref_as_expression(clk));
  xls_vast_always_base* out_aff = nullptr;
  char* err2 = nullptr;
  ASSERT_TRUE(xls_vast_generate_loop_add_always_ff(loop, sens_list, 1, &out_aff,
                                                   &err2));
  ASSERT_NE(out_aff, nullptr);
  ASSERT_EQ(err2, nullptr);

  // Emit and compare.
  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

TEST(XlsCApiTest, VastNestedGenerateLoops) {
  const std::string_view kWantEmitted = R"(module top;
  wire [3:0][2:0] src;
  wire [3:0][2:0] dst;
  for (genvar i = 0; i < 4; i = i + 1) begin : outer
    for (genvar j = 0; j < 3; j = j + 1) begin : inner
      assign dst[i][j] = src[i][j];
    end
  end
endmodule
)";

  xls_vast_verilog_file* f =
      xls_vast_make_verilog_file(xls_vast_file_type_verilog);
  ASSERT_NE(f, nullptr);
  absl::Cleanup free_file([&] { xls_vast_verilog_file_free(f); });

  xls_vast_verilog_module* m = xls_vast_verilog_file_add_module(f, "top");
  ASSERT_NE(m, nullptr);

  // Create packed array type with element bit width 1 and dimensions [4][3].
  xls_vast_data_type* u1 =
      xls_vast_verilog_file_make_bit_vector_type(f, 1, /*is_signed=*/false);
  const std::vector<int64_t> dims = {4, 3};
  xls_vast_data_type* arr_t = xls_vast_verilog_file_make_packed_array_type(
      f, u1, dims.data(), dims.size());

  xls_vast_logic_ref* src = xls_vast_verilog_module_add_wire(m, "src", arr_t);
  xls_vast_logic_ref* dst = xls_vast_verilog_module_add_wire(m, "dst", arr_t);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);

  // Add outer and inner generate loops.
  xls_vast_literal* zero_lit = xls_vast_verilog_file_make_plain_literal(f, 0);
  xls_vast_literal* four_lit = xls_vast_verilog_file_make_plain_literal(f, 4);
  xls_vast_literal* three_lit = xls_vast_verilog_file_make_plain_literal(f, 3);
  xls_vast_expression* zero_expr = xls_vast_literal_as_expression(zero_lit);
  xls_vast_expression* four_expr = xls_vast_literal_as_expression(four_lit);
  xls_vast_expression* three_expr = xls_vast_literal_as_expression(three_lit);

  xls_vast_generate_loop* outer = xls_vast_verilog_module_add_generate_loop(
      m, "i", zero_expr, four_expr, "outer");
  ASSERT_NE(outer, nullptr);

  xls_vast_logic_ref* i_ref = xls_vast_generate_loop_get_genvar(outer);
  ASSERT_NE(i_ref, nullptr);
  xls_vast_expression* i_expr = xls_vast_logic_ref_as_expression(i_ref);

  xls_vast_generate_loop* inner = xls_vast_generate_loop_add_generate_loop(
      outer, "j", zero_expr, three_expr, "inner");
  ASSERT_NE(inner, nullptr);
  xls_vast_logic_ref* j_ref = xls_vast_generate_loop_get_genvar(inner);
  ASSERT_NE(j_ref, nullptr);
  xls_vast_expression* j_expr = xls_vast_logic_ref_as_expression(j_ref);

  // Add assign dst[i][j] = src[i][j]; inside inner loop.
  xls_vast_index* dst_i = xls_vast_verilog_file_make_index(
      f, xls_vast_logic_ref_as_indexable_expression(dst), i_expr);
  xls_vast_index* dst_ij = xls_vast_verilog_file_make_index(
      f, xls_vast_index_as_indexable_expression(dst_i), j_expr);
  xls_vast_index* src_i = xls_vast_verilog_file_make_index(
      f, xls_vast_logic_ref_as_indexable_expression(src), i_expr);
  xls_vast_index* src_ij = xls_vast_verilog_file_make_index(
      f, xls_vast_index_as_indexable_expression(src_i), j_expr);

  xls_vast_statement* assign_stmt =
      xls_vast_generate_loop_add_continuous_assignment(
          inner, xls_vast_index_as_expression(dst_ij),
          xls_vast_index_as_expression(src_ij));
  ASSERT_NE(assign_stmt, nullptr);

  char* emitted = xls_vast_verilog_file_emit(f);
  ASSERT_NE(emitted, nullptr);
  absl::Cleanup free_emitted([&] { xls_c_str_free(emitted); });
  EXPECT_EQ(std::string_view{emitted}, kWantEmitted);
}

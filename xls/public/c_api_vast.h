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

// VAST (Verilog AST) APIs
//
// Note that these are expected to be *less* stable than other public C APIs,
// as they are exposing a useful implementation library present within XLS.
//
// Per usual, in a general sense, no promises are made around API or ABI
// stability overall. However, seems worth noting these are effectively
// "protected" APIs, use with particular caution around stability. See
// `xls/protected/BUILD` for how we tend to think about "protected" APIs in the
// project.

#ifndef XLS_PUBLIC_C_API_VAST_H_
#define XLS_PUBLIC_C_API_VAST_H_

#include <stddef.h>  // NOLINT(modernize-deprecated-headers)
#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

#include "xls/public/c_api_format_preference.h"

extern "C" {

// Opaque structs.
struct xls_vast_verilog_file;
struct xls_vast_verilog_module;
struct xls_vast_node;
struct xls_vast_expression;
struct xls_vast_logic_ref;
struct xls_vast_data_type;
struct xls_vast_indexable_expression;
struct xls_vast_slice;
struct xls_vast_literal;
struct xls_vast_instantiation;
struct xls_vast_continuous_assignment;
struct xls_vast_comment;
struct xls_vast_inline_verilog_statement;
struct xls_vast_macro_ref;
struct xls_vast_macro_statement;
struct xls_vast_always_base;
struct xls_vast_statement;
struct xls_vast_statement_block;
struct xls_vast_generate_loop;
struct xls_vast_module_port;
struct xls_vast_def;
struct xls_vast_parameter_ref;
struct xls_vast_conditional;
struct xls_vast_case_statement;
struct xls_vast_localparam_ref;
struct xls_vast_blank_line;
struct xls_vast_inline_verilog_statement;

// Note: We define the enum with a fixed width integer type for clarity of the
// exposed ABI.
typedef int32_t xls_vast_file_type;
enum {
  xls_vast_file_type_verilog,
  xls_vast_file_type_system_verilog,
};

typedef int32_t xls_vast_operator_kind;
enum {
  // unary operators
  xls_vast_operator_kind_negate,
  xls_vast_operator_kind_bitwise_not,
  xls_vast_operator_kind_logical_not,
  xls_vast_operator_kind_and_reduce,
  xls_vast_operator_kind_or_reduce,
  xls_vast_operator_kind_xor_reduce,

  // binary operators
  xls_vast_operator_kind_add,
  xls_vast_operator_kind_logical_and,
  xls_vast_operator_kind_bitwise_and,
  xls_vast_operator_kind_ne,
  xls_vast_operator_kind_case_ne,
  xls_vast_operator_kind_eq,
  xls_vast_operator_kind_case_eq,
  xls_vast_operator_kind_ge,
  xls_vast_operator_kind_gt,
  xls_vast_operator_kind_le,
  xls_vast_operator_kind_lt,
  xls_vast_operator_kind_div,
  xls_vast_operator_kind_mod,
  xls_vast_operator_kind_mul,
  xls_vast_operator_kind_power,
  xls_vast_operator_kind_bitwise_or,
  xls_vast_operator_kind_logical_or,
  xls_vast_operator_kind_bitwise_xor,
  xls_vast_operator_kind_shll,
  xls_vast_operator_kind_shra,
  xls_vast_operator_kind_shrl,
  xls_vast_operator_kind_sub,
  xls_vast_operator_kind_ne_x,
  xls_vast_operator_kind_eq_x,
};

// Note: caller owns the returned verilog file object, to be freed by
// `xls_vast_verilog_file_free`.
struct xls_vast_verilog_file* xls_vast_make_verilog_file(
    xls_vast_file_type file_type);

void xls_vast_verilog_file_free(struct xls_vast_verilog_file* f);

struct xls_vast_verilog_module* xls_vast_verilog_file_add_module(
    struct xls_vast_verilog_file* f, const char* name);

// -- VerilogFile::Make*Type

struct xls_vast_data_type* xls_vast_verilog_file_make_scalar_type(
    struct xls_vast_verilog_file* f);

struct xls_vast_data_type* xls_vast_verilog_file_make_bit_vector_type(
    struct xls_vast_verilog_file* f, int64_t bit_count, bool is_signed);

struct xls_vast_data_type*
xls_vast_verilog_file_make_bit_vector_type_with_expression(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* expression,
    bool is_signed);

struct xls_vast_data_type* xls_vast_verilog_file_make_integer_type(
    struct xls_vast_verilog_file* f, bool is_signed);

struct xls_vast_data_type* xls_vast_verilog_file_make_int_type(
    struct xls_vast_verilog_file* f, bool is_signed);

// Convenience creators for typed Defs of integer/int kinds.
struct xls_vast_def* xls_vast_verilog_file_make_integer_def(
    struct xls_vast_verilog_file* f, const char* name, bool is_signed);
struct xls_vast_def* xls_vast_verilog_file_make_int_def(
    struct xls_vast_verilog_file* f, const char* name, bool is_signed);

struct xls_vast_data_type* xls_vast_verilog_file_make_extern_package_type(
    struct xls_vast_verilog_file* f, const char* package_name,
    const char* entity_name);
struct xls_vast_data_type* xls_vast_verilog_file_make_extern_type(
    struct xls_vast_verilog_file* f, const char* entity_name);

struct xls_vast_data_type* xls_vast_verilog_file_make_packed_array_type(
    struct xls_vast_verilog_file* f, xls_vast_data_type* element_type,
    const int64_t* packed_dims, size_t packed_dims_count);

// Creates an unpacked array type.
//
// Example (SystemVerilog):
//   element_type = bit_vector_type(8) and unpacked_dims = {2, 3} yields a type
//   that emits like: `[7:0] <ident>[2][3]`.
struct xls_vast_data_type* xls_vast_verilog_file_make_unpacked_array_type(
    struct xls_vast_verilog_file* f, xls_vast_data_type* element_type,
    const int64_t* unpacked_dims, size_t unpacked_dims_count);

// -- Module::Add*

void xls_vast_verilog_module_add_member_instantiation(
    struct xls_vast_verilog_module* m, struct xls_vast_instantiation* member);
void xls_vast_verilog_module_add_member_continuous_assignment(
    struct xls_vast_verilog_module* m,
    struct xls_vast_continuous_assignment* member);
void xls_vast_verilog_module_add_member_comment(
    struct xls_vast_verilog_module* m, struct xls_vast_comment* comment);
void xls_vast_verilog_module_add_member_blank_line(
    struct xls_vast_verilog_module* m, struct xls_vast_blank_line* blank);
void xls_vast_verilog_module_add_member_inline_statement(
    struct xls_vast_verilog_module* m,
    struct xls_vast_inline_verilog_statement* stmt);

// Adds a macro statement (e.g. `FOO(...);) to the module.
void xls_vast_verilog_module_add_member_macro_statement(
    struct xls_vast_verilog_module* m,
    struct xls_vast_macro_statement* statement);

struct xls_vast_logic_ref* xls_vast_verilog_module_add_input(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type);
struct xls_vast_logic_ref* xls_vast_verilog_module_add_output(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type);
// Adds input/output ports using SystemVerilog `logic` for the declaration.
struct xls_vast_logic_ref* xls_vast_verilog_module_add_logic_input(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type);
struct xls_vast_logic_ref* xls_vast_verilog_module_add_logic_output(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type);
struct xls_vast_logic_ref* xls_vast_verilog_module_add_wire(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type);
struct xls_vast_generate_loop* xls_vast_verilog_module_add_generate_loop(
    struct xls_vast_verilog_module* m, const char* genvar_name,
    struct xls_vast_expression* init, struct xls_vast_expression* limit,
    const char* label);
struct xls_vast_expression* xls_vast_verilog_module_add_parameter_port(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_expression* rhs);
struct xls_vast_expression* xls_vast_verilog_module_add_typed_parameter_port(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type, struct xls_vast_expression* rhs);
// TODO(cdleary): 2024-09-05 Add xls_vast_verilog_module_add_wire_with_expr

// Adds a module parameter with the given name and RHS expression.
// Returns a handle to the created parameter reference.
struct xls_vast_parameter_ref* xls_vast_verilog_module_add_parameter(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_expression* rhs);

// Adds a module parameter with the given name and RHS expression.
// Returns a handle to the created parameter reference.
struct xls_vast_localparam_ref* xls_vast_verilog_module_add_localparam(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_expression* rhs);

// Note: returned value is owned by the caller, free via `xls_c_str_free`.
char* xls_vast_verilog_module_get_name(struct xls_vast_verilog_module* m);
// Returns the ports that are present on the given module.
//
// Note: the returned array is owned by the caller, to be freed by
// `xls_vast_verilog_module_free_ports`.
struct xls_vast_module_port** xls_vast_verilog_module_get_ports(
    struct xls_vast_verilog_module* m, size_t* out_count);

typedef int32_t xls_vast_module_port_direction;
enum {
  xls_vast_module_port_direction_input,
  xls_vast_module_port_direction_output,
  xls_vast_module_port_direction_inout,
};

xls_vast_module_port_direction xls_vast_verilog_module_port_get_direction(
    struct xls_vast_module_port* port);

xls_vast_def* xls_vast_verilog_module_port_get_def(
    struct xls_vast_module_port* port);

// Note: returned value is owned by the caller, free via `xls_c_str_free`.
char* xls_vast_def_get_name(struct xls_vast_def* def);

// Returns the data type of the given `def`.
xls_vast_data_type* xls_vast_def_get_data_type(struct xls_vast_def* def);

bool xls_vast_data_type_width_as_int64(struct xls_vast_data_type* type,
                                       int64_t* out_width, char** error_out);

bool xls_vast_data_type_flat_bit_count_as_int64(struct xls_vast_data_type* type,
                                                int64_t* out_flat_bit_count,
                                                char** error_out);

// Returns the width expression for the type; note that scalars (e.g. "wire
// foo;") and integers return a nullptr.
xls_vast_expression* xls_vast_data_type_width(struct xls_vast_data_type* type);

bool xls_vast_data_type_is_signed(struct xls_vast_data_type* type);

typedef int32_t xls_vast_data_kind;
enum {
  xls_vast_data_kind_reg,
  xls_vast_data_kind_wire,
  xls_vast_data_kind_logic,
  xls_vast_data_kind_integer,
  xls_vast_data_kind_int,
  xls_vast_data_kind_user,
  xls_vast_data_kind_untyped_enum,
  xls_vast_data_kind_genvar,
};

// Creates a generic definition (Def) with the given name, data kind, and type.
// Useful for creating typed parameters via
// xls_vast_verilog_module_add_parameter_with_def.
struct xls_vast_def* xls_vast_verilog_file_make_def(
    struct xls_vast_verilog_file* f, const char* name, xls_vast_data_kind kind,
    struct xls_vast_data_type* type);

// Adds a module parameter with an explicit Def (type/kind) and RHS expression.
// Returns a handle to the created parameter reference.
struct xls_vast_parameter_ref* xls_vast_verilog_module_add_parameter_with_def(
    struct xls_vast_verilog_module* m, struct xls_vast_def* def,
    struct xls_vast_expression* rhs);
struct xls_vast_localparam_ref* xls_vast_verilog_module_add_localparam_with_def(
    struct xls_vast_verilog_module* m, struct xls_vast_def* def,
    struct xls_vast_expression* rhs);

// Frees the ports array returned by `xls_vast_verilog_module_get_ports`.
void xls_vast_verilog_module_free_ports(struct xls_vast_module_port** ports,
                                        size_t count);

struct xls_vast_continuous_assignment*
xls_vast_verilog_file_make_continuous_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

struct xls_vast_comment* xls_vast_verilog_file_make_comment(
    struct xls_vast_verilog_file* f, const char* text);

struct xls_vast_blank_line* xls_vast_verilog_file_make_blank_line(
    struct xls_vast_verilog_file* f);

struct xls_vast_inline_verilog_statement*
xls_vast_verilog_file_make_inline_verilog_statement(
    struct xls_vast_verilog_file* f, const char* text);

// Creates a MacroRef expression: `NAME or `NAME(args...)
struct xls_vast_macro_ref* xls_vast_verilog_file_make_macro_ref(
    struct xls_vast_verilog_file* f, const char* name);
struct xls_vast_macro_ref* xls_vast_verilog_file_make_macro_ref_with_args(
    struct xls_vast_verilog_file* f, const char* name,
    struct xls_vast_expression** args, size_t arg_count);

// Cast: MacroRef -> Expression
struct xls_vast_expression* xls_vast_macro_ref_as_expression(
    struct xls_vast_macro_ref* ref);

// Creates a MacroStatement from a MacroRef: e.g. `NAME(...);
struct xls_vast_macro_statement* xls_vast_verilog_file_make_macro_statement(
    struct xls_vast_verilog_file* f, struct xls_vast_macro_ref* ref,
    bool emit_semicolon);

struct xls_vast_instantiation* xls_vast_verilog_file_make_instantiation(
    struct xls_vast_verilog_file* f, const char* module_name,
    const char* instance_name, const char** parameter_port_names,
    struct xls_vast_expression** parameter_expressions, size_t parameter_count,
    const char** connection_port_names,
    struct xls_vast_expression** connection_expressions,
    size_t connection_count);

void xls_vast_verilog_file_add_include(struct xls_vast_verilog_file* f,
                                       const char* path);

// Adds a blank line to the file.
void xls_vast_verilog_file_add_blank_line(struct xls_vast_verilog_file* f);

// Adds a comment to the file.
void xls_vast_verilog_file_add_comment(struct xls_vast_verilog_file* f,
                                       const char* text);

struct xls_vast_concat* xls_vast_verilog_file_make_concat(
    struct xls_vast_verilog_file* f, struct xls_vast_expression** elements,
    size_t element_count);

// Creates a replicated concatenation expression: {replication{elements...}}.
// For single-element replication, pass element_count=1.
struct xls_vast_concat* xls_vast_verilog_file_make_replicated_concat(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* replication,
    struct xls_vast_expression** elements, size_t element_count);

// Convenience: replicated concatenation with an integer replication count.
struct xls_vast_concat* xls_vast_verilog_file_make_replicated_concat_i64(
    struct xls_vast_verilog_file* f, int64_t replication_count,
    struct xls_vast_expression** elements, size_t element_count);

// Creates an array assignment pattern expression: `'{a, b, c}`.
struct xls_vast_expression* xls_vast_verilog_file_make_array_assignment_pattern(
    struct xls_vast_verilog_file* f, struct xls_vast_expression** elements,
    size_t element_count);

struct xls_vast_slice* xls_vast_verilog_file_make_slice_i64(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject, int64_t hi, int64_t lo);

struct xls_vast_slice* xls_vast_verilog_file_make_slice(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject,
    struct xls_vast_expression* hi, struct xls_vast_expression* lo);

struct xls_vast_expression* xls_vast_verilog_file_make_unary(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* arg,
    xls_vast_operator_kind op);

struct xls_vast_expression* xls_vast_verilog_file_make_binary(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs, xls_vast_operator_kind op);

struct xls_vast_expression* xls_vast_verilog_file_make_ternary(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* cond,
    struct xls_vast_expression* consequent,
    struct xls_vast_expression* alternate);

struct xls_vast_expression* xls_vast_verilog_file_make_width_cast(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* width,
    struct xls_vast_expression* value);

// Creates a SystemVerilog type cast expression: <type>'(<value>)
struct xls_vast_expression* xls_vast_verilog_file_make_type_cast(
    struct xls_vast_verilog_file* f, struct xls_vast_data_type* type,
    struct xls_vast_expression* value);

struct xls_vast_index* xls_vast_verilog_file_make_index_i64(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject, int64_t index);

struct xls_vast_index* xls_vast_verilog_file_make_index(
    struct xls_vast_verilog_file* f,
    struct xls_vast_indexable_expression* subject,
    struct xls_vast_expression* index);

struct xls_vast_literal* xls_vast_verilog_file_make_plain_literal(
    struct xls_vast_verilog_file* f, int32_t value);

// Creates a VAST literal with an arbitrary bit count.
//
// Returns an error if the given format preference is invalid.
bool xls_vast_verilog_file_make_literal(struct xls_vast_verilog_file* f,
                                        struct xls_bits* bits,
                                        xls_format_preference format_preference,
                                        bool emit_bit_count, char** error_out,
                                        struct xls_vast_literal** literal_out);

// Creates unsized literal expressions: '1, '0, 'X
struct xls_vast_expression* xls_vast_verilog_file_make_unsized_one_literal(
    struct xls_vast_verilog_file* f);
struct xls_vast_expression* xls_vast_verilog_file_make_unsized_zero_literal(
    struct xls_vast_verilog_file* f);
struct xls_vast_expression* xls_vast_verilog_file_make_unsized_x_literal(
    struct xls_vast_verilog_file* f);

// Casts to turn the given node to an expression, where possible.
struct xls_vast_expression* xls_vast_literal_as_expression(
    struct xls_vast_literal* v);
struct xls_vast_expression* xls_vast_logic_ref_as_expression(
    struct xls_vast_logic_ref* v);
struct xls_vast_expression* xls_vast_slice_as_expression(
    struct xls_vast_slice* v);
struct xls_vast_expression* xls_vast_concat_as_expression(
    struct xls_vast_concat* v);
struct xls_vast_expression* xls_vast_index_as_expression(
    struct xls_vast_index* v);
struct xls_vast_expression* xls_vast_parameter_ref_as_expression(
    struct xls_vast_parameter_ref* v);
struct xls_vast_expression* xls_vast_localparam_ref_as_expression(
    struct xls_vast_localparam_ref* v);
struct xls_vast_expression* xls_vast_indexable_expression_as_expression(
    struct xls_vast_indexable_expression* v);

struct xls_vast_indexable_expression*
xls_vast_logic_ref_as_indexable_expression(
    struct xls_vast_logic_ref* logic_ref);

struct xls_vast_indexable_expression*
xls_vast_parameter_ref_as_indexable_expression(
    struct xls_vast_parameter_ref* parameter_ref);
struct xls_vast_logic_ref* xls_vast_generate_loop_get_genvar(
    struct xls_vast_generate_loop* loop);

// Adds a nested generate loop inside the given generate loop.
struct xls_vast_generate_loop* xls_vast_generate_loop_add_generate_loop(
    struct xls_vast_generate_loop* loop, const char* genvar_name,
    struct xls_vast_expression* init, struct xls_vast_expression* limit,
    const char* label);

// Adds a blank line inside the given generate loop.
void xls_vast_generate_loop_add_blank_line(struct xls_vast_generate_loop* loop);

// Adds a comment inside the given generate loop.
void xls_vast_generate_loop_add_comment(struct xls_vast_generate_loop* loop,
                                        struct xls_vast_comment* comment);

// Adds an instantiation inside the given generate loop.
void xls_vast_generate_loop_add_instantiation(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_instantiation* instantiation);

// Adds an inline verilog statement inside the given generate loop.
void xls_vast_generate_loop_add_inline_verilog_statement(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_inline_verilog_statement* stmt);

// Adds an always_comb block inside the given generate loop.
// Returns true on success; on failure returns false and sets error_out.
bool xls_vast_generate_loop_add_always_comb(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_always_base** out_always_comb, char** error_out);

// Adds an always_ff block inside the given generate loop.
// 'sensitivity_list_elements' is an array of expressions.
// Returns true on success; on failure returns false and sets error_out.
bool xls_vast_generate_loop_add_always_ff(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_expression** sensitivity_list_elements,
    size_t sensitivity_list_count, struct xls_vast_always_base** out_always_ff,
    char** error_out);

// Adds a localparam item declaration inside the given generate loop.
struct xls_vast_localparam_ref* xls_vast_generate_loop_add_localparam(
    struct xls_vast_generate_loop* loop, const char* name,
    struct xls_vast_expression* rhs);

// Adds a localparam item using an existing def inside the given generate loop.
struct xls_vast_localparam_ref* xls_vast_generate_loop_add_localparam_with_def(
    struct xls_vast_generate_loop* loop, struct xls_vast_def* def,
    struct xls_vast_expression* rhs);

// Adds a continuous assignment inside the given generate loop.
struct xls_vast_statement* xls_vast_generate_loop_add_continuous_assignment(
    struct xls_vast_generate_loop* loop, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

struct xls_vast_macro_statement* xls_vast_generate_loop_add_macro_statement(
    struct xls_vast_generate_loop* loop,
    struct xls_vast_macro_statement* statement);

// Note: returned value is owned by the caller, free via `xls_c_str_free`.
char* xls_vast_logic_ref_get_name(struct xls_vast_logic_ref* logic_ref);

struct xls_vast_indexable_expression* xls_vast_index_as_indexable_expression(
    struct xls_vast_index* index);

// Gets the statement block associated with an always_base construct (like
// always_ff).
struct xls_vast_statement_block* xls_vast_always_base_get_statement_block(
    struct xls_vast_always_base* always_base);

// Adds a non-blocking assignment statement (lhs <= rhs) to a statement block
// and returns a pointer to the created statement.
struct xls_vast_statement* xls_vast_statement_block_add_nonblocking_assignment(
    struct xls_vast_statement_block* block, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

// Adds a blocking assignment statement (lhs = rhs) to a statement block and
// returns a pointer to the created statement.
struct xls_vast_statement* xls_vast_statement_block_add_blocking_assignment(
    struct xls_vast_statement_block* block, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

// Adds a continuous assignment statement (assign lhs = rhs;) to a statement
// block and returns a pointer to the created statement.
struct xls_vast_statement* xls_vast_statement_block_add_continuous_assignment(
    struct xls_vast_statement_block* block, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

struct xls_vast_statement* xls_vast_statement_block_add_comment_text(
    struct xls_vast_statement_block* block, const char* text);

struct xls_vast_statement* xls_vast_statement_block_add_blank_line(
    struct xls_vast_statement_block* block);

struct xls_vast_statement* xls_vast_statement_block_add_inline_text(
    struct xls_vast_statement_block* block, const char* text);

// Emits/formats the contents of the given verilog file to a string.
//
// Note: caller owns the returned string, to be freed by `xls_c_str_free`.
char* xls_vast_verilog_file_emit(const struct xls_vast_verilog_file* f);

// Emits an expression to a string and returns an owned C string; caller must
// free with xls_c_str_free.
char* xls_vast_expression_emit(struct xls_vast_expression* expr);

// Adds an always_ff block to the module.
// 'sensitivity_list_elements' is an array of expressions, typically created
// using 'xls_vast_verilog_file_make_pos_edge' or
// 'xls_vast_verilog_file_make_neg_edge'. Returns true on success. On failure,
// returns false and sets error_out. The caller is responsible for freeing
// error_out if it is not NULL.
bool xls_vast_verilog_module_add_always_ff(
    struct xls_vast_verilog_module* m,
    struct xls_vast_expression** sensitivity_list_elements,
    size_t sensitivity_list_count, struct xls_vast_always_base** out_always_ff,
    char** error_out);

// Adds an always @ block to the module (Verilog-2001 style).
// 'sensitivity_list_elements' is an array of expressions.
// Returns true on success. On failure, returns false and sets error_out.
// The caller is responsible for freeing error_out if it is not NULL.
bool xls_vast_verilog_module_add_always_at(
    struct xls_vast_verilog_module* m,
    struct xls_vast_expression** sensitivity_list_elements,
    size_t sensitivity_list_count, struct xls_vast_always_base** out_always_at,
    char** error_out);

// Adds an always_comb block to the module (SystemVerilog).
// Returns true on success. On failure, returns false and sets error_out. The
// caller is responsible for freeing error_out if it is not NULL.
bool xls_vast_verilog_module_add_always_comb(
    struct xls_vast_verilog_module* m,
    struct xls_vast_always_base** out_always_comb, char** error_out);

// Adds an inout port to the module.
struct xls_vast_logic_ref* xls_vast_verilog_module_add_inout(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type);

// Adds a register (reg) definition to the module.
// Returns true on success. On failure, returns false and sets error_out.
// The caller is responsible for freeing error_out if it is not NULL.
bool xls_vast_verilog_module_add_reg(struct xls_vast_verilog_module* m,
                                     const char* name,
                                     struct xls_vast_data_type* type,
                                     struct xls_vast_logic_ref** out_reg_ref,
                                     char** error_out);

// Adds a `logic` variable definition to the module.
// Returns true on success. On failure, returns false and sets error_out.
// The caller is responsible for freeing error_out if it is not NULL.
bool xls_vast_verilog_module_add_logic(
    struct xls_vast_verilog_module* m, const char* name,
    struct xls_vast_data_type* type, struct xls_vast_logic_ref** out_logic_ref,
    char** error_out);

// Creates a positive edge expression (e.g., "posedge clk").
// 'signal_expr' is typically a logic_ref_as_expression.
struct xls_vast_expression* xls_vast_verilog_file_make_pos_edge(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* signal_expr);

// Creates a non-blocking assignment statement (lhs <= rhs).
struct xls_vast_statement* xls_vast_verilog_file_make_nonblocking_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

// Creates a blocking assignment statement (lhs = rhs).
struct xls_vast_statement* xls_vast_verilog_file_make_blocking_assignment(
    struct xls_vast_verilog_file* f, struct xls_vast_expression* lhs,
    struct xls_vast_expression* rhs);

// Adds a conditional (if) with the given condition to a statement block and
// returns a handle to the created conditional.
struct xls_vast_conditional* xls_vast_statement_block_add_conditional(
    struct xls_vast_statement_block* block, struct xls_vast_expression* cond);

// Returns the 'then' statement block of the given conditional.
struct xls_vast_statement_block* xls_vast_conditional_get_then_block(
    struct xls_vast_conditional* cond);

// Adds an else-if clause to the given conditional with the provided
// condition and returns the associated statement block.
struct xls_vast_statement_block* xls_vast_conditional_add_else_if(
    struct xls_vast_conditional* cond, struct xls_vast_expression* expr_cond);

// Adds an else clause (no condition) to the given conditional and returns the
// associated statement block. Must be called at most once.
struct xls_vast_statement_block* xls_vast_conditional_add_else(
    struct xls_vast_conditional* cond);

// Adds a conditional (if) with the given condition at module scope.
struct xls_vast_conditional* xls_vast_verilog_module_add_conditional(
    struct xls_vast_verilog_module* m, struct xls_vast_expression* cond);

// Adds a conditional (if) inside a generate loop.
struct xls_vast_conditional* xls_vast_generate_loop_add_conditional(
    struct xls_vast_generate_loop* loop, struct xls_vast_expression* cond);

// Adds a case statement with the given selector to a statement block and
// returns a handle to the created case statement.
struct xls_vast_case_statement* xls_vast_statement_block_add_case(
    struct xls_vast_statement_block* block,
    struct xls_vast_expression* selector);

// Adds a case item with the given match expression to the case statement and
// returns the associated statement block for that item.
struct xls_vast_statement_block* xls_vast_case_statement_add_item(
    struct xls_vast_case_statement* case_stmt,
    struct xls_vast_expression* match_expr);

// Adds a default case item to the case statement and returns the associated
// statement block. Must be called at most once.
struct xls_vast_statement_block* xls_vast_case_statement_add_default(
    struct xls_vast_case_statement* case_stmt);

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_VAST_H_

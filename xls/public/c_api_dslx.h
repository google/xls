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

// DSLX (Domain Specific Language "X") APIs
//
// Note that these are expected to be *less* stable than other public C APIs,
// as they are exposing a useful implementation library present within XLS.
//
// Per usual, in a general sense, no promises are made around API or ABI
// stability overall. However, seems worth noting these are effectively
// "protected" APIs, use with particular caution around stability. See
// `xls/protected/BUILD` for how we tend to think about "protected" APIs in the
// project.

#ifndef XLS_PUBLIC_C_API_DSLX_H_
#define XLS_PUBLIC_C_API_DSLX_H_

#include <stddef.h>  // NOLINT(modernize-deprecated-headers)
#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

extern "C" {

typedef int32_t xls_dslx_type_definition_kind;
enum {
  xls_dslx_type_definition_kind_type_alias,
  xls_dslx_type_definition_kind_struct_def,
  xls_dslx_type_definition_kind_enum_def,
  xls_dslx_type_definition_kind_colon_ref,
  xls_dslx_type_definition_kind_proc_def,
  xls_dslx_type_definition_kind_use_tree_entry,
};

typedef int32_t xls_dslx_module_member_kind;
enum {
  xls_dslx_module_member_kind_function,
  xls_dslx_module_member_kind_proc,
  xls_dslx_module_member_kind_test_function,
  xls_dslx_module_member_kind_test_proc,
  xls_dslx_module_member_kind_quick_check,
  xls_dslx_module_member_kind_type_alias,
  xls_dslx_module_member_kind_struct_def,
  xls_dslx_module_member_kind_proc_def,
  xls_dslx_module_member_kind_enum_def,
  xls_dslx_module_member_kind_constant_def,
  xls_dslx_module_member_kind_import,
  xls_dslx_module_member_kind_const_assert,
  xls_dslx_module_member_kind_impl,
  xls_dslx_module_member_kind_verbatim_node,
  xls_dslx_module_member_kind_use,
};

// Opaque structs.
struct xls_dslx_typechecked_module;
struct xls_dslx_import_data;
struct xls_dslx_module;
struct xls_dslx_type_definition;
struct xls_dslx_struct_def;
struct xls_dslx_enum_def;
struct xls_dslx_type_alias;
struct xls_dslx_type_info;
struct xls_dslx_type;
struct xls_dslx_type_annotation;
struct xls_dslx_constant_def;
struct xls_dslx_function;
struct xls_dslx_quickcheck;
struct xls_dslx_function;
struct xls_dslx_param;
struct xls_dslx_expr;
struct xls_dslx_module_member;
struct xls_dslx_type_dim;

struct xls_dslx_import_data* xls_dslx_import_data_create(
    const char* dslx_stdlib_path, const char* additional_search_paths[],
    size_t additional_search_paths_count);

void xls_dslx_import_data_free(struct xls_dslx_import_data*);

bool xls_dslx_parse_and_typecheck(
    const char* text, const char* path, const char* module_name,
    struct xls_dslx_import_data* import_data, char** error_out,
    struct xls_dslx_typechecked_module** result_out);

void xls_dslx_typechecked_module_free(struct xls_dslx_typechecked_module* tm);

struct xls_dslx_module* xls_dslx_typechecked_module_get_module(
    struct xls_dslx_typechecked_module*);
struct xls_dslx_type_info* xls_dslx_typechecked_module_get_type_info(
    struct xls_dslx_typechecked_module*);

// Returns the imported TypeInfo associated with the given module if present
// within the import graph rooted at `type_info`; otherwise returns nullptr.
struct xls_dslx_type_info* xls_dslx_type_info_get_imported_type_info(
    struct xls_dslx_type_info* type_info, struct xls_dslx_module* module);

int64_t xls_dslx_module_get_member_count(struct xls_dslx_module*);

struct xls_dslx_module_member* xls_dslx_module_get_member(
    struct xls_dslx_module*, int64_t);

struct xls_dslx_constant_def* xls_dslx_module_member_get_constant_def(
    struct xls_dslx_module_member*);

struct xls_dslx_struct_def* xls_dslx_module_member_get_struct_def(
    struct xls_dslx_module_member*);

struct xls_dslx_enum_def* xls_dslx_module_member_get_enum_def(
    struct xls_dslx_module_member*);

struct xls_dslx_type_alias* xls_dslx_module_member_get_type_alias(
    struct xls_dslx_module_member*);

// Returns the function AST node from the given module member if it is a
// function; otherwise returns nullptr.
struct xls_dslx_function* xls_dslx_module_member_get_function(
    struct xls_dslx_module_member*);

// Returns whether the given DSLX function is parametric.
bool xls_dslx_function_is_parametric(struct xls_dslx_function*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_function_get_identifier(struct xls_dslx_function*);

// Returns the number of parameters for the given DSLX function.
int64_t xls_dslx_function_get_param_count(struct xls_dslx_function* fn);

// Returns the i-th parameter of the given DSLX function.
// The returned pointer is borrowed and tied to the lifetime of the underlying
// function/module objects.
struct xls_dslx_param* xls_dslx_function_get_param(struct xls_dslx_function* fn,
                                                   int64_t index);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_param_get_name(struct xls_dslx_param* p);

// Returns the syntactic type annotation of this parameter as written in the
// DSLX source.
struct xls_dslx_type_annotation* xls_dslx_param_get_type_annotation(
    struct xls_dslx_param* p);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_function_to_string(struct xls_dslx_function* fn);

// Returns the QuickCheck AST node from the given module member. The caller
// should ensure the module member kind is
// `xls_dslx_module_member_kind_quick_check`.
struct xls_dslx_quickcheck* xls_dslx_module_member_get_quickcheck(
    struct xls_dslx_module_member*);

// Retrieves the underlying function associated with the given QuickCheck.
struct xls_dslx_function* xls_dslx_quickcheck_get_function(
    struct xls_dslx_quickcheck*);

// Returns true iff the QuickCheck has the `exhaustive` test-cases specifier.
bool xls_dslx_quickcheck_is_exhaustive(struct xls_dslx_quickcheck*);

// Retrieves the test-case count for the QuickCheck. Returns true and sets
// `*result_out` when the QuickCheck has a counted test-case specifier; returns
// false when the QuickCheck is marked exhaustive (in which case
// `*result_out` is not modified).
bool xls_dslx_quickcheck_get_count(struct xls_dslx_quickcheck*,
                                   int64_t* result_out);

int64_t xls_dslx_module_get_type_definition_count(
    struct xls_dslx_module* module);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_module_get_name(struct xls_dslx_module*);

xls_dslx_type_definition_kind xls_dslx_module_get_type_definition_kind(
    struct xls_dslx_module* module, int64_t i);

struct xls_dslx_struct_def* xls_dslx_module_get_type_definition_as_struct_def(
    struct xls_dslx_module* module, int64_t i);

struct xls_dslx_enum_def* xls_dslx_module_get_type_definition_as_enum_def(
    struct xls_dslx_module* module, int64_t i);

struct xls_dslx_type_alias* xls_dslx_module_get_type_definition_as_type_alias(
    struct xls_dslx_module* module, int64_t i);

// -- type_definition

struct xls_dslx_colon_ref* xls_dslx_type_defintion_get_colon_ref(
    struct xls_dslx_type_definition*);
struct xls_dslx_type_alias* xls_dslx_type_definition_get_type_alias(
    struct xls_dslx_type_definition*);

// -- constant_def

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_constant_def_get_name(struct xls_dslx_constant_def*);

struct xls_dslx_expr* xls_dslx_constant_def_get_value(
    struct xls_dslx_constant_def*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_constant_def_to_string(struct xls_dslx_constant_def*);

// -- struct_def

// Note: the return value is owned by the caller and must be freed via
// `xls_c_str_free`.
char* xls_dslx_struct_def_get_identifier(struct xls_dslx_struct_def*);

bool xls_dslx_struct_def_is_parametric(struct xls_dslx_struct_def*);
int64_t xls_dslx_struct_def_get_member_count(struct xls_dslx_struct_def*);

struct xls_dslx_struct_member* xls_dslx_struct_def_get_member(
    struct xls_dslx_struct_def*, int64_t);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_struct_member_get_name(struct xls_dslx_struct_member*);

struct xls_dslx_type_annotation* xls_dslx_struct_member_get_type(
    struct xls_dslx_struct_member*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_struct_def_to_string(struct xls_dslx_struct_def*);

// -- enum_def (AST node)

// Note: the return value is owned by the caller and must be freed via
// `xls_c_str_free`.
char* xls_dslx_enum_def_get_identifier(struct xls_dslx_enum_def*);

int64_t xls_dslx_enum_def_get_member_count(struct xls_dslx_enum_def*);

struct xls_dslx_enum_member* xls_dslx_enum_def_get_member(
    struct xls_dslx_enum_def*, int64_t);

struct xls_dslx_type_annotation* xls_dslx_enum_def_get_underlying(
    struct xls_dslx_enum_def*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_enum_member_get_name(struct xls_dslx_enum_member*);

struct xls_dslx_expr* xls_dslx_enum_member_get_value(
    struct xls_dslx_enum_member*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_enum_def_to_string(struct xls_dslx_enum_def*);

// Returns the owning module for the given expression AST node.
struct xls_dslx_module* xls_dslx_expr_get_owner_module(
    struct xls_dslx_expr* expr);

// -- type_annotation

// Attempts to convert the given type annotation to a TypeRefTypeAnnotation --
// returns nullptr if the conversion is not viable.
struct xls_dslx_type_ref_type_annotation*
xls_dslx_type_annotation_get_type_ref_type_annotation(
    struct xls_dslx_type_annotation*);

// -- type_ref_type_annotation

struct xls_dslx_type_ref* xls_dslx_type_ref_type_annotation_get_type_ref(
    struct xls_dslx_type_ref_type_annotation*);

// -- type_ref

struct xls_dslx_type_definition* xls_dslx_type_ref_get_type_definition(
    struct xls_dslx_type_ref*);

// -- type_definition

struct xls_dslx_colon_ref* xls_dslx_type_definition_get_colon_ref(
    struct xls_dslx_type_definition*);

// -- import

int64_t xls_dslx_import_get_subject_count(struct xls_dslx_import*);
char* xls_dslx_import_get_subject(struct xls_dslx_import*, int64_t);

// -- colon_ref

struct xls_dslx_import* xls_dslx_colon_ref_resolve_import_subject(
    struct xls_dslx_colon_ref*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_colon_ref_get_attr(struct xls_dslx_colon_ref*);

// -- type_alias

// Note: the return value is owned by the caller and must be freed via
// `xls_c_str_free`.
char* xls_dslx_type_alias_get_identifier(struct xls_dslx_type_alias*);

struct xls_dslx_type_annotation* xls_dslx_type_alias_get_type_annotation(
    struct xls_dslx_type_alias*);

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_type_alias_to_string(struct xls_dslx_type_alias*);

// -- interp_value

// Note: return value is owned by the caller, free via `xls_c_str_free`.
char* xls_dslx_interp_value_to_string(struct xls_dslx_interp_value*);

bool xls_dslx_interp_value_convert_to_ir(struct xls_dslx_interp_value* v,
                                         char** error_out,
                                         struct xls_value** result_out);

void xls_dslx_interp_value_free(struct xls_dslx_interp_value*);

// -- type_info (deduced type information)

// Note: if there is no type information available for the given entity these
// may return null; however, if type checking has completed successfully this
// should not occur in practice.

const struct xls_dslx_type* xls_dslx_type_info_get_type_struct_def(
    struct xls_dslx_type_info*, struct xls_dslx_struct_def*);

const struct xls_dslx_type* xls_dslx_type_info_get_type_struct_member(
    struct xls_dslx_type_info*, struct xls_dslx_struct_member*);

const struct xls_dslx_type* xls_dslx_type_info_get_type_enum_def(
    struct xls_dslx_type_info*, struct xls_dslx_enum_def*);

const struct xls_dslx_type* xls_dslx_type_info_get_type_constant_def(
    struct xls_dslx_type_info*, struct xls_dslx_constant_def*);

const struct xls_dslx_type* xls_dslx_type_info_get_type_type_annotation(
    struct xls_dslx_type_info*, struct xls_dslx_type_annotation*);

// Note: the outparam is owned by the caller and must be freed via
// `xls_dslx_interp_value_free`.
bool xls_dslx_type_info_get_const_expr(
    struct xls_dslx_type_info* type_info, struct xls_dslx_expr* expr,
    char** error_out, struct xls_dslx_interp_value** result_out);

// -- type (deduced type information)

bool xls_dslx_type_get_total_bit_count(const struct xls_dslx_type*,
                                       char** error_out, int64_t* result_out);

// Returns whether the given type is a bits-like type with signedness 'true'.
bool xls_dslx_type_is_signed_bits(const struct xls_dslx_type*, char** error_out,
                                  bool* result_out);

bool xls_dslx_type_to_string(const struct xls_dslx_type*, char** error_out,
                             char** result_out);

// Note: on success the caller owns `is_signed` and `size` and must free them
// via `xls_dslx_type_dim_free`.
bool xls_dslx_type_is_bits_like(struct xls_dslx_type*,
                                struct xls_dslx_type_dim** is_signed,
                                struct xls_dslx_type_dim** size);

bool xls_dslx_type_is_enum(const struct xls_dslx_type*);

bool xls_dslx_type_is_struct(const struct xls_dslx_type*);

bool xls_dslx_type_is_array(const struct xls_dslx_type*);

// Precondition: xls_dslx_type_is_enum
struct xls_dslx_enum_def* xls_dslx_type_get_enum_def(struct xls_dslx_type*);

// Precondition: xls_dslx_type_is_struct
struct xls_dslx_struct_def* xls_dslx_type_get_struct_def(struct xls_dslx_type*);

// Precondition: xls_dslx_type_is_array
struct xls_dslx_type* xls_dslx_type_array_get_element_type(
    struct xls_dslx_type*);

// Note: returned xls_dslx_type_dim is owned by the caller and must be
// deallocated.
struct xls_dslx_type_dim* xls_dslx_type_array_get_size(struct xls_dslx_type*);

// -- type_dim (deduced type information)

bool xls_dslx_type_dim_is_parametric(struct xls_dslx_type_dim*);

// Determines whether `function` requires the implicit-token calling
// convention. Returns true on success and sets `*result_out` to the answer.
// On failure (e.g. no information recorded for that function) returns false
// and populates `*error_out` with a description (caller must free via
// `xls_c_str_free`).
bool xls_dslx_type_info_get_requires_implicit_token(
    struct xls_dslx_type_info* type_info, struct xls_dslx_function* function,
    char** error_out, bool* result_out);

bool xls_dslx_type_dim_get_as_bool(struct xls_dslx_type_dim*, char** error_out,
                                   bool* result_out);

bool xls_dslx_type_dim_get_as_int64(struct xls_dslx_type_dim*, char** error_out,
                                    int64_t* result_out);

void xls_dslx_type_dim_free(struct xls_dslx_type_dim*);

// Precondition: `type` must be an array type.
struct xls_dslx_type* xls_dslx_type_array_get_element_type(
    struct xls_dslx_type* type);

struct xls_dslx_type_dim* xls_dslx_type_array_get_size(
    struct xls_dslx_type* type);

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_DSLX_H_

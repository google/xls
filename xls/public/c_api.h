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

#ifndef XLS_PUBLIC_C_API_H_
#define XLS_PUBLIC_C_API_H_

#include <stddef.h>  // NOLINT(modernize-deprecated-headers)
#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

// C API that exposes the functionality in various public headers in a way that
// C-based FFI facilities can easily wrap.
//
// Note that StatusOr from C++ is generally translated as:
//      StatusOr<T> MyFunction(...) =>
//      bool MyFunction(..., char** error_out, T* out)
//
// The boolean return value indicates "ok" -- if not ok, the `error_out` value
// will be populated with an error string indicating what went wrong -- the
// string will be owned by the caller and will need to be deallocated in the
// case of error.
//
// Caller-owned C strings are created using C standard library facilities and
// thus should be deallocated via `free`.
//
// **WARNING**: These are *not* meant to be *ABI-stable* -- assume you have to
// re-compile against this header for any given XLS commit.

extern "C" {

// Opaque structs.
struct xls_value;
struct xls_package;
struct xls_function;
struct xls_type;
struct xls_function_type;

void xls_init_xls(const char* usage, int argc, char* argv[]);

bool xls_convert_dslx_to_ir(const char* dslx, const char* path,
                            const char* module_name,
                            const char* dslx_stdlib_path,
                            const char* additional_search_paths[],
                            size_t additional_search_paths_count,
                            char** error_out, char** ir_out);

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out);

bool xls_optimize_ir(const char* ir, const char* top, char** error_out,
                     char** ir_out);

bool xls_mangle_dslx_name(const char* module_name, const char* function_name,
                          char** error_out, char** mangled_out);

// Parses a string that represents a typed XLS value; e.g. `bits[32]:0x42`.
bool xls_parse_typed_value(const char* input, char** error_out,
                           struct xls_value** xls_value_out);

// Returns a new token XLS value which the caller must free.
struct xls_value* xls_value_make_token();

// Returns a new `bits[1]:1` XLS value which the caller must free.
struct xls_value* xls_value_make_true();

// Returns a new `bits[1]:0` XLS value which the caller must free.
struct xls_value* xls_value_make_false();

// Returns a string representation of the given value `v`.
bool xls_value_to_string(const struct xls_value* v, char** string_out);

// Returns whether `v` is equal to `w`.
bool xls_value_eq(const struct xls_value* v, const struct xls_value* w);

// Note: We define the format preference enum with a fixed width integer type
// for clarity of the exposed ABI.
typedef int32_t xls_format_preference;
enum {
  xls_format_preference_default,
  xls_format_preference_binary,
  xls_format_preference_signed_decimal,
  xls_format_preference_unsigned_decimal,
  xls_format_preference_hex,
  xls_format_preference_plain_binary,
  xls_format_preference_plain_hex,
};

// Returns a format preference enum value from a string specifier; i.e.
// `xls_format_preference_from_string("hex")` returns the value of
// `xls_format_preference_hex` -- this is particularly useful for language
// bindings that don't parse the C headers to determine enumerated values.
bool xls_format_preference_from_string(const char* s, char** error_out,
                                       xls_format_preference* result_out);

// Returns the given value `v` converted to a string by way of the given
// `format_preference`.
bool xls_value_to_string_format_preference(
    const struct xls_value* v, xls_format_preference format_preference,
    char** error_out, char** result_out);

// Deallocates a value, e.g. one as created by `xls_parse_typed_value`.
void xls_value_free(struct xls_value* v);

void xls_package_free(struct xls_package* p);

// Frees the given `c_str` -- the C string should have been allocated by the
// XLS library where ownership was passed back to the caller.
//
// `c_str` may be null, in which case this function does nothing.
//
// e.g. `xls_convert_dslx_to_ir` gives back `ir_out` which can be deallocated
// by this function.
//
// This function is primarily useful when the underlying allocator may be
// different between the caller and the XLS library (otherwise the caller could
// just call `free` directly).
void xls_c_str_free(char* c_str);

// Returns a string representation of the given IR package `p`.
bool xls_package_to_string(const struct xls_package* p, char** string_out);

// Parses IR text to a package.

// Note: `filename` may be nullptr.
bool xls_parse_ir_package(const char* ir, const char* filename,
                          char** error_out,
                          struct xls_package** xls_package_out);

// Returns a function contained within the given `package`.
//
// Note: the returned function does not need to be freed, it is tied to the
// package's lifetime.
bool xls_package_get_function(struct xls_package* package,
                              const char* function_name, char** error_out,
                              struct xls_function** result_out);

// Returns the type of the given value, as owned by the given package.
//
// Note: the returned type does not need to be freed, it is tied to the
// package's lifetime.
bool xls_package_get_type_for_value(struct xls_package* package,
                                    struct xls_value* value, char** error_out,
                                    struct xls_type** result_out);

// Returns the string representation of the type.
bool xls_type_to_string(struct xls_type* type, char** error_out,
                        char** result_out);

// Returns the type of the given function.
//
// Note: the returned type does not need to be freed, it is tied to the
// package's lifetime.
bool xls_function_get_type(struct xls_function* function, char** error_out,
                           struct xls_function_type** xls_fn_type_out);

// Returns the name of the given function `function` -- `string_out` is owned
// by the caller and must be freed.
bool xls_function_get_name(struct xls_function* function, char** error_out,
                           char** string_out);

// Returns a string representation of the given `xls_function_type`.
bool xls_function_type_to_string(struct xls_function_type* xls_function_type,
                                 char** error_out, char** string_out);

// Interprets the given `function` using the given `args` (an array of size
// `argc`) -- interpretation runs to a function result placed in `result_out`,
// or `error_out` is populated and false is returned in the event of an error.
bool xls_interpret_function(struct xls_function* function, size_t argc,
                            const struct xls_value** args, char** error_out,
                            struct xls_value** result_out);

// -- VAST (Verilog AST) APIs
//
// Note that these are expected to be *less* stable than the above APIs, as
// they are exposing a useful implementation library present within XLS.
//
// Per usual, in a general sense, no promises are made around API or ABI
// stability overall. However, seems worth noting these are effectively
// "protected" APIs, use with particular caution around stability. See
// `xls/protected/BUILD` for how we tend to think about "protected" APIs in the
// project.

// Opaque structs.
struct xls_vast_verilog_file;
struct xls_vast_verilog_module;

// Note: We define the enum with a fixed width integer type for clarity of the
// exposed ABI.
typedef int32_t xls_vast_file_type;
enum {
  xls_vast_file_type_verilog,
  xls_vast_file_type_system_verilog,
};

struct xls_vast_verilog_file* xls_vast_make_verilog_file(
    xls_vast_file_type file_type);
void xls_vast_verilog_file_free(struct xls_vast_verilog_file* f);

struct xls_vast_verilog_module* xls_vast_verilog_file_add_module(
    struct xls_vast_verilog_file* f, const char* name);
void xls_vast_verilog_file_add_include(struct xls_vast_verilog_file* f,
                                       const char* path);

char* xls_vast_verilog_file_emit(const struct xls_vast_verilog_file* f);

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_H_

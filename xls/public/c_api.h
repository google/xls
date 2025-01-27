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

#include "xls/public/c_api_dslx.h"
#include "xls/public/c_api_format_preference.h"
#include "xls/public/c_api_vast.h"

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
struct xls_function_base;
struct xls_type;
struct xls_function_type;
struct xls_schedule_and_codegen_result;

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

// Args:
//   p: The package to schedule and codegen.
//   scheduling_options_flags_proto: The scheduling options flags proto in
//   textproto form. codegen_flags_proto: The codegen options flags proto in
//   textproto form. error_out: Populated with a string message on error
//   (outparam). result_out: The result output (outparam).
//
// Returns `true` on success, `false` on error.
bool xls_schedule_and_codegen_package(
    struct xls_package* p, const char* scheduling_options_flags_proto,
    const char* codegen_flags_proto, bool with_delay_model, char** error_out,
    struct xls_schedule_and_codegen_result** result_out);

// Note: the returned string is owned by the caller and must be freed via
// `xls_c_str_free`.
char* xls_schedule_and_codegen_result_get_verilog_text(
    const struct xls_schedule_and_codegen_result* result);

void xls_schedule_and_codegen_result_free(
    struct xls_schedule_and_codegen_result* result);

// Parses a string that represents a typed XLS value; e.g. `bits[32]:0x42`.
bool xls_parse_typed_value(const char* input, char** error_out,
                           struct xls_value** xls_value_out);

// Returns a new token XLS value which the caller must free.
struct xls_value* xls_value_make_token();

// Returns a new `bits[1]:1` XLS value which the caller must free.
struct xls_value* xls_value_make_true();

// Returns a new tuple-kind XLS value which the caller must free.
struct xls_value* xls_value_make_tuple(size_t element_count,
                                       struct xls_value** elements);

// Attempts to extract a "bits" value from the given XLS value -- the resulting
// `bits_out` is owned by the caller and must be freed via `xls_bits_free()` on
// success.
bool xls_value_get_bits(const struct xls_value* value, char** error_out,
                        struct xls_bits** bits_out);

int64_t xls_bits_get_bit_count(const struct xls_bits* bits);

// Returns a string representation of the given bits value.
//
// Note: the returned string is owned by the caller and must be freed via
// `xls_c_str_free`.
char* xls_bits_to_debug_string(const struct xls_bits* bits);

// Helper routine for making an unsigned bits value using a value that fits in a
// 64-bit word.
//
// `bit_count` must be large enough to hold the value or an error is returned.
bool xls_bits_make_ubits(int64_t bit_count, uint64_t value, char** error_out,
                         struct xls_bits** bits_out);

// As above but for a signed bit value -- `bit_count` must be large enough to
// hold the value assuming sign extension or an error is returned.
bool xls_bits_make_sbits(int64_t bit_count, int64_t value, char** error_out,
                         struct xls_bits** bits_out);

void xls_bits_free(struct xls_bits* bits);

bool xls_bits_eq(const struct xls_bits* a, const struct xls_bits* b);

// Returns the bit at the given index, where `index` is a zero-is-lsb value.
//
// That is, if the bits value is `0b01`, then `xls_bits_get_bit(bits, 0)`
// returns 1, `xls_bits_get_bit(bits, 1)` returns 0.
bool xls_bits_get_bit(const struct xls_bits* bits, int64_t index);

struct xls_bits* xls_bits_width_slice(const struct xls_bits* bits,
                                      int64_t start, int64_t width);

struct xls_bits* xls_bits_shift_left_logical(const struct xls_bits* bits,
                                             int64_t shift_amount);

struct xls_bits* xls_bits_shift_right_logical(const struct xls_bits* bits,
                                              int64_t shift_amount);

struct xls_bits* xls_bits_shift_right_arithmetic(const struct xls_bits* bits,
                                                 int64_t shift_amount);

struct xls_bits* xls_bits_negate(const struct xls_bits* bits);

struct xls_bits* xls_bits_abs(const struct xls_bits* bits);

struct xls_bits* xls_bits_not(const struct xls_bits* bits);

struct xls_bits* xls_bits_add(const struct xls_bits* lhs,
                              const struct xls_bits* rhs);

struct xls_bits* xls_bits_sub(const struct xls_bits* lhs,
                              const struct xls_bits* rhs);

struct xls_bits* xls_bits_and(const struct xls_bits* lhs,
                              const struct xls_bits* rhs);

struct xls_bits* xls_bits_or(const struct xls_bits* lhs,
                             const struct xls_bits* rhs);

struct xls_bits* xls_bits_xor(const struct xls_bits* lhs,
                              const struct xls_bits* rhs);

struct xls_bits* xls_bits_umul(const struct xls_bits* lhs,
                               const struct xls_bits* rhs);

struct xls_bits* xls_bits_smul(const struct xls_bits* lhs,
                               const struct xls_bits* rhs);

// Returns a new `bits[1]:0` XLS value which the caller must free.
struct xls_value* xls_value_make_false();

// Returns a string representation of the given value `v`.
bool xls_value_to_string(const struct xls_value* v, char** string_out);

// Returns whether `v` is equal to `w`.
bool xls_value_eq(const struct xls_value* v, const struct xls_value* w);

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

bool xls_bits_to_string(const struct xls_bits* bits,
                        xls_format_preference format_preference,
                        bool include_bit_count, char** error_out,
                        char** result_out);

// Deallocates a value, e.g. one as created by `xls_parse_typed_value`.
void xls_value_free(struct xls_value* v);

struct xls_value* xls_value_from_bits(const struct xls_bits* bits);

// Flattens the given value to a sequence of bits in a bits "buffer" value.
//
// Note that in a tuple or array the earlier fields/members are stored in the
// most significant bits of the result value.
//
// Note: the returned bits buffer is owned by the caller and must be freed via
// `xls_bits_free`.
struct xls_bits* xls_value_flatten_to_bits(const struct xls_value* v);

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

// Returns the "top" (i.e. entry point) function base (i.e. function or proc) of
// the given package.
struct xls_function_base* xls_package_get_top(struct xls_package* p);

// Sets the top function base of the given package to the function base with the
// given name.
bool xls_package_set_top_by_name(struct xls_package* p, const char* name,
                                 char** error_out);

// Parses IR text to a package.
//
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

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_H_

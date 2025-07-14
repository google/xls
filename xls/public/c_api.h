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
#include "xls/public/c_api_ir_builder.h"
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

typedef int32_t xls_value_kind;
enum {
  xls_value_kind_invalid,
  xls_value_kind_bits,
  xls_value_kind_array,
  xls_value_kind_tuple,
  xls_value_kind_token,
};

// Opaque structs.
struct xls_bits;
struct xls_bits_rope;
struct xls_function;
struct xls_function_base;
struct xls_function_jit;
struct xls_function_type;
struct xls_package;
struct xls_schedule_and_codegen_result;
struct xls_type;
struct xls_value;

void xls_init_xls(const char* usage, int argc, char* argv[]);

bool xls_convert_dslx_to_ir(const char* dslx, const char* path,
                            const char* module_name,
                            const char* dslx_stdlib_path,
                            const char* additional_search_paths[],
                            size_t additional_search_paths_count,
                            char** error_out, char** ir_out);

// As above, but also takes `enable_warnings` and `disable_warnings` which
// are arrays of warning names to enable or disable respectively against the
// default warning set.
//
// Precondition: if `warnings_out` is provided then `warnings_out_count` must
// also be provided.
//
// Note that if the `warnings_out_count` is populated with zero (i.e. there were
// no warnings) then we can return `nullptr` for `warnings_out`.
bool xls_convert_dslx_to_ir_with_warnings(
    const char* dslx, const char* path, const char* module_name,
    const char* dslx_stdlib_path, const char* additional_search_paths[],
    size_t additional_search_paths_count, const char* enable_warnings[],
    size_t enable_warnings_count, const char* disable_warnings[],
    size_t disable_warnings_count, bool warnings_as_errors,
    char*** warnings_out, size_t* warnings_out_count, char** error_out,
    char** ir_out);

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out);

// As above `xls_convert_dslx_to_ir_with_warnings`, but for a filesystem path
// instead of DSLX text.
bool xls_convert_dslx_path_to_ir_with_warnings(
    const char* path, const char* dslx_stdlib_path,
    const char* additional_search_paths[], size_t additional_search_paths_count,
    const char* enable_warnings[], size_t enable_warnings_count,
    const char* disable_warnings[], size_t disable_warnings_count,
    bool warnings_as_errors, char*** warnings_out, size_t* warnings_out_count,
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

bool xls_value_make_ubits(int64_t bit_count, uint64_t value, char** error_out,
                          struct xls_value** xls_value_out);

// Returns a new bits rope with the given bit count. The caller must free the
// returned `bits_rope` via `xls_bits_rope_free` if it is non-null.
struct xls_bits_rope* xls_create_bits_rope(int64_t bit_count);

// Appends the given bits to the given bits rope.
// Note the bits rope must have enough space to hold the bits.
void xls_bits_rope_append_bits(struct xls_bits_rope* bits_rope,
                               const struct xls_bits* bits);

// Returns the bits in the given bits rope. The returned bits are owned by
// the caller and must be freed via `xls_bits_free`. This API must be called
// after all bits have been appended to the bits rope, i.e. the bits rope must
// be "full".
struct xls_bits* xls_bits_rope_get_bits(struct xls_bits_rope* bits_rope);

// Returns the kind of the given value. The caller must free the returned
// `error_out` string via `xls_c_str_free` if an error is returned.
bool xls_value_get_kind(const struct xls_value* value, char** error_out,
                        xls_value_kind* kind_out);

bool xls_value_make_sbits(int64_t bit_count, int64_t value, char** error_out,
                          struct xls_value** xls_value_out);

// Returns a new array-kind XLS value which the caller must free. There must be
// at least one element and all provided elements must be of the same type
// otherwise an error result is returned.
bool xls_value_make_array(size_t element_count, struct xls_value** elements,
                          char** error_out, struct xls_value** result_out);

// Returns a new token XLS value which the caller must free.
struct xls_value* xls_value_make_token();

// Returns a new `bits[1]:1` XLS value which the caller must free.
struct xls_value* xls_value_make_true();

// Returns a new tuple-kind XLS value which the caller must free.
struct xls_value* xls_value_make_tuple(size_t element_count,
                                       struct xls_value** elements);

// Returns a clone of the given value -- the caller must free the returned value
// via `xls_value_free`.
struct xls_value* xls_value_clone(const struct xls_value* value);

// Returns the element at the given index in the value -- the value must be an
// aggregate (i.e. tuple or array) or an error is returned.
bool xls_value_get_element(const struct xls_value* value, size_t index,
                           char** error_out, struct xls_value** element_out);

// Returns the number of elements in the given value -- the value must be an
// aggregate (i.e. tuple or array) or an error is returned.
bool xls_value_get_element_count(const struct xls_value* value,
                                 char** error_out, int64_t* count_out);

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

// Returns a new bits value from the given bytes array. On success,
// the result is saved in `bits_out` and the caller must free the
// returned `bits_out` via `xls_bits_free`. If the conversion fails,
// `error_out` is populated with an error message and the caller must free the
// `error_out` pointer with `xls_c_str_free`.
bool xls_bits_make_bits_from_bytes(size_t bit_count, const uint8_t* bytes,
                                   size_t byte_count, char** error_out,
                                   struct xls_bits** bits_out);

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
void xls_bits_rope_free(xls_bits_rope* b);

bool xls_bits_eq(const struct xls_bits* a, const struct xls_bits* b);

// Returns the bit at the given index, where `index` is a zero-is-lsb value.
//
// That is, if the bits value is `0b01`, then `xls_bits_get_bit(bits, 0)`
// returns 1, `xls_bits_get_bit(bits, 1)` returns 0.
bool xls_bits_get_bit(const struct xls_bits* bits, int64_t index);

// Converts the given bits value to a byte array, whose length is saved in
// byte_count_out. The caller must free the returned `bytes_out` pointer
// via `xls_bytes_free` if xls_bits_to_bytes returns true.
// If the conversion fails and it returns false, error_out is populated
// with an error message and the caller must free the
// `error_out` pointer with `xls_c_str_free`.
bool xls_bits_to_bytes(const struct xls_bits* bits, char** error_out,
                       uint8_t** bytes_out, size_t* byte_count_out);

// Converts the given bits value to a signed or unsigned integer 64-bit integer
// value. If the conversion is not possible, false is returned, `error_out`
// contains the error message and the caller must free the `error_out` pointer
// with `xls_c_str_free`.
bool xls_bits_to_uint64(const struct xls_bits* bits, char** error_out,
                        uint64_t* value_out);
bool xls_bits_to_int64(const struct xls_bits* bits, char** error_out,
                       int64_t* value_out);
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

// Returns a value (box) that wraps the given bits value.
//
// No ownership is taken over the given bits value. See
// `xls_value_from_bits_owned` if you want to "gift" the value to the API.
struct xls_value* xls_value_from_bits(const struct xls_bits* bits);

// As above but takes ownership of the bits value (so it should no longer be
// freed by the caller).
struct xls_value* xls_value_from_bits_owned(struct xls_bits* bits);

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

// Frees an array of C strings that were provided by this XLS public API.
void xls_c_strs_free(char** c_strs, size_t count);

// Frees the given `bytes` -- the bytes should have been allocated by the
// XLS library where ownership was passed back to the caller.
void xls_bytes_free(uint8_t* bytes);

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

// Returns all the functions contained within the given `package`.
//
// If successful, true us returned and the contained function pointers are
// saved in the array `result_out` which is owned by the caller and must be
// freed with `xls_function_ptr_array_free`. The length of the array is saved in
// `count_out`.
// If an error occurs, false is returned and `error_out` is populated with an
// error message, which must be freed with `xls_c_str_free`.
// Note: the returned functions do not need to be freed, they are tied to the
// package's lifetime.
bool xls_package_get_functions(struct xls_package* package, char** error_out,
                               struct xls_function*** result_out,
                               size_t* count_out);

// Returns the type of the given value, as owned by the given package.
//
// Note: the returned type does not need to be freed, it is tied to the
// package's lifetime.
bool xls_package_get_type_for_value(struct xls_package* package,
                                    struct xls_value* value, char** error_out,
                                    struct xls_type** result_out);

// Returns the kind of the given type.
bool xls_type_get_kind(struct xls_type* type, char** error_out,
                       xls_value_kind* kind_out);

// Returns the string representation of the type.
bool xls_type_to_string(struct xls_type* type, char** error_out,
                        char** result_out);

// Returns the count of bits required to represent the type.
int64_t xls_type_get_flat_bit_count(struct xls_type* type);

// Returns the number of Bits types contained in the type.
int64_t xls_type_get_leaf_count(struct xls_type* type);

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

// Returns the name of the parameter at the given `index` in `function` --
// `name_out` is owned by the caller and must be freed.
bool xls_function_get_param_name(struct xls_function* function, size_t index,
                                 char** error_out, char** name_out);

// Returns a string representation of the given `xls_function_type`.
bool xls_function_type_to_string(struct xls_function_type* xls_function_type,
                                 char** error_out, char** string_out);

// Returns the number of parameters for the given function type.
int64_t xls_function_type_get_param_count(struct xls_function_type* type);

// Returns the type of the parameter at the given index.
// On success `param_type_out` is populated and the function returns true.
// If the index is out of range, returns false and `error_out` is populated.
bool xls_function_type_get_param_type(struct xls_function_type* type,
                                      size_t index, char** error_out,
                                      struct xls_type** param_type_out);

// Returns the return type of the given function type.
struct xls_type* xls_function_type_get_return_type(
    struct xls_function_type* type);

// Interprets the given `function` using the given `args` (an array of size
// `argc`) -- interpretation runs to a function result placed in `result_out`,
// or `error_out` is populated and false is returned in the event of an error.
bool xls_interpret_function(struct xls_function* function, size_t argc,
                            const struct xls_value* const* args,
                            char** error_out, struct xls_value** result_out);

bool xls_make_function_jit(struct xls_function* function, char** error_out,
                           struct xls_function_jit** result_out);

void xls_function_jit_free(struct xls_function_jit* jit);

struct xls_trace_message {
  char* message;
  int64_t verbosity;
};

// Runs the given `jit` function with the given `args` (an array of size `argc`)
// and returns the result in `result_out`.
//
// Note:
// * `trace_messages_out` should be freed by the caller via
//   `xls_trace_messages_free`.
// * `assert_messages_out` should be freed by the caller via
//   `xls_c_strs_free`.
bool xls_function_jit_run(struct xls_function_jit* jit, size_t argc,
                          const struct xls_value* const* args, char** error_out,
                          struct xls_trace_message** trace_messages_out,
                          size_t* trace_messages_count_out,
                          char*** assert_messages_out,
                          size_t* assert_messages_count_out,
                          struct xls_value** result_out);

// frees the array of  `xls_function` pointers -- the function should have been
// allocated by the XLS library where ownership was passed back to the caller.
void xls_function_ptr_array_free(struct xls_function** functions);

void xls_trace_messages_free(struct xls_trace_message* trace_messages,
                             size_t count);

// Returns the SMTLIB2-compliant string representation of the given function
// (expressed as a Z3 bit-vector expression of the function return value).
//
// The result is the Z3-printed SMT-LIB2 form of a lambda that binds every
// function parameter and yields the function body expression (using bit-vector
// operators).  For example, a two-parameter 32-bit adder becomes:
//   (lambda ((x (_ BitVec 32)) (y (_ BitVec 32))) (bvadd x y))
// The returned string is owned by the caller and must be freed with
// `xls_c_str_free`.
//
// Returns `true` on success and populates `string_out`; otherwise returns
// `false` and populates `error_out`.
bool xls_function_to_z3_smtlib(struct xls_function* function, char** error_out,
                               char** string_out);

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_H_

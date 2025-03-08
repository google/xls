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

// XLS IR builder APIs -- this provides the ability to programmatically build up
// XLS IR.

#ifndef XLS_PUBLIC_C_API_IR_BUILDER_H_
#define XLS_PUBLIC_C_API_IR_BUILDER_H_

#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

extern "C" {

struct xls_package;
struct xls_function;
struct xls_type;
struct xls_builder_base;
struct xls_function_builder;
struct xls_bvalue;

// Note: `xls_package_free` is available in `c_api.h`.
struct xls_package* xls_package_create(const char* name);

// Note: for type creation APIs the returned value is owned by the package and
// tied to its lifetime, it does not need to be freed.

struct xls_type* xls_package_get_bits_type(struct xls_package* package,
                                           int64_t bit_count);

struct xls_type* xls_package_get_tuple_type(struct xls_package* package,
                                            struct xls_type** members,
                                            int64_t member_count);

struct xls_function_builder* xls_function_builder_create(
    const char* name, struct xls_package* package, bool should_verify);

// Note that the returned value is the same underlying value as the given
// `fn_builder` -- therefore, the returned value should only be used within the
// lifetime of the given `fn_builder`.
struct xls_builder_base* xls_function_builder_as_builder_base(
    struct xls_function_builder* fn_builder);

void xls_function_builder_free(struct xls_function_builder* builder);

void xls_bvalue_free(struct xls_bvalue* bvalue);

struct xls_bvalue* xls_function_builder_add_parameter(
    struct xls_function_builder* builder, const char* name,
    struct xls_type* type);

bool xls_function_builder_build(struct xls_function_builder* builder,
                                char** error_out,
                                struct xls_function** function_out);

bool xls_function_builder_build_with_return_value(
    struct xls_function_builder* builder, struct xls_bvalue* return_value,
    char** error_out, struct xls_function** function_out);

// -- xls_builder_base
// Notes that apply to all of these functions:
// * `name` is optional and can be nullptr
// * the bvalue return value can only be used with the builder that it was
//   created for
// * the bvalues returned (unfortunately) must all be freed by the caller, as
//   there is no ABI friendly way to return the bvalues by value as is done in
//   the C++ API
//
// Note that the builder API is "fluent/monadic style" where error status is
// accumulated internally until the ultimate attempt to build or similar (e.g.
// inspecting the type of some built value that chained from an error), which is
// why these function signatures do not produce errors.

struct xls_bvalue* xls_builder_base_add_and(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name);

struct xls_bvalue* xls_builder_base_add_or(struct xls_builder_base* builder,
                                           struct xls_bvalue* lhs,
                                           struct xls_bvalue* rhs,
                                           const char* name);

struct xls_bvalue* xls_builder_base_add_not(struct xls_builder_base* builder,
                                            struct xls_bvalue* value,
                                            const char* name);

struct xls_bvalue* xls_builder_base_add_literal(
    struct xls_builder_base* builder, struct xls_value* value,
    const char* name);

struct xls_bvalue* xls_builder_base_add_tuple(struct xls_builder_base* builder,
                                              struct xls_bvalue** operands,
                                              int64_t operand_count,
                                              const char* name);

struct xls_bvalue* xls_builder_base_add_tuple_index(
    struct xls_builder_base* builder, struct xls_bvalue* tuple, int64_t index,
    const char* name);

struct xls_bvalue* xls_builder_base_add_bit_slice(
    struct xls_builder_base* builder, struct xls_bvalue* value, int64_t start,
    int64_t width, const char* name);

struct xls_bvalue* xls_builder_base_add_dynamic_bit_slice(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    struct xls_bvalue* start, int64_t width, const char* name);

struct xls_bvalue* xls_builder_base_add_concat(struct xls_builder_base* builder,
                                               struct xls_bvalue** operands,
                                               int64_t operand_count,
                                               const char* name);

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_IR_BUILDER_H_

// Copyright 2026 The XLS Authors
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

#ifndef XLS_PUBLIC_C_API_IR_ANALYSIS_H_
#define XLS_PUBLIC_C_API_IR_ANALYSIS_H_

#include <stddef.h>  // NOLINT(modernize-deprecated-headers)
#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

// Protected C API for IR analysis (known bits / range information).
//
// Like the rest of the XLS C API, results are returned via outparams and
// `error_out` is populated with an error string (owned by the caller) on
// failure. Error strings must be freed via `xls_c_str_free`.
//
// **WARNING**: This API is *not* meant to be *ABI-stable* -- assume you have to
// re-compile against this header for any given XLS commit.
//
// In this API, node queries are keyed by `xls::Node::id()` (an `int64_t`). The
// C API does not yet expose a node enumeration facility; callers are expected
// to obtain node ids out-of-band (e.g. from IR text dumps).

extern "C" {

// Forward declarations for types defined in other C API headers.
struct xls_bits;
struct xls_package;

// Opaque analysis handle. Owns analysis state and (optionally) the IR package.
struct xls_ir_analysis;

// Opaque interval set handle (bits-typed values only in this API).
struct xls_interval_set;

// Creates an analysis handle from an existing package.
//
// Precondition: `p` must have a top function base set.
//
// Note: The returned analysis handle does not take ownership of `p`; the
// package must outlive the analysis handle.
bool xls_ir_analysis_create_from_package(struct xls_package* p,
                                         char** error_out,
                                         struct xls_ir_analysis** out);

void xls_ir_analysis_free(struct xls_ir_analysis* a);

// Returns known-bits information for the bits-typed node with the given id.
//
// On success, returns a `known_mask` and `known_value` pair as `xls_bits*`:
// - `known_mask` has a 1 bit where the corresponding bit in the node is known.
// - `known_value` holds the known bit values (and zero in unknown positions).
//
// The returned `xls_bits*` values are owned by the caller and must be freed via
// `xls_bits_free`.
//
// Returns an error if `node_id` is not found, or if the node is not bits-typed.
bool xls_ir_analysis_get_known_bits_for_node_id(
    const struct xls_ir_analysis* a, int64_t node_id, char** error_out,
    struct xls_bits** known_mask_out, struct xls_bits** known_value_out);

// Returns range information for the bits-typed node with the given id.
//
// The result is returned as an opaque interval set. Use the interval set APIs
// below to inspect it.
//
// Returns an error if `node_id` is not found, or if the node is not bits-typed.
bool xls_ir_analysis_get_intervals_for_node_id(
    const struct xls_ir_analysis* a, int64_t node_id, char** error_out,
    struct xls_interval_set** intervals_out);

// Interval set inspection APIs.
int64_t xls_interval_set_get_interval_count(const struct xls_interval_set* s);

// Returns the inclusive bounds of the i-th interval.
//
// The returned bounds are owned by the caller and must be freed via
// `xls_bits_free`.
bool xls_interval_set_get_interval_bounds(const struct xls_interval_set* s,
                                          int64_t i, char** error_out,
                                          struct xls_bits** lo_out,
                                          struct xls_bits** hi_out);

void xls_interval_set_free(struct xls_interval_set* s);

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_IR_ANALYSIS_H_

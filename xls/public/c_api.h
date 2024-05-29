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

#include <stddef.h>

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

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_H_

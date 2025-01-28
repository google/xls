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

// Implementation helper functions for implementing the public C API -- note
// these are not helpers for /using/ the C API, and so will not need to be
// included by client code.

#ifndef XLS_PUBLIC_C_API_IMPL_HELPERS_H_
#define XLS_PUBLIC_C_API_IMPL_HELPERS_H_

#include <cstddef>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/format_preference.h"
#include "xls/public/c_api_format_preference.h"

namespace xls {

// Converts a C-API-given format preference value into the XLS internal enum --
// since these values can be out of normal enum class range when passed to the
// API, we validate here.
bool FormatPreferenceFromC(xls_format_preference c_pref,
                           xls::FormatPreference* cpp_pref, char** error_out);

// Helper function that we can use to adapt to the common C API pattern when
// we're returning an `absl::StatusOr<std::string>` value.
bool ReturnStringHelper(absl::StatusOr<std::string>& to_return,
                        char** error_out, char** value_out);

// Returns the payload of `s` as a C string allocated by libc and owned by the
// caller.
char* ToOwnedCString(std::string_view s);

// Converts the C representation of filesystem search paths into a C++
// representation.
std::vector<std::filesystem::path> ToCpp(const char* additional_search_paths[],
                                         size_t additional_search_paths_count);

}  // namespace xls

#endif  // XLS_PUBLIC_C_API_IMPL_HELPERS_H_

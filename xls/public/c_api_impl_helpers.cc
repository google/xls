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

#include "xls/public/c_api_impl_helpers.h"

#include <filesystem>  // NOLINT
#include <vector>

#include "absl/log/check.h"

namespace xls {

std::vector<std::filesystem::path> ToCpp(const char* additional_search_paths[],
                                         size_t additional_search_paths_count) {
  std::vector<std::filesystem::path> additional_search_paths_cpp;
  additional_search_paths_cpp.reserve(additional_search_paths_count);
  for (size_t i = 0; i < additional_search_paths_count; ++i) {
    const char* additional_search_path = additional_search_paths[i];
    CHECK(additional_search_path != nullptr);
    additional_search_paths_cpp.push_back(additional_search_path);
  }
  return additional_search_paths_cpp;
}

char* ToOwnedCString(const std::string& s) { return strdup(s.c_str()); }

// Helper function that we can use to adapt to the common C API pattern when
// we're returning an `absl::StatusOr<std::string>` value.
bool ReturnStringHelper(absl::StatusOr<std::string>& to_return,
                        char** error_out, char** value_out) {
  if (to_return.ok()) {
    *value_out = ToOwnedCString(to_return.value());
    *error_out = nullptr;
    return true;
  }

  *value_out = nullptr;
  *error_out = ToOwnedCString(to_return.status().ToString());
  return false;
}

// Converts a C-API-given format preference value into the XLS internal enum --
// since these values can be out of normal enum class range when passed to the
// API, we validate here.
bool FormatPreferenceFromC(xls_format_preference c_pref,
                           xls::FormatPreference* cpp_pref, char** error_out) {
  switch (c_pref) {
    case xls_format_preference_default:
      *cpp_pref = xls::FormatPreference::kDefault;
      break;
    case xls_format_preference_binary:
      *cpp_pref = xls::FormatPreference::kBinary;
      break;
    case xls_format_preference_signed_decimal:
      *cpp_pref = xls::FormatPreference::kSignedDecimal;
      break;
    case xls_format_preference_unsigned_decimal:
      *cpp_pref = xls::FormatPreference::kUnsignedDecimal;
      break;
    case xls_format_preference_hex:
      *cpp_pref = xls::FormatPreference::kHex;
      break;
    case xls_format_preference_plain_binary:
      *cpp_pref = xls::FormatPreference::kPlainBinary;
      break;
    case xls_format_preference_plain_hex:
      *cpp_pref = xls::FormatPreference::kPlainHex;
      break;
    default:
      *error_out = ToOwnedCString(
          absl::StrFormat("Invalid format preference value: %d", c_pref));
      return false;
  }
  return true;
}

}  // namespace xls

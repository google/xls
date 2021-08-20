// Copyright 2020 The XLS Authors
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

#include "xls/ir/format_preference.h"

#include "absl/strings/str_format.h"

namespace xls {

absl::string_view FormatPreferenceToString(FormatPreference preference) {
  switch (preference) {
    case FormatPreference::kDefault:
      return "default";
    case FormatPreference::kBinary:
      return "binary";
    case FormatPreference::kDecimal:
      return "decimal";
    case FormatPreference::kHex:
      return "hex";
    case FormatPreference::kPlainBinary:
      return "plain_binary";
    case FormatPreference::kPlainHex:
      return "plain_hex";
    default:
      return "<invalid format preference>";
  }
}

absl::StatusOr<FormatPreference> FormatPreferenceFromString(
    absl::string_view s) {
  if (s == "default") {
    return FormatPreference::kDefault;
  } else if (s == "binary") {
    return FormatPreference::kBinary;
  } else if (s == "hex") {
    return FormatPreference::kHex;
  } else if (s == "decimal") {
    return FormatPreference::kDecimal;
  } else if (s == "plain_binary") {
    return FormatPreference::kPlainBinary;
  } else if (s == "plain_hex") {
    return FormatPreference::kPlainHex;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid format preference: \"%s\"", s));
}

}  // namespace xls

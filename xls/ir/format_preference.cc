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

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace xls {

std::string_view FormatPreferenceToString(FormatPreference preference) {
  switch (preference) {
    case FormatPreference::kDefault:
      return "default";
    case FormatPreference::kBinary:
      return "binary";
    case FormatPreference::kSignedDecimal:
      return "signed-decimal";
    case FormatPreference::kUnsignedDecimal:
      return "unsigned-decimal";
    case FormatPreference::kHex:
      return "hex";
    case FormatPreference::kPlainBinary:
      return "plain_binary";
    case FormatPreference::kPlainHex:
      return "plain_hex";
  }

  return "<invalid format preference>";
}

std::string_view FormatPreferenceToXlsSpecifier(FormatPreference preference) {
  switch (preference) {
    case FormatPreference::kDefault:
      return "{}";
    case FormatPreference::kBinary:
      return "{:#b}";
    case FormatPreference::kUnsignedDecimal:
      return "{:u}";
    case FormatPreference::kSignedDecimal:
      return "{:d}";
    case FormatPreference::kHex:
      return "{:#x}";
    case FormatPreference::kPlainBinary:
      return "{:b}";
    case FormatPreference::kPlainHex:
      return "{:x}";
  }

  return "<invalid format preference>";
}

std::string_view FormatPreferenceToVerilogSpecifier(
    FormatPreference preference) {
  switch (preference) {
    case FormatPreference::kDefault:
      // This is probably wrong, but it isn't clear what to do. Alternatives:
      // - bake in a format at IR conversion time
      // - add a "default format" argument to Verilog code generation
      // - decide that the default format (or at least the format associated
      // with {}) is decimal after all (matches Rust)
      return "%d";
    case FormatPreference::kSignedDecimal:
      return "%d";
    case FormatPreference::kUnsignedDecimal:
      return "%u";
    // Technically, the binary and hex format specifications are slightly wrong
    // because Verilog simulators don't break up long values with underscores as
    // XLS does. Practically speaking, though, it isn't worth doing a complex,
    // fragmented rendering just for getting that.
    case FormatPreference::kBinary:
      return "0b%b";
    case FormatPreference::kHex:
      return "0x%h";
    case FormatPreference::kPlainBinary:
      return "%b";
    case FormatPreference::kPlainHex:
      return "%h";
  }

  return "<invalid format preference>";
}

absl::StatusOr<FormatPreference> FormatPreferenceFromString(
    std::string_view s) {
  if (s == "default") {
    return FormatPreference::kDefault;
  }
  if (s == "binary") {
    return FormatPreference::kBinary;
  }
  if (s == "hex") {
    return FormatPreference::kHex;
  }
  if (s == "signed-decimal") {
    return FormatPreference::kSignedDecimal;
  }
  if (s == "unsigned-decimal") {
    return FormatPreference::kUnsignedDecimal;
  }
  if (s == "plain_binary") {
    return FormatPreference::kPlainBinary;
  }
  if (s == "plain_hex") {
    return FormatPreference::kPlainHex;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid format preference: \"%s\"", s));
}

}  // namespace xls

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

constexpr std::string_view kBinaryName = "binary";
constexpr std::string_view kDefaultName = "default";
constexpr std::string_view kHexName = "hex";
constexpr std::string_view kPlainBinaryName = "plain_binary";
constexpr std::string_view kPlainHexName = "plain_hex";
constexpr std::string_view kSignedDecimalName = "signed-decimal";
constexpr std::string_view kUnsignedDecimalName = "unsigned-decimal";
constexpr std::string_view kZeroPaddedBinaryName = "zero-padded-binary";
constexpr std::string_view kZeroPaddedHexName = "zero-padded-hex";

std::string_view FormatPreferenceToString(FormatPreference preference) {
  switch (preference) {
    case FormatPreference::kDefault:
      return kDefaultName;
    case FormatPreference::kBinary:
      return kBinaryName;
    case FormatPreference::kSignedDecimal:
      return kSignedDecimalName;
    case FormatPreference::kUnsignedDecimal:
      return kUnsignedDecimalName;
    case FormatPreference::kHex:
      return kHexName;
    case FormatPreference::kPlainBinary:
      return kPlainBinaryName;
    case FormatPreference::kPlainHex:
      return kPlainHexName;
    case FormatPreference::kZeroPaddedBinary:
      return kZeroPaddedBinaryName;
    case FormatPreference::kZeroPaddedHex:
      return kZeroPaddedHexName;
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
    case FormatPreference::kZeroPaddedBinary:
      return "{:0b}";
    case FormatPreference::kZeroPaddedHex:
      return "{:0x}";
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
    case FormatPreference::kZeroPaddedBinary:
      return "%b";
    case FormatPreference::kZeroPaddedHex:
      return "%h";
  }

  return "<invalid format preference>";
}

absl::StatusOr<FormatPreference> FormatPreferenceFromString(
    std::string_view s) {
  if (s == kDefaultName) {
    return FormatPreference::kDefault;
  }
  if (s == kBinaryName) {
    return FormatPreference::kBinary;
  }
  if (s == kHexName) {
    return FormatPreference::kHex;
  }
  if (s == kSignedDecimalName) {
    return FormatPreference::kSignedDecimal;
  }
  if (s == kUnsignedDecimalName) {
    return FormatPreference::kUnsignedDecimal;
  }
  if (s == kPlainBinaryName) {
    return FormatPreference::kPlainBinary;
  }
  if (s == kPlainHexName) {
    return FormatPreference::kPlainHex;
  }
  if (s == kZeroPaddedBinaryName) {
    return FormatPreference::kZeroPaddedBinary;
  }
  if (s == kZeroPaddedHexName) {
    return FormatPreference::kZeroPaddedHex;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid format preference: \"%s\"", s));
}

}  // namespace xls

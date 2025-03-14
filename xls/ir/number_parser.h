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

// Parse routines for turning numerical strings into Bits and integral values.

#ifndef XLS_IR_NUMBER_PARSER_H_
#define XLS_IR_NUMBER_PARSER_H_

#include <cstdint>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls {

// Flag value to indicate that parsing a number to Bits should use the minimum
// width necessary to represent the produced value.
constexpr int64_t kMinimumBitCount = -1;

enum class Sign : uint8_t {
  kUnspecified,
  kPositive,
  kNegative,
};

// Returns a signedness indicating whether the literal token is negative (or if
// the signedness is unspecified), and a Bits containing the magnitude. The Bits
// value is the minimum width necessary to hold the (necessarily) unsigned
// magnitude.
absl::StatusOr<std::pair<Sign, Bits>> GetSignAndMagnitude(
    std::string_view input);

// Parses the given string as a number and returns the result as a Bits
// value. Number may be represented in decimal, hexadecimal (prefixed with
// '0x'), or binary (prefixed with '0b'). Hexadecimal and binary numbers can be
// arbitrarily wide. Decimal numbers are limited to 64 bits. Numbers can include
// underscores (e.g., "0xdead_beef").
//
// For negative values (leading '-' in parsed string), the width of the Bits
// value is exactly wide enough to hold the value as a twos-complement number
// (with a minimum width of 1). For non-negative numbers, the width of Bits
// value is exactly wide enough to hold the value as an unsigned
// number. Examples:
//
//   "0" => UBits(0, 1)
//   "-1" => SBits(-1, 1) (equivalently UBits(1, 1))
//   "0b011" => UBits(3, 2)
//   "10" => UBits(10, 4)
//   "-10" => SBits(-10, 5) (equivalently UBits(22, 5))
//   "0x17" => UBits(0x17, 5)
absl::StatusOr<Bits> ParseNumber(std::string_view input);

// Parses the string as a number and returns the value as a (u)int64_t. Returns
// an error if the number is not representable as a (u)int64_t.
absl::StatusOr<int64_t> ParseNumberAsInt64(std::string_view input);
absl::StatusOr<uint64_t> ParseNumberAsUint64(std::string_view input);

absl::StatusOr<bool> ParseNumberAsBool(std::string_view input);

// Parse an unsigned number but the given input does not include any
// format-specifying prefix (e.g., "0x" for hexadecimal). Rather, the format is
// specified by argument.
absl::StatusOr<Bits> ParseUnsignedNumberWithoutPrefix(
    std::string_view input, FormatPreference format = FormatPreference::kHex,
    int64_t bit_count = kMinimumBitCount);

}  // namespace xls

#endif  // XLS_IR_NUMBER_PARSER_H_

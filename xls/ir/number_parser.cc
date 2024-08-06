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

#include "xls/ir/number_parser.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/string_to_int.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls {

// Parses the given input as an unsigned number (with no format prefix) of the
// given format. 'orig_string' is the string used in error message.
static absl::StatusOr<Bits> ParseUnsignedNumberHelper(
    std::string_view input, FormatPreference format,
    std::string_view orig_string, int64_t bit_count = kMinimumBitCount) {
  if (format == FormatPreference::kDefault) {
    return absl::InvalidArgumentError("Cannot specify default format.");
  }

  std::string numeric_string = absl::StrReplaceAll(input, {{"_", ""}});
  if (numeric_string.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not convert %s to a number", orig_string));
  }

  if (format == FormatPreference::kUnsignedDecimal) {
    Bits result(UBits(0, 32));
    Bits bits_10(UBits(10, 8));
    for (int i = 0; i < numeric_string.size(); i++) {
      result = bits_ops::UMul(result, bits_10);
      uint32_t magnitude;
      std::string digit_str(1, numeric_string.at(i));
      if (!absl::SimpleAtoi(digit_str, &magnitude)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Could not convert %s to 32-bit decimal number", orig_string));
      }
      Bits new_char(UBits(magnitude, result.bit_count()));

      result = bits_ops::Add(result, new_char);
      result = bits_ops::DropLeadingZeroes(result);
    }
    return result;
  }

  int base;
  int base_bits;
  std::string base_name;
  if (format == FormatPreference::kBinary) {
    base = 2;
    base_bits = 1;
    base_name = "binary";
  } else if (format == FormatPreference::kHex) {
    base = 16;
    base_bits = 4;
    base_name = "hexadecimal";
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid format: %d", format));
  }

  // Walk through string 64 bits at a time (16 hexadecimal symbols or 64
  // binary symbols) and convert each to a separate Bits value. Then
  // concatenate them all together.
  const int64_t step_size = 64 / base_bits;
  std::vector<Bits> chunks;
  for (int64_t i = 0; i < numeric_string.size(); i = i + step_size) {
    int64_t chunk_length =
        std::min<int64_t>(step_size, numeric_string.size() - i);
    absl::StatusOr<uint64_t> chunk_value_or =
        StrTo64Base(numeric_string.substr(i, chunk_length), base);
    if (!chunk_value_or.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Could not convert %s to %s number: %s", orig_string,
                          base_name, chunk_value_or.status().message()));
    }
    chunks.push_back(UBits(*chunk_value_or, chunk_length * base_bits));
  }

  Bits unnarrowed = bits_ops::Concat(chunks);
  if (bit_count == kMinimumBitCount) {
    // Narrow the Bits value to be just wide enough to hold the value.
    int64_t new_width = unnarrowed.bit_count() - unnarrowed.CountLeadingZeros();
    return unnarrowed.Slice(0, new_width);
  }
  if (bit_count > unnarrowed.bit_count()) {
    BitsRope rope(bit_count);
    rope.push_back(unnarrowed);
    rope.push_back(Bits(bit_count - unnarrowed.bit_count()));
    return rope.Build();
  }
  return unnarrowed.Slice(0, bit_count);
}

absl::StatusOr<Bits> ParseUnsignedNumberWithoutPrefix(std::string_view input,
                                                      FormatPreference format,
                                                      int64_t bit_count) {
  return ParseUnsignedNumberHelper(input, format, /*orig_string=*/input,
                                   bit_count);
}

absl::StatusOr<std::pair<bool, Bits>> GetSignAndMagnitude(
    std::string_view input) {
  // Literal numbers can be one of:
  //   1) decimal numbers, eg '123'
  //   2) binary numbers, eg '0b010101'
  //   3) hexadecimal numbers, eg '0xdeadbeef'
  // Binary and hexadecimal numbers can be arbitrarily large. Decimal numbers
  // must fit in 64 bits.
  if (input.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot parse empty string as a number."));
  }
  bool is_negative = false;
  // The substring containing the actual numeric characters.
  std::string_view numeric_substring = input;
  if (input[0] == '-') {
    is_negative = true;
    numeric_substring = numeric_substring.substr(1);
  }
  FormatPreference format = FormatPreference::kUnsignedDecimal;
  if (numeric_substring.size() >= 2 && numeric_substring[0] == '0') {
    char base_char = numeric_substring[1];
    if (base_char == 'b') {
      format = FormatPreference::kBinary;
    } else if (base_char == 'x') {
      format = FormatPreference::kHex;
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid numeric base %#x = '%c'. Expected 'b' or 'x'.", base_char,
          base_char));
    }
    numeric_substring = numeric_substring.substr(2);
  }
  XLS_ASSIGN_OR_RETURN(Bits value,
                       ParseUnsignedNumberHelper(numeric_substring, format,
                                                 /*orig_string=*/input));
  return std::make_pair(is_negative, value);
}

absl::StatusOr<Bits> ParseNumber(std::string_view input) {
  std::pair<bool, Bits> pair;
  XLS_ASSIGN_OR_RETURN(pair, GetSignAndMagnitude(input));
  bool is_negative = pair.first;
  const Bits& magnitude = pair.second;
  if (is_negative && !magnitude.IsZero()) {
    Bits result = bits_ops::Negate(
        bits_ops::ZeroExtend(magnitude, magnitude.bit_count() + 1));
    // We want to return the narrowest Bits object which can hold the (negative)
    // twos-complement number. Shave off all but one of the leading ones.
    int64_t leading_ones = 0;
    for (int64_t i = result.bit_count() - 1;
         i >= 0 && static_cast<int>(result.Get(i)) == 1; --i) {
      ++leading_ones;
    }
    XLS_RET_CHECK_GT(leading_ones, 0);
    return result.Slice(0, result.bit_count() - leading_ones + 1);
  }
  return magnitude;
}

absl::StatusOr<uint64_t> ParseNumberAsUint64(std::string_view input) {
  std::pair<bool, Bits> pair;
  XLS_ASSIGN_OR_RETURN(pair, GetSignAndMagnitude(input));
  bool is_negative = pair.first;
  if ((is_negative && !pair.second.IsZero()) || pair.second.bit_count() > 64) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Value is not representable as an uint64_t: %s", input));
  }
  return pair.second.ToUint64();
}

absl::StatusOr<int64_t> ParseNumberAsInt64(std::string_view input) {
  std::pair<bool, Bits> pair;
  XLS_ASSIGN_OR_RETURN(pair, GetSignAndMagnitude(input));
  bool is_negative = pair.first;
  Bits magnitude = pair.second;
  auto not_representable = [&]() {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value is not representable as an int64_t: %s", input));
  };
  if (!is_negative) {
    // A non-negative number must fit in 63 bits to be represented as a 64-bit
    // twos-complement number.
    if (magnitude.bit_count() > 63) {
      return not_representable();
    }
    return bits_ops::ZeroExtend(magnitude, 64).ToInt64();
  }

  // To be representable as a 64-bit twos -complement negative number the
  // magnitude must be representable in 63 bits OR the magnitude must be equal
  // to the magnitude of the most negative number (0x80000...).
  if (magnitude.bit_count() >= 64) {
    if ((magnitude.bit_count() == 64) && magnitude.msb() &&
        (magnitude.PopCount() == 1)) {
      return std::numeric_limits<int64_t>::min();
    }
    return not_representable();
  }
  // At this point 'magnitude' is an unsigned number of 63 or fewer bits. Zero
  // extend and negate for the result.
  return bits_ops::Negate(bits_ops::ZeroExtend(magnitude, 64)).ToInt64();
}

absl::StatusOr<bool> ParseNumberAsBool(std::string_view input) {
  if (input == "true") {
    return true;
  }
  if (input == "false") {
    return false;
  }
  std::pair<bool, Bits> pair;
  XLS_ASSIGN_OR_RETURN(pair, GetSignAndMagnitude(input));
  Bits magnitude = pair.second;
  bool is_negative = pair.first;
  auto not_representable = [&]() {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value is not representable as a bool: %s", input));
  };
  if (is_negative) {
    return not_representable();
  }
  if (magnitude.bit_count() > 1) {
    return not_representable();
  }
  return magnitude.Get(0);
}

}  // namespace xls

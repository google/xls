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

#include "xls/ir/bits.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"

namespace xls {

/* static */ int64_t Bits::MinBitCountSigned(int64_t value) {
  if (value == 0) {
    return 0;
  }
  if (value < 0) {
    // The expression -(value + 1) cannot overflow when value is negative.
    return MinBitCountUnsigned(-(value + 1)) + 1;
  }
  return MinBitCountUnsigned(value) + 1;
}

absl::StatusOr<Bits> UBitsWithStatus(uint64_t value, int64_t bit_count) {
  if (Bits::MinBitCountUnsigned(value) > bit_count) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value %#x requires %d bits to fit in an unsigned "
                        "datatype, but attempting to fit in %d bits.",
                        value, Bits::MinBitCountUnsigned(value), bit_count));
  }
  return Bits(InlineBitmap::FromWord(value, bit_count, /*fill=*/false));
}

absl::StatusOr<Bits> SBitsWithStatus(int64_t value, int64_t bit_count) {
  if (Bits::MinBitCountSigned(value) > bit_count) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value %#x requires %d bits to fit in an signed "
                        "datatype, but attempting to fit in %d bits.",
                        value, Bits::MinBitCountSigned(value), bit_count));
  }
  bool fill = (value >> 63);
  return Bits(InlineBitmap::FromWord(value, bit_count, fill));
}

Bits::Bits(absl::Span<bool const> bits) : bitmap_(bits.size()) {
  for (int64_t i = 0; i < bits.size(); ++i) {
    bitmap_.Set(i, bits[i]);
  }
}

void Bits::SetRange(int64_t start_index, int64_t end_index, bool value) {
  bitmap_.SetRange(start_index, end_index, value);
}

/* static */
Bits Bits::AllOnes(int64_t bit_count) {
  return bit_count == 0 ? Bits() : SBits(-1, bit_count);
}

/* static */
Bits Bits::MaxSigned(int64_t bit_count) {
  return Bits::AllOnes(bit_count).UpdateWithSet(bit_count - 1, false);
}

/* static */
Bits Bits::MinSigned(int64_t bit_count) {
  return Bits::PowerOfTwo(bit_count - 1, bit_count);
}

absl::InlinedVector<bool, 1> Bits::ToBitVector() const {
  absl::InlinedVector<bool, 1> bits(bit_count());
  for (int64_t i = 0; i < bit_count(); ++i) {
    bits[i] = bitmap_.Get(i);
  }
  return bits;
}

/* static */
Bits Bits::PowerOfTwo(int64_t set_bit_index, int64_t bit_count) {
  Bits result(bit_count);
  result.bitmap_.Set(set_bit_index, true);
  return result;
}

bool Bits::IsOne() const { return PopCount() == 1 && Get(0); }

int64_t Bits::PopCount() const {
  int64_t count = 0;
  for (int64_t i = 0; i < bit_count(); ++i) {
    if (Get(i)) {
      ++count;
    }
  }
  return count;
}

int64_t Bits::CountLeadingZeros() const {
  for (int64_t i = 0; i < bit_count(); ++i) {
    if (Get(bit_count() - 1 - i)) {
      return i;
    }
  }
  return bit_count();
}

int64_t Bits::CountLeadingOnes() const {
  for (int64_t i = 0; i < bit_count(); ++i) {
    if (!Get(bit_count() - 1 - i)) {
      return i;
    }
  }
  return bit_count();
}

int64_t Bits::CountTrailingZeros() const {
  for (int64_t i = 0; i < bit_count(); ++i) {
    if (Get(i)) {
      return i;
    }
  }
  return bit_count();
}

int64_t Bits::CountTrailingOnes() const {
  for (int64_t i = 0; i < bit_count(); ++i) {
    if (!Get(i)) {
      return i;
    }
  }
  return bit_count();
}

bool Bits::HasSingleRunOfSetBits(int64_t* leading_zero_count,
                                 int64_t* set_bit_count,
                                 int64_t* trailing_zero_count) const {
  int64_t leading_zeros = CountLeadingZeros();
  XLS_CHECK_GE(leading_zeros, 0);
  int64_t trailing_zeros = CountTrailingZeros();
  XLS_CHECK_GE(trailing_zeros, 0);
  if (bit_count() == trailing_zeros) {
    XLS_CHECK_EQ(leading_zeros, bit_count());
    return false;
  }
  for (int64_t i = trailing_zeros; i < bit_count() - leading_zeros; ++i) {
    if (Get(i) != 1) {
      return false;
    }
  }
  *leading_zero_count = leading_zeros;
  *trailing_zero_count = trailing_zeros;
  *set_bit_count = bit_count() - leading_zeros - trailing_zeros;
  XLS_CHECK_GE(*set_bit_count, 0);
  return true;
}

bool Bits::FitsInUint64() const { return FitsInNBitsUnsigned(64); }

bool Bits::FitsInInt64() const { return FitsInNBitsSigned(64); }

bool Bits::FitsInNBitsUnsigned(int64_t n) const {
  // All bits at and above bit 'n' must be zero.
  for (int64_t i = n; i < bit_count(); ++i) {
    if (Get(i)) {
      return false;
    }
  }
  return true;
}

bool Bits::FitsInNBitsSigned(int64_t n) const {
  if (n == 0) {
    return IsZero();
  }

  // All bits at and above bit N-1 must be the same.
  for (int64_t i = n - 1; i < bit_count(); ++i) {
    if (Get(i) != msb()) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<uint64_t> Bits::ToUint64() const {
  if (bit_count() == 0) {
    // By convention, an empty Bits has a numeric value of zero.
    return 0;
  }
  if (!FitsInUint64()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Bits value cannot be represented as an unsigned 64-bit value: ",
        ToString()));
  }
  return bitmap_.GetWord(0);
}

absl::StatusOr<uint64_t> Bits::WordToUint64(int64_t word_number) const {
  if (bit_count() == 0) {
    // By convention, an empty Bits has a numeric value of zero.
    return 0;
  }

  return bitmap_.GetWord(word_number);
}

absl::StatusOr<int64_t> Bits::ToInt64() const {
  if (!FitsInInt64()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Bits value cannot be represented as a signed 64-bit value: ",
        ToString()));
  }

  uint64_t word = bitmap_.GetWord(0);
  if (msb() && bit_count() < 64) {
    word |= -1ULL << bit_count();
  }

  return absl::bit_cast<int64_t>(word);
}

Bits Bits::Slice(int64_t start, int64_t width) const {
  XLS_CHECK_GE(width, 0);
  XLS_CHECK_LE(start + width, bit_count())
      << "start: " << start << " width: " << width;
  Bits result(width);
  for (int64_t i = 0; i < width; ++i) {
    if (Get(start + i)) {
      result.bitmap_.Set(i, true);
    }
  }
  return result;
}

std::string Bits::ToString(FormatPreference preference,
                           bool include_bit_count) const {
  if (preference == FormatPreference::kDefault) {
    if (bit_count() <= 64) {
      preference = FormatPreference::kUnsignedDecimal;
    } else {
      preference = FormatPreference::kHex;
    }
  }
  std::string result;
  if (preference == FormatPreference::kBinary) {
    result = "0b";
  } else if (preference == FormatPreference::kHex) {
    result = "0x";
  }
  absl::StrAppend(&result, ToRawDigits(preference));
  if (include_bit_count) {
    absl::StrAppendFormat(&result, " [%d bits]", bit_count());
  }
  return result;
}

std::string Bits::ToRawDigits(FormatPreference preference,
                              bool emit_leading_zeros) const {
  XLS_CHECK_NE(preference, FormatPreference::kDefault);
  if (preference == FormatPreference::kSignedDecimal) {
    // Leading zeros don't make a lot of sense in decimal format as there is no
    // clean correspondence between decimal digits and binary digits.
    XLS_CHECK(!emit_leading_zeros)
        << "emit_leading_zeros not supported for decimal format.";

    // TODO(google/xls#461): 2019-04-03 Add support for arbitrary width decimal
    // emission.
    XLS_CHECK(FitsInInt64())
        << "Decimal output not supported for values which do "
           "not fit in an int64_t";
    return absl::StrCat(ToInt64().value());
  }

  if (preference == FormatPreference::kUnsignedDecimal) {
    // Leading zeros don't make a lot of sense in decimal format as there is no
    // clean correspondence between decimal digits and binary digits.
    XLS_CHECK(!emit_leading_zeros)
        << "emit_leading_zeros not supported for decimal format.";
    // TODO(google/xls#461): 2019-04-03 Add support for arbitrary width decimal
    // emission.
    XLS_CHECK(FitsInUint64())
        << "Decimal output not supported for values which do "
           "not fit in a uint64_t";
    return absl::StrCat(ToUint64().value());
  }
  if (bit_count() == 0) {
    return "0";
  }

  const bool binary_format = (preference == FormatPreference::kBinary) ||
                             (preference == FormatPreference::kPlainBinary);
  const bool hex_format = (preference == FormatPreference::kHex) ||
                          (preference == FormatPreference::kPlainHex);
  const bool plain_format = (preference == FormatPreference::kPlainBinary) ||
                            (preference == FormatPreference::kPlainHex);

  XLS_CHECK(binary_format || hex_format);

  const int64_t digit_width = binary_format ? 1 : 4;
  const int64_t digit_count = CeilOfRatio(bit_count(), digit_width);
  // Include separators every 4 digits (to break up binary and hex numbers),
  // unless we are printing in a plain format.
  const bool include_separators = !plain_format;
  const int64_t kSeparatorPeriod = 4;

  std::string result;
  bool eliding_leading_zeros = !emit_leading_zeros;
  for (int64_t digit_no = digit_count - 1; digit_no >= 0; --digit_no) {
    // If including separators, add one every kSeparatorPeriod digits.
    if (include_separators && ((digit_no + 1) % kSeparatorPeriod == 0) &&
        !result.empty()) {
      absl::StrAppend(&result, "_");
    }
    // Slice out a Bits which contains 1 digit.
    int64_t start = digit_no * digit_width;
    int64_t width = std::min(digit_width, bit_count() - start);
    // As single digit necessarily fits in a uint64_t, so the value() is safe.
    uint64_t digit_value = Slice(start, width).ToUint64().value();
    if (digit_value == 0 && eliding_leading_zeros && digit_no != 0) {
      continue;
    }
    eliding_leading_zeros = false;
    absl::StrAppend(&result, absl::StrFormat("%x", digit_value));
  }
  return result;
}

}  // namespace xls

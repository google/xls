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

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"

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
  bool fill = (value >> 63) != 0;
  return Bits(InlineBitmap::FromWord(value, bit_count, fill));
}

Bits::Bits(absl::Span<bool const> bits)
    : bitmap_(InlineBitmap::FromBitsLsbIs0(bits)) {}

void Bits::SetRange(int64_t start_index, int64_t end_index, bool value) {
  bitmap_.SetRange(start_index, end_index, value);
}

/* static */ Bits Bits::AllOnes(int64_t bit_count) {
  return Bits::FromBitmap(InlineBitmap(bit_count, /*fill=*/true));
}

/* static */ Bits Bits::MaxSigned(int64_t bit_count) {
  if (bit_count == 0) {
    return SBits(0, 0);
  }
  return Bits::AllOnes(bit_count).UpdateWithSet(bit_count - 1, false);
}

/* static */ Bits Bits::MinSigned(int64_t bit_count) {
  if (bit_count == 0) {
    return SBits(0, 0);
  }
  return Bits::PowerOfTwo(bit_count - 1, bit_count);
}

std::string Bits::ToDebugString() const {
  std::string result = "0b";
  auto bits = ToBitVector();
  for (auto it = bits.rbegin(); it != bits.rend(); ++it) {
    absl::StrAppendFormat(&result, "%d", *it);
  }
  return result;
}

absl::InlinedVector<bool, 1> Bits::ToBitVector() const {
  absl::InlinedVector<bool, 1> bits(bit_count());
  for (int64_t i = 0; i < bit_count(); ++i) {
    bits[i] = bitmap_.Get(i);
  }
  return bits;
}

/* static */ Bits Bits::PowerOfTwo(int64_t set_bit_index, int64_t bit_count) {
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
  CHECK_GE(leading_zeros, 0);
  int64_t trailing_zeros = CountTrailingZeros();
  CHECK_GE(trailing_zeros, 0);
  if (bit_count() == trailing_zeros) {
    CHECK_EQ(leading_zeros, bit_count());
    return false;
  }
  for (int64_t i = trailing_zeros; i < bit_count() - leading_zeros; ++i) {
    if (!Get(i)) {
      return false;
    }
  }
  *leading_zero_count = leading_zeros;
  *trailing_zero_count = trailing_zeros;
  *set_bit_count = bit_count() - leading_zeros - trailing_zeros;
  CHECK_GE(*set_bit_count, 0);
  return true;
}

bool Bits::FitsInUint64() const { return FitsInNBitsUnsigned(64); }

bool Bits::FitsInInt64Unsigned() const { return FitsInNBitsUnsigned(63); }

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
        ToDebugString()));
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
        ToDebugString()));
  }

  uint64_t word = bitmap_.GetWord(0);
  if (msb() && bit_count() < 64) {
    word |= -1ULL << bit_count();
  }

  return absl::bit_cast<int64_t>(word);
}

absl::StatusOr<int64_t> Bits::UnsignedToInt64() const {
  if (bit_count() == 0) {
    // By convention, an empty Bits has a numeric value of zero.
    return 0;
  }
  if (!FitsInInt64Unsigned()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsigned Bits value cannot be represented as a signed 64-bit value: ",
        ToDebugString()));
  }
  DCHECK_LE(bitmap_.GetWord(0),
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  return static_cast<int64_t>(bitmap_.GetWord(0));
}

Bits Bits::Slice(int64_t start, int64_t width) && {
  CHECK_GE(width, 0);
  CHECK_LE(start + width, bit_count())
      << "start: " << start << " width: " << width;
  if (start == 0) {
    // Do the fast truncate.
    //
    // This is the most common slice so make it fast.
    return Bits::FromBitmap(std::move(bitmap_).WithSize(width));
  }
  InlineBitmap bm(width);
  bm.Overwrite(bitmap_, width, /*w_offset=*/0, /*r_offset=*/start);
  return Bits::FromBitmap(std::move(bm));
}

Bits Bits::Slice(int64_t start, int64_t width) const& {
  CHECK_GE(width, 0);
  CHECK_LE(start + width, bit_count())
      << "start: " << start << " width: " << width;
  if (start == 0) {
    // Do the fast truncate.
    //
    // This is the most common slice so make it fast.
    return Bits::FromBitmap(bitmap_.WithSize(width));
  }
  InlineBitmap bm(width);
  bm.Overwrite(bitmap_, width, /*w_offset=*/0, /*r_offset=*/start);
  return Bits::FromBitmap(std::move(bm));
}

}  // namespace xls

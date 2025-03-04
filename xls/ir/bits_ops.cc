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

#include "xls/ir/bits_ops.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/bits_util.h"
#include "xls/common/math_util.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/big_int.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/op.h"

namespace xls {
namespace bits_ops {
namespace {

// Converts the given bits value to signed value of the given bit count. Uses
// truncation or sign-extension to narrow/widen the value.
Bits TruncateOrSignExtend(Bits bits, int64_t bit_count) {
  if (bits.bit_count() == bit_count) {
    return bits;
  }
  if (bits.bit_count() < bit_count) {
    return SignExtend(std::move(bits), bit_count);
  }
  return Truncate(std::move(bits), bit_count);
}

}  // namespace

int64_t UnsignedBitsToSaturatedInt64(const Bits& bits) {
  if (bits.FitsInNBitsUnsigned(63)) {
    return bits.ToUint64().value();
  }
  return std::numeric_limits<int64_t>::max();
}

Bits And(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    return UBits(lhs.ToUint64().value() & rhs.ToUint64().value(),
                 lhs.bit_count());
  }
  std::vector<uint8_t> bytes = lhs.ToBytes();
  std::vector<uint8_t> rhs_bytes = rhs.ToBytes();
  for (int64_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = bytes[i] & rhs_bytes[i];
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryAnd(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64_t i = 1; i < operands.size(); ++i) {
    accum = And(accum, operands[i]);
  }
  return accum;
}

Bits Or(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64_t lhs_int = lhs.ToUint64().value();
    uint64_t rhs_int = rhs.ToUint64().value();
    uint64_t result = (lhs_int | rhs_int);
    return UBits(result, lhs.bit_count());
  }
  std::vector<uint8_t> bytes = lhs.ToBytes();
  std::vector<uint8_t> rhs_bytes = rhs.ToBytes();
  for (int64_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = bytes[i] | rhs_bytes[i];
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryOr(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64_t i = 1; i < operands.size(); ++i) {
    accum = Or(accum, operands[i]);
  }
  return accum;
}

Bits Xor(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64_t lhs_int = lhs.ToUint64().value();
    uint64_t rhs_int = rhs.ToUint64().value();
    uint64_t result = (lhs_int ^ rhs_int);
    return UBits(result, lhs.bit_count());
  }
  std::vector<uint8_t> bytes = lhs.ToBytes();
  std::vector<uint8_t> rhs_bytes = rhs.ToBytes();
  for (int64_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = bytes[i] ^ rhs_bytes[i];
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryXor(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64_t i = 1; i < operands.size(); ++i) {
    accum = Xor(accum, operands[i]);
  }
  return accum;
}

Bits Nand(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    return UBits(~(lhs.ToUint64().value() & rhs.ToUint64().value()) &
                     Mask(lhs.bit_count()),
                 lhs.bit_count());
  }
  std::vector<uint8_t> bytes = lhs.ToBytes();
  std::vector<uint8_t> rhs_bytes = rhs.ToBytes();
  for (int64_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = ~(bytes[i] & rhs_bytes[i]);
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryNand(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64_t i = 1; i < operands.size(); ++i) {
    accum = And(accum, operands[i]);
  }
  return Not(accum);
}

Bits Nor(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    return UBits(~(lhs.ToUint64().value() | rhs.ToUint64().value()) &
                     Mask(lhs.bit_count()),
                 lhs.bit_count());
  }
  std::vector<uint8_t> bytes = lhs.ToBytes();
  std::vector<uint8_t> rhs_bytes = rhs.ToBytes();
  for (int64_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = ~(bytes[i] | rhs_bytes[i]);
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryNor(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64_t i = 1; i < operands.size(); ++i) {
    accum = Or(accum, operands[i]);
  }
  return Not(accum);
}

Bits Not(const Bits& bits) {
  if (bits.bit_count() <= 64) {
    return UBits((~bits.ToUint64().value()) & Mask(bits.bit_count()),
                 bits.bit_count());
  }
  std::vector<uint8_t> bytes = bits.ToBytes();
  for (uint8_t& byte : bytes) {
    byte = ~byte;
  }
  return Bits::FromBytes(bytes, bits.bit_count());
}

Bits AndReduce(const Bits& operand) {
  // Is every bit set?
  return operand.IsAllOnes() ? UBits(1, 1) : UBits(0, 1);
}

Bits OrReduce(const Bits& operand) {
  // Is any bit set?
  return operand.IsZero() ? UBits(0, 1) : UBits(1, 1);
}

Bits XorReduce(const Bits& operand) {
  // Are there an odd number of bits set?
  return operand.PopCount() & 1 ? UBits(1, 1) : UBits(0, 1);
}

Bits Add(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64_t lhs_int = lhs.ToUint64().value();
    uint64_t rhs_int = rhs.ToUint64().value();
    uint64_t result = (lhs_int + rhs_int) & Mask(lhs.bit_count());
    return UBits(result, lhs.bit_count());
  }

  Bits sum = BigInt::Add(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs))
                 .ToSignedBits();
  return TruncateOrSignExtend(std::move(sum), lhs.bit_count());
}

Bits Sub(const Bits& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64_t lhs_int = lhs.ToUint64().value();
    uint64_t rhs_int = rhs.ToUint64().value();
    uint64_t result = (lhs_int - rhs_int) & Mask(lhs.bit_count());
    return UBits(result, lhs.bit_count());
  }
  Bits diff = BigInt::Sub(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs))
                  .ToSignedBits();
  return TruncateOrSignExtend(std::move(diff), lhs.bit_count());
}

Bits Increment(Bits x) {
  InlineBitmap result = std::move(x).bitmap();
  for (int64_t i = 0; i < result.word_count(); ++i) {
    uint64_t word = result.GetWord(i);
    if (word < std::numeric_limits<uint64_t>::max()) {
      result.SetWord(i, word + 1);
      break;
    }
    result.SetWord(i, 0);
  }
  return Bits::FromBitmap(std::move(result));
}

Bits Decrement(Bits x) {
  InlineBitmap result = std::move(x).bitmap();
  for (int64_t i = 0; i < result.word_count(); ++i) {
    uint64_t word = result.GetWord(i);
    if (word > 0) {
      result.SetWord(i, word - 1);
      break;
    }
    result.SetWord(i, ~uint64_t{0});
  }
  return Bits::FromBitmap(std::move(result));
}

Bits SMul(const Bits& lhs, const Bits& rhs) {
  const int64_t result_width = lhs.bit_count() + rhs.bit_count();
  if (result_width <= 64) {
    int64_t lhs_int = lhs.ToInt64().value();
    int64_t rhs_int = rhs.ToInt64().value();
    int64_t result = lhs_int * rhs_int;
    return SBits(result, result_width);
  }

  BigInt product =
      BigInt::Mul(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
  return product.ToSignedBitsWithBitCount(result_width).value();
}

Bits UMul(const Bits& lhs, const Bits& rhs) {
  const int64_t result_width = lhs.bit_count() + rhs.bit_count();
  if (result_width <= 64) {
    uint64_t lhs_int = lhs.ToUint64().value();
    uint64_t rhs_int = rhs.ToUint64().value();
    uint64_t result = lhs_int * rhs_int;
    return UBits(result, result_width);
  }

  BigInt product =
      BigInt::Mul(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
  return product.ToUnsignedBitsWithBitCount(result_width).value();
}

Bits UDiv(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return Bits::AllOnes(lhs.bit_count());
  }
  BigInt quotient =
      BigInt::Div(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
  return ZeroExtend(quotient.ToUnsignedBits(), lhs.bit_count());
}

Bits UMod(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return Bits(rhs.bit_count());
  }
  BigInt modulo =
      BigInt::Mod(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
  return ZeroExtend(modulo.ToUnsignedBits(), rhs.bit_count());
}

Bits SDiv(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    if (lhs.bit_count() == 0) {
      return UBits(0, 0);
    }
    if (SLessThan(lhs, UBits(0, lhs.bit_count()))) {
      // Divide by zero and lhs is negative.  Return largest magnitude negative
      // number: 0b1000...000.
      return Concat({UBits(1, 1), UBits(0, lhs.bit_count() - 1)});
    }
    // Divide by zero and lhs is non-negative. Return largest positive number:
    // 0b0111...111.
    return ZeroExtend(Bits::AllOnes(lhs.bit_count() - 1), lhs.bit_count());
  }
  BigInt quotient =
      BigInt::Div(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
  return TruncateOrSignExtend(quotient.ToSignedBits(), lhs.bit_count());
}

Bits SMod(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return Bits(rhs.bit_count());
  }
  BigInt modulo = BigInt::Mod(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
  return TruncateOrSignExtend(modulo.ToSignedBits(), rhs.bit_count());
}

bool UEqual(const Bits& lhs, const Bits& rhs) { return UCmp(lhs, rhs) == 0; }

bool UEqual(const Bits& lhs, int64_t rhs) {
  CHECK_GE(rhs, 0);
  if (!lhs.FitsInNBitsUnsigned(63)) {
    return false;
  }
  return static_cast<int64_t>(*lhs.ToUint64()) == rhs;
}

bool UGreaterThanOrEqual(const Bits& lhs, const Bits& rhs) {
  return UCmp(lhs, rhs) >= 0;
}

bool UGreaterThan(const Bits& lhs, const Bits& rhs) {
  return UCmp(lhs, rhs) > 0;
}

bool ULessThanOrEqual(const Bits& lhs, const Bits& rhs) {
  return UCmp(lhs, rhs) <= 0;
}

bool ULessThan(const Bits& lhs, const Bits& rhs) { return UCmp(lhs, rhs) < 0; }

int64_t UCmp(const Bits& lhs, const Bits& rhs) {
  return lhs.bitmap().UCmp(rhs.bitmap());
}

const Bits& UMin(const Bits& lhs, const Bits& rhs) {
  return ULessThan(lhs, rhs) ? lhs : rhs;
}

const Bits& UMax(const Bits& lhs, const Bits& rhs) {
  return ULessThan(lhs, rhs) ? rhs : lhs;
}

bool UGreaterThanOrEqual(const Bits& lhs, int64_t rhs) {
  CHECK_GE(rhs, 0);
  if (!lhs.FitsInNBitsUnsigned(63)) {
    return true;
  }
  return static_cast<int64_t>(*lhs.ToUint64()) >= rhs;
}

bool UGreaterThan(const Bits& lhs, int64_t rhs) {
  CHECK_GE(rhs, 0);
  if (!lhs.FitsInNBitsUnsigned(63)) {
    return true;
  }
  return static_cast<int64_t>(*lhs.ToUint64()) > rhs;
}

bool ULessThanOrEqual(const Bits& lhs, int64_t rhs) {
  CHECK_GE(rhs, 0);
  if (!lhs.FitsInNBitsUnsigned(63)) {
    return false;
  }
  return static_cast<int64_t>(*lhs.ToUint64()) <= rhs;
}

bool ULessThan(const Bits& lhs, int64_t rhs) {
  CHECK_GE(rhs, 0);
  if (!lhs.FitsInNBitsUnsigned(63)) {
    return false;
  }
  return static_cast<int64_t>(*lhs.ToUint64()) < rhs;
}

bool SEqual(const Bits& lhs, const Bits& rhs) {
  return BigInt::MakeSigned(lhs) == BigInt::MakeSigned(rhs);
}

bool SEqual(const Bits& lhs, int64_t rhs) {
  if (!lhs.FitsInInt64()) {
    return false;
  }
  return *lhs.ToInt64() == rhs;
}

bool SGreaterThanOrEqual(const Bits& lhs, const Bits& rhs) {
  return !SLessThan(lhs, rhs);
}

bool SGreaterThan(const Bits& lhs, const Bits& rhs) {
  return !SLessThanOrEqual(lhs, rhs);
}

bool SLessThanOrEqual(const Bits& lhs, const Bits& rhs) {
  return SEqual(lhs, rhs) || SLessThan(lhs, rhs);
}

bool SLessThan(const Bits& lhs, const Bits& rhs) {
  if (lhs.bit_count() <= 64 && rhs.bit_count() <= 64) {
    return lhs.ToInt64().value() < rhs.ToInt64().value();
  }
  return BigInt::LessThan(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
}

bool SGreaterThanOrEqual(const Bits& lhs, int64_t rhs) {
  if (lhs.FitsInInt64()) {
    return *lhs.ToInt64() >= rhs;
  }
  // LHS is less than int64_t's minimum value (and thus rhs) if its MSB is set;
  // otherwise it's greater than the maximum value (and thus rhs).
  return !lhs.msb();
}

bool SGreaterThan(const Bits& lhs, int64_t rhs) {
  if (lhs.FitsInInt64()) {
    return *lhs.ToInt64() > rhs;
  }
  // LHS is less than int64_t's minimum value (and thus rhs) if its MSB is set;
  // otherwise it's greater than the maximum value (and thus rhs).
  return !lhs.msb();
}

bool SLessThanOrEqual(const Bits& lhs, int64_t rhs) {
  if (lhs.FitsInInt64()) {
    return *lhs.ToInt64() <= rhs;
  }
  // LHS is less than int64_t's minimum value (and thus rhs) if its MSB is set;
  // otherwise it's greater than the maximum value (and thus rhs).
  return lhs.msb();
}

bool SLessThan(const Bits& lhs, int64_t rhs) {
  if (lhs.FitsInInt64()) {
    return *lhs.ToInt64() < rhs;
  }
  // LHS is less than int64_t's minimum value (and thus rhs) if its MSB is set;
  // otherwise it's greater than the maximum value (and thus rhs).
  return lhs.msb();
}

Bits ZeroExtend(Bits bits, int64_t new_bit_count) {
  CHECK_GE(new_bit_count, 0);
  CHECK_GE(new_bit_count, bits.bit_count());
  return Bits::FromBitmap(std::move(bits).bitmap().WithSize(new_bit_count));
}

Bits SignExtend(Bits bits, int64_t new_bit_count) {
  CHECK_GE(new_bit_count, 0);
  CHECK_GE(new_bit_count, bits.bit_count());
  bool sign_bit = bits.bit_count() > 0 && bits.GetFromMsb(0);
  return Bits::FromBitmap(
      std::move(bits).bitmap().WithSize(new_bit_count, /*new_data=*/sign_bit));
}

Bits Concat(absl::Span<const Bits> inputs) {
  int64_t new_bit_count = 0;
  for (const Bits& bits : inputs) {
    new_bit_count += bits.bit_count();
  }
  // Iterate in reverse order because the first input becomes the
  // most-significant bits.
  BitsRope rope(new_bit_count);
  for (int64_t i = 0; i < inputs.size(); ++i) {
    rope.push_back(inputs[inputs.size() - i - 1]);
  }
  return rope.Build();
}

Bits Negate(const Bits& bits) {
  if (bits.bit_count() < 64) {
    return UBits((-bits.ToInt64().value()) & Mask(bits.bit_count()),
                 bits.bit_count());
  }
  Bits negated = BigInt::Negate(BigInt::MakeSigned(bits)).ToSignedBits();
  return TruncateOrSignExtend(std::move(negated), bits.bit_count());
}

Bits Abs(const Bits& bits) {
  return (bits.GetFromMsb(0)) ? Negate(bits) : bits;
}

Bits ShiftLeftLogical(const Bits& bits, int64_t shift_amount) {
  CHECK_GE(shift_amount, 0);
  shift_amount = std::min(shift_amount, bits.bit_count());
  return Concat(
      {bits.Slice(0, bits.bit_count() - shift_amount), UBits(0, shift_amount)});
}

Bits ShiftRightLogical(const Bits& bits, int64_t shift_amount) {
  CHECK_GE(shift_amount, 0);
  shift_amount = std::min(shift_amount, bits.bit_count());
  return Concat({UBits(0, shift_amount),
                 bits.Slice(shift_amount, bits.bit_count() - shift_amount)});
}

Bits ShiftRightArith(const Bits& bits, int64_t shift_amount) {
  CHECK_GE(shift_amount, 0);
  shift_amount = std::min(shift_amount, bits.bit_count());
  return Concat(
      {bits.msb() ? Bits::AllOnes(shift_amount) : UBits(0, shift_amount),
       bits.Slice(shift_amount, bits.bit_count() - shift_amount)});
}

Bits OneHotLsbToMsb(const Bits& bits) {
  if (bits.IsZero()) {
    return Bits::PowerOfTwo(bits.bit_count(), bits.bit_count() + 1);
  }

  InlineBitmap result(bits.bit_count() + 1);
  for (auto it = bits.begin(); it != bits.end(); ++it) {
    if (*it) {
      result.Set(std::distance(bits.begin(), it));
      break;
    }
  }
  return Bits::FromBitmap(std::move(result));
}

Bits OneHotMsbToLsb(const Bits& bits) {
  if (bits.IsZero()) {
    return Bits::PowerOfTwo(bits.bit_count(), bits.bit_count() + 1);
  }

  InlineBitmap result(bits.bit_count() + 1);
  for (auto it = bits.end() - 1; it >= bits.begin(); --it) {
    if (*it) {
      result.Set(std::distance(bits.begin(), it));
      break;
    }
  }
  return Bits::FromBitmap(std::move(result));
}

Bits Reverse(const Bits& bits) {
  InlineBitmap reversed_bitmap(bits.bit_count());
  auto last = bits.end() - 1;
  for (auto it = bits.begin(); it != bits.end(); ++it) {
    if (*it) {
      reversed_bitmap.Set(std::distance(it, last));
    }
  }
  return Bits::FromBitmap(std::move(reversed_bitmap));
}

Bits DropLeadingZeroes(const Bits& bits) {
  if (bits.IsZero()) {
    return Bits();
  }

  auto first_one = bits.end() - 1;
  while (first_one >= bits.begin() && !*first_one) {
    --first_one;
  }
  return bits.Slice(0, std::distance(bits.begin(), first_one) + 1);
}

Bits Truncate(Bits bits, int64_t size) {
  CHECK_GE(bits.bit_count(), size);
  return std::move(bits).Slice(0, size);
}

Bits BitSliceUpdate(const Bits& to_update, int64_t start,
                    const Bits& update_value) {
  if (start >= to_update.bit_count()) {
    // Start index is entirely out-of-bounds. The return value is simply the
    // input data operand.
    return to_update;
  }

  // Construct the result as the sliced concatenation of three slices:
  //   (1) slice of some least-significant bits of to_update.
  //   (2) slice of update_value
  //   (3) slice of some most-significant bits of to_update.
  // One or more of these slices may be zero-width.
  Bits lsb_slice = to_update.Slice(/*start=*/0, /*width=*/start);
  Bits update_slice = update_value.Slice(
      /*start=*/0,
      /*width=*/std::min(update_value.bit_count(),
                         to_update.bit_count() - start));
  int64_t msb_start =
      std::min(to_update.bit_count(), start + update_value.bit_count());
  Bits msb_slice = to_update.Slice(
      /*start=*/msb_start,
      /*width=*/std::max(int64_t{0}, to_update.bit_count() - msb_start));
  return Concat({msb_slice, update_slice, lsb_slice});
}

Bits LongestCommonPrefixLSB(absl::Span<const Bits> bits_span) {
  if (bits_span.empty()) {
    return Bits();
  }

  int64_t input_size = bits_span[0].bit_count();
  for (const Bits& bits : bits_span) {
    CHECK_EQ(bits.bit_count(), input_size);
  }

  int64_t first_difference = input_size;
  std::vector<Bits::Iterator> iterators;
  iterators.reserve(bits_span.size());
  for (const Bits& bits : bits_span) {
    iterators.push_back(bits.begin());
  }
  for (; iterators[0] != bits_span[0].end(); iterators[0]++) {
    for (auto& it : absl::MakeSpan(iterators).subspan(1)) {
      if (*it++ != *iterators[0]) {
        first_difference = std::distance(bits_span[0].begin(), iterators[0]);
        break;
      }
    }
    if (first_difference != input_size) {
      break;
    }
  }
  return bits_span[0].Slice(0, first_difference);
}

Bits LongestCommonPrefixMSB(absl::Span<const Bits> bits_span) {
  if (bits_span.empty()) {
    return Bits();
  }

  int64_t input_size = bits_span[0].bit_count();
  for (const Bits& bits : bits_span) {
    CHECK_EQ(bits.bit_count(), input_size);
  }

  int64_t first_difference = -1;
  std::vector<Bits::Iterator> iterators;
  iterators.reserve(bits_span.size());
  for (const Bits& bits : bits_span) {
    iterators.push_back(bits.end() - 1);
  }
  for (; iterators[0] >= bits_span[0].begin(); --iterators[0]) {
    for (auto& it : absl::MakeSpan(iterators).subspan(1)) {
      if (*it-- != *iterators[0]) {
        first_difference = std::distance(bits_span[0].begin(), iterators[0]);
        break;
      }
    }
    if (first_difference != -1) {
      break;
    }
  }
  return bits_span[0].Slice(first_difference + 1,
                            input_size - first_difference - 1);
}

}  // namespace bits_ops

Bits LogicalOpIdentity(Op op, int64_t width) {
  switch (op) {
    case Op::kAnd:
    case Op::kNand:
      return Bits::AllOnes(width);
    case Op::kOr:
    case Op::kNor:
    case Op::kXor:
      return Bits(width);
    default:
      LOG(FATAL) << "NaryOpIdentity got non-nary op:" << OpToString(op);
  }
}

Bits DoLogicalOp(Op op, absl::Span<const Bits> operands) {
  CHECK_GT(operands.size(), 0);
  switch (op) {
    case Op::kAnd:
      return bits_ops::NaryAnd(operands);
    case Op::kOr:
      return bits_ops::NaryOr(operands);
    case Op::kXor:
      return bits_ops::NaryXor(operands);
    case Op::kNand:
      return bits_ops::NaryNand(operands);
    case Op::kNor:
      return bits_ops::NaryNor(operands);
    default:
      LOG(FATAL) << "DoNaryBitOp got non-nary op: " << OpToString(op);
  }
}

Bits operator&(const Bits& lhs, const Bits& rhs) {
  return bits_ops::And(lhs, rhs);
}
Bits operator|(const Bits& lhs, const Bits& rhs) {
  return bits_ops::Or(lhs, rhs);
}
Bits operator^(const Bits& lhs, const Bits& rhs) {
  return bits_ops::Xor(lhs, rhs);
}
Bits operator~(const Bits& bits) { return bits_ops::Not(bits); }

Bits MulpOffsetForSimulation(int64_t result_size, int64_t shift_size) {
  int64_t offset_lsbs_size = std::max(int64_t{0}, result_size - 2);
  int64_t msbs_size = result_size - offset_lsbs_size;
  int64_t offset_lsbs_shift_amount =
      std::max(int64_t{0}, std::min(offset_lsbs_size - 1, shift_size));
  Bits offset_bits = bits_ops::ShiftRightLogical(
      Bits::AllOnes(offset_lsbs_size), offset_lsbs_shift_amount);
  if (msbs_size > 0) {
    offset_bits = bits_ops::Concat({Bits::MaxSigned(msbs_size), offset_bits});
  }
  return offset_bits;
}

std::string BitsToRawDigits(const Bits& bits, FormatPreference preference,
                            bool emit_leading_zeros) {
  CHECK_NE(preference, FormatPreference::kDefault);
  if (preference == FormatPreference::kZeroPaddedBinary ||
      preference == FormatPreference::kZeroPaddedHex) {
    emit_leading_zeros = true;
  }
  if (preference == FormatPreference::kSignedDecimal) {
    // Leading zeros don't make a lot of sense in decimal format as there is no
    // clean correspondence between decimal digits and binary digits.
    CHECK(!emit_leading_zeros)
        << "emit_leading_zeros not supported for decimal format.";

    return BigInt::MakeSigned(bits).ToDecimalString();
  }

  if (preference == FormatPreference::kUnsignedDecimal) {
    // Leading zeros don't make a lot of sense in decimal format as there is no
    // clean correspondence between decimal digits and binary digits.
    CHECK(!emit_leading_zeros)
        << "emit_leading_zeros not supported for decimal format.";

    return BigInt::MakeUnsigned(bits).ToDecimalString();
  }
  if (bits.bit_count() == 0) {
    return "0";
  }

  const bool binary_format = preference == FormatPreference::kBinary ||
                             preference == FormatPreference::kPlainBinary ||
                             preference == FormatPreference::kZeroPaddedBinary;
  const bool hex_format = preference == FormatPreference::kHex ||
                          preference == FormatPreference::kPlainHex ||
                          preference == FormatPreference::kZeroPaddedHex;
  const bool plain_format = (preference == FormatPreference::kPlainBinary) ||
                            (preference == FormatPreference::kPlainHex);

  CHECK(binary_format || hex_format);

  const int64_t digit_width = binary_format ? 1 : 4;
  const int64_t digit_count = CeilOfRatio(bits.bit_count(), digit_width);
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
    int64_t width = std::min(digit_width, bits.bit_count() - start);
    // As single digit necessarily fits in a uint64_t, so the value() is safe.
    uint64_t digit_value = bits.Slice(start, width).ToUint64().value();
    if (digit_value == 0 && eliding_leading_zeros && digit_no != 0) {
      continue;
    }
    eliding_leading_zeros = false;
    absl::StrAppend(&result, absl::StrFormat("%x", digit_value));
  }
  return result;
}

std::string BitsToString(const Bits& bits, FormatPreference preference,
                         bool include_bit_count) {
  if (preference == FormatPreference::kDefault) {
    if (bits.bit_count() <= 64) {
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
  absl::StrAppend(&result, BitsToRawDigits(bits, preference));
  if (include_bit_count) {
    absl::StrAppendFormat(&result, " [%d bits]", bits.bit_count());
  }
  return result;
}

}  // namespace xls

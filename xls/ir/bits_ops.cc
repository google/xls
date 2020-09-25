// Copyright 2020 Google LLC
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

#include <vector>

#include "xls/common/logging/logging.h"
#include "xls/ir/big_int.h"

namespace xls {
namespace bits_ops {
namespace {

// Converts the given bits value to signed value of the given bit count. Uses
// truncation or sign-extension to narrow/widen the value.
Bits TruncateOrSignExtend(const Bits& bits, int64 bit_count) {
  if (bits.bit_count() == bit_count) {
    return bits;
  } else if (bits.bit_count() < bit_count) {
    return SignExtend(bits, bit_count);
  } else {
    return bits.Slice(0, bit_count);
  }
}

}  // namespace

Bits And(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    return UBits(lhs.ToUint64().value() & rhs.ToUint64().value(),
                 lhs.bit_count());
  }
  std::vector<uint8> bytes = lhs.ToBytes();
  std::vector<uint8> rhs_bytes = rhs.ToBytes();
  for (int64 i = 0; i < bytes.size(); ++i) {
    bytes[i] = bytes[i] & rhs_bytes[i];
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryAnd(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64 i = 1; i < operands.size(); ++i) {
    accum = And(accum, operands[i]);
  }
  return accum;
}

Bits Or(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64 lhs_int = lhs.ToUint64().value();
    uint64 rhs_int = rhs.ToUint64().value();
    uint64 result = (lhs_int | rhs_int);
    return UBits(result, lhs.bit_count());
  }
  std::vector<uint8> bytes = lhs.ToBytes();
  std::vector<uint8> rhs_bytes = rhs.ToBytes();
  for (int64 i = 0; i < bytes.size(); ++i) {
    bytes[i] = bytes[i] | rhs_bytes[i];
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryOr(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64 i = 1; i < operands.size(); ++i) {
    accum = Or(accum, operands[i]);
  }
  return accum;
}

Bits Xor(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64 lhs_int = lhs.ToUint64().value();
    uint64 rhs_int = rhs.ToUint64().value();
    uint64 result = (lhs_int ^ rhs_int);
    return UBits(result, lhs.bit_count());
  }
  std::vector<uint8> bytes = lhs.ToBytes();
  std::vector<uint8> rhs_bytes = rhs.ToBytes();
  for (int64 i = 0; i < bytes.size(); ++i) {
    bytes[i] = bytes[i] ^ rhs_bytes[i];
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryXor(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64 i = 1; i < operands.size(); ++i) {
    accum = Xor(accum, operands[i]);
  }
  return accum;
}

Bits Nand(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    return UBits(~(lhs.ToUint64().value() & rhs.ToUint64().value()) &
                     Mask(lhs.bit_count()),
                 lhs.bit_count());
  }
  std::vector<uint8> bytes = lhs.ToBytes();
  std::vector<uint8> rhs_bytes = rhs.ToBytes();
  for (int64 i = 0; i < bytes.size(); ++i) {
    bytes[i] = ~(bytes[i] & rhs_bytes[i]);
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryNand(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64 i = 1; i < operands.size(); ++i) {
    accum = And(accum, operands[i]);
  }
  return Not(accum);
}

Bits Nor(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    return UBits(~(lhs.ToUint64().value() | rhs.ToUint64().value()) &
                     Mask(lhs.bit_count()),
                 lhs.bit_count());
  }
  std::vector<uint8> bytes = lhs.ToBytes();
  std::vector<uint8> rhs_bytes = rhs.ToBytes();
  for (int64 i = 0; i < bytes.size(); ++i) {
    bytes[i] = ~(bytes[i] | rhs_bytes[i]);
  }
  return Bits::FromBytes(bytes, lhs.bit_count());
}

Bits NaryNor(absl::Span<const Bits> operands) {
  Bits accum = operands.at(0);
  for (int64 i = 1; i < operands.size(); ++i) {
    accum = Or(accum, operands[i]);
  }
  return Not(accum);
}

Bits Not(const Bits& bits) {
  if (bits.bit_count() <= 64) {
    return UBits((~bits.ToUint64().value()) & Mask(bits.bit_count()),
                 bits.bit_count());
  }
  std::vector<uint8> bytes = bits.ToBytes();
  for (uint8& byte : bytes) {
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
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64 lhs_int = lhs.ToUint64().value();
    uint64 rhs_int = rhs.ToUint64().value();
    uint64 result = (lhs_int + rhs_int) & Mask(lhs.bit_count());
    return UBits(result, lhs.bit_count());
  }

  Bits sum = BigInt::Add(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs))
                 .ToSignedBits();
  return TruncateOrSignExtend(sum, lhs.bit_count());
}

Bits Sub(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64 lhs_int = lhs.ToUint64().value();
    uint64 rhs_int = rhs.ToUint64().value();
    uint64 result = (lhs_int - rhs_int) & Mask(lhs.bit_count());
    return UBits(result, lhs.bit_count());
  }
  Bits diff = BigInt::Sub(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs))
                  .ToSignedBits();
  return TruncateOrSignExtend(diff, lhs.bit_count());
}

Bits Mul(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (lhs.bit_count() <= 64) {
    uint64 lhs_int = lhs.ToUint64().value();
    uint64 rhs_int = rhs.ToUint64().value();
    uint64 result = (lhs_int * rhs_int) & Mask(lhs.bit_count());
    return UBits(result, lhs.bit_count());
  }

  BigInt product =
      BigInt::Mul(BigInt::MakeSigned(SignExtend(lhs, lhs.bit_count())),
                  BigInt::MakeSigned(SignExtend(rhs, lhs.bit_count())));
  return product.ToSignedBitsWithBitCount(lhs.bit_count() * 2)
      .value()
      .Slice(0, lhs.bit_count());
}

Bits SMul(const Bits& lhs, const Bits& rhs) {
  const int64 result_width = lhs.bit_count() + rhs.bit_count();
  if (result_width <= 64) {
    int64 lhs_int = lhs.ToInt64().value();
    int64 rhs_int = rhs.ToInt64().value();
    int64 result = lhs_int * rhs_int;
    return SBits(result, result_width);
  }

  BigInt product =
      BigInt::Mul(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
  return product.ToSignedBitsWithBitCount(result_width).value();
}

Bits UMul(const Bits& lhs, const Bits& rhs) {
  const int64 result_width = lhs.bit_count() + rhs.bit_count();
  if (result_width <= 64) {
    uint64 lhs_int = lhs.ToUint64().value();
    uint64 rhs_int = rhs.ToUint64().value();
    uint64 result = lhs_int * rhs_int;
    return UBits(result, result_width);
  }

  BigInt product =
      BigInt::Mul(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
  return product.ToUnsignedBitsWithBitCount(result_width).value();
}

Bits UDiv(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (rhs.IsZero()) {
    return Bits::AllOnes(lhs.bit_count());
  }
  BigInt quotient =
      BigInt::Div(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
  return ZeroExtend(quotient.ToUnsignedBits(), lhs.bit_count());
}

Bits UMod(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (rhs.IsZero()) {
    return Bits(lhs.bit_count());
  }
  BigInt modulo =
      BigInt::Mod(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
  return ZeroExtend(modulo.ToUnsignedBits(), lhs.bit_count());
}

Bits SDiv(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (rhs.IsZero()) {
    if (lhs.bit_count() == 0) {
      return UBits(0, 0);
    }
    if (SLessThan(lhs, UBits(0, lhs.bit_count()))) {
      // Divide by zero and lhs is negative.  Return largest magnitude negative
      // number: 0b1000...000.
      return Concat({UBits(1, 1), UBits(0, lhs.bit_count() - 1)});
    } else {
      // Divide by zero and lhs is non-negative. Return largest positive number:
      // 0b0111...111.
      return ZeroExtend(Bits::AllOnes(lhs.bit_count() - 1), lhs.bit_count());
    }
  }
  BigInt quotient =
      BigInt::Div(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
  return TruncateOrSignExtend(quotient.ToSignedBits(), lhs.bit_count());
}

Bits SMod(const Bits& lhs, const Bits& rhs) {
  XLS_CHECK_EQ(lhs.bit_count(), rhs.bit_count());
  if (rhs.IsZero()) {
    return Bits(lhs.bit_count());
  }
  BigInt modulo = BigInt::Mod(BigInt::MakeSigned(lhs), BigInt::MakeSigned(rhs));
  return TruncateOrSignExtend(modulo.ToSignedBits(), lhs.bit_count());
}

bool UEqual(const Bits& lhs, const Bits& rhs) {
  return BigInt::MakeUnsigned(lhs) == BigInt::MakeUnsigned(rhs);
}

bool UEqual(const Bits& lhs, int64 rhs) {
  XLS_CHECK_GE(rhs, 0);
  return UEqual(lhs, UBits(rhs, 64));
}

bool UGreaterThanOrEqual(const Bits& lhs, const Bits& rhs) {
  return !ULessThan(lhs, rhs);
}

bool UGreaterThan(const Bits& lhs, const Bits& rhs) {
  return !ULessThanOrEqual(lhs, rhs);
}

bool ULessThanOrEqual(const Bits& lhs, const Bits& rhs) {
  return UEqual(lhs, rhs) ||
         BigInt::LessThan(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
}

bool ULessThan(const Bits& lhs, const Bits& rhs) {
  return BigInt::LessThan(BigInt::MakeUnsigned(lhs), BigInt::MakeUnsigned(rhs));
}

bool UGreaterThanOrEqual(const Bits& lhs, int64 rhs) {
  XLS_CHECK_GE(rhs, 0);
  return UGreaterThanOrEqual(lhs, UBits(rhs, 64));
}

bool UGreaterThan(const Bits& lhs, int64 rhs) {
  XLS_CHECK_GE(rhs, 0);
  return UGreaterThan(lhs, UBits(rhs, 64));
}

bool ULessThanOrEqual(const Bits& lhs, int64 rhs) {
  XLS_CHECK_GE(rhs, 0);
  return ULessThanOrEqual(lhs, UBits(rhs, 64));
}

bool ULessThan(const Bits& lhs, int64 rhs) {
  XLS_CHECK_GE(rhs, 0);
  return ULessThan(lhs, UBits(rhs, 64));
}

bool SEqual(const Bits& lhs, const Bits& rhs) {
  return BigInt::MakeSigned(lhs) == BigInt::MakeSigned(rhs);
}

bool SEqual(const Bits& lhs, int64 rhs) { return SEqual(lhs, SBits(rhs, 64)); }

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

bool SGreaterThanOrEqual(const Bits& lhs, int64 rhs) {
  return SGreaterThanOrEqual(lhs, SBits(rhs, 64));
}

bool SGreaterThan(const Bits& lhs, int64 rhs) {
  return SGreaterThan(lhs, SBits(rhs, 64));
}

bool SLessThanOrEqual(const Bits& lhs, int64 rhs) {
  return SLessThanOrEqual(lhs, SBits(rhs, 64));
}

bool SLessThan(const Bits& lhs, int64 rhs) {
  return SLessThan(lhs, SBits(rhs, 64));
}

Bits ZeroExtend(const Bits& bits, int64 new_bit_count) {
  XLS_CHECK_GE(new_bit_count, 0);
  XLS_CHECK_GE(new_bit_count, bits.bit_count());
  return Concat({UBits(0, new_bit_count - bits.bit_count()), bits});
}

Bits SignExtend(const Bits& bits, int64 new_bit_count) {
  XLS_CHECK_GE(new_bit_count, 0);
  XLS_CHECK_GE(new_bit_count, bits.bit_count());
  const int64 ext_width = new_bit_count - bits.bit_count();
  return Concat(
      {bits.msb() ? Bits::AllOnes(ext_width) : Bits(ext_width), bits});
}

Bits Concat(absl::Span<const Bits> inputs) {
  int64 new_bit_count = 0;
  for (const Bits& bits : inputs) {
    new_bit_count += bits.bit_count();
  }
  // Iterate in reverse order because the first input becomes the
  // most-significant bits.
  BitsRope rope(new_bit_count);
  for (int64 i = 0; i < inputs.size(); ++i) {
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
  return TruncateOrSignExtend(negated, bits.bit_count());
}

Bits ShiftLeftLogical(const Bits& bits, int64 shift_amount) {
  XLS_CHECK_GE(shift_amount, 0);
  shift_amount = std::min(shift_amount, bits.bit_count());
  return Concat(
      {bits.Slice(0, bits.bit_count() - shift_amount), UBits(0, shift_amount)});
}

Bits ShiftRightLogical(const Bits& bits, int64 shift_amount) {
  XLS_CHECK_GE(shift_amount, 0);
  shift_amount = std::min(shift_amount, bits.bit_count());
  return Concat({UBits(0, shift_amount),
                 bits.Slice(shift_amount, bits.bit_count() - shift_amount)});
}

Bits ShiftRightArith(const Bits& bits, int64 shift_amount) {
  XLS_CHECK_GE(shift_amount, 0);
  shift_amount = std::min(shift_amount, bits.bit_count());
  return Concat(
      {bits.msb() ? Bits::AllOnes(shift_amount) : UBits(0, shift_amount),
       bits.Slice(shift_amount, bits.bit_count() - shift_amount)});
}

Bits OneHotLsbToMsb(const Bits& bits) {
  for (int64 i = 0; i < bits.bit_count(); ++i) {
    if (bits.Get(i)) {
      return Bits::PowerOfTwo(i, bits.bit_count() + 1);
    }
  }
  return Bits::PowerOfTwo(bits.bit_count(), bits.bit_count() + 1);
}

Bits OneHotMsbToLsb(const Bits& bits) {
  for (int64 i = bits.bit_count() - 1; i >= 0; --i) {
    if (bits.Get(i)) {
      return Bits::PowerOfTwo(i, bits.bit_count() + 1);
    }
  }
  return Bits::PowerOfTwo(bits.bit_count(), bits.bit_count() + 1);
}

Bits Reverse(const Bits& bits) {
  auto bits_vector = bits.ToBitVector();
  std::reverse(bits_vector.begin(), bits_vector.end());
  return Bits(bits_vector);
}

}  // namespace bits_ops

Bits LogicalOpIdentity(Op op, int64 width) {
  switch (op) {
    case Op::kAnd:
    case Op::kNand:
      return Bits::AllOnes(width);
    case Op::kOr:
    case Op::kNor:
    case Op::kXor:
      return Bits(width);
    default:
      XLS_LOG(FATAL) << "NaryOpIdentity got non-nary op:" << OpToString(op);
  }
}

Bits DoLogicalOp(Op op, absl::Span<const Bits> operands) {
  XLS_CHECK_GT(operands.size(), 0);
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
      XLS_LOG(FATAL) << "DoNaryBitOp got non-nary op: " << OpToString(op);
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

}  // namespace xls

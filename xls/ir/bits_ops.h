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

#ifndef XLS_IR_BITS_OPS_H_
#define XLS_IR_BITS_OPS_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/op.h"

namespace xls {
namespace bits_ops {

// Returns the given bits as an int64_t, returning std::nullopt if the result
// would be too large to fit in an int64_t.
std::optional<int64_t> TryUnsignedBitsToInt64(const Bits& bits);

// Returns the given bits as an int64_t, saturating to the maximum value if the
// result would be too large to fit in an int64_t.
int64_t UnsignedBitsToSaturatedInt64(const Bits& bits);

// Various bitwise operations. The width of the lhs and rhs must be equal, and
// the returned Bits object has the same width as the input.
Bits And(const Bits& lhs, const Bits& rhs);
Bits NaryAnd(absl::Span<const Bits> operands);
Bits Or(const Bits& lhs, const Bits& rhs);
Bits NaryOr(absl::Span<const Bits> operands);
Bits Xor(const Bits& lhs, const Bits& rhs);
Bits NaryXor(absl::Span<const Bits> operands);
Bits Not(const Bits& bits);
Bits Nand(const Bits& lhs, const Bits& rhs);
Bits NaryNand(absl::Span<const Bits> operands);
Bits Nor(const Bits& lhs, const Bits& rhs);
Bits NaryNor(absl::Span<const Bits> operands);

// Reducing bitwise operations. All of these produce single-bit result values.
Bits AndReduce(const Bits& operand);
Bits OrReduce(const Bits& operand);
Bits XorReduce(const Bits& operand);

// Various arithmetic operations. The width of all inputs must be equal, and the
// returned Bits object is truncated to the same width as the input.
Bits Increment(Bits x);
Bits Decrement(Bits x);
Bits Add(const Bits& lhs, const Bits& rhs);
Bits Sub(const Bits& lhs, const Bits& rhs);

// Signed/unsigned multiplication. The rhs and lhs can be different widths.
// The width of the result of the operation is the sum of the widths of the
// operands.
Bits SMul(const Bits& lhs, const Bits& rhs);
Bits UMul(const Bits& lhs, const Bits& rhs);

// Performs an (un)signed divide with round toward zero. The rhs and lhs can be
// different widths, and the returned result is the same width as the left
// operand. For UDiv, if the rhs is zero the result is all ones. For SDiv, if
// the rhs is zero the result is the maximal positive/negative value depending
// upon whether the lhs is positive/negative.
Bits SDiv(const Bits& lhs, const Bits& rhs);
Bits UDiv(const Bits& lhs, const Bits& rhs);

// Performs the (un)signed modulus operation. The rhs and lhs can be
// different widths, and the returned result is the same width as the right
// operand. For signed modulus, the sign of the result matches the sign of the
// left operand. If the right operand is zero, the result is zero for both
// signed and unsigned modulus.
Bits SMod(const Bits& lhs, const Bits& rhs);
Bits UMod(const Bits& lhs, const Bits& rhs);

// Various unsigned comparison operations. lhs and rhs can be different widths.
bool UEqual(const Bits& lhs, const Bits& rhs);
bool UGreaterThanOrEqual(const Bits& lhs, const Bits& rhs);
bool UGreaterThan(const Bits& lhs, const Bits& rhs);
bool ULessThanOrEqual(const Bits& lhs, const Bits& rhs);
bool ULessThan(const Bits& lhs, const Bits& rhs);

// Returns negative if lhs < rhs, 0 if lhs == rhs, and positive if lhs > rhs.
int64_t UCmp(const Bits& lhs, const Bits& rhs);

const Bits& UMin(const Bits& lhs, const Bits& rhs);
const Bits& UMax(const Bits& lhs, const Bits& rhs);

// Overloads for unsigned comparisons against an int64_t. CHECK fails if 'rhs'
// if negative because this is an unsigned comparison. We do not use an uint64_t
// to avoid surprising conversions. For example, with an uint64_t argument, the
// following would be true: LessThan(Ubits(42, 16), -1).
bool UEqual(const Bits& lhs, int64_t rhs);
bool UGreaterThanOrEqual(const Bits& lhs, int64_t rhs);
bool UGreaterThan(const Bits& lhs, int64_t rhs);
bool ULessThanOrEqual(const Bits& lhs, int64_t rhs);
bool ULessThan(const Bits& lhs, int64_t rhs);

// Various signed comparison operations.  lhs and rhs can be different widths.
bool SEqual(const Bits& lhs, const Bits& rhs);
bool SGreaterThanOrEqual(const Bits& lhs, const Bits& rhs);
bool SGreaterThan(const Bits& lhs, const Bits& rhs);
bool SLessThanOrEqual(const Bits& lhs, const Bits& rhs);
bool SLessThan(const Bits& lhs, const Bits& rhs);

bool SEqual(const Bits& lhs, int64_t rhs);
bool SGreaterThanOrEqual(const Bits& lhs, int64_t rhs);
bool SGreaterThan(const Bits& lhs, int64_t rhs);
bool SLessThanOrEqual(const Bits& lhs, int64_t rhs);
bool SLessThan(const Bits& lhs, int64_t rhs);

// Zero/sign extend 'bits' to the new bit count and return the result.
//
// Check-fails if new_bit_count is not >= bits.bit_count().
Bits ZeroExtend(Bits bits, int64_t new_bit_count);
Bits SignExtend(Bits bits, int64_t new_bit_count);

// Shift Left/Right Logical or Arithmetic (shift right only). The width of the
// returned Bits object is the same as the input.
Bits ShiftLeftLogical(const Bits& bits, int64_t shift_amount);
Bits ShiftRightLogical(const Bits& bits, int64_t shift_amount);
Bits ShiftRightArith(const Bits& bits, int64_t shift_amount);

// Performs a twos-complement negate. The width of the returned bits object is
// the same as the input. In case of negating the minimal negative number
// (e.g., bit pattern 0x800...) the value overflows (e.g. produces bit pattern
// 0x800...).
Bits Negate(const Bits& bits);

// Returns the absolute value of the given [signed] Bits.
Bits Abs(const Bits& bits);

// Concatenates the argument bits together. The zero-th index element in the
// span becomes the most significant bits in the returned Bits object.
Bits Concat(absl::Span<const Bits> inputs);

// Performs an operation equivalent to the XLS IR Op::kOneHot operation.
Bits OneHotLsbToMsb(const Bits& bits);
Bits OneHotMsbToLsb(const Bits& bits);

inline int64_t CountLeadingOnes(const Bits& bits) {
  return Not(bits).CountLeadingZeros();
}
inline int64_t CountTrailingOnes(const Bits& bits) {
  return Not(bits).CountTrailingZeros();
}

// Returns a Bits object with the bits of the argument in reverse order. That
// is, most-significant bit becomes least-significant bit, etc.
Bits Reverse(const Bits& bits);

// Returns a Bits object with leading zeroes stripped off, e.g.,
// bits[8]:0b00001010 would become bits[4]:1010.
// Returns a zero-bit/empty Bits for a zero-valued input.
Bits DropLeadingZeroes(const Bits& bits);

Bits Truncate(Bits bits, int64_t size);

// Returns a Bits object with the sequence of bits starting at index 'start'
// replaced with update_value. Any out-of-bounds updated bits are ignored.
Bits BitSliceUpdate(const Bits& to_update, int64_t start,
                    const Bits& update_value);

// Computes the longest common prefix of any number of `Bits`s, starting at the
// least-significant end of each bit string.
// CHECK fails if the `Bits`s don't all have the same bit width.
Bits LongestCommonPrefixLSB(absl::Span<const Bits> bits_span);

// Computes the longest common prefix of any number of `Bits`s, starting at the
// most-significant end of each bit string.
// CHECK fails if the `Bits`s don't all have the same bit width.
Bits LongestCommonPrefixMSB(absl::Span<const Bits> bits_span);

}  // namespace bits_ops

// Returns the identity value of the given width for the given logical Op (e.g.,
// Op::kAnd).
Bits LogicalOpIdentity(Op op, int64_t width);

// Returns the result of applying the given logical Op (e.g., Op::kAnd) to the
// operands. Must have at least one operand.
Bits DoLogicalOp(Op op, absl::Span<const Bits> operands);

// Operator overloads.
Bits operator&(const Bits& lhs, const Bits& rhs);
Bits operator|(const Bits& lhs, const Bits& rhs);
Bits operator^(const Bits& lhs, const Bits& rhs);
Bits operator~(const Bits& bits);

// Note: the following is not a "generically useful" bit op like the above
// declarations, but it is used in multiple places and operates purely on bits.
// For now, this seems like the best place to put it, but it is not really like
// the other things in this file.

// Computes a fixed offset for mulp operations, e.g. smulp(a, b) = (OFFSET,
// a*b-OFFSET). The purpose of this offset is to prevent naive mulp
// implementations from masking incorrect behavior.
//
// smulp and umulp return a 2-tuple with elements that can be any values
// satisfying smulp(a, b).0 + smulp(a, b).1 = a * b. Our simulation environments
// have to choose a particular implementation, and some are better than others.
// The naive (0, a*b) can hide the unsoundness of some optimizations: in
// particular, sign_ext(add(a, b)) and add(sign_ext(a), sign_ext(b)) are
// equivlent if a is 0 (i.e. the naive smulp implementation), but are not
// equivalent in general. Mulp implementations should choose a value OFFSET and
// return (OFFSET, a*b-OFFSET), and the OFFSET value should guarantee that some
// input values will produce different results if you extend before summing the
// mulp results. One way to guarantee this is to choose offset with 01 as MSBs.
// As long as a carry-in can occur for the inputs (guaranteed as long as both
// OFFSET and a*b are not all zero after their MSBs), the extend-then-add
// expression will evaluate to different values than the add-then-extend
// expression in some cases. Each different runtime environment should choose
// different LSBs to further protect against incorrect smulp usages that happen
// to produce correct outputs most of the time.
//
// Note that even though the above discusses signed multiplies, it should still
// be used for umulps. It is possible for smulp's to get lowered into umulps and
// the offsets for signed multiplies can still produce a carry out that will
// result in mismatch for umulp(zero_ext(a), zero_ext(b)) != zero_ext(umulp(a,
// b)).
//
// Args:
//  result_size: the size of each element in the result tuple of the mulp.
//  shift_size: the amount the right-shift the LSBs by, should be distinct for
//  each runtime.
Bits MulpOffsetForSimulation(int64_t result_size, int64_t shift_size);

// Emits the bits value as a string of digits. Supported FormatPreferences
// are: kDecimal, kHex, and kBinary. The string is not prefixed (e.g., no
// leading "0x"). Hexadecimal and binary numbers have '_' separators inserted
// every 4th digit. For example, the decimal number 1000 in binary:
// 111_1110_1000. If emit_leading_zeros is true, binary and hexadecimal
// formatted number will include leading zero up to the width of the
// underlying Bits object. For example, if emit_leading_zeros is true:
// Bits(42, 11) will result in '02a' and '000_0010_1010' for binary and
// hexadecimal format respectively.
std::string BitsToRawDigits(const Bits& bits, FormatPreference preference,
                            bool emit_leading_zeros = false);

// Returns a string representation of the Bits object. A kDefault format
// preference emits a decimal string if the bit count is less than or equal to
// 64, or a hexadecimal string otherwise. If include_bit_count is true, then
// also emit the bit width as a suffix; example: "0xabcd [16 bits]".
std::string BitsToString(
    const Bits& bits, FormatPreference preference = FormatPreference::kDefault,
    bool include_bit_count = false);

// Implementation note: the operator<< and AbslStringify definitions for the
// Bits datatype are defined here so that we can layer on top of BigNum
// functionality, which avoids having a massive translation unit for all
// bits-like-things (and bits.h stays a simple bitmap structure).

inline std::ostream& operator<<(std::ostream& os, const Bits& bits) {
  os << BitsToString(bits, FormatPreference::kDefault,
                     /*include_bit_count=*/true);
  return os;
}

template <typename Sink>
void AbslStringify(Sink& sink, const Bits& bits) {
  sink.Append(BitsToString(bits));
}

}  // namespace xls

#endif  // XLS_IR_BITS_OPS_H_

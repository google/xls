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

#ifndef THIRD_PARTY_XLS_IR_BITS_OPS_H_
#define THIRD_PARTY_XLS_IR_BITS_OPS_H_

#include "xls/ir/bits.h"
#include "xls/ir/op.h"

namespace xls {
namespace bits_ops {

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

// Reducing bitwise operations.
Bits AndReduce(const Bits& operand);
Bits OrReduce(const Bits& operand);
Bits XorReduce(const Bits& operand);

// Various arithmetic operations. The width of the lhs and rhs must be equal,
// and the returned Bits object is truncated to the same width as the input.
Bits Add(const Bits& lhs, const Bits& rhs);
Bits Sub(const Bits& lhs, const Bits& rhs);

// Signed/unsigned multiplication. The rhs and lhs can be different widths. The
// width of the result of the operation is the sum of the widths of the
// operands.
Bits SMul(const Bits& lhs, const Bits& rhs);
Bits UMul(const Bits& lhs, const Bits& rhs);

// Performs an (un)signed divide with round toward zero. The width of the lhs
// and rhs must be equal, and the returned result is the same width as the
// inputs. For UDiv, if the rhs is zero the result is all ones. For SDiv, if the
// rhs is zero the result is the maximal positive/negative value depending upon
// whether the lhs is positive/negative.
Bits SDiv(const Bits& lhs, const Bits& rhs);
Bits UDiv(const Bits& lhs, const Bits& rhs);

// Various unsigned comparison operations.
bool UGreaterThanOrEqual(const Bits& lhs, const Bits& rhs);
bool UGreaterThan(const Bits& lhs, const Bits& rhs);
bool ULessThanOrEqual(const Bits& lhs, const Bits& rhs);
bool ULessThan(const Bits& lhs, const Bits& rhs);

// Overloads for unsigned comparisons against an int64. CHECK fails if 'rhs' if
// negative because this is an unsigned comparison. We do not use an uint64 to
// avoid surprising conversions. For example, with an uint64 argument, the
// following would be true: LessThan(Ubits(42, 16), -1).
bool UGreaterThanOrEqual(const Bits& lhs, int64 rhs);
bool UGreaterThan(const Bits& lhs, int64 rhs);
bool ULessThanOrEqual(const Bits& lhs, int64 rhs);
bool ULessThan(const Bits& lhs, int64 rhs);

// Various signed comparison operations.
bool SGreaterThanOrEqual(const Bits& lhs, const Bits& rhs);
bool SGreaterThan(const Bits& lhs, const Bits& rhs);
bool SLessThanOrEqual(const Bits& lhs, const Bits& rhs);
bool SLessThan(const Bits& lhs, const Bits& rhs);

bool SGreaterThanOrEqual(const Bits& lhs, int64 rhs);
bool SGreaterThan(const Bits& lhs, int64 rhs);
bool SLessThanOrEqual(const Bits& lhs, int64 rhs);
bool SLessThan(const Bits& lhs, int64 rhs);

// Zero/sign extend 'bits' to the new bit count and return the result.
//
// Check-fails if new_bit_count is not >= bits.bit_count().
Bits ZeroExtend(const Bits& bits, int64 new_bit_count);
Bits SignExtend(const Bits& bits, int64 new_bit_count);

// Shift Left/Right Logical or Arithmetic (shift right only). The width of the
// returned Bits object is the same as the input.
Bits ShiftLeftLogical(const Bits& bits, int64 shift_amount);
Bits ShiftRightLogical(const Bits& bits, int64 shift_amount);
Bits ShiftRightArith(const Bits& bits, int64 shift_amount);

// Performs a twos-complement negate. The width of the returned bits object is
// the same as the input. In case of negating the minimal negative number
// (e.g., bit pattern 0x800...) the value overflows (e.g. produces bit pattern
// 0x800...).
Bits Negate(const Bits& bits);

// Concatenates the argument bits together. The zero-th index element in the
// span becomes the most significant bits in the returned Bits object.
Bits Concat(absl::Span<const Bits> inputs);

// Performs an operation equivalent to the XLS IR Op::kOneHot operation.
Bits OneHotLsbToMsb(const Bits& bits);
Bits OneHotMsbToLsb(const Bits& bits);

inline int64 CountLeadingOnes(const Bits& bits) {
  return Not(bits).CountLeadingZeros();
}
inline int64 CountTrailingOnes(const Bits& bits) {
  return Not(bits).CountTrailingZeros();
}

// Returns a Bits object with the bits of the argument in reverse order. That
// is, most-significant bit becomes least-significant bit, etc.
Bits Reverse(const Bits& bits);

}  // namespace bits_ops

// Returns the identity value of the given width for the given logical Op (e.g.,
// Op::kAnd).
Bits LogicalOpIdentity(Op op, int64 width);

// Returns the result of applying the given logical Op (e.g., Op::kAnd) to the
// operands. Must have at least one operand.
Bits DoLogicalOp(Op op, absl::Span<const Bits> operands);

// Operator overloads.
Bits operator&(const Bits& lhs, const Bits& rhs);
Bits operator|(const Bits& lhs, const Bits& rhs);
Bits operator^(const Bits& lhs, const Bits& rhs);
Bits operator~(const Bits& bits);

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_BITS_OPS_H_

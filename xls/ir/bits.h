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

#ifndef THIRD_PARTY_XLS_IR_BITS_H_
#define THIRD_PARTY_XLS_IR_BITS_H_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "absl/base/casts.h"
#include "absl/strings/str_format.h"
#include "xls/common/bits_util.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/statusor.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bit_push_buffer.h"
#include "xls/ir/format_preference.h"

namespace xls {
namespace internal {

// Helper for use in initializer list; checks that lhs >= rhs and returns lhs as
// a result.
template <typename T>
T CheckGe(T lhs, T rhs) {
  XLS_CHECK_GE(lhs, rhs);
  return lhs;
}

}  // namespace internal

// Helper class that is used to represent a literal value that is a vector of
// bits with a given width (bit_count).
class Bits {
 public:
  Bits() : Bits(0) {}

  // Creates a zero-initialized bits object with the given bit width.
  explicit Bits(int64 bit_count)
      : bitmap_(internal::CheckGe(bit_count, int64{0})) {}

  // Creates a bits object with the given bits values. The n-th element of the
  // argument becomes the n-th bit of the Bits object. A std::vector<bool>
  // cannot be passed in because of std::vector<bool> specialization. Use
  // absl::InlinedVector instead.
  explicit Bits(absl::Span<bool const> bits);

  // Returns a Bits object with all ones. If 'bit_count' is zero, an empty Bits
  // object is returned as, vacuously, an empty bits object has all of its bits
  // set to one.
  static Bits AllOnes(int64 bit_count);

  // These functions create Bits objects with the maximum or minimum signed
  // value with the specified width.
  static Bits MaxSigned(int64 bit_count);
  static Bits MinSigned(int64 bit_count);

  // Returns a Bits object with exactly one bit set.
  static Bits PowerOfTwo(int64 set_bit_index, int64 bit_count);

  // Constructs a Bits object from a vector of bytes. Bytes are in big endian
  // order where the byte zero is the most significant byte. The size of 'bytes'
  // must be at least than bit_count / 8. Any bits beyond 'bit_count' in 'bytes'
  // are ignored.
  static Bits FromBytes(absl::Span<const uint8> bytes, int64 bit_count);

  // Note: we flatten into the pushbuffer with the MSb pushed first.
  void FlattenTo(BitPushBuffer* buffer) const {
    for (int64 i = 0; i < bit_count(); ++i) {
      buffer->PushBit(Get(bit_count() - i - 1));
    }
  }

  // Return the Bits object as a vector of bytes. Bytes are in big endian order
  // where the byte zero of the returned vector is the most significant
  // byte. The returned vector has size ceil(bit_count()/8). Any bits in the
  // returned vector beyond the bit_count()-th are set to zero.
  std::vector<uint8> ToBytes() const {
    std::vector<uint8> v(bitmap_.byte_count());
    ToBytes(absl::MakeSpan(v));
    return v;
  }

  // Implements ToBytes() as above by inserting values into a user-provided raw
  // byte buffer.
  //
  // "big_endian" determines whether the least significant byte from this bits
  // object ends up in the 0th byte (on false), or in the N-1th byte (on true).
  // This can be useful to control when calling machine code.
  void ToBytes(absl::Span<uint8> bytes, bool big_endian = true) const {
    int64 byte_count = bitmap_.byte_count();
    XLS_DCHECK(bytes.size() >= byte_count);
    // Use raw access to avoid evaluating ABSL_ASSERT on every reference.
    uint8* buffer = bytes.data();
    for (int64 i = 0; i < byte_count; ++i) {
      int64 index = big_endian ? byte_count - i - 1 : i;
      buffer[index] = bitmap_.GetByte(i);
    }
    // Mask off final byte.
    if (bit_count() % 8 != 0) {
      buffer[big_endian ? 0 : byte_count - 1] &= Mask(bit_count() % 8);
    }
  }

  // Return the Bits object as a vector of bits. The n-th element of the
  // returned vector is the n-th bit of the Bits object. Returns
  // absl::InlinedVector to avoid std::vector<bool> specialization.
  absl::InlinedVector<bool, 1> ToBitVector() const;

  int64 bit_count() const { return bitmap_.bit_count(); }

  // Returns a string representation of the Bits object. A kDefault format
  // preference emits a decimal string if the bit count is less than or equal to
  // 64, or a hexadecimal string otherwise. If include_bit_count is true, then
  // also emit the bit width as a suffix; example: "0xabcd [16 bits]".
  std::string ToString(FormatPreference preference = FormatPreference::kDefault,
                       bool include_bit_count = false) const;

  // Emits the bits value as an string of digits. Supported FormatPreferences
  // are: kDecimal, kHex, and kBinary. The string is not prefixed (e.g., no
  // leading "0x"). Hexadecimal and binary numbers have '_' separators inserted
  // every 4th digit. For example, the decimal number 1000 in binary:
  // 111_1110_1000. If emit_leading_zeros is true, binary and hexadecimal
  // formatted number will include leading zero up to the width of the
  // underlying Bits object. For example, if emit_leading_zeros is true:
  // Bits(42, 11) will result in '02a' and '000_0010_1010' for binary and
  // hexadecimal format respectively.
  std::string ToRawDigits(FormatPreference preference,
                          bool emit_leading_zeros = false) const;

  // Return the most-significant bit.
  bool msb() const { return bit_count() == 0 ? false : Get(bit_count() - 1); }

  // Get/Set individual bits in the underlying Bitmap.
  // LSb is bit 0, MSb is at location bit_count() - 1.
  bool Get(int64 index) const { return bitmap_.Get(index); }

  // As above, but retrieves with index "0" starting at the MSb side of the bit
  // vector.
  bool GetFromMsb(int64 index) const {
    XLS_DCHECK_LT(index, bit_count());
    return bitmap_.Get(bit_count() - index - 1);
  }

  ABSL_MUST_USE_RESULT
  Bits UpdateWithSet(int64 index, bool value) const {
    Bits clone = *this;
    clone.bitmap_.Set(index, value);
    return clone;
  }

  // Returns whether the bits are all zeros/ones.
  bool IsAllOnes() const { return bitmap_.IsAllOnes(); }
  bool IsAllZeros() const { return bitmap_.IsAllZeroes(); }

  // Returns true if the bits interpreted as an unsigned number is equal to one.
  bool IsOne() const;

  // Returns true if the Bits value as an unsigned number is a power of two.
  bool IsPowerOfTwo() const { return PopCount() == 1; }

  // Returns the number of ones set in the Bits value.
  int64 PopCount() const;

  // Counts the number of contiguous zero (ones) bits present from the MSb
  // (towards the LSb). Result is `0 <= x <= bits.bit_count()`.
  int64 CountLeadingZeros() const;
  int64 CountLeadingOnes() const;

  // Counts the number of contiguous zero (ones) bits present from the LSb
  // (towards the MSb). Result is `0 <= x <= bits.bit_count()`.
  int64 CountTrailingZeros() const;
  int64 CountTrailingOnes() const;

  // Checks whether the bits value has a single contiguous run of set bits; i.e.
  // matches:
  //
  //    0b 0* 1+ 0*
  //
  // And returns the leading zero count in leading_zero_count, the number of set
  // bits in set_bit_count, and the number of trailing zeros in
  // trailing_zero_count if true is returned.
  bool HasSingleRunOfSetBits(int64* leading_zero_count, int64* set_bit_count,
                             int64* trailing_zero_count) const;

  // Returns true if the (unsigned/signed) value held by the Bits object fits in
  // 'n' bits.
  bool FitsInNBitsUnsigned(int64 n) const;
  bool FitsInNBitsSigned(int64 n) const;

  // Returns true if the (unsigned/signed) value held by the Bits object fits in
  // uint64/int64.
  bool FitsInUint64() const;
  bool FitsInInt64() const;

  // Converts the value held by this "bits" object into a uint64 (int64).
  // ToUint64 interprets the bits as unsigned. ToInt64 interprets the bits in
  // twos-complement representation. Returns an error if the *value* cannot be
  // represented in 64 bits (the width of the Bits object can be arbitrarily
  // large).
  xabsl::StatusOr<uint64> ToUint64() const;
  xabsl::StatusOr<int64> ToInt64() const;

  // Returns whether this "bits" object is identical to the other in both
  // bit_count() and held value.
  bool operator==(const Bits& other) const { return bitmap_ == other.bitmap_; }
  bool operator!=(const Bits& other) const { return !(*this == other); }

  // Slices a range of bits from the Bits object. 'start' is the first index in
  // the slice. 'start' is zero-indexed with zero being the LSb (same indexing
  // as Get/Set). 'width' is the number of bits to slice out and is the
  // bit_count of the result.
  Bits Slice(int64 start, int64 width) const;

  // Returns the minimum number of bits required to store the given value as an
  // unsigned number.
  static int64 MinBitCountUnsigned(uint64 value) {
    return value == 0 ? 0 : FloorOfLog2(value) + 1;
  }

  // Returns the minimum number of bits required to store the given value as
  // twos complement signed number.
  static int64 MinBitCountSigned(int64 value);

 private:
  friend class BitsRope;
  friend xabsl::StatusOr<Bits> UBitsWithStatus(uint64, int64);
  friend xabsl::StatusOr<Bits> SBitsWithStatus(int64, int64);

  explicit Bits(InlineBitmap&& bitmap) : bitmap_(bitmap) {}

  InlineBitmap bitmap_;
};

// Helper for "stringing together" bits objects into a final result, avoiding
// intermediate allocations.
class BitsRope {
 public:
  explicit BitsRope(int64 total_bit_count) : bitmap_(total_bit_count) {}

  // Pushes the bits object into the bit string being built.
  //
  // Note that bit 0 of the first bits object pushed becomes the LSb of the
  // final result; so something like:
  //
  //    rope.push_back(a_1 a_0)
  //    rope.push_back(b_2 b_1 b_0)
  //
  // Creates a resulting rope:
  //
  //    b_2 b_1 b_0 a_1 a_0
  //
  // So b.Get(0) is now at result.Get(2).
  void push_back(const Bits& bits) {
    for (int64 i = 0; i < bits.bit_count(); ++i) {
      bitmap_.Set(index_ + i, bits.Get(i));
    }
    index_ += bits.bit_count();
  }

  void push_back(bool bit) { bitmap_.Set(index_++, bit); }

  Bits Build() {
    XLS_CHECK_EQ(index_, bitmap_.bit_count());
    return Bits{std::move(bitmap_)};
  }

 private:
  InlineBitmap bitmap_;
  int64 index_ = 0;
};

// Creates an Bits object which holds the given unsigned/signed value. Width
// must be large enough to hold the value. The bits object itelf has no
// signedness, but the UBits and SBits factories affect how the minimum bit
// width is computed for checking against 'width' and whether the value is
// zero or sign extended.
xabsl::StatusOr<Bits> UBitsWithStatus(uint64 value, int64 bit_count);
xabsl::StatusOr<Bits> SBitsWithStatus(int64 value, int64 bit_count);
inline Bits UBits(uint64 value, int64 bit_count) {
  return UBitsWithStatus(value, bit_count).value();
}
inline Bits SBits(int64 value, int64 bit_count) {
  return SBitsWithStatus(value, bit_count).value();
}

inline std::ostream& operator<<(std::ostream& os, const Bits& bits) {
  os << bits.ToString(FormatPreference::kDefault, /*include_bit_count=*/true);
  return os;
}

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_BITS_H_

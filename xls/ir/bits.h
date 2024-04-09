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

#ifndef XLS_IR_BITS_H_
#define XLS_IR_BITS_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bit_push_buffer.h"

namespace xls {
namespace internal {

// Helper for use in initializer list; checks that lhs >= rhs and returns lhs as
// a result.
template <typename T>
T CheckGe(T lhs, T rhs) {
  CHECK_GE(lhs, rhs);
  return lhs;
}

}  // namespace internal

// Helper class that is used to represent a literal value that is a vector of
// bits with a given width (bit_count).
class Bits {
 public:
  Bits() : Bits(0) {}

  // Creates a zero-initialized bits object with the given bit width.
  explicit Bits(int64_t bit_count)
      : bitmap_(internal::CheckGe(bit_count, int64_t{0})) {}

  // Creates a bits object with the given bits values. The n-th element of the
  // argument becomes the n-th bit of the Bits object. A std::vector<bool>
  // cannot be passed in because of std::vector<bool> specialization. Use
  // absl::InlinedVector instead.
  explicit Bits(absl::Span<bool const> bits);

  // Sets the values of a range. The range is defined as:
  // [lower_index, upper_index).
  void SetRange(int64_t start_index, int64_t end_index, bool value = true);

  // Returns a Bits object with all ones. If 'bit_count' is zero, an empty Bits
  // object is returned as, vacuously, an empty bits object has all of its bits
  // set to one.
  static Bits AllOnes(int64_t bit_count);

  // These functions create Bits objects with the maximum or minimum signed
  // value with the specified width.
  static Bits MaxSigned(int64_t bit_count);
  static Bits MinSigned(int64_t bit_count);

  // Returns a Bits object with exactly one bit set.
  static Bits PowerOfTwo(int64_t set_bit_index, int64_t bit_count);

  // Constructs a Bits object from a vector of bytes. Bytes are in big endian
  // order where the byte zero is the most significant byte. The size of 'bytes'
  // must be at least than bit_count / 8. Any bits beyond 'bit_count' in 'bytes'
  // are ignored.
  static Bits FromBytes(absl::Span<const uint8_t> bytes, int64_t bit_count) {
    return Bits(InlineBitmap::FromBytes(bit_count, bytes));
  }

  // Constructs a Bits object from a bitmap.
  static Bits FromBitmap(InlineBitmap bitmap) {
    return Bits(std::move(bitmap));
  }

  // Note: we flatten into the pushbuffer with the MSb pushed first.
  void FlattenTo(BitPushBuffer* buffer) const {
    for (int64_t i = 0; i < bit_count(); ++i) {
      buffer->PushBit(Get(bit_count() - i - 1));
    }
  }

  // Return the Bits object as a vector of bytes. Bytes are in little endian
  // order where the byte zero of the returned vector is the least significant
  // byte. The returned vector has size ceil(bit_count()/8). Any bits in the
  // returned vector beyond the bit_count()-th are set to zero.
  std::vector<uint8_t> ToBytes() const {
    std::vector<uint8_t> v(bitmap_.byte_count());
    ToBytes(absl::MakeSpan(v));
    return v;
  }

  // Implements ToBytes() as above by inserting values into a user-provided raw
  // byte buffer.
  void ToBytes(absl::Span<uint8_t> bytes) const {
    bitmap_.WriteBytesToBuffer(bytes);
  }

  // Returns the underlying bitmap contents as a binary string (suitable for
  // debugging printouts in e.g. error messages) -- this is used for any error
  // reporting that needs to happen at the `Bits` level of functionality --
  // nicer pretty printing is available in `bits_ops.h`.
  std::string ToDebugString() const;

  // Returns the Bits object as a vector of bits. The n-th element of the
  // returned vector is the n-th bit of the Bits object, i.e. the msb is at the
  // end of the vector.
  //
  // Returns absl::InlinedVector to avoid std::vector<bool> specialization.
  absl::InlinedVector<bool, 1> ToBitVector() const;

  int64_t bit_count() const { return bitmap_.bit_count(); }

  // Return the most-significant bit.
  bool msb() const { return bit_count() == 0 ? false : Get(bit_count() - 1); }

  // Get/Set individual bits in the underlying Bitmap.
  // LSb is bit 0, MSb is at location bit_count() - 1.
  bool Get(int64_t index) const { return bitmap_.Get(index); }

  // As above, but retrieves with index "0" starting at the MSb side of the bit
  // vector.
  bool GetFromMsb(int64_t index) const {
    DCHECK_LT(index, bit_count());
    return bitmap_.Get(bit_count() - index - 1);
  }

  ABSL_MUST_USE_RESULT
  Bits UpdateWithSet(int64_t index, bool value) const {
    Bits clone = *this;
    clone.bitmap_.Set(index, value);
    return clone;
  }

  // Returns whether the bits are all ones.
  bool IsAllOnes() const { return bitmap_.IsAllOnes(); }

  // Returns true if the bits interpreted as an unsigned number is equal to one.
  bool IsOne() const;

  // Returns true if the bits value is zero.
  bool IsZero() const { return bitmap_.IsAllZeroes(); }

  // Returns true if the Bits value as an unsigned number is a power of two.
  bool IsPowerOfTwo() const { return PopCount() == 1; }

  // Returns the number of ones set in the Bits value.
  int64_t PopCount() const;

  // Counts the number of contiguous zero (ones) bits present from the MSb
  // (towards the LSb). Result is `0 <= x <= bits.bit_count()`.
  int64_t CountLeadingZeros() const;
  int64_t CountLeadingOnes() const;

  // Counts the number of contiguous zero (ones) bits present from the LSb
  // (towards the MSb). Result is `0 <= x <= bits.bit_count()`.
  int64_t CountTrailingZeros() const;
  int64_t CountTrailingOnes() const;

  // Checks whether the bits value has a single contiguous run of set bits; i.e.
  // matches:
  //
  //    0b 0* 1+ 0*
  //
  // And returns the leading zero count in leading_zero_count, the number of set
  // bits in set_bit_count, and the number of trailing zeros in
  // trailing_zero_count if true is returned.
  bool HasSingleRunOfSetBits(int64_t* leading_zero_count,
                             int64_t* set_bit_count,
                             int64_t* trailing_zero_count) const;

  // Returns true if the (unsigned/signed) value held by the Bits object fits in
  // 'n' bits.
  bool FitsInNBitsUnsigned(int64_t n) const;
  bool FitsInNBitsSigned(int64_t n) const;

  // Returns true if the (unsigned/signed) value held by the Bits object fits in
  // uint64_t/int64_t.
  bool FitsInUint64() const;
  bool FitsInInt64() const;

  // Converts the value held by this "bits" object into a uint64_t (int64_t).
  // ToUint64 interprets the bits as unsigned. ToInt64 interprets the bits in
  // twos-complement representation. Returns an error if the *value* cannot be
  // represented in 64 bits (the width of the Bits object can be arbitrarily
  // large).
  absl::StatusOr<uint64_t> ToUint64() const;
  absl::StatusOr<int64_t> ToInt64() const;

  // Extracts the "word_number"th u64 from this value, as in Bitmap::GetWord.
  // (Zero-bit values get 0 by convention.)
  absl::StatusOr<uint64_t> WordToUint64(int64_t word_number) const;

  // Returns whether this "bits" object is identical to the other in both
  // bit_count() and held value.
  bool operator==(const Bits& other) const { return bitmap_ == other.bitmap_; }
  bool operator!=(const Bits& other) const { return !(*this == other); }

  // Slices a range of bits from the Bits object. 'start' is the first index in
  // the slice. 'start' is zero-indexed with zero being the LSb (same indexing
  // as Get/Set). 'width' is the number of bits to slice out and is the
  // bit_count of the result.
  Bits Slice(int64_t start, int64_t width) const&;
  Bits Slice(int64_t start, int64_t width) &&;

  // Returns the minimum number of bits required to store the given value as an
  // unsigned number.
  static int64_t MinBitCountUnsigned(uint64_t value) {
    return value == 0 ? 0 : FloorOfLog2(value) + 1;
  }

  // Returns the minimum number of bits required to store the given value as
  // twos complement signed number.
  static int64_t MinBitCountSigned(int64_t value);

  const InlineBitmap& bitmap() const& { return bitmap_; }
  InlineBitmap&& bitmap() && { return std::move(bitmap_); }

  template <typename H>
  friend H AbslHashValue(H h, const Bits& bits) {
    return H::combine(std::move(h), bits.bitmap_);
  }

 private:
  friend class BitsRope;
  friend absl::StatusOr<Bits> UBitsWithStatus(uint64_t, int64_t);
  friend absl::StatusOr<Bits> SBitsWithStatus(int64_t, int64_t);

  explicit Bits(InlineBitmap&& bitmap) : bitmap_(bitmap) {}

  InlineBitmap bitmap_;
};

// Helper for "stringing together" bits objects into a final result, avoiding
// intermediate allocations.
class BitsRope {
 public:
  explicit BitsRope(int64_t total_bit_count) : bitmap_(total_bit_count) {}

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
    for (int64_t i = 0; i < bits.bit_count(); ++i) {
      bitmap_.Set(index_ + i, bits.Get(i));
    }
    index_ += bits.bit_count();
  }

  void push_back(bool bit) { bitmap_.Set(index_++, bit); }

  Bits Build() {
    CHECK_EQ(index_, bitmap_.bit_count());
    return Bits{std::move(bitmap_)};
  }

 private:
  InlineBitmap bitmap_;
  int64_t index_ = 0;
};

// Creates an Bits object which holds the given unsigned/signed value. Width
// must be large enough to hold the value. The bits object itelf has no
// signedness, but the UBits and SBits factories affect how the minimum bit
// width is computed for checking against 'width' and whether the value is
// zero or sign extended.
absl::StatusOr<Bits> UBitsWithStatus(uint64_t value, int64_t bit_count);
absl::StatusOr<Bits> SBitsWithStatus(int64_t value, int64_t bit_count);
inline Bits UBits(uint64_t value, int64_t bit_count) {
  return UBitsWithStatus(value, bit_count).value();
}
inline Bits SBits(int64_t value, int64_t bit_count) {
  return SBitsWithStatus(value, bit_count).value();
}

}  // namespace xls

#endif  // XLS_IR_BITS_H_

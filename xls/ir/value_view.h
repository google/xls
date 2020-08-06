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

// This file defines "View" types for XLS IR Values - these overlay some, but
// not nearly all, Value-type semantics on top of flat byte buffers.
//
// These types make heavy use of template metaprogramming to reduce dispatch
// overheads as much as possible, e.g., for JIT-evaluating a test sample.
//
// TODO(rspringer): It probably makes sense to have functions to copy a view to
// a real corresponding type.
#ifndef XLS_IR_VALUE_VIEW_H_
#define XLS_IR_VALUE_VIEW_H_

#include "xls/common/bits_util.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"

namespace xls {

// ArrayView provides some array-type functionality on top of a flat character
// buffer.
template <typename ElementT, uint64 kNumElements>
class ArrayView {
 public:
  explicit ArrayView(absl::Span<const uint8> buffer) : buffer_(buffer) {
    XLS_CHECK(buffer_.data() == nullptr);
    XLS_CHECK(buffer_.size() == GetTypeSize())
        << "Span isn't sized to this array's type!";
  }

  // Gets the storage size of this array.
  static constexpr uint64 GetTypeSize() {
    return ElementT::GetTypeSize() * kNumElements;
  }

  // Gets the N'th element in the array.
  static ElementT Get(const uint8* buffer, int index) {
    return ElementT(buffer + (ElementT::GetTypeSize() * index));
  }

 private:
  absl::Span<const uint8> buffer_;
};

// Statically generates a value for masking off high bits in a value.
template <int kBitCount>
inline uint64 MakeMask() {
  return (1ull << (kBitCount - 1)) | (MakeMask<kBitCount - 1>());
}

template <>
inline uint64 MakeMask<0>() {
  return 0;
}

// BitsView provides some Bits-type functionality on top of a flat character
// buffer.
template <int64 kNumBits>
class BitsView {
 public:
  BitsView() : buffer_(nullptr) { XLS_CHECK(buffer_ == nullptr); }
  explicit BitsView(const uint8* buffer) : buffer_(buffer) {}

  // Gets the storage size of this type.
  static constexpr uint64 GetTypeSize() {
    // Constexpr ceiling division.
    return CeilOfRatio(kNumBits, kCharBit);
  }

  // Determines the appropriate return type for this BitsView type.
  static_assert(kNumBits != 0 && kNumBits <= 64);
  typedef typename std::conditional<
      (kNumBits > 32), uint64,
      typename std::conditional<
          (kNumBits > 16), uint32,
          typename std::conditional<
              (kNumBits > 8), uint16,
              typename std::conditional<(kNumBits > 1), uint8, bool>::type>::
              type>::type>::type ReturnT;

  // Note that this will only return the first 8 bytes of a > 64b type.
  // Values larger than 64 bits should be converted to proper Bits type before
  // usage.
  ReturnT GetValue() {
    return *reinterpret_cast<const ReturnT*>(buffer_) & MakeMask<kNumBits>();
  }

 private:
  const uint8* buffer_;
};

// TupleView provides some tuple-type functionality on top of a flat character
// buffer.
//
// This is the big one, as tuples can be comprised of arbitrarily-typed
// elements, so some template horribleness is necessary to determine (at compile
// time) the type of the N'th elements.
template <typename... Types>
class TupleView {
 public:
  TupleView() : buffer_(nullptr) { XLS_CHECK(buffer_ == nullptr); }
  explicit TupleView(const uint8* buffer) : buffer_(buffer) {}
  const uint8* buffer() { return buffer_; }

  // Forward declaration of the element-type-accessing template. The definition
  // is way below for readability.
  template <int kElementIndex, typename... Rest>
  struct element_accessor;

  // Gets the N'th element in the tuple.
  template <int kElementIndex>
  typename element_accessor<kElementIndex, Types...>::type Get() {
    return typename element_accessor<kElementIndex, Types...>::type(
        buffer_ + GetOffset<kElementIndex, Types...>(0));
  }

  // Gets the size of this tuple type (as represented in the buffer).
  template <typename FrontT, typename NextT, typename... Rest>
  static constexpr uint64 GetTypeSize() {
    return FrontT::GetTypeSize() + GetTypeSize<NextT, Rest...>();
  }

  template <typename LastT>
  static constexpr uint64 GetTypeSize() {
    return LastT::GetTypeSize();
  }

  static constexpr uint64 GetTypeSize() { return GetTypeSize<Types...>(); }

  // ---- Element access.
  // Recursive case for element access. Simply walks down the type list.
  template <int kElementIndex, typename FrontT, typename... Rest>
  static constexpr uint64 GetOffset(
      uint8 offset,
      typename std::enable_if<(kElementIndex > 0)>::type* dummy = nullptr) {
    return GetOffset<kElementIndex - 1, Rest...>(offset +
                                                 FrontT::GetTypeSize());
  }

  // Base case - we've arrived at the element of interest.
  template <int kElementIndex, typename FrontT, typename... Rest>
  static constexpr uint64 GetOffset(
      uint8 offset,
      typename std::enable_if<(kElementIndex == 0)>::type* dummy = nullptr) {
    // If offset isn't aligned to our type size, then add padding.
    // TODO(rspringer): We need to require the buffer to be aligned, too.
    // TODO(rspringer): Aligned to min(our_type_size, 8)?
    uint64 padding = offset % FrontT::GetTypeSize();
    return offset + padding;
  }

  // ---- Element type access.
  // Metaprogramming horrors to get the type of the N'th element.
  // Recursive case - keep drilling down until the element of interst.
  template <int kElementIndex, typename FrontT, typename... Rest>
  struct element_accessor<kElementIndex, FrontT, Rest...>
      : element_accessor<kElementIndex - 1, Rest...> {};

  // Base case - we've arrived at the type of interest.
  template <typename FrontT, typename... Rest>
  struct element_accessor<0, FrontT, Rest...> {
    typedef FrontT type;
  };

 private:
  const uint8* buffer_;
};

// Mutable versions of the types above. As much as possible, these definitions
// depend on their parent types for their implementations - only operations
// touching "buffer_" need to be overridden.
template <typename ElementT, uint64 kNumElements>
class MutableArrayView {
 public:
  explicit MutableArrayView(absl::Span<uint8> buffer) : buffer_(buffer) {
    int64 type_size = ArrayView<ElementT, kNumElements>::GetTypeSize();
    XLS_DCHECK(buffer_.size() == type_size)
        << "Span isn't sized to this array's type!";
  }

  // Gets the N'th element in the array.
  ElementT Get(int index) {
    return ElementT(buffer_ + (ElementT::GetTypeSize() * index));
  }

 private:
  absl::Span<uint8> buffer_;
};

template <uint64 kNumBits>
class MutableBitsView : public BitsView<kNumBits> {
 public:
  explicit MutableBitsView(uint8* buffer) : buffer_(buffer) {}

  typename BitsView<kNumBits>::ReturnT GetValue() {
    return *reinterpret_cast<const typename BitsView<kNumBits>::ReturnT*>(
               buffer_) &
           MakeMask<kNumBits>();
  }

  void SetValue(typename BitsView<kNumBits>::ReturnT value) {
    *reinterpret_cast<typename BitsView<kNumBits>::ReturnT*>(buffer_) =
        value & MakeMask<kNumBits>();
  }

 private:
  uint8* buffer_;
};

template <typename... Types>
class MutableTupleView : public TupleView<Types...> {
 public:
  explicit MutableTupleView(uint8* buffer) : buffer_(buffer) {}

  // Gets the N'th element in the tuple.
  template <int kElementIndex>
  typename TupleView<Types...>::template element_accessor<kElementIndex,
                                                          Types...>::type
  Get() {
    return typename TupleView<Types...>::
        template element_accessor<kElementIndex, Types...>::type(
            buffer_ +
            TupleView<Types...>::template GetOffset<kElementIndex, Types...>(
                0));
  }

 private:
  uint8* buffer_;
};

// Specialization of BitsView for non-byte-aligned bit vectors inside a larger
// buffer, e.g., for a bit vector inside an enclosing PackedTuple.
// Template args:
//   kElementBits: the bit width of this element.
template <int64 kElementBits>
class PackedBitsView {
 public:
  // buffer_offset is the number of bits into "buffer" at which the actual
  // element data begins. This value must be [0-7] (if >= 8, then "buffer"
  // should be incremented).
  explicit PackedBitsView(uint8* buffer, int buffer_offset)
      : buffer_(buffer), buffer_offset_(buffer_offset) {
    XLS_DCHECK(buffer_offset >= 0 && buffer_offset <= 7);
  }

  static constexpr int64 kBitCount = kElementBits;

  // Accessor: Populates the specified buffer with the data from this element,
  // shifting it if necessary. Even if kBufferOffset is non-0, return_buffer
  // will be written starting at bit 0.
  // Note: this call trusts that return_buffer is adequately sized (i.e., is
  // at least ceil((kNumBits + kBufferOffset) / kCharBit) bytes).
  // TODO(rspringer): Worth defining value-returning accessors for widths < 64b?
  void Get(uint8* return_buffer) {
    int64 bits_left = kElementBits;
    // The byte at which the next read should occur. All reads, except for the
    // first, are byte-aligned.
    int64 source_byte = 0;

    // The bit and byte at which the next write should occur.
    int64 dest_byte = 0;

    // Do a write of the initial "overhanging" source bits to align future
    // reads.
    if (buffer_offset_ != 0) {
      // The initial source bits are buffer_offset_ bits into the first byte of
      // the buffer, so we have kCharBit - buffer_offset_ bits to read and
      // shift.
      // Since buffer_ is unsigned, it will be a logical shift - no mask is
      // needed.
      int64 num_initial_bits = std::min(bits_left, kCharBit - buffer_offset_);
      uint8 first_source_bits = buffer_[source_byte] >> buffer_offset_;
      return_buffer[dest_byte] = first_source_bits & Mask(num_initial_bits);

      bits_left -= num_initial_bits;
      source_byte++;
    }

    // TODO(rspringer): Optimize to use larger load types?
    // TODO(rspringer): For-loop-ize this to make it more constexpr? Or can we
    // rely on the compiler to do it for us?
    while (bits_left) {
      uint8 byte = buffer_[source_byte];
      source_byte++;

      // Easy case: dest_bit == 0; this is an aligned full-byte write.
      if (buffer_offset_ == 0) {
        if (bits_left < kCharBit) {
          constexpr int kLeftover = (kElementBits % kCharBit);
          return_buffer[dest_byte] = byte & MakeMask<kLeftover>();
        } else {
          return_buffer[dest_byte] = byte;
        }
        dest_byte++;
        bits_left -= std::min(bits_left, kCharBit);
        continue;
      }

      // Hard case. Write a low and high chunk.
      // Write the low chunk.
      int64 num_low_bits = std::min(bits_left, buffer_offset_);
      uint8 low_bits = byte & Mask(num_low_bits);
      return_buffer[dest_byte] |= low_bits << (kCharBit - buffer_offset_);
      dest_byte++;
      bits_left -= num_low_bits;

      // And, if necessary, the high chunk in the next byte.
      if (bits_left) {
        int64 num_high_bits = std::min(bits_left, kCharBit - num_low_bits);
        uint8 high_bits = (byte >> buffer_offset_) & Mask(num_high_bits);
        return_buffer[dest_byte] = high_bits;
        bits_left -= num_high_bits;
      }
    }
  }

 private:
  uint8* buffer_;
  const int64 buffer_offset_;
};

// Specialization of ArrayView for packed elements, similar to PackedBitsView
// above.
template <typename ElementT, int64 kNumElements>
class PackedArrayView {
 public:
  PackedArrayView(uint8* buffer, int64 buffer_offset)
      : buffer_(buffer), buffer_offset_(buffer_offset) {}

  // Returns the element at the given index in the array.
  ElementT Get(int index) {
    assert(index < kNumElements);
    int64 bit_increment = index * ElementT::kBitCount + buffer_offset_;
    int64 byte_offset = bit_increment / kCharBit;
    int64 bit_offset = bit_increment % kCharBit;
    return ElementT(buffer_ + byte_offset, bit_offset);
  }

 private:
  uint8* buffer_;
  int64 buffer_offset_;
};

}  // namespace xls

#endif  // XLS_IR_VALUE_VIEW_H_

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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/types/span.h"
#include "xls/common/bits_util.h"
#include "xls/common/math_util.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

// ArrayView provides some array-type functionality on top of a flat character
// buffer.
template <typename ElementT, uint64_t kNumElements>
class ArrayView {
 public:
  explicit ArrayView(absl::Span<const uint8_t> buffer) : buffer_(buffer) {
    CHECK(buffer_.data() == nullptr);
    CHECK(buffer_.size() == GetTypeSize())
        << "Span isn't sized to this array's type!";
  }

  const uint8_t* buffer() { return buffer_.data(); }

  // Gets the storage size of this array.
  static constexpr uint64_t GetTypeSize() {
    return ElementT::GetTypeSize() * kNumElements;
  }

  // Gets the N'th element in the array.
  static ElementT Get(const uint8_t* buffer, int index) {
    return ElementT(buffer + (ElementT::GetTypeSize() * index));
  }

 private:
  absl::Span<const uint8_t> buffer_;
};

// Statically generates a value for masking off high bits in a value.
template <int kBitCount>
inline uint64_t MakeMask() {
  return (1ull << (kBitCount - 1)) | (MakeMask<kBitCount - 1>());
}

template <>
inline uint64_t MakeMask<0>() {
  return 0;
}

// BitsView provides some Bits-type functionality on top of a flat character
// buffer.
template <int64_t kNumBits>
class BitsView {
 public:
  BitsView() : buffer_(nullptr) { CHECK(buffer_ == nullptr); }
  explicit BitsView(const uint8_t* buffer) : buffer_(buffer) {}
  const uint8_t* buffer() { return buffer_; }

  // Determines the appropriate return type for this BitsView type.
  using ReturnT = std::conditional_t<
      (kNumBits > 128), std::array<uint8_t, (kNumBits - 1) / 8 + 1>,
      std::conditional_t<
          (kNumBits > 64), absl::uint128,
          std::conditional_t<
              (kNumBits > 32), uint64_t,
              std::conditional_t<
                  (kNumBits > 16), uint32_t,
                  std::conditional_t<
                      (kNumBits > 8), uint16_t,
                      std::conditional_t<(kNumBits > 1), uint8_t, bool>>>>>>;

  // Gets the storage size of this type.
  static constexpr uint64_t GetTypeSize() { return sizeof(ReturnT); }

  template <int N = kNumBits>
  std::enable_if_t<(N <= 128), ReturnT> GetValue() {
    return *reinterpret_cast<const ReturnT*>(buffer_) & MakeMask<kNumBits>();
  }

 private:
  const uint8_t* buffer_;
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
  TupleView() : buffer_(nullptr) { CHECK(buffer_ == nullptr); }
  explicit TupleView(const uint8_t* buffer) : buffer_(buffer) {}
  const uint8_t* buffer() { return buffer_; }

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
  static constexpr uint64_t GetTypeSize() {
    return FrontT::GetTypeSize() + GetTypeSize<NextT, Rest...>();
  }

  template <typename LastT>
  static constexpr uint64_t GetTypeSize() {
    return LastT::GetTypeSize();
  }

  static constexpr uint64_t GetTypeSize() { return GetTypeSize<Types...>(); }

  // ---- Element access.
  // Recursive case for element access. Simply walks down the type list.
  template <int kElementIndex, typename FrontT, typename... Rest>
  static constexpr uint64_t GetOffset(
      uint8_t offset,
      typename std::enable_if<(kElementIndex > 0)>::type* dummy = nullptr) {
    return GetOffset<kElementIndex - 1, Rest...>(offset +
                                                 FrontT::GetTypeSize());
  }

  // Base case - we've arrived at the element of interest.
  template <int kElementIndex, typename FrontT, typename... Rest>
  static constexpr uint64_t GetOffset(
      uint8_t offset,
      typename std::enable_if<(kElementIndex == 0)>::type* dummy = nullptr) {
    // If offset isn't aligned to our type size, then add padding.
    // TODO(rspringer): We need to require the buffer to be aligned, too.
    // TODO(rspringer): Aligned to min(our_type_size, 8)?
    return RoundUpToNearest<uint64_t>(offset, FrontT::GetTypeSize());
  }

  // ---- Element type access.
  // Metaprogramming horrors to get the type of the N'th element.
  // Recursive case - keep drilling down until the element of interest.
  template <int kElementIndex, typename FrontT, typename... Rest>
  struct element_accessor<kElementIndex, FrontT, Rest...>
      : element_accessor<kElementIndex - 1, Rest...> {};

  // Base case - we've arrived at the type of interest.
  template <typename FrontT, typename... Rest>
  struct element_accessor<0, FrontT, Rest...> {
    typedef FrontT type;
  };

 private:
  const uint8_t* buffer_;
};

// Mutable versions of the types above. As much as possible, these definitions
// depend on their parent types for their implementations - only operations
// touching "buffer_" need to be overridden.
template <typename ElementT, uint64_t kNumElements>
class MutableArrayView : public ArrayView<ElementT, kNumElements> {
 public:
  explicit MutableArrayView(absl::Span<uint8_t> buffer)
      : ArrayView<ElementT, kNumElements>(buffer) {}

  uint8_t* mutable_buffer() { return const_cast<uint8_t*>(this->buffer()); }

  // Gets the Nth element in the array.
  ElementT Get(int index) {
    return ElementT(mutable_buffer() + (ElementT::GetTypeSize() * index));
  }
};

template <uint64_t kNumBits>
class MutableBitsView : public BitsView<kNumBits> {
 public:
  explicit MutableBitsView(uint8_t* buffer) : BitsView<kNumBits>(buffer) {}

  uint8_t* mutable_buffer() { return const_cast<uint8_t*>(this->buffer()); }

  typename BitsView<kNumBits>::ReturnT GetValue() {
    return *reinterpret_cast<const typename BitsView<kNumBits>::ReturnT*>(
               mutable_buffer()) &
           MakeMask<kNumBits>();
  }

  void SetValue(typename BitsView<kNumBits>::ReturnT value) {
    *reinterpret_cast<typename BitsView<kNumBits>::ReturnT*>(mutable_buffer()) =
        value & MakeMask<kNumBits>();
  }
};

template <typename... Types>
class MutableTupleView : public TupleView<Types...> {
 public:
  explicit MutableTupleView(uint8_t* buffer) : TupleView<Types...>(buffer) {}

  uint8_t* mutable_buffer() { return const_cast<uint8_t*>(this->buffer()); }

  // Gets the N'th element in the tuple.
  template <int kElementIndex>
  typename TupleView<Types...>::template element_accessor<kElementIndex,
                                                          Types...>::type
  Get() {
    return typename TupleView<Types...>::
        template element_accessor<kElementIndex, Types...>::type(
            mutable_buffer() +
            TupleView<Types...>::template GetOffset<kElementIndex, Types...>(
                0));
  }

 private:
  uint8_t* mutable_buffer_;
};

// Specialization of BitsView for non-byte-aligned bit vectors inside a larger
// buffer, e.g., for a bit vector inside an enclosing PackedTuple.
// Template args:
//   kElementBits: the bit width of this element.
template <int64_t kElementBits>
class PackedBitsView {
 public:
  // buffer_offset is the number of bits into "buffer" at which the actual
  // element data begins. This value must be [0-7] (if >= 8, then "buffer"
  // should be incremented).
  PackedBitsView(uint8_t* buffer, int buffer_offset)
      : buffer_(buffer), buffer_offset_(buffer_offset) {
    DCHECK(buffer_offset >= 0 && buffer_offset <= 7);
  }

  // Returns the XLS IR Type corresponding to this packed view.
  static Type* GetFullType(Package* p) { return p->GetBitsType(kElementBits); }

  const uint8_t* buffer() { return buffer_; }
  uint8_t* mutable_buffer() { return buffer_; }

  static constexpr int64_t kBitCount = kElementBits;

  // Accessor: Populates the specified buffer with the data from this element,
  // shifting it if necessary. Even if kBufferOffset is non-0, return_buffer
  // will be written starting at bit 0.
  // Note: this call trusts that return_buffer is adequately sized (i.e., is
  // at least ceil((kNumBits + kBufferOffset) / kCharBit) bytes).
  // TODO(rspringer): Worth defining value-returning accessors for widths < 64b?
  void Get(uint8_t* return_buffer) {
    int64_t bits_left = kElementBits;
    // The byte at which the next read should occur. All reads, except for the
    // first, are byte-aligned.
    int64_t source_byte = 0;

    // The bit and byte at which the next write should occur.
    int64_t dest_byte = 0;

    // Do a write of the initial "overhanging" source bits to align future
    // reads.
    if (buffer_offset_ != 0) {
      // The initial source bits are buffer_offset_ bits into the first byte of
      // the buffer, so we have kCharBit - buffer_offset_ bits to read and
      // shift.
      // Since buffer_ is unsigned, it will be a logical shift - no mask is
      // needed.
      int64_t num_initial_bits = std::min(bits_left, kCharBit - buffer_offset_);
      uint8_t first_source_bits = buffer_[source_byte] >> buffer_offset_;
      return_buffer[dest_byte] = first_source_bits & Mask(num_initial_bits);

      bits_left -= num_initial_bits;
      source_byte++;
    }

    // TODO(rspringer): Optimize to use larger load types?
    // TODO(rspringer): For-loop-ize this to make it more constexpr? Or can we
    // rely on the compiler to do it for us?
    while (bits_left) {
      uint8_t byte = buffer_[source_byte];
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
      int64_t num_low_bits = std::min(bits_left, buffer_offset_);
      uint8_t low_bits = byte & Mask(num_low_bits);
      return_buffer[dest_byte] |= low_bits << (kCharBit - buffer_offset_);
      dest_byte++;
      bits_left -= num_low_bits;

      // And, if necessary, the high chunk in the next byte.
      if (bits_left) {
        int64_t num_high_bits = std::min(bits_left, kCharBit - num_low_bits);
        uint8_t high_bits = (byte >> buffer_offset_) & Mask(num_high_bits);
        return_buffer[dest_byte] = high_bits;
        bits_left -= num_high_bits;
      }
    }
  }

 private:
  uint8_t* buffer_;
  const int64_t buffer_offset_;
};

// Specialization of ArrayView for packed elements, similar to PackedBitsView
// above.
template <typename ElementT, int64_t kNumElements>
class PackedArrayView {
 public:
  static constexpr int64_t kBitCount = ElementT::kBitCount * kNumElements;

  // *Somewhat* magical converter, which will grab the guts of an array with an
  // appropriate width integral type and turn it into this packed array view
  // type. This allows us to pass std::arrays to APIs which take fixed size
  // arrays of that bit count with more type safety than casting to a `void*`.
  //
  // For example:
  //    absl::Status PopulateArray(std::array<int32_t, 64>* a) {
  //      return populate_array_jit->Run(
  //        PackedArrayView<PackedBitsView<32>, 64>(a));
  //    }
  template <typename ArrayElementT,
            typename =
                std::enable_if_t<std::is_integral<ArrayElementT>::value &&
                                 std::numeric_limits<ArrayElementT>::digits +
                                         std::is_signed<ArrayElementT>::value ==
                                     ElementT::kBitCount>>
  explicit PackedArrayView(std::array<ArrayElementT, kNumElements>* array)
      : PackedArrayView(absl::bit_cast<uint8_t*>(array->data()), 0) {}

  PackedArrayView(uint8_t* buffer, int64_t buffer_offset)
      : buffer_(buffer), buffer_offset_(buffer_offset) {}

  const uint8_t* buffer() { return buffer_; }
  uint8_t* mutable_buffer() { return buffer_; }

  // Returns the XLS IR Type corresponding to this packed view.
  static Type* GetFullType(Package* p) {
    return p->GetArrayType(kNumElements, ElementT::GetFullType(p));
  }

  // Returns the element at the given index in the array.
  ElementT Get(int index) {
    DCHECK_LT(index, kNumElements);
    int64_t bit_increment = index * ElementT::kBitCount + buffer_offset_;
    int64_t byte_offset = bit_increment / kCharBit;
    int64_t bit_offset = bit_increment % kCharBit;
    return ElementT(buffer_ + byte_offset, bit_offset);
  }

 private:
  uint8_t* buffer_;
  int64_t buffer_offset_;
};

// Specialization of TupleView for packed elements, similar to PackedArrayView
// above.
template <typename... Types>
class PackedTupleView {
 public:
  // buffer_offset is the number of bits into "buffer" at which the actual
  // element data begins. This value must be [0-7] (if >= 8, then "buffer"
  // should be incremented).
  PackedTupleView(uint8_t* buffer, int64_t buffer_offset)
      : buffer_(buffer), buffer_offset_(buffer_offset) {}

  const uint8_t* buffer() { return buffer_; }
  uint8_t* mutable_buffer() { return buffer_; }

  // Recursive templates - determine the size (in bits) of this packed tuple.
  template <typename FrontT, typename... ElemTypes>
  struct TupleBitSizer {
    static constexpr int64_t value =
        FrontT::kBitCount + TupleBitSizer<ElemTypes...>::value;
  };

  // Base case!
  template <typename LastT>
  struct TupleBitSizer<LastT> {
    static constexpr int64_t value = LastT::kBitCount;
  };

  static constexpr int64_t kBitCount = TupleBitSizer<Types...>::value;

  // Helper template routines to get the XLS IR Types for this tuple's elements.
  template <typename FrontT, typename... RestT>
  static void GetFullTypes(
      Package* p, std::vector<Type*>& types,
      typename std::enable_if<(sizeof...(RestT) != 0)>::type* dummy = nullptr) {
    types.push_back(FrontT::GetFullType(p));
    GetFullTypes<RestT...>(p, types);
  }

  template <typename FrontT, typename... RestT>
  static void GetFullTypes(
      Package* p, std::vector<Type*>& types,
      typename std::enable_if<(sizeof...(RestT) == 0)>::type* dummy = nullptr) {
    types.push_back(FrontT::GetFullType(p));
  }

  // Returns the XLS IR Type corresponding to this packed view.
  static TupleType* GetFullType(Package* p) {
    std::vector<Type*> types;
    GetFullTypes<Types...>(p, types);
    return p->GetTupleType(types);
  }

 private:
  // Forward declaration of the element-type-accessing template. The definition
  // is way below for readability.
  template <int kElementIndex, typename... Rest>
  struct ElementAccessor;

 public:
  // Gets the N'th element in the tuple.
  template <int kElementIndex>
  typename ElementAccessor<kElementIndex, Types...>::type Get() {
    static_assert(kElementIndex < sizeof...(Types));

    constexpr int64_t kStartBitOffset =
        GetStartBitOffset<0, kElementIndex, Types...>(0);
    return typename ElementAccessor<kElementIndex, Types...>::type(
        buffer_ + (kStartBitOffset / kCharBit),
        ((buffer_offset_ + kStartBitOffset) % kCharBit));
  }

 private:
  uint8_t* buffer_;
  int64_t buffer_offset_;

  // ---- Element offset calculation.
  // Tuples are declared MSb to LSb, so we need to extract types in "reverse"
  // order to find the N'th element. Since we "peel" types off the front of the
  // template parameter pack, that means we start counting bit widths once we
  // hit the element of interest.
  template <int kCurrentIndex, int kTargetIndex, typename FrontT,
            typename NextT, typename... Rest>
  static constexpr int64_t GetStartBitOffset(
      int64_t offset,
      typename std::enable_if<(kCurrentIndex != sizeof...(Types) - 1)>::type*
          dummy = nullptr) {
    if (kCurrentIndex < kTargetIndex) {
      return GetStartBitOffset<kCurrentIndex + 1, kTargetIndex, NextT, Rest...>(
          offset);
    }
    return NextT::kBitCount +
           GetStartBitOffset<kCurrentIndex + 1, kTargetIndex, NextT, Rest...>(
               offset);
  }

  template <int kCurrentIndex, int kTargetIndex, typename FrontT,
            typename... Rest>
  static constexpr int64_t GetStartBitOffset(
      int64_t offset,
      typename std::enable_if<(kCurrentIndex == sizeof...(Types) - 1)>::type*
          dummy = nullptr) {
    return offset;
  }

  // ---- Element type access.
  // Metaprogramming horrors to get the type of the N'th element.
  // Recursive case - keep drilling down until the element of interst.
  template <int kElementIndex, typename FrontT, typename... Rest>
  struct ElementAccessor<kElementIndex, FrontT, Rest...>
      : ElementAccessor<kElementIndex - 1, Rest...> {};

  // Base case - we've arrived at the type of interest.
  template <typename FrontT, typename... Rest>
  struct ElementAccessor<0, FrontT, Rest...> {
    typedef FrontT type;
  };
};

// Some common view types.
using PackedFloat = PackedTupleView</* sign */ PackedBitsView<1>,
                                    /* exponent */ PackedBitsView<8>,
                                    /* mantissa */ PackedBitsView<23>>;
using PackedDouble = PackedTupleView</* sign */ PackedBitsView<1>,
                                     /* exponent */ PackedBitsView<11>,
                                     /* mantissa */ PackedBitsView<52>>;

}  // namespace xls

#endif  // XLS_IR_VALUE_VIEW_H_

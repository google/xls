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
    XLS_DCHECK(buffer_.size() == GetTypeSize())
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
  return (1 << kBitCount) | (MakeMask<kBitCount - 1>());
}

template <>
inline uint64 MakeMask<0>() {
  return 1;
}

// BitsView provides some Bits-type functionality on top of a flat character
// buffer.
template <uint64 kNumBits>
class BitsView {
 public:
  explicit BitsView(const uint8* buffer) : buffer_(buffer) {}

  // Gets the storage size of this type.
  static constexpr uint64 GetTypeSize() {
    constexpr uint64 kCharBit = 8;

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
    return *reinterpret_cast<const ReturnT*>(buffer_) &
           MakeMask<kNumBits - 1>();
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
  template <typename FrontT, typename... Rest>
  static constexpr uint64 GetTypeSize() {
    return FrontT::GetTypeSize() + GetTypeSize<Rest...>();
  }

  template <typename LastT>
  static constexpr uint64 GetTypeSize() {
    return LastT::GetTypeSize();
  }

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

}  // namespace xls

#endif  // XLS_IR_VALUE_VIEW_H_

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

#ifndef XLS_DATA_STRUCTURES_INLINE_BITMAP_H_
#define XLS_DATA_STRUCTURES_INLINE_BITMAP_H_

#include <algorithm>
#include <compare>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/bits_util.h"
#include "xls/common/endian.h"
#include "xls/common/math_util.h"
#include "xls/common/test_macros.h"

namespace xls {

class BitmapView;

// A bitmap that has 64-bits of inline storage by default.
class InlineBitmap {
 public:
  // How many bits are held in one word.
  static constexpr int64_t kWordBits = 64;
  // How many bytes are held in one word.
  static constexpr int64_t kWordBytes = 8;
  // Constructs an InlineBitmap of width `bit_count` using the bits in
  // `word`. If `bit_count` is greater than 64, then all high bits are set to
  // `fill`.
  static InlineBitmap FromWord(uint64_t word, int64_t bit_count,
                               bool fill = false) {
    InlineBitmap result(bit_count, fill);
    if (bit_count != 0) {
      result.data_[0] = word & result.MaskForWord(0);
    }
    return result;
  }

  // Constructs a bitmap of width `bit_count` using the bytes (in little-endian
  // layout). Bytes be of size at least `ceil(bit_count / 8)`.
  //
  // Implementation note: this byte mapping scheme works because, when we memcpy
  // bytes into the array of uint64_t words, any "trailing bytes" end up in the
  // "low byte" positions of the last word, and then our mask applies to that
  // last word. E.g. for 65 bits:
  //
  //     /--- word 0 least signifiant byte
  //     v
  //    {b0, b1, b2, b3, ..., b7, b8}
  //     ^----- word 0 data ---^  ^- word 1 data (partial)
  //                              ^
  //                               \- 0b0000_0001 == last word mask value 0x1
  static InlineBitmap FromBytes(int64_t bit_count,
                                absl::Span<const uint8_t> bytes) {
    InlineBitmap result(bit_count, false);
    int64_t byte_count = CeilOfRatio(bit_count, int64_t{8});
    CHECK_GE(bytes.size(), byte_count) << "bit_count: " << bit_count;
    // memcpy() requires valid pointers even when the number of bytes copied is
    // zero, and an empty absl::Span's data() pointer may not be valid. Guard
    // the memcpy with a check that the span is not empty.
    if (!result.data_.empty()) {
      std::memcpy(result.data_.data(), bytes.data(), byte_count);
    }
    result.MaskLastWord();
    return result;
  }

  // Constructs a bitmap of width `bits.size()` using the given bits,
  // interpreting index 0 as the *least* significant bit.
  //
  // Note: if you find you want an overload that accepts a `std::vector<bool>`,
  // consider using an `absl::InlinedVector<bool, N>` as storage instead, as it
  // can be converted to span.
  static InlineBitmap FromBitsLsbIs0(absl::Span<bool const> bits) {
    InlineBitmap result(bits.size(), /*fill=*/false);
    int64_t bit_idx = 0;
    uint64_t* word = result.data_.data();
    for (bool bit : bits) {
      *word |= static_cast<uint64_t>(bit) << bit_idx;
      if (++bit_idx >= kWordBits) {
        ++word;
        bit_idx = 0;
      }
    }
    return result;
  }

  // As above, but index 0 of the span is the most significant bit.
  static InlineBitmap FromBitsMsbIs0(absl::Span<bool const> bits) {
    InlineBitmap result(bits.size(), /*fill=*/false);
    for (int64_t i = 0; i < bits.size(); ++i) {
      result.Set(bits.size() - i - 1, bits[i]);
    }
    return result;
  }

  explicit InlineBitmap(int64_t bit_count, bool fill = false)
      : bit_count_(bit_count),
        data_(CeilOfRatio(bit_count, kWordBits),
              fill ? ~uint64_t{0} : uint64_t{0}) {
    DCHECK_GE(bit_count, 0);
    // If we initialized our data to zero, no need to mask out the bits past the
    // end of the bitmap; they're already zero.
    if (fill) {
      MaskLastWord();
    }
  }

  // Create a new InlineBitmap with the same content but 'new_size' bits. Bits
  // beyond what this bitmap holds are 'new_data'.
  InlineBitmap&& WithSize(int64_t new_size, bool new_data = false) && {
    int64_t new_word_size = CeilOfRatio(new_size, kWordBits);
    if (new_size > bit_count_) {
      int64_t old_wordcount = word_count();
      int64_t last_wordno = old_wordcount - 1;
      uint64_t mask = MaskForWord(last_wordno);
      if (new_word_size > old_wordcount) {
        data_.resize(new_word_size, new_data ? ~uint64_t{0} : uint64_t{0});
      }
      if (bit_count_ != 0) {
        if (new_data) {
          data_[last_wordno] |= ~mask;
        } else {
          data_[last_wordno] &= mask;
        }
      }
      bit_count_ = new_size;
    } else {
      data_.resize(new_word_size);
      bit_count_ = new_size;
    }
    MaskLastWord();
    return std::move(*this);
  }
  // Create a new InlineBitmap with the same content but 'new_size' bits. Bits
  // beyond what this bitmap holds are 'new_data'.
  InlineBitmap WithSize(int64_t new_size, bool new_data = false) const& {
    // TODO(allight): We should be able to do this without copying the whole
    // *this in some cases.
    InlineBitmap bm = *this;
    return std::move(bm).WithSize(new_size, new_data);
  }

  bool operator==(const InlineBitmap& other) const {
    if (bit_count_ != other.bit_count_) {
      return false;
    }
    for (int64_t wordno = 0; wordno < word_count(); ++wordno) {
      if (data_[wordno] != other.data_[wordno]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const InlineBitmap& other) const { return !(*this == other); }

  int64_t bit_count() const { return bit_count_; }

  bool empty() const { return bit_count_ == 0; }

  bool IsAllOnes() const {
    for (int64_t wordno = 0; wordno < word_count(); ++wordno) {
      if (data_[wordno] != MaskForWord(wordno)) {
        return false;
      }
    }
    return true;
  }
  bool IsAllZeroes() const {
    for (int64_t wordno = 0; wordno < word_count(); ++wordno) {
      if (data_[wordno] != 0) {
        return false;
      }
    }
    return true;
  }
  inline bool Get(int64_t index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, bit_count());
    uint64_t word = data_[index / kWordBits];
    uint64_t bitno = index % kWordBits;
    return (word >> bitno) & 1ULL;
  }
  inline void Set(int64_t index, bool value = true) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, bit_count());
    uint64_t& word = data_[index / kWordBits];
    uint64_t bitno = index % kWordBits;
    if (value) {
      word |= 1ULL << bitno;
    } else {
      word &= ~(1ULL << bitno);
    }
  }
  // Sets the values of a range. The range is defined as:
  // [lower_index, upper_index).
  inline void SetRange(int64_t lower_index, int64_t upper_index,
                       bool value = true) {
    DCHECK_GE(lower_index, 0);
    DCHECK_LE(upper_index, bit_count());
    for (int64_t index = lower_index; index < upper_index; ++index) {
      Set(index, value);
    }
  }
  // Sets all the values of the bitmap to false.
  inline void SetAllBitsToFalse() {
    std::fill(data_.begin(), data_.end(), 0ULL);
  }

  // Fast path for users of the InlineBitmap to get at the 64-bit word that
  // backs a group of 64 bits.
  uint64_t GetWord(int64_t wordno) const {
    if (wordno == 0 && word_count() == 0) {
      return 0;
    }
    DCHECK_LT(wordno, word_count());
    return data_[wordno];
  }
  void SetWord(int64_t wordno, uint64_t value) {
    DCHECK_LT(wordno, word_count());
    data_[wordno] = value & MaskForWord(wordno);
  }

  // Sets a byte in the data underlying the bitmap.
  //
  // Setting byte i as {b_7, b_6, b_5, ..., b_0} sets the bit at i*8 to b_0, the
  // bit at i*8+1 to b_1, and so on.
  //
  // Note: the byte-to-word mapping is defined as little endian; i.e. if the
  // bytes are set via SetByte() and then the words are observed via GetWord(),
  // the user will observe that byte 0 is mapped to the least significant bits
  // of word 0, byte 7 is mapped to the most significant bits of word 0, byte 8
  // is mapped to the least significant bits of word 1, and so on.
  void SetByte(int64_t byteno, uint8_t value) {
    DCHECK_LT(byteno, byte_count());
    CHECK(kEndianness == Endianness::kLittleEndian);
    absl::bit_cast<uint8_t*>(data_.data())[byteno] = value;
    // Ensure the data is appropriately masked in case this byte writes to that
    // region of bits.
    MaskLastWord();
  }

  // Returns the byte at the given offset. Byte order is little-endian.
  uint8_t GetByte(int64_t byteno) const {
    DCHECK_LT(byteno, byte_count());
    CHECK(kEndianness == Endianness::kLittleEndian);
    return absl::bit_cast<uint8_t*>(data_.data())[byteno];
  }

  // Writes the underlying bytes of the inline bit map to the given buffer. Byte
  // order is little-endian. Writes out Ceil(bit_count_ / 8) number of bytes.
  void WriteBytesToBuffer(absl::Span<uint8_t> bytes) const {
    CHECK(kEndianness == Endianness::kLittleEndian);
    // memcpy() requires valid pointers even when the number of bytes copied is
    // zero, and an empty absl::Span's data() pointer may not be valid. Guard
    // the memcpy with a check that the span is not empty.
    if (!bytes.empty()) {
      std::memcpy(bytes.data(), data_.data(),
                  CeilOfRatio(bit_count_, int64_t{8}));
    }
  }

  // Compares against another InlineBitmap as if they were unsigned
  // two's complement integers. If equal, returns 0. If this is greater than
  // other, returns 1. If this is less than other, returns -1.
  int64_t UCmp(const InlineBitmap& other) const {
    // If this InlineBitmap is longer than other, check if any of the excess
    // bits are set; if so, this is bigger.
    int64_t my_word_idx = word_count() - 1;
    int64_t other_word_idx = other.word_count() - 1;
    for (; my_word_idx > other_word_idx; --my_word_idx) {
      if (GetWord(my_word_idx) > 0) {
        return 1;
      }
    }
    // Do the same if other is longer than this.
    for (; other_word_idx > my_word_idx; --other_word_idx) {
      if (other.GetWord(other_word_idx) > 0) {
        return -1;
      }
    }

    // Compare word-by-word; due to masking on creation, this is guaranteed to
    // be accurate and should be much faster than going bit-by-bit.
    for (int64_t wordno = my_word_idx; wordno >= 0; --wordno) {
      const uint64_t my_word = GetWord(wordno);
      const uint64_t other_word = other.GetWord(wordno);
      if (my_word > other_word) {
        return 1;
      }
      if (my_word < other_word) {
        return -1;
      }
    }
    return 0;
  }

  // Sets this bitmap to the union (bitwise 'or') of this bitmap and `other`.
  void Union(const InlineBitmap& other) {
    CHECK_EQ(bit_count(), other.bit_count());
    for (int64_t i = 0; i < data_.size(); ++i) {
      data_[i] |= other.data_[i];
    }
  }

  // Sets this bitmap to the bitwise 'and' of this bitmap and `other`.
  void Intersect(const InlineBitmap& other) {
    CHECK_EQ(bit_count(), other.bit_count());
    for (int64_t i = 0; i < data_.size(); ++i) {
      data_[i] &= other.data_[i];
    }
  }

  int64_t byte_count() const { return CeilOfRatio(bit_count_, int64_t{8}); }
  int64_t word_count() const { return data_.size(); }

  template <typename H>
  friend H AbslHashValue(H h, const InlineBitmap& ib) {
    return H::combine(std::move(h), ib.bit_count_, ib.data_);
  }

  // Overwrite 'cnt' bits starting at w_offset with the bits in 'other' starting
  // at 'r_offset'. Returns an error code if either of 'w_offset + cnt' or
  // 'r_offset + cnt' is out of bounds.
  void Overwrite(const InlineBitmap& other, int64_t cnt, int64_t w_offset = 0,
                 int64_t r_offset = 0);

 private:
  XLS_FRIEND_TEST(InlineBitmapTest, MaskForWord);
  friend uint64_t GetWordBitsAtForTest(const InlineBitmap& ib,
                                       int64_t bit_offset);

  // Gets the kWordBits bits following bit_offset with 'Get(bit_offset)' being
  // the LSB, Get(bit_offset + 1) being the next lsb etc.
  int64_t GetWordBitsAt(int64_t bit_offset) const;

  void MaskLastWord() {
    if (word_count() == 0) {
      return;
    }
    int64_t last_wordno = word_count() - 1;
    uint64_t mask = MaskForWord(last_wordno);
    data_[last_wordno] &= mask;
  }

  // Creates a mask for the valid bits in word "wordno".
  uint64_t MaskForWord(int64_t wordno) const {
    int64_t remainder = bit_count_ % kWordBits;
    return ((wordno < word_count() - 1) || remainder == 0) ? Mask(kWordBits)
                                                           : Mask(remainder);
  }

  int64_t bit_count_;
  absl::InlinedVector<uint64_t, 1> data_;

 public:
  class Iterator {
   public:
    using difference_type = int64_t;
    using value_type = bool;
    using reference = bool;
    using iterator_category = std::random_access_iterator_tag;

    bool operator*() const { return ((*word_it_) >> bit_index_) & int64_t{1}; }
    bool operator[](int64_t index) const { return *(*this + index); }

    Iterator& operator+=(int64_t n) {
      bit_index_ += n;
      int64_t word_delta = FloorOfRatio(bit_index_, kWordBits);
      if (word_delta != 0) {
        word_it_ += word_delta;
        bit_index_ -= word_delta * kWordBits;
        DCHECK_GE(bit_index_, 0);
        DCHECK_LT(bit_index_, kWordBits);
      }
      return *this;
    }
    Iterator& operator++() { return *this += 1; }
    Iterator operator++(int) {
      Iterator result = *this;
      ++*this;
      return result;
    }

    friend Iterator operator+(Iterator it, int64_t n) {
      Iterator result = it;
      result += n;
      return result;
    }
    friend Iterator operator+(int64_t n, Iterator it) { return it + n; }

    Iterator& operator-=(int64_t n) { return *this += -n; }
    Iterator& operator--() { return *this += -1; }
    Iterator operator--(int) {
      Iterator result = *this;
      --*this;
      return result;
    }

    friend Iterator operator-(Iterator it, int64_t n) {
      Iterator result = it;
      result -= n;
      return result;
    }

    friend int64_t operator-(Iterator b, Iterator a) {
      return (b.word_it_ - a.word_it_) * kWordBits +
             (b.bit_index_ - a.bit_index_);
    }

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.word_it_ == b.word_it_ && a.bit_index_ == b.bit_index_;
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return !(a == b);
    }

    friend std::strong_ordering operator<=>(const Iterator& a,
                                            const Iterator& b) {
      if (a.word_it_ != b.word_it_) {
        return a.word_it_ <=> b.word_it_;
      }
      return a.bit_index_ <=> b.bit_index_;
    }

   private:
    friend class InlineBitmap;

    Iterator(decltype(InlineBitmap::data_)::const_iterator word_it,
             int64_t bit_index)
        : word_it_(word_it), bit_index_(bit_index) {}

    decltype(InlineBitmap::data_)::const_iterator word_it_;
    int64_t bit_index_;
  };

  Iterator begin() const { return Iterator(data_.begin(), 0); }
  Iterator end() const {
    if (bit_count_ % kWordBits == 0) {
      return Iterator(data_.end(), 0);
    }
    return Iterator(data_.end() - 1, bit_count_ % kWordBits);
  }

  Iterator cbegin() const { return Iterator(data_.cbegin(), 0); }
  Iterator cend() const {
    if (bit_count_ % kWordBits == 0) {
      return Iterator(data_.cend(), 0);
    }
    return Iterator(data_.cend() - 1, bit_count_ % kWordBits);
  }
};

class BitmapView {
 public:
  explicit BitmapView(const InlineBitmap& bitmap,
                      std::optional<int64_t> start_bit = std::nullopt,
                      std::optional<int64_t> bit_count = std::nullopt)
      : bitmap_(bitmap) {
    start_bit_ = start_bit.value_or(0);
    CHECK_LE(start_bit_, bitmap_.bit_count());
    bit_count_ = bit_count.value_or(bitmap_.bit_count() - start_bit_);
  }

  bool Get(int64_t bit_index) const {
    CHECK_LT(bit_index, bit_count_);
    return bitmap_.Get(start_bit_ + bit_index);
  }

  BitmapView Slice(int64_t start_bit, int64_t bit_count) const {
    return BitmapView(bitmap_, start_bit_ + start_bit, bit_count);
  }

  InlineBitmap ToBitmap() const {
    InlineBitmap result(bit_count_, false);
    result.Overwrite(bitmap_, bit_count_, /*w_offset=*/0,
                     /*r_offset=*/start_bit_);
    return result;
  }

  int64_t bit_count() const { return bit_count_; }

 private:
  const InlineBitmap& bitmap_;
  int64_t start_bit_;
  int64_t bit_count_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_INLINE_BITMAP_H_

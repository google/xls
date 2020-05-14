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

#ifndef THIRD_PARTY_XLS_DATA_STRUCTURES_INLINE_BITMAP_H_
#define THIRD_PARTY_XLS_DATA_STRUCTURES_INLINE_BITMAP_H_

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "xls/common/bits_util.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"

namespace xls {

// A bitmap that has 64-bits of inline storage by default.
class InlineBitmap {
 public:
  static InlineBitmap FromWord(uint64 word, int64 bit_count, bool fill) {
    InlineBitmap result(bit_count, fill);
    if (bit_count != 0) {
      result.data_[0] = word & result.MaskForWord(0);
    }
    return result;
  }

  explicit InlineBitmap(int64 bit_count, bool fill = false)
      : bit_count_(bit_count),
        data_(CeilOfRatio(bit_count, kWordBits),
              fill ? -1ULL : 0ULL) {
    XLS_DCHECK_GE(bit_count, 0);
  }

  bool operator==(const InlineBitmap& other) const {
    if (bit_count_ != other.bit_count_) {
      return false;
    }
    for (int64 wordno = 0; wordno < word_count(); ++wordno) {
      uint64 mask = MaskForWord(wordno);
      uint64 lhs = (data_[wordno] & mask);
      uint64 rhs = (other.data_[wordno] & mask);
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const InlineBitmap& other) const { return !(*this == other); }

  int64 bit_count() const { return bit_count_; }
  bool IsAllOnes() const {
    for (int64 wordno = 0; wordno < word_count(); ++wordno) {
      uint64 mask = MaskForWord(wordno);
      if ((data_[wordno] & mask) != mask) {
        return false;
      }
    }
    return true;
  }
  bool IsAllZeroes() const {
    for (int64 wordno = 0; wordno < word_count(); ++wordno) {
      uint64 mask = MaskForWord(wordno);
      if ((data_[wordno] & mask) != 0) {
        return false;
      }
    }
    return true;
  }
  bool Get(int64 index) const {
    XLS_DCHECK_GE(index, 0);
    XLS_DCHECK_LT(index, bit_count());
    uint64 word = data_[index / kWordBits];
    uint64 bitno = index % kWordBits;
    return (word >> bitno) & 1ULL;
  }
  void Set(int64 index, bool value) {
    XLS_DCHECK_GE(index, 0);
    XLS_DCHECK_LT(index, bit_count());
    uint64& word = data_[index / kWordBits];
    uint64 bitno = index % kWordBits;
    if (value) {
      word |= 1ULL << bitno;
    } else {
      word &= ~(1ULL << bitno);
    }
  }

  // Fast path for users of the InlineBitmap to get at the 64-bit word that
  // backs a group of 64 bits.
  uint64 GetWord(int64 wordno) const {
    if (wordno == 0 && word_count() == 0) {
      return 0;
    }
    XLS_DCHECK_LT(wordno, word_count());
    return data_[wordno];
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
  void SetByte(int64 byteno, uint8 value) {
    XLS_DCHECK_LT(byteno, byte_count());
    // Implementation note: this relies on the endianness of the machine.
    absl::bit_cast<uint8*>(data_.data())[byteno] = value;
    // Ensure the data is appropriately masked in case this byte writes to that
    // region of bits.
    MaskLastWord();
  }

  uint8 GetByte(int64 byteno) const {
    XLS_DCHECK_LT(byteno, byte_count());
    // Implementation note: this relies on the endianness of the machine.
    return absl::bit_cast<uint8*>(data_.data())[byteno];
  }

  int64 byte_count() const { return CeilOfRatio(bit_count_, int64{8}); }

 private:
  static constexpr int64 kWordBits = 64;
  static constexpr int64 kWordBytes = 8;
  int64 word_count() const { return data_.size(); }

  void MaskLastWord() {
    int64 last_wordno = word_count() - 1;
    data_[last_wordno] &= MaskForWord(last_wordno);
  }

  // Creates a mask for the valid bits in word "wordno".
  uint64 MaskForWord(int64 wordno) const {
    int64 remainder = bit_count_ % kWordBits;
    return ((wordno < word_count() - 1) || remainder == 0) ? Mask(kWordBits)
                                                           : Mask(remainder);
  }

  int64 bit_count_;
  absl::InlinedVector<uint64, 1> data_;
};

}  // namespace xls

#endif  // THIRD_PARTY_XLS_DATA_STRUCTURES_INLINE_BITMAP_H_

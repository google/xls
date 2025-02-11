// Copyright 2025 The XLS Authors
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

#include "xls/data_structures/inline_bitmap.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/check.h"
#include "xls/common/bits_util.h"

namespace xls {

namespace {}  // namespace

void InlineBitmap::Overwrite(const InlineBitmap& other, int64_t cnt,
                             int64_t w_offset, int64_t r_offset) {
  CHECK_GE(cnt, 0) << "negative cnt";
  if (cnt == 0) {
    return;
  }
  CHECK_LE(w_offset + cnt, bit_count()) << "out of bounds write";
  CHECK_LE(r_offset + cnt, other.bit_count()) << "out of bounds read";
  CHECK(static_cast<const void*>(this) != static_cast<const void*>(&other))
      << "Memmove not supported.";
  int64_t word_no = w_offset / kWordBits;
  // Copy the first word and align writing to word boundary.
  if (w_offset % kWordBits != 0) {
    int64_t w_bit_offset = w_offset % kWordBits;
    uint64_t cur_word = other.GetWordBitsAt(r_offset);
    uint64_t low_bits = GetWord(word_no) & Mask(w_bit_offset);
    uint64_t high_bits = (cnt + w_bit_offset) >= kWordBits
                             ? 0
                             : GetWord(word_no) & (~Mask(w_bit_offset + cnt));
    int64_t written = std::min(cnt, kWordBits - w_bit_offset);
    SetWord(word_no, low_bits | high_bits |
                         ((cur_word & Mask(written)) << w_bit_offset));
    word_no++;
    w_offset += written;
    r_offset += written;
    cnt -= written;
  }
  // Handle all the intermediate words
  for (; cnt - kWordBits >= 0;
       cnt -= kWordBits, word_no++, r_offset += kWordBits) {
    SetWord(word_no, other.GetWordBitsAt(r_offset));
  }

  // Handle the remaining bits.
  if (cnt > 0) {
    uint64_t existing_word_high = GetWord(word_no) & (~Mask(cnt));
    SetWord(word_no,
            (other.GetWordBitsAt(r_offset) & Mask(cnt)) | existing_word_high);
  }
}

int64_t InlineBitmap::GetWordBitsAt(int64_t bit_offset) const {
  int64_t bits_off = bit_offset % kWordBits;
  int64_t start_word_num = bit_offset / kWordBits;
  if (bits_off == 0) {
    return GetWord(start_word_num);
  }
  // NB Uint to get zero-extend
  uint64_t start_word = GetWord(start_word_num);
  uint64_t low_bits = start_word >> bits_off;
  if (start_word_num + 1 >= word_count()) {
    // Cycles into an unset word so just assume zeros.
    return low_bits;
  }
  uint64_t high_word = GetWord(start_word_num + 1);
  uint64_t high_bits = high_word << (kWordBits - bits_off);
  return high_bits | low_bits;
}

}  // namespace xls

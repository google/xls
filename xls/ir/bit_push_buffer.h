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

#ifndef XLS_IR_BIT_PUSH_BUFFER_H_
#define XLS_IR_BIT_PUSH_BUFFER_H_

#include <cstdint>
#include <vector>

#include "xls/data_structures/inline_bitmap.h"

namespace xls {

// Mutable structure for pushing bits into; e.g. in the serializing/flattening
// process where we're about to push values over the wire.
class BitPushBuffer {
 public:
  // Pushes a bit into the buffer -- see GetUint8Data() comment below on the
  // ordering with which these pushed bits are returned in the byte sequence.
  void PushBit(bool bit) { bitmap_.push_back(bit); }

  InlineBitmap ToBitmap() const {
    return InlineBitmap::FromBitsMsbIs0(bitmap_);
  }

  // Retrieves the pushed bits as a sequence of bytes.
  //
  // The first-pushed bit goes into the MSb of the 0th byte.
  //
  // i.e. if we just push a single `1` bit, the byte that comes out of this
  // function is `0x80`.
  //
  // The final byte, if it is partial, will have padding zeroes in the least
  // significant bits, as shown above.
  std::vector<uint8_t> GetUint8DataWithLsbPadding() const;

  // As above, but the zero-padding is placed in the high bits of the first
  // byte.
  //
  // i.e. if we just push a single `1` bit, the byte that comes out of this
  // function is `0x01`.
  //
  // The first byte, if it is partial, will have padding zeroes in the most
  // significant bits, as shown above.
  std::vector<uint8_t> GetUint8DataWithMsbPadding() const;

  bool empty() const { return bitmap_.empty(); }

  // Returns the number of bytes required to store the currently-pushed bits.
  int64_t size_in_bytes() const { return CeilOfRatio(bitmap_.size(), 8UL); }

  // Returns the number of bits currently in the buffer.
  int64_t size_in_bits() const { return bitmap_.size(); }

  // Returns a binary string representation of the currently-pushed bits; e.g.:
  // ```c++
  // BitPushBuffer buffer;
  // buffer.PushBit(true);
  // buffer.PushBit(false);
  // buffer.ToString() == "0b10"
  // ```
  std::string ToString() const;

 private:
  absl::InlinedVector<bool, 64> bitmap_;
};

}  // namespace xls

#endif  // XLS_IR_BIT_PUSH_BUFFER_H_

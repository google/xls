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

#include "xls/common/math_util.h"

namespace xls {

// Mutable structure for pushing bits into; e.g. in the serializing/flattening
// process where we're about to push values over the wire.
class BitPushBuffer {
 public:
  // Pushes a bit into the buffer -- see GetUint8Data() comment below on the
  // ordering with which these pushed bits are returned in the byte sequence.
  void PushBit(bool bit) { bitmap_.push_back(bit); }

  // Retrieves the pushed bits as a sequence of bytes.
  //
  // The first-pushed bit goes into the MSb of the 0th byte. Concordantly, the
  // final byte, if it is partial, will have padding zeroes in the least
  // significant bits.
  std::vector<uint8_t> GetUint8Data() const {
    // Implementation note: bitmap does not expose its underlying storage.
    std::vector<uint8_t> result;
    result.resize(CeilOfRatio(bitmap_.size(), 8UL), 0);
    for (int64_t i = 0; i < static_cast<int64_t>(bitmap_.size()); ++i) {
      result[i / 8] |= bitmap_[i] << (7 - i % 8);
    }
    return result;
  }

  bool empty() const { return bitmap_.empty(); }

  // Returns the number of bytes required to store the currently-pushed bits.
  int64_t size_in_bytes() const { return CeilOfRatio(bitmap_.size(), 8UL); }

 private:
  std::vector<bool> bitmap_;
};

}  // namespace xls

#endif  // XLS_IR_BIT_PUSH_BUFFER_H_

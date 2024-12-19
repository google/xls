// Copyright 2024 The XLS Authors
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

#include "xls/ir/bit_push_buffer.h"

#include "xls/common/math_util.h"

namespace xls {

std::vector<uint8_t> BitPushBuffer::GetUint8DataWithLsbPadding() const {
  // Implementation note: bitmap does not expose its underlying storage.
  std::vector<uint8_t> result;
  result.resize(CeilOfRatio(bitmap_.size(), 8UL), 0);
  for (int64_t i = 0; i < static_cast<int64_t>(bitmap_.size()); ++i) {
    result[i / 8] |= bitmap_[i] << (7 - i % 8);
  }
  return result;
}

std::vector<uint8_t> BitPushBuffer::GetUint8DataWithMsbPadding() const {
  std::vector<uint8_t> result;
  result.resize(CeilOfRatio(bitmap_.size(), 8UL), 0);
  int64_t msbyte_padding_bits =
      bitmap_.size() % 8 == 0 ? 0 : (8 - bitmap_.size() % 8);
  int64_t msbyte_populated_bits = 8 - msbyte_padding_bits;
  for (int64_t source_bit_index = 0;
       source_bit_index < static_cast<int64_t>(bitmap_.size());
       ++source_bit_index) {
    bool bit_value = bitmap_[source_bit_index];
    int64_t target_bit_index =
        source_bit_index < msbyte_populated_bits
            ? msbyte_populated_bits - source_bit_index - 1
            : 7 - ((source_bit_index + msbyte_padding_bits) % 8);
    int64_t target_byte_index =
        source_bit_index < msbyte_populated_bits
            ? 0
            : (source_bit_index + msbyte_padding_bits) / 8;
    result[target_byte_index] |= bit_value << target_bit_index;
  }
  return result;
}

std::string BitPushBuffer::ToString() const {
  std::string result = "0b";
  for (bool bit : bitmap_) {
    result += bit ? '1' : '0';
  }
  return result;
}

}  // namespace xls

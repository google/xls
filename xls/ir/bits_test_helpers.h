// Copyright 2023 The XLS Authors
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

#ifndef XLS_IR_BITS_TEST_HELPERS_H_
#define XLS_IR_BITS_TEST_HELPERS_H_

#include <cstdint>

#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"

namespace xls {

// Create a Bits of the given bit count with the prime number index bits set to
// one.
inline Bits PrimeBits(int64_t bit_count) {
  auto is_prime = [](int64_t n) {
    if (n < 2) {
      return false;
    }
    for (int64_t i = 2; i * i < n; ++i) {
      if (n % i == 0) {
        return false;
      }
    }
    return true;
  };

  InlineBitmap bitmap(bit_count, /*fill=*/false);
  for (int64_t i = 0; i < bit_count; ++i) {
    if (is_prime(i)) {
      bitmap.Set(i, true);
    }
  }
  return Bits::FromBitmap(bitmap);
}

}  // namespace xls

#endif  // XLS_IR_BITS_TEST_HELPERS_H_

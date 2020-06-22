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

#ifndef XLS_COMMON_BITS_UTIL_H_
#define XLS_COMMON_BITS_UTIL_H_

#include "xls/common/logging/logging.h"

namespace xls {

// Helper that generates an (unsigned) mask with "bit_count" low bits set.
inline uint64 Mask(int64 bit_count) {
  XLS_DCHECK_GE(bit_count, 0);
  XLS_DCHECK_LE(bit_count, 64);
  return bit_count == 64 ? -1ULL : (1ULL << bit_count) - 1;
}

// Swaps the byte order of the input vector.
inline void ByteSwap(absl::Span<uint8> input) {
  std::reverse(input.begin(), input.end());
}

}  // namespace xls

#endif  // XLS_COMMON_BITS_UTIL_H_

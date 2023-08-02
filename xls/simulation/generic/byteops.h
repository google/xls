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

#ifndef XLS_SIMULATION_GENERIC_BYTEOPS_H_
#define XLS_SIMULATION_GENERIC_BYTEOPS_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic::byteops {

// Given a span of bytes, reads Size bytes starting from a specified
// offset and assembles them into a value of integer type T following
// little-endian byte order
template <typename Word, uint64_t Size = sizeof(Word)>
absl::StatusOr<Word> bytes_read_word(absl::Span<const uint8_t> bytes,
                                     uint64_t offset) {
  static_assert(Size >= 1, "Size should be positive");
  static_assert(std::numeric_limits<Word>::is_integer,
                "Word must be an integer");
  static_assert(sizeof(Word) >= Size, "Word is too small for specified Size");
  Word ret = 0;
  if (bytes.size() == 0 || bytes.size() - 1 < offset)
    return absl::InternalError("Offset: " + std::to_string(offset) +
                               " is outside byte array");
  // Assume little-endian order
  for (uint64_t i = 0; i < Size; i++) {
    if (offset + Size - 1 - i >= bytes.size()) continue;
    if (i > 0) ret <<= 8;
    ret |= bytes[offset + Size - 1 - i];
  }
  return ret;
}

// Given a span of bytes, takes value of integer T and writes it to the
// span at a specified offset following little-endian byte order
template <typename Word, uint64_t Size = sizeof(Word)>
absl::Status bytes_write_word(absl::Span<uint8_t> bytes, uint64_t offset,
                              Word value) {
  static_assert(Size >= 1, "Size should be positive");
  static_assert(std::numeric_limits<Word>::is_integer,
                "Word must be an integer");
  static_assert(sizeof(Word) >= Size, "Word is too small for specified Size");
  using UWord = std::make_unsigned_t<Word>;
  if (bytes.size() == 0 || bytes.size() - 1 < offset)
    return absl::InternalError("Offset: " + std::to_string(offset) +
                               " is outside byte array");
  UWord uval = value;
  // Assume little-endian order
  for (uint64_t i = 0; i < Size && offset + i < bytes.size(); i++) {
    bytes[offset + i] = uval & 0xFF;
    uval >>= 8;
  }
  return absl::OkStatus();
}

}  // namespace xls::simulation::generic::byteops

#endif  // XLS_SIMULATION_GENERIC_BYTEOPS_H_

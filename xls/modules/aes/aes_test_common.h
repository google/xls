// Copyright 2022 The XLS Authors
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
#ifndef XLS_MODULES_AES_AES_TEST_COMMON_H_
#define XLS_MODULES_AES_AES_TEST_COMMON_H_

// Utilities common to AES module testing.
// These will likely be expanded/modified once tests exist for other key sizes.
#include <arpa/inet.h>

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/ir/value.h"

namespace xls::aes {

constexpr int32_t kBlockBits = 128;
constexpr int32_t kBlockBytes =
    kBlockBits / std::numeric_limits<uint8_t>::digits;
constexpr int32_t kInitVectorBits = 96;
constexpr int32_t kInitVectorBytes =
    kInitVectorBits / std::numeric_limits<uint8_t>::digits;

using Block = std::array<uint8_t, kBlockBytes>;
using InitVector = std::array<uint8_t, kInitVectorBytes>;
using AuthData = std::vector<Block>;

// Returns a string representation of the given block in a format suitable for
// printing/debugging.
std::string FormatBlock(const Block& block);
std::string FormatBlocks(const std::vector<Block>& block, int indent);

// Returns a string representation of the initialization vector for AES counter
// modes.
std::string FormatInitVector(const InitVector& iv);

// Prints an appropriate error message on a block value mismatch.
void PrintFailure(const Block& expected_block, const Block& actual_block,
                  int32_t index, bool ciphertext);

Value InitVectorToValue(const InitVector& iv);

// Converts the given block into an XLS value.
absl::StatusOr<Value> BlockToValue(const Block& block);

// Converts the given value into a block.
absl::StatusOr<Block> ValueToBlock(const Value& value);

// Converts the given key into an XLS value.
// TODO(rspringer): A user should never have to do this - any transformations of
// this sort should be handled "internally", by some operation that combines
// data from an llvm::DataLayout and a call to JitRuntime::BlitValueToBuffer().
template <int kKeyBytes>
absl::StatusOr<Value> KeyToValue(const std::array<uint8_t, kKeyBytes>& key) {
  constexpr int32_t kKeyWords = kKeyBytes / 4;
  std::vector<Value> key_values;
  key_values.reserve(kKeyWords);
  for (int32_t i = 0; i < kKeyWords; i++) {
    uint32_t q = i * 4;
    // XLS is big-endian, so we have to populate the key in "reverse" order.
    uint32_t key_word = static_cast<uint32_t>(key[q + 3]) |
                        static_cast<uint32_t>(key[q + 2]) << 8 |
                        static_cast<uint32_t>(key[q + 1]) << 16 |
                        static_cast<uint32_t>(key[q]) << 24;
    key_values.push_back(Value(UBits(key_word, /*bit_count=*/32)));
  }
  return Value::Array(key_values);
}

// Populates the given buffer with the value of "key". In DSLX land, a key is
// four sequential u32s, but XLS is big-endian, so we need to re-lay out our
// bits appropriately.
template <int kKeyBytes, int kKeyWords = kKeyBytes / 4>
void KeyToBuffer(const std::array<uint8_t, kKeyBytes>& key,
                 std::array<uint32_t, kKeyWords>* buffer) {
  for (int32_t i = 0; i < 4; i++) {
    uint32_t key_word =
        htonl(*reinterpret_cast<const uint32_t*>(key.data() + i * 4));
    buffer->data()[i] = key_word;
  }
}

// Returns a string representation of the given key in a format suitable for
// printing/debugging.
template <int kKeyBytes>
std::string FormatKey(const std::array<uint8_t, kKeyBytes>& key) {
  return absl::StrJoin(key, ", ", [](std::string* out, uint8_t key_byte) {
    absl::StrAppend(out, absl::StrFormat("0x%02x", key_byte));
  });
}

}  // namespace xls::aes

#endif  // XLS_MODULES_AES_AES_TEST_COMMON_H_

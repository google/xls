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

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/value.h"

namespace xls::aes {

constexpr int32_t kMaxKeyBits = 256;
constexpr int32_t kMaxKeyBytes = kMaxKeyBits / 8;

constexpr int32_t kBlockBits = 128;
constexpr int32_t kBlockBytes =
    kBlockBits / std::numeric_limits<uint8_t>::digits;
constexpr int32_t kInitVectorBits = 96;
constexpr int32_t kInitVectorBytes =
    kInitVectorBits / std::numeric_limits<uint8_t>::digits;

using Block = std::array<uint8_t, kBlockBytes>;
using AuthData = std::vector<Block>;
using InitVector = std::array<uint8_t, kInitVectorBytes>;
using Key = std::array<uint8_t, kMaxKeyBytes>;

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
absl::StatusOr<Value> KeyToValue(const Key& key);

// Returns a string representation of the given key in a format suitable for
// printing/debugging.
std::string FormatKey(const Key& key);

}  // namespace xls::aes

#endif  // XLS_MODULES_AES_AES_TEST_COMMON_H_

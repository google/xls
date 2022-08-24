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
#include "xls/modules/aes/aes_128_test_common.h"

#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"

namespace xls::aes {

std::string FormatKey(const std::vector<uint8_t>& key) {
  return absl::StrFormat(
      "0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, "
      "0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x",
      key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8],
      key[9], key[10], key[11], key[12], key[13], key[14], key[15]);
}

std::string FormatBlock(const Block& block) {
  return absl::StrFormat(
      "0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, "
      "0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x",
      block[0], block[1], block[2], block[3], block[4], block[5], block[6],
      block[7], block[8], block[9], block[10], block[11], block[12], block[13],
      block[14], block[15]);
}

std::string FormatBlocks(const std::vector<Block>& blocks, int indent) {
  std::vector<std::string> pieces;
  pieces.reserve(blocks.size());
  for (const Block& block : blocks) {
    pieces.push_back(
        absl::StrCat(std::string(indent, ' '), FormatBlock(block)));
  }

  return absl::StrJoin(pieces, "\n");
}

std::string FormatInitVector(const InitVector& iv) {
  return absl::StrFormat(
      "0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, "
      "0x%02x, 0x%02x, 0x%02x, 0x%02x",
      iv[0], iv[1], iv[2], iv[3], iv[4], iv[5], iv[6], iv[7], iv[8], iv[9],
      iv[10], iv[11]);
}

void PrintFailure(const Block& expected_block, const Block& actual_block,
                  const std::vector<uint8_t>& key, int32_t index,
                  bool ciphertext) {
  std::string type_str = ciphertext ? "ciphertext" : "plaintext";
  std::cout << "Mismatch in " << type_str << " at byte " << index << ": "
            << std::hex << "expected: 0x"
            << static_cast<uint32_t>(expected_block[index]) << "; actual: 0x"
            << static_cast<uint32_t>(actual_block[index]) << std::endl;
}

absl::StatusOr<Value> KeyToValue(const std::vector<uint8_t>& key) {
  constexpr int32_t kKeyWords = kKeyBits / 32;
  std::vector<Value> key_values;
  key_values.reserve(kKeyWords);
  for (int32_t i = 0; i < kKeyWords; i++) {
    uint32_t q = i * kKeyWords;
    // XLS is big-endian, so we have to populate the key in "reverse" order.
    uint32_t key_word = static_cast<uint32_t>(key[q + 3]) |
                        static_cast<uint32_t>(key[q + 2]) << 8 |
                        static_cast<uint32_t>(key[q + 1]) << 16 |
                        static_cast<uint32_t>(key[q]) << 24;
    key_values.push_back(Value(UBits(key_word, /*bit_count=*/32)));
  }
  return Value::Array(key_values);
}

absl::StatusOr<Value> BlockToValue(const Block& block) {
  std::vector<Value> block_rows;
  block_rows.reserve(4);
  for (int32_t i = 0; i < 4; i++) {
    std::vector<Value> block_cols;
    block_cols.reserve(4);
    for (int32_t j = 0; j < 4; j++) {
      block_cols.push_back(Value(UBits(block[i * 4 + j], /*bit_count=*/8)));
    }
    block_rows.push_back(Value::ArrayOrDie(block_cols));
  }
  return Value::Array(block_rows);
}

absl::StatusOr<Block> ValueToBlock(const Value& value) {
  Block result;
  XLS_ASSIGN_OR_RETURN(std::vector<Value> result_rows, value.GetElements());
  for (int32_t i = 0; i < 4; i++) {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> result_cols,
                         result_rows[i].GetElements());
    for (int32_t j = 0; j < 4; j++) {
      XLS_ASSIGN_OR_RETURN(result[i * 4 + j], result_cols[j].bits().ToUint64());
    }
  }

  return result;
}

}  // namespace xls::aes

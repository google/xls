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

// Sample generator for AES encryption tests.
//
// Does not yet support:
//  - Pure AES mode
//  - CTR mode
//  - User specification of key, iv, etc.
//  - Output in formats other than DSLX.
//
// These will be added when/if needed.
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "openssl/aead.h"  // NOLINT
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/modules/aes/aes_test_common.h"

namespace xls::aes {

// Either plain single-block AES encryption, AES-CTR mode, or AES-GCM mode.
enum class EncryptionMode { AES, CTR, GCM };

struct Sample {
  Key key;
  int key_bytes;
  InitVector iv;
  std::vector<Block> aad;
  std::vector<Block> msg;
};

struct Result {
  std::vector<Block> msg;
  std::optional<Block> auth_tag;
};

static absl::StatusOr<Result> RunGcm(const Sample& sample) {
  EVP_AEAD_CTX* ref_ctx =
      EVP_AEAD_CTX_new(EVP_aead_aes_256_gcm(), sample.key.data(),
                       /*key_len=*/32, /*tag_len=*/16);

  size_t max_result_size = sample.msg.size() * kBlockBytes +
                           EVP_AEAD_max_overhead(EVP_aead_aes_256_gcm());

  auto msg_buffer =
      std::make_unique<uint8_t[]>(sample.msg.size() * kBlockBytes);
  for (int i = 0; i < sample.msg.size(); i++) {
    memcpy(&msg_buffer[i * kBlockBytes], sample.msg[i].data(), kBlockBytes);
  }

  auto aad_buffer =
      std::make_unique<uint8_t[]>(sample.aad.size() * kBlockBytes);
  for (int i = 0; i < sample.aad.size(); i++) {
    memcpy(&aad_buffer[i * kBlockBytes], sample.aad[i].data(), kBlockBytes);
  }

  auto result_buffer = std::make_unique<uint8_t[]>(max_result_size);
  size_t actual_result_size;
  int success =
      EVP_AEAD_CTX_seal(ref_ctx, result_buffer.get(), &actual_result_size,
                        max_result_size, sample.iv.data(), kInitVectorBytes,
                        msg_buffer.get(), sample.msg.size() * kBlockBytes,
                        aad_buffer.get(), sample.aad.size() * kBlockBytes);
  if (!static_cast<bool>(success)) {
    return absl::UnknownError("Error encrypting sample.");
  }

  Result result;
  result.msg.resize(sample.msg.size());
  for (int i = 0; i < sample.msg.size(); i++) {
    memcpy(result.msg[i].data(), &result_buffer[i * kBlockBytes], kBlockBytes);
  }
  result.auth_tag.emplace(Block());
  memcpy(result.auth_tag.value().data(),
         &result_buffer[sample.msg.size() * kBlockBytes], kBlockBytes);

  return result;
}

static Block CreateDataBlock(absl::BitGen* bitgen) {
  Block block;
  for (int i = 0; i < kBlockBytes; i++) {
    block[i] = absl::Uniform(*bitgen, 0, 256);
  }
  return block;
}

static Key CreateKey(absl::BitGen* bitgen, int key_bytes) {
  Key key;
  key.fill(0);
  for (int i = 0; i < key_bytes; i++) {
    key[i] = absl::Uniform(*bitgen, 0, 256);
  }
  return key;
}

static InitVector CreateInitVector(absl::BitGen* bitgen) {
  InitVector iv;
  for (int i = 0; i < kInitVectorBytes; i++) {
    iv[i] = absl::Uniform(*bitgen, 0, 256);
  }

  return iv;
}

static std::string DslxFormatKey(const Key& key, int key_bytes) {
  std::string indent_1(8, ' ');
  std::string indent_2(12, ' ');
  std::vector<std::string> pieces;
  pieces.push_back(absl::StrCat(indent_1, "let key = Key:["));
  for (int i = 0; i < key_bytes; i += 4) {
    pieces.push_back(
        absl::StrFormat("%su8:0x%02x, u8:0x%02x, u8:0x%02x, u8:0x%02x,",
                        indent_2, key[i], key[i + 1], key[i + 2], key[i + 3]));
  }
  if (key_bytes == 16) {
    pieces.push_back(absl::StrCat(indent_2, "..."));
  }
  pieces.push_back(absl::StrCat(indent_1, "];"));
  return absl::StrJoin(pieces, "\n");
}

static std::string DslxFormatIv(const InitVector& iv) {
  std::string indent(8, ' ');
  std::string iv_str = "0x";
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2; j++) {
      absl::StrAppend(&iv_str, absl::StrFormat("%02x", iv[i * 2 + j]));
    }
    if (i != 5) {
      absl::StrAppend(&iv_str, "_");
    }
  }
  return absl::StrCat(indent, "let iv = InitVector:", iv_str, ";");
}

// Only prints the data, none of the surrounding formatting.
static std::string DslxFormatBlock(const Block& block, int indent) {
  std::string indent_str(indent * 4, ' ');
  std::vector<std::string> pieces;
  for (int i = 0; i < 16; i += 4) {
    pieces.push_back(absl::StrFormat(
        "%su8[4]:[u8:0x%02x, u8:0x%02x, u8:0x%02x, u8:0x%02x],", indent_str,
        block[i], block[i + 1], block[i + 2], block[i + 3]));
  }
  return absl::StrJoin(pieces, "\n");
}

static std::string DslxFormatBlocks(const std::vector<Block>& blocks,
                                    std::string_view var_name) {
  std::string indent_1(8, ' ');
  std::string indent_2(12, ' ');
  std::vector<std::string> pieces;

  if (blocks.size() == 1) {
    pieces.push_back(absl::StrFormat("%slet %s = Block:[", indent_1, var_name));
    pieces.push_back(DslxFormatBlock(blocks[0], /*indent=*/3));
  } else {
    pieces.push_back(absl::StrFormat("%slet %s = Block[%d]:[", indent_1,
                                     var_name, blocks.size()));
    for (int i = 0; i < blocks.size(); i++) {
      pieces.push_back(absl::StrCat(indent_2, "Block:["));
      pieces.push_back(DslxFormatBlock(blocks[i], /*indent=*/4));
      pieces.push_back(absl::StrCat(indent_2, "],"));
    }
  }
  pieces.push_back(absl::StrCat(indent_1, "];"));
  return absl::StrJoin(pieces, "\n");
}

static std::string DslxFormatOutput(const Sample& sample,
                                    const Result& result) {
  std::vector<std::string> pieces;
  pieces.push_back(DslxFormatKey(sample.key, sample.key_bytes));
  pieces.push_back(DslxFormatIv(sample.iv));
  if (!sample.aad.empty()) {
    pieces.push_back(DslxFormatBlocks(sample.aad, "aad"));
  }
  if (!sample.msg.empty()) {
    pieces.push_back(DslxFormatBlocks(sample.msg, "msg"));
  }
  if (!result.msg.empty()) {
    pieces.push_back(DslxFormatBlocks(result.msg, "expected_msg"));
  }
  if (result.auth_tag.has_value()) {
    pieces.push_back(
        DslxFormatBlocks({result.auth_tag.value()}, "expected_auth_tag"));
  }

  return absl::StrJoin(pieces, "\n");
}

static absl::Status RealMain(EncryptionMode mode, int key_bits,
                             int num_aad_blocks, int num_msg_blocks) {
  absl::BitGen bitgen;
  Sample sample;
  sample.key = CreateKey(&bitgen, key_bits / 8);
  sample.key_bytes = key_bits / 8;
  sample.iv = CreateInitVector(&bitgen);
  sample.aad.reserve(num_aad_blocks);
  for (int i = 0; i < num_aad_blocks; i++) {
    sample.aad.push_back(CreateDataBlock(&bitgen));
  }
  sample.msg.reserve(num_msg_blocks);
  for (int i = 0; i < num_msg_blocks; i++) {
    sample.msg.push_back(CreateDataBlock(&bitgen));
  }

  XLS_ASSIGN_OR_RETURN(Result result, RunGcm(sample));

  std::cout << DslxFormatOutput(sample, result) << '\n';

  return absl::OkStatus();
}

static bool AbslParseFlag(std::string_view text, xls::aes::EncryptionMode* out,
                          std::string* error) {
  if (text == "aes") {
    *out = xls::aes::EncryptionMode::AES;
    return true;
  }

  if (text == "ctr") {
    *out = xls::aes::EncryptionMode::CTR;
    return true;
  }

  if (text == "gcm") {
    *out = xls::aes::EncryptionMode::GCM;
    return true;
  }

  return false;
}

static std::string AbslUnparseFlag(xls::aes::EncryptionMode in) {
  if (in == xls::aes::EncryptionMode::AES) {
    return "aes";
  }

  if (in == xls::aes::EncryptionMode::CTR) {
    return "ctr";
  }

  if (in == xls::aes::EncryptionMode::GCM) {
    return "gcm";
  }

  return "unknown";
}

}  // namespace xls::aes

ABSL_FLAG(xls::aes::EncryptionMode, mode, xls::aes::EncryptionMode::AES,
          "Encryption mode: pure AES, AES-CTR, or AES-GCM, "
          "as the lowercase strings \"aes\", \"ctr\", and \"gcm\", "
          "respectively.");
ABSL_FLAG(int, key_bits, 256,
          "Key size to use (in bits). 128, 192, and 256 bits are options.");
ABSL_FLAG(int, aad_blocks, 1, "Number of AAD blocks to generate.");
ABSL_FLAG(int, msg_blocks, 1, "Number of msg blocks to generate.");

int main(int argc, char** argv) {
  std::vector<std::string_view> args = xls::InitXls(argv[0], argc, argv);

  int key_bits = absl::GetFlag(FLAGS_key_bits);
  if (key_bits != 128 && key_bits != 192 && key_bits != 256) {
    std::cout << "Key size must be 128, 192, or 256 bits.";
    return 1;
  }

  int num_aad_blocks = absl::GetFlag(FLAGS_aad_blocks);
  if (num_aad_blocks < 0) {
    std::cout << "--aad_blocks must be >= 0.";
    return 1;
  }

  int num_msg_blocks = absl::GetFlag(FLAGS_msg_blocks);
  if (num_msg_blocks < 0) {
    std::cout << "--msg_blocks must be >= 0.";
    return 1;
  }

  return xls::ExitStatus(xls::aes::RealMain(absl::GetFlag(FLAGS_mode), key_bits,
                                            num_aad_blocks, num_msg_blocks));
}

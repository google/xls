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

// Test to compare the outputs of a reference vs. XLS AES implementation.
// Currently only supports AES-128 in CBC mode, but that may be expanded in the
// future.
//
// The code is unique_ptr heavy, but that'll change once templated over
// various key lengths.
#include <cstdint>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "openssl/aes.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value.h"
#include "xls/modules/aes/aes_128_decrypt_cc.h"
#include "xls/modules/aes/aes_128_encrypt_cc.h"

constexpr int32_t kKeyBits = 128;
constexpr int32_t kKeyBytes = kKeyBits / 8;
constexpr int32_t kBlockBytes = 16;

ABSL_FLAG(int32_t, num_samples, 1000,
          "The number of (randomly-generated) blocks to test.");

namespace xls {

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

absl::StatusOr<Value> BlockToValue(const std::vector<uint8_t>& plaintext) {
  std::vector<Value> block_rows;
  block_rows.reserve(4);
  for (int32_t i = 0; i < 4; i++) {
    std::vector<Value> block_cols;
    block_cols.reserve(4);
    for (int32_t j = 0; j < 4; j++) {
      block_cols.push_back(Value(UBits(plaintext[i * 4 + j], /*bit_count=*/8)));
    }
    block_rows.push_back(Value::ArrayOrDie(block_cols));
  }
  return Value::Array(block_rows);
}

absl::StatusOr<std::vector<uint8_t>> ValueToBlock(const Value& value) {
  std::vector<uint8_t> result(kBlockBytes, 0);
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

absl::StatusOr<std::vector<uint8_t>> XlsEncrypt(
    const std::vector<uint8_t>& key, const std::vector<uint8_t>& plaintext) {
  XLS_ASSIGN_OR_RETURN(Value key_value, KeyToValue(key));
  XLS_ASSIGN_OR_RETURN(Value block_value, BlockToValue(plaintext));
  XLS_ASSIGN_OR_RETURN(Value result_value,
                       xls::aes::aes_encrypt(key_value, block_value));
  return ValueToBlock(result_value);
}

absl::StatusOr<std::vector<uint8_t>> XlsDecrypt(
    const std::vector<uint8_t>& key, const std::vector<uint8_t>& plaintext) {
  XLS_ASSIGN_OR_RETURN(Value key_value, KeyToValue(key));
  XLS_ASSIGN_OR_RETURN(Value block_value, BlockToValue(plaintext));
  XLS_ASSIGN_OR_RETURN(Value result_value,
                       xls::aes::aes_decrypt(key_value, block_value));
  return ValueToBlock(result_value);
}

std::vector<uint8_t> ReferenceEncrypt(const std::vector<uint8_t>& key,
                                      const std::vector<uint8_t>& plaintext) {
  std::vector<uint8_t> ciphertext(kBlockBytes, 0);

  // Needed because the key is modified during operation.
  uint8_t local_key[kKeyBytes];
  memcpy(local_key, key.data(), kKeyBytes);

  // OpenSSL doesn't have a GCM implementation, so we'll have to use something
  // else once we get there.
  AES_KEY aes_key;
  XLS_QCHECK_EQ(AES_set_encrypt_key(local_key, kKeyBits, &aes_key), 0);
  AES_encrypt(plaintext.data(), ciphertext.data(), &aes_key);
  return ciphertext;
}

std::vector<uint8_t> ReferenceDecrypt(const std::vector<uint8_t>& key,
                                      const std::vector<uint8_t>& ciphertext) {
  std::vector<uint8_t> plaintext(kBlockBytes, 0);

  // Needed because the key is modified during operation.
  uint8_t local_key[kKeyBytes];
  memcpy(local_key, key.data(), kKeyBytes);

  AES_KEY aes_key;
  XLS_QCHECK_EQ(AES_set_decrypt_key(local_key, kKeyBits, &aes_key), 0);
  AES_decrypt(ciphertext.data(), plaintext.data(), &aes_key);
  return plaintext;
}

// `block` had better have 16 elements!
std::string FormatBlock(const std::vector<uint8_t> block) {
  return absl::StrFormat(
      "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, "
      "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x",
      block[0], block[1], block[2], block[3], block[4], block[5], block[6],
      block[7], block[8], block[9], block[10], block[11], block[12], block[13],
      block[14], block[15]);
}

std::string FormatKey(std::vector<uint8_t>& key) {
  return absl::StrFormat(
      "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, "
      "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x",
      key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8],
      key[9], key[10], key[11], key[12], key[13], key[14], key[15]);
}

void PrintFailure(const std::vector<uint8_t>& expected_block,
                  const std::vector<uint8_t>& actual_block,
                  const std::vector<uint8_t>& key, int32_t index,
                  bool ciphertext) {
  std::string type_str = ciphertext ? "ciphertext" : "plaintext";
  std::cout << "Mismatch in " << type_str << " at byte " << index << ": "
            << std::hex << "expected: 0x"
            << static_cast<uint32_t>(expected_block[index]) << "; actual: 0x"
            << static_cast<uint32_t>(actual_block[index]) << std::endl;
}

// Returns false on error (will terminate further runs).
absl::StatusOr<bool> RunSample(const std::vector<uint8_t>& input,
                               const std::vector<uint8_t>& key,
                               absl::Duration* xls_encrypt_dur,
                               absl::Duration* xls_decrypt_dur) {
  XLS_VLOG(2) << "Plaintext: " << FormatBlock(input) << std::endl;

  std::vector<uint8_t> reference_ciphertext = ReferenceEncrypt(key, input);

  absl::Time start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(std::vector<uint8_t> xls_ciphertext,
                       XlsEncrypt(key, input));
  *xls_encrypt_dur += absl::Now() - start_time;

  XLS_VLOG(2) << "Reference ciphertext: " << FormatBlock(reference_ciphertext)
              << std::endl;
  XLS_VLOG(2) << "XLS ciphertext: " << FormatBlock(xls_ciphertext) << std::endl;

  // Verify the ciphertexts match, to ensure we're actually doing the encryption
  // properly.
  for (int32_t i = 0; i < kBlockBytes; i++) {
    if (reference_ciphertext[i] != xls_ciphertext[i]) {
      PrintFailure(reference_ciphertext, xls_ciphertext, key, i,
                   /*ciphertext=*/true);
      return false;
    }
  }

  start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(std::vector<uint8_t> xls_decrypted,
                       XlsDecrypt(key, input));
  *xls_decrypt_dur += absl::Now() - start_time;

  XLS_VLOG(2) << "Decrypted plaintext: " << FormatBlock(xls_decrypted)
              << std::endl;

  // We can just compare the XLS result to the input to verify we're decrypting
  // right.
  for (int32_t i = 0; i < kBlockBytes; i++) {
    if (reference_ciphertext[i] != xls_ciphertext[i]) {
      PrintFailure(input, xls_decrypted, key, i, /*ciphertext=*/false);
      return false;
    }
  }

  return true;
}

absl::Status RealMain(int32_t num_samples) {
  std::vector<uint8_t> input(kBlockBytes, 0);
  std::vector<uint8_t> key(kKeyBytes, 0);
  absl::BitGen bitgen;
  absl::Duration xls_encrypt_dur;
  absl::Duration xls_decrypt_dur;
  for (int32_t i = 0; i < num_samples; i++) {
    for (int32_t j = 0; j < kKeyBytes; j++) {
      key[j] = absl::Uniform(bitgen, 0, 256);
    }
    for (int32_t j = 0; j < kBlockBytes; j++) {
      input[j] = absl::Uniform(bitgen, 0, 256);
    }

    XLS_ASSIGN_OR_RETURN(bool proceed, RunSample(input, key, &xls_encrypt_dur,
                                                 &xls_decrypt_dur));
    if (!proceed) {
      std::cout << "Plaintext: " << FormatBlock(input) << std::endl;
      std::cout << "Key: " << FormatKey(key) << std::endl;
      break;
    }
  }

  std::cout << "Successfully ran " << num_samples << " samples." << std::endl;
  std::cout << "Avg. XLS encryption time: " << xls_encrypt_dur / num_samples
            << std::endl;
  std::cout << "Avg. XLS decryption time: " << xls_decrypt_dur / num_samples
            << std::endl;

  return absl::OkStatus();
}

}  // namespace xls

int32_t main(int32_t argc, char** argv) {
  std::vector<absl::string_view> args = xls::InitXls(argv[0], argc, argv);
  XLS_QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_num_samples)));
  return 0;
}

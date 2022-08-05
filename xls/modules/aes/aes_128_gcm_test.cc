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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "openssl/aes.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/modules/aes/aes_128_gcm_jit_wrapper.h"

constexpr int kKeyBits = 128;
constexpr int kKeyBytes = kKeyBits / 8;

namespace xls {

std::unique_ptr<uint8_t[]> ReferenceEncrypt(uint8_t* plaintext, uint8_t* key) {
  auto ciphertext = std::make_unique<uint8_t[]>(16);

  // Needed because the key is modified during operation.
  uint8_t local_key[kKeyBytes];
  memcpy(local_key, key, kKeyBytes);

  // OpenSSL doesn't have a GCM implementation, so we'll have to use something
  // else once we get there.
  AES_KEY aes_key;
  XLS_QCHECK_EQ(AES_set_encrypt_key(local_key, kKeyBits, &aes_key), 0);
  AES_encrypt(plaintext, ciphertext.get(), &aes_key);
  return ciphertext;
}

absl::StatusOr<std::unique_ptr<uint8_t[]>> XlsEncrypt(Aes128Gcm* jit,
                                                      uint8_t* plaintext,
                                                      uint8_t* key) {
  std::vector<Value> key_values;
  key_values.reserve(4);
  for (int i = 0; i < 4; i++) {
    uint32_t q = i * 4;
    // XLS is big-endian, so we have to populate the key in "reverse" order.
    uint32_t key_word = static_cast<uint32_t>(key[q + 3]) |
                        static_cast<uint32_t>(key[q + 2]) << 8 |
                        static_cast<uint32_t>(key[q + 1]) << 16 |
                        static_cast<uint32_t>(key[q]) << 24;
    key_values.push_back(Value(UBits(key_word, /*bit_count=*/32)));
  }
  Value key_value(Value::ArrayOrDie(key_values));

  std::vector<Value> block_rows;
  block_rows.reserve(4);
  for (int i = 0; i < 4; i++) {
    std::vector<Value> block_cols;
    block_cols.reserve(4);
    for (int j = 0; j < 4; j++) {
      block_cols.push_back(Value(UBits(plaintext[i * 4 + j], /*bit_count=*/8)));
    }
    block_rows.push_back(Value::ArrayOrDie(block_cols));
  }
  Value block_value(Value::ArrayOrDie(block_rows));

  XLS_ASSIGN_OR_RETURN(Value result_value, jit->Run(key_value, block_value));

  auto result = std::make_unique<uint8_t[]>(16);
  XLS_ASSIGN_OR_RETURN(std::vector<Value> result_rows,
                       result_value.GetElements());
  for (int i = 0; i < 4; i++) {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> result_cols,
                         result_rows[i].GetElements());
    for (int j = 0; j < 4; j++) {
      XLS_ASSIGN_OR_RETURN(result[i * 4 + j], result_cols[j].bits().ToUint64());
    }
  }

  return result;
}

std::unique_ptr<uint8_t[]> ReferenceDecrypt(const uint8_t* ciphertext,
                                            const uint8_t* key) {
  auto plaintext = std::make_unique<uint8_t[]>(16);

  // Needed because the key is modified during operation.
  uint8_t local_key[kKeyBytes];

  memcpy(local_key, key, kKeyBytes);

  AES_KEY aes_key;
  XLS_QCHECK_EQ(AES_set_decrypt_key(local_key, kKeyBits, &aes_key), 0);
  AES_decrypt(ciphertext, plaintext.get(), &aes_key);
  return plaintext;
}

// `block` had better have 16 elements!
std::string FormatBlock(uint8_t* block) {
  return absl::StrFormat(
      "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, "
      "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x",
      block[0], block[1], block[2], block[3], block[4], block[5], block[6],
      block[7], block[8], block[9], block[10], block[11], block[12], block[13],
      block[14], block[15]);
}

// This test will be expanded to run multiple samples once the XLS
// implementation supports decryption.
absl::Status RealMain() {
  uint8_t input[16] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                       0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
  uint8_t key[16] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                     0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};

  XLS_VLOG(2) << "Plaintext: " << FormatBlock(input) << std::endl;

  std::unique_ptr<uint8_t[]> reference_ciphertext =
      ReferenceEncrypt(input, key);
  XLS_ASSIGN_OR_RETURN(auto jit, Aes128Gcm::Create());
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<uint8_t[]> xls_ciphertext,
                       XlsEncrypt(jit.get(), input, key));

  XLS_VLOG(2) << "Reference ciphertext: "
              << FormatBlock(reference_ciphertext.get()) << std::endl;
  XLS_VLOG(2) << "XLS ciphertext: " << FormatBlock(xls_ciphertext.get())
              << std::endl;

  for (int i = 0; i < 16; i++) {
    XLS_QCHECK_EQ(xls_ciphertext[i], reference_ciphertext[i])
        << "Mismatch at byte " << i << std::hex
        << ": XLS: " << xls_ciphertext[i] << " vs. " << reference_ciphertext[i];
  }

  std::unique_ptr<uint8_t[]> reference_decrypted =
      ReferenceDecrypt(reference_ciphertext.get(), key);
  XLS_VLOG(2) << "Reference decrypted: "
              << FormatBlock(reference_decrypted.get()) << std::endl;

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> args = xls::InitXls(argv[0], argc, argv);
  XLS_QCHECK_OK(xls::RealMain());
  return 0;
}

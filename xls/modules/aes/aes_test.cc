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
#include "openssl/aes.h"

#include <cstdint>
#include <iostream>
#include <ostream>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/modules/aes/aes_decrypt_cc.h"
#include "xls/modules/aes/aes_encrypt_cc.h"
#include "xls/modules/aes/aes_test_common.h"

ABSL_FLAG(int32_t, num_samples, 1000,
          "The number of (randomly-generated) blocks to test.");

namespace xls::aes {

static absl::StatusOr<Block> XlsEncrypt(const Key& key, int key_bytes,
                                        const Block& plaintext) {
  // Not sure why Clang isn't able to infer the "32" correctly, but w/e.
  XLS_ASSIGN_OR_RETURN(Value key_value, KeyToValue(key));
  XLS_ASSIGN_OR_RETURN(Value block_value, BlockToValue(plaintext));
  Value width_value(UBits(key_bytes == 16 ? 0 : 2, /*bit_count=*/2));
  XLS_ASSIGN_OR_RETURN(Value result_value,
                       xls::aes::encrypt(key_value, width_value, block_value));
  return ValueToBlock(result_value);
}

static absl::StatusOr<Block> XlsDecrypt(const Key& key, int key_bytes,
                                        const Block& ciphertext) {
  XLS_ASSIGN_OR_RETURN(Value key_value, KeyToValue(key));
  XLS_ASSIGN_OR_RETURN(Value block_value, BlockToValue(ciphertext));
  Value width_value(UBits(key_bytes == 128 ? 0 : 2, /*bit_count=*/2));
  XLS_ASSIGN_OR_RETURN(Value result_value,
                       xls::aes::decrypt(key_value, width_value, block_value));
  return ValueToBlock(result_value);
}

static Block ReferenceEncrypt(const Key& key, int key_bytes,
                              const Block& plaintext) {
  Block ciphertext;

  // Needed because the key is modified during operation.
  uint8_t local_key[kMaxKeyBytes];
  memcpy(local_key, key.data(), key_bytes);

  AES_KEY aes_key;
  QCHECK_EQ(AES_set_encrypt_key(local_key, key_bytes * 8, &aes_key), 0);
  AES_encrypt(plaintext.data(), ciphertext.data(), &aes_key);
  return ciphertext;
}

// Returns false on error (will terminate further runs).
static absl::StatusOr<bool> RunSample(const Block& input, const Key& key,
                                      int key_bytes,
                                      absl::Duration* xls_encrypt_dur,
                                      absl::Duration* xls_decrypt_dur) {
  Block reference_ciphertext = ReferenceEncrypt(key, key_bytes, input);

  absl::Time start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(Block xls_ciphertext, XlsEncrypt(key, key_bytes, input));
  *xls_encrypt_dur += absl::Now() - start_time;

  VLOG(2) << "Reference ciphertext: " << FormatBlock(reference_ciphertext)
          << '\n';
  VLOG(2) << "XLS ciphertext: " << FormatBlock(xls_ciphertext) << '\n';

  // Verify the ciphertexts match, to ensure we're actually doing the encryption
  // properly.
  for (int32_t i = 0; i < kBlockBytes; i++) {
    if (reference_ciphertext[i] != xls_ciphertext[i]) {
      PrintFailure(reference_ciphertext, xls_ciphertext, i,
                   /*ciphertext=*/true);
      return false;
    }
  }

  start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(Block xls_decrypted, XlsDecrypt(key, key_bytes, input));
  *xls_decrypt_dur += absl::Now() - start_time;

  VLOG(2) << "Decrypted plaintext: " << FormatBlock(xls_decrypted) << '\n';

  // We can just compare the XLS result to the input to verify we're decrypting
  // right.
  for (int32_t i = 0; i < kBlockBytes; i++) {
    if (reference_ciphertext[i] != xls_ciphertext[i]) {
      PrintFailure(input, xls_decrypted, i, /*ciphertext=*/false);
      return false;
    }
  }

  return true;
}

static absl::Status RunTest(int32_t key_bits, int32_t num_samples) {
  int key_bytes = key_bits / 8;
  Block input;
  Key key;
  key.fill(0);
  absl::BitGen bitgen;
  absl::Duration xls_encrypt_dur;
  absl::Duration xls_decrypt_dur;
  for (int32_t i = 0; i < num_samples; i++) {
    for (int32_t j = 0; j < key_bytes; j++) {
      key[j] = absl::Uniform(bitgen, 0, 256);
    }
    for (int32_t j = 0; j < kBlockBytes; j++) {
      input[j] = absl::Uniform(bitgen, 0, 256);
    }

    XLS_ASSIGN_OR_RETURN(
        bool proceed,
        RunSample(input, key, key_bytes, &xls_encrypt_dur, &xls_decrypt_dur));
    if (!proceed) {
      std::cout << "Plaintext: " << FormatBlock(input) << '\n';
      std::cout << "Key: " << FormatKey(key) << '\n';
      return absl::InternalError(
          absl::StrCat(key_bits, "-bit validation failed."));
    }
  }

  std::cout << "Successfully ran " << num_samples << " " << key_bits
            << "-bit samples." << '\n';
  std::cout << "Avg. XLS encryption time: " << xls_encrypt_dur / num_samples
            << '\n';
  std::cout << "Avg. XLS decryption time: " << xls_decrypt_dur / num_samples
            << '\n';

  return absl::OkStatus();
}

static absl::Status RealMain(int32_t num_samples) {
  XLS_RETURN_IF_ERROR(RunTest(/*key_bits=*/128, num_samples));
  return RunTest(/*key_bits=*/256, num_samples);
}

}  // namespace xls::aes

int32_t main(int32_t argc, char** argv) {
  std::vector<std::string_view> args = xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::aes::RealMain(absl::GetFlag(FLAGS_num_samples)));
}

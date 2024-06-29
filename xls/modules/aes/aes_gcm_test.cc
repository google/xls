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

// Test of the XLS GCM mode implementation against a reference (in this
// case, BoringSSL's implementation).
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "openssl/aead.h"
#include "openssl/base.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/modules/aes/aes_gcm_wrapper.h"
#include "xls/modules/aes/aes_test_common.h"

// TODO(rspringer): This is a bit slow. Seems like we should be able to compute
// 1k samples a lot faster than 2 minutes.

ABSL_FLAG(int, num_samples, 100, "The number of samples to execute.");
ABSL_FLAG(bool, print_traces, false,
          "If true, print any trace! or trace_fmt! messages.");

namespace xls::aes {

constexpr int kMaxAadBlocks = 128;
constexpr int kMaxPtxtBlocks = 128;
constexpr int kTagBits = 128;
constexpr int kTagBytes = kTagBits / 8;

constexpr std::string_view kCmdChannelName = "aes_gcm__command_in";
constexpr std::string_view kDataInChannelName = "aes_gcm__data_r";
constexpr std::string_view kDataOutChannelName = "aes_gcm__data_s";

struct JitData {
  std::unique_ptr<Package> package;
  std::unique_ptr<ProcRuntime> runtime;
};

struct SampleData {
  EVP_AEAD_CTX* openssl_ctxt;
  Key key;
  int key_bits;
  InitVector init_vector;
  std::vector<Block> input_data;
  AuthData aad;
};

struct Result {
  std::vector<Block> output_data;
  Block auth_tag;
};

static absl::StatusOr<Value> CreateCommandValue(const SampleData& sample_data,
                                                bool encrypt) {
  std::vector<Value> command_elements;
  // encrypt
  command_elements.push_back(Value(UBits(static_cast<uint64_t>(encrypt), 1)));
  // msg_blocks
  command_elements.push_back(Value(UBits(sample_data.input_data.size(), 32)));
  // aad_blocks
  command_elements.push_back(Value(UBits(sample_data.aad.size(), 32)));
  // Key
  XLS_ASSIGN_OR_RETURN(Value key_value, KeyToValue(sample_data.key));
  command_elements.push_back(key_value);
  // Key width
  command_elements.push_back(
      Value(UBits(sample_data.key_bits == 128 ? 0 : 2, /*bit_count=*/2)));
  // IV
  command_elements.push_back(InitVectorToValue(sample_data.init_vector));
  return Value::Tuple(command_elements);
}

static absl::StatusOr<Result> XlsEncrypt(JitData* jit_data,
                                         const SampleData& sample_data,
                                         bool encrypt) {
  // Create (and send) the initial command.
  Package* package = jit_data->package.get();
  ProcRuntime* runtime = jit_data->runtime.get();
  XLS_ASSIGN_OR_RETURN(Channel * cmd_channel,
                       package->GetChannel(kCmdChannelName));
  XLS_ASSIGN_OR_RETURN(Value command, CreateCommandValue(sample_data, encrypt));
  XLS_RETURN_IF_ERROR(
      runtime->queue_manager().GetQueue(cmd_channel).Write(command));

  // Then send all input data: the AAD followed by the message body.
  XLS_ASSIGN_OR_RETURN(Channel * data_in_channel,
                       package->GetChannel(kDataInChannelName));
  for (int i = 0; i < sample_data.aad.size(); i++) {
    XLS_ASSIGN_OR_RETURN(Value block_value, BlockToValue(sample_data.aad[i]));
    XLS_RETURN_IF_ERROR(
        runtime->queue_manager().GetQueue(data_in_channel).Write(block_value));
  }

  for (int i = 0; i < sample_data.input_data.size(); i++) {
    XLS_ASSIGN_OR_RETURN(Value block_value,
                         BlockToValue(sample_data.input_data[i]));
    XLS_RETURN_IF_ERROR(
        runtime->queue_manager().GetQueue(data_in_channel).Write(block_value));
  }

  // Tick the network until we have all results.
  Result result;
  XLS_ASSIGN_OR_RETURN(Channel * data_out_channel,
                       package->GetChannel(kDataOutChannelName));
  int msg_blocks_left = sample_data.input_data.size();
  while (true) {
    XLS_RETURN_IF_ERROR(runtime->Tick());
    std::optional<Value> maybe_value =
        runtime->queue_manager().GetQueue(data_out_channel).Read();
    if (!maybe_value.has_value()) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Block block, ValueToBlock(maybe_value.value()));
    if (msg_blocks_left > 0) {
      result.output_data.push_back(block);
      msg_blocks_left--;
    } else {
      result.auth_tag = block;
      break;
    }
  }

  return result;
}

static absl::StatusOr<Result> ReferenceEncrypt(const SampleData& sample) {
  int num_ptxt_blocks = sample.input_data.size();
  int num_aad_blocks = sample.aad.size();
  size_t max_result_size;
  if (sample.key_bits == 128) {
    max_result_size = num_ptxt_blocks * kBlockBytes +
                      EVP_AEAD_max_overhead(EVP_aead_aes_128_gcm());
  } else {
    max_result_size = num_ptxt_blocks * kBlockBytes +
                      EVP_AEAD_max_overhead(EVP_aead_aes_256_gcm());
  }

  auto ptxt_buffer = std::make_unique<uint8_t[]>(num_ptxt_blocks * kBlockBytes);
  for (int i = 0; i < num_ptxt_blocks; i++) {
    memcpy(&ptxt_buffer[i * kBlockBytes], sample.input_data[i].data(),
           kBlockBytes);
  }

  auto aad_buffer = std::make_unique<uint8_t[]>(num_aad_blocks * kBlockBytes);
  for (int i = 0; i < num_aad_blocks; i++) {
    memcpy(&aad_buffer[i * kBlockBytes], sample.aad[i].data(), kBlockBytes);
  }

  uint8_t local_init_vector[kInitVectorBytes];
  memcpy(local_init_vector, sample.init_vector.data(), kInitVectorBytes);

  auto result_buffer = std::make_unique<uint8_t[]>(max_result_size);
  size_t actual_result_size;
  int success = EVP_AEAD_CTX_seal(
      sample.openssl_ctxt, result_buffer.get(), &actual_result_size,
      max_result_size, local_init_vector, kInitVectorBytes, ptxt_buffer.get(),
      sample.input_data.size() * kBlockBytes, aad_buffer.get(),
      sample.aad.size() * kBlockBytes);
  if (!static_cast<bool>(success)) {
    return absl::UnknownError("Error reference-encrypting sample.");
  }

  Result result;
  result.output_data.resize(sample.input_data.size());
  for (int i = 0; i < sample.input_data.size(); i++) {
    memcpy(result.output_data[i].data(), &result_buffer[i * kBlockBytes],
           kBlockBytes);
  }
  memcpy(result.auth_tag.data(),
         &result_buffer[sample.input_data.size() * kBlockBytes], kBlockBytes);

  return result;
}

static bool CompareBlock(const Block& expected, const Block& actual,
                         std::string_view failure_msg, bool is_ciphertext) {
  for (int byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
    if (expected[byte_idx] != actual[byte_idx]) {
      std::cout << failure_msg << '\n';
      PrintFailure(expected, actual, byte_idx, is_ciphertext);
      return false;
    }
  }

  return true;
}

static absl::StatusOr<bool> RunSample(JitData* jit_data,
                                      const SampleData& sample_data) {
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "Input plaintext:\n"
              << FormatBlocks(sample_data.input_data, /*indent=*/4);
    LOG(INFO) << "Input AAD:\n" << FormatBlocks(sample_data.aad, /*indent=*/4);
  }

  XLS_ASSIGN_OR_RETURN(Result reference_encrypted,
                       ReferenceEncrypt(sample_data));
  XLS_ASSIGN_OR_RETURN(Result xls_encrypted,
                       XlsEncrypt(jit_data, sample_data, true));

  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "Reference ciphertext:\n"
              << FormatBlocks(reference_encrypted.output_data,
                              /*indent=*/4);
    LOG(INFO) << "Reference auth tag:\n    "
              << FormatBlock(reference_encrypted.auth_tag);
    LOG(INFO) << "XLS ciphertext:\n"
              << FormatBlocks(xls_encrypted.output_data, /*indent=*/4);
    LOG(INFO) << "XLS auth tag:\n    " << FormatBlock(xls_encrypted.auth_tag);
  }

  if (reference_encrypted.output_data.size() !=
      xls_encrypted.output_data.size()) {
    std::cout << "Reference & XLS ciphertext differed in sizes: "
              << "reference: " << reference_encrypted.output_data.size()
              << ", XLS: " << xls_encrypted.output_data.size() << '\n';
    return false;
  }

  for (int block_idx = 0; block_idx < reference_encrypted.output_data.size();
       block_idx++) {
    const Block& reference = reference_encrypted.output_data[block_idx];
    const Block& xls = xls_encrypted.output_data[block_idx];
    if (!CompareBlock(
            reference, xls,
            absl::StrCat("Error comparing to reference block ", block_idx, ":"),
            /*is_ciphertext=*/true)) {
      return false;
    }
  }

  if (!CompareBlock(
          reference_encrypted.auth_tag, xls_encrypted.auth_tag,
          "Error comparing encryption auth tags:", /*is_ciphertext=*/true)) {
    return false;
  }

  SampleData decrypt_sample{sample_data.openssl_ctxt,  sample_data.key,
                            sample_data.key_bits,      sample_data.init_vector,
                            xls_encrypted.output_data, sample_data.aad};
  XLS_ASSIGN_OR_RETURN(Result xls_decrypted,
                       XlsEncrypt(jit_data, decrypt_sample, false));
  if (xls_decrypted.output_data.size() != xls_encrypted.output_data.size()) {
    std::cout << "Input plaintext and XLS deciphered text differed in sizes: "
              << "input: " << sample_data.input_data.size()
              << ", decrypted: " << xls_decrypted.output_data.size() << '\n';
    return false;
  }

  for (int block_idx = 0; block_idx < sample_data.input_data.size();
       block_idx++) {
    const Block& plaintext = sample_data.input_data[block_idx];
    const Block& decrypted = xls_decrypted.output_data[block_idx];
    if (!CompareBlock(
            plaintext, decrypted,
            absl::StrCat("Error comparing to plaintext block ", block_idx, ":"),
            /*is_ciphertext=*/true)) {
      return false;
    }
  }

  if (!CompareBlock(
          xls_encrypted.auth_tag, xls_decrypted.auth_tag,
          "Error comparing decryption auth tags:", /*is_ciphertext=*/true)) {
    return false;
  }

  return true;
}

static absl::StatusOr<JitData> CreateJitData() {
  XLS_ASSIGN_OR_RETURN((std::unique_ptr<wrapped::AesGcm> aes_gcm),
                       wrapped::AesGcm::Create());
  auto [package, runtime] = wrapped::AesGcm::TakeRuntime(std::move(aes_gcm));
  return JitData{.package = std::move(package), .runtime = std::move(runtime)};
}

static absl::Status RunTest(int num_samples, int key_bits) {
  int key_bytes = key_bits / 8;
  SampleData sample_data;
  sample_data.key.fill(0);
  sample_data.key_bits = key_bits;
  XLS_ASSIGN_OR_RETURN(JitData jit_data, CreateJitData());

  absl::BitGen bitgen;
  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    for (int byte_idx = 0; byte_idx < key_bytes; byte_idx++) {
      sample_data.key[byte_idx] = absl::Uniform(bitgen, 0, 256);
    }

    EVP_AEAD_CTX* ref_ctx;
    if (key_bits == 128) {
      ref_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(), sample_data.key.data(),
                                 key_bytes, kTagBytes);
    } else {
      ref_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_256_gcm(), sample_data.key.data(),
                                 key_bytes, kTagBytes);
    }
    if (ref_ctx == nullptr) {
      return absl::InternalError("Unable to create reference context.");
    }
    auto cleanup = absl::Cleanup([&ref_ctx]() { EVP_AEAD_CTX_free(ref_ctx); });
    sample_data.openssl_ctxt = ref_ctx;

    int num_ptxt_blocks = absl::Uniform(bitgen, 1, kMaxPtxtBlocks + 1);
    sample_data.input_data.resize(num_ptxt_blocks);
    for (int block_idx = 0; block_idx < num_ptxt_blocks; block_idx++) {
      for (int byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
        sample_data.input_data[block_idx][byte_idx] =
            absl::Uniform(bitgen, 0, 256);
      }
    }

    int num_aad_blocks = absl::Uniform(bitgen, 1, kMaxAadBlocks + 1);
    sample_data.aad.resize(num_aad_blocks);
    for (int block_idx = 0; block_idx < num_aad_blocks; block_idx++) {
      for (int byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
        sample_data.aad[block_idx][byte_idx] = absl::Uniform(bitgen, 0, 256);
      }
    }

    for (int byte_idx = 0; byte_idx < kInitVectorBytes; byte_idx++) {
      sample_data.init_vector[byte_idx] = absl::Uniform(bitgen, 0, 256);
    }

    XLS_ASSIGN_OR_RETURN(bool proceed, RunSample(&jit_data, sample_data));
    if (!proceed) {
      std::cout << "Key      : " << FormatKey(sample_data.key) << '\n';
      std::cout << "IV       : " << FormatInitVector(sample_data.init_vector)
                << '\n';
      std::cout << "Plaintext: " << '\n'
                << FormatBlocks(sample_data.input_data, /*indent=*/4) << '\n';
      std::cout << "AAD: " << '\n'
                << FormatBlocks(sample_data.aad, /*indent=*/4) << '\n';
      return absl::InternalError(
          absl::StrCat("Testing failed at sample ", sample_idx, "."));
    }
  }

  std::cout << "AES-GCM: Successfully ran " << num_samples << " " << key_bits
            << "-bit samples." << '\n';

  return absl::OkStatus();
}

static absl::Status RealMain(int num_samples) {
  XLS_RETURN_IF_ERROR(RunTest(num_samples, /*key_bits=*/128));
  return RunTest(num_samples, /*key_bits=*/256);
}

}  // namespace xls::aes

int main(int argc, char** argv) {
  std::vector<std::string_view> args = xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::aes::RealMain(absl::GetFlag(FLAGS_num_samples)));
}

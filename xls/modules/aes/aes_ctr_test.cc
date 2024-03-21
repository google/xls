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

// Test to compare the outputs of a reference vs. XLS implementation of CTR
// mode using the AES block cipher.
#include <arpa/inet.h>

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "openssl/aes.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_parser.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/modules/aes/aes_test_common.h"

constexpr std::string_view kEncrypterIrPath = "xls/modules/aes/aes_ctr.ir";

ABSL_FLAG(int32_t, num_samples, 1000,
          "The number of (randomly-generated) blocks to test.");
ABSL_FLAG(bool, print_traces, false,
          "If true, print any trace! or trace_fmt! messages.");

namespace xls::aes {

struct SampleData {
  Key key;
  int key_bytes;
  // Held as a uN[96] in the DSLX.
  InitVector iv;
  std::vector<Block> input_blocks;
};

// Holds together all the data needed for ProcJit management.
struct JitData {
  std::unique_ptr<Package> package;
  Proc* proc;
  std::unique_ptr<SerialProcRuntime> proc_runtime;
};

// In the DSLX, the IV is treated as a uN[96], so we potentially need to swap
// its byte ordering as well.
static void IvToBuffer(const InitVector& iv,
                       std::array<uint8_t, kInitVectorBytes>* buffer) {
  for (int i = kInitVectorBytes - 1; i >= 0; i--) {
    buffer->data()[i] = iv[kInitVectorBytes - 1 - i];
  }
}

static void PrintTraceMessages(const InterpreterEvents& events) {
  if (absl::GetFlag(FLAGS_print_traces) && !events.trace_msgs.empty()) {
    std::cout << "Trace messages:" << '\n';
    for (const auto& tm : events.trace_msgs) {
      std::cout << " - " << tm.message << '\n';
    }
  }
}

static absl::StatusOr<std::vector<Block>> XlsEncrypt(
    const SampleData& sample_data, JitData* jit_data) {
  const std::string_view kCmdChannel = "aes_ctr__command_in";
  const std::string_view kInputDataChannel = "aes_ctr__ptxt_in";
  const std::string_view kOutputDataChannel = "aes_ctr__ctxt_out";

  // TODO(rspringer): Find a better way to collect queue IDs than IR inspection:
  // numbering is not guaranteed! Perhaps GetQueueByName?
  // TODO(rspringer): This is not safe! We're mapping between native and XLS
  // types! Find a better way to do this! We're manually inserting padding to
  // anticipate Clang/LLVM data layout! Bad!
  struct CtrCommand {
    uint32_t msg_bytes;
    Key key;
    uint32_t key_width;
    uint64_t padding_1;
    InitVector init_vector;
    uint32_t padding_2;
    uint32_t initial_ctr;
    uint32_t ctr_stride;
    uint64_t padding_3;
  };
  uint32_t num_blocks = sample_data.input_blocks.size();
  uint32_t num_bytes = num_blocks * kBlockBytes;
  CtrCommand command;
  command.msg_bytes = num_bytes;
  memcpy(command.key.data(), sample_data.key.data(), sample_data.key_bytes);
  command.key_width = sample_data.key_bytes == 16 ? 0 : 2;
  IvToBuffer(sample_data.iv, &command.init_vector);
  command.initial_ctr = 0;
  command.ctr_stride = 1;

  XLS_ASSIGN_OR_RETURN(Channel * cmd_channel,
                       jit_data->package->GetChannel(kCmdChannel));
  XLS_ASSIGN_OR_RETURN(JitChannelQueueManager * qm,
                       jit_data->proc_runtime->GetJitChannelQueueManager());
  JitChannelQueue& cmd_queue = qm->GetJitQueue(cmd_channel);
  cmd_queue.WriteRaw(reinterpret_cast<uint8_t*>(&command));

  XLS_ASSIGN_OR_RETURN(Channel * input_data_channel,
                       jit_data->package->GetChannel(kInputDataChannel));
  JitChannelQueue& input_data_queue = qm->GetJitQueue(input_data_channel);
  input_data_queue.WriteRaw(sample_data.input_blocks[0].data());

  XLS_RETURN_IF_ERROR(jit_data->proc_runtime->Tick());
  PrintTraceMessages(
      jit_data->proc_runtime->GetInterpreterEvents(jit_data->proc));

  // TODO(rspringer): Set this up to handle partial blocks.
  for (int i = 1; i < num_blocks; i++) {
    input_data_queue.WriteRaw(sample_data.input_blocks[i].data());
    XLS_RETURN_IF_ERROR(jit_data->proc_runtime->Tick());
    PrintTraceMessages(
        jit_data->proc_runtime->GetInterpreterEvents(jit_data->proc));
  }

  // Finally, read out the ciphertext.
  XLS_ASSIGN_OR_RETURN(Channel * output_data_channel,
                       jit_data->package->GetChannel(kOutputDataChannel));
  JitChannelQueue& output_data_queue = qm->GetJitQueue(output_data_channel);
  std::vector<Block> blocks;
  blocks.resize(num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    QCHECK(output_data_queue.ReadRaw(blocks[i].data()));
  }

  return blocks;
}

static std::vector<Block> ReferenceEncrypt(const SampleData& sample_data) {
  // Needed because the key and iv are modified during operation.
  uint8_t local_key[kMaxKeyBytes];
  memcpy(local_key, sample_data.key.data(), kMaxKeyBytes);

  uint8_t local_iv[kMaxKeyBytes];
  memcpy(local_iv, sample_data.iv.data(), kInitVectorBytes);

  // The BoringSSL implementation expects a 128-bit IV, instead of the 96-bit
  // one that XLS uses, so we pad it out with zeroes.
  for (int i = 0; i < 4; i++) {
    local_iv[kInitVectorBytes + i] = 0;
  }

  AES_KEY aes_key;
  QCHECK_EQ(AES_set_encrypt_key(local_key, sample_data.key_bytes * 8, &aes_key),
            0);

  int num_blocks = sample_data.input_blocks.size();
  auto input_buffer = std::make_unique<uint8_t[]>(num_blocks * kBlockBytes);
  for (int i = 0; i < num_blocks; i++) {
    memcpy(&input_buffer[i * kBlockBytes], sample_data.input_blocks[i].data(),
           kBlockBytes);
  }

  auto output_buffer = std::make_unique<uint8_t[]>(num_blocks * kBlockBytes);
  uint8_t ecount[kMaxKeyBytes] = {0};
  uint32_t num = 0;
  AES_ctr128_encrypt(input_buffer.get(), output_buffer.get(),
                     sample_data.input_blocks.size() * kBlockBytes, &aes_key,
                     local_iv, ecount, &num);

  std::vector<Block> output(num_blocks, Block{0});
  for (int i = 0; i < num_blocks; i++) {
    memcpy(output[i].data(), &output_buffer[i * kBlockBytes], kBlockBytes);
  }
  return output;
}

// Returns false on error (will terminate further runs).
static absl::StatusOr<bool> RunSample(JitData* jit_data,
                                      const SampleData& sample_data,
                                      absl::Duration* xls_encrypt_dur) {
  std::vector<Block> reference_ciphertext = ReferenceEncrypt(sample_data);

  absl::Time start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(std::vector<Block> xls_ciphertext,
                       XlsEncrypt(sample_data, jit_data));
  *xls_encrypt_dur += absl::Now() - start_time;

  VLOG(1) << "Input plaintext:\n"
          << FormatBlocks(sample_data.input_blocks, /*indent=*/4);
  VLOG(1) << "Reference ciphertext:\n"
          << FormatBlocks(reference_ciphertext, /*indent=*/4);
  VLOG(1) << "XLS ciphertext:\n" << FormatBlocks(xls_ciphertext, /*indent=*/4);

  if (reference_ciphertext.size() != xls_ciphertext.size()) {
    std::cout << "Error: XLS and reference ciphertexts differ in num blocks: "
              << "XLS: " << xls_ciphertext.size()
              << ", ref: " << reference_ciphertext.size() << '\n';
    return false;
  }

  for (int32_t block_idx = 0; block_idx < reference_ciphertext.size();
       block_idx++) {
    for (int32_t byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
      if (reference_ciphertext[block_idx][byte_idx] !=
          xls_ciphertext[block_idx][byte_idx]) {
        std::cout << "Error comparing block " << block_idx << ":" << '\n';
        PrintFailure(reference_ciphertext[block_idx], xls_ciphertext[block_idx],
                     byte_idx, /*ciphertext=*/true);
        return false;
      }
    }
  }

  // Now decryption!
  SampleData decryption_data;
  decryption_data.key = sample_data.key;
  decryption_data.key_bytes = sample_data.key_bytes;
  decryption_data.iv = sample_data.iv;
  decryption_data.input_blocks = xls_ciphertext;
  start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(std::vector<Block> xls_plaintext,
                       XlsEncrypt(decryption_data, jit_data));

  VLOG(1) << "XLS plaintext:\n" << FormatBlocks(xls_plaintext, /*indent=*/4);
  *xls_encrypt_dur += absl::Now() - start_time;
  if (sample_data.input_blocks.size() != xls_plaintext.size()) {
    std::cout << "Error: XLS decrypted plaintext and input plaintext differ "
              << "in num blocks XLS: " << xls_ciphertext.size()
              << ", ref: " << sample_data.input_blocks.size() << '\n';
    return false;
  }

  for (int32_t block_idx = 0; block_idx < sample_data.input_blocks.size();
       block_idx++) {
    for (int32_t byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
      if (sample_data.input_blocks[block_idx][byte_idx] !=
          xls_plaintext[block_idx][byte_idx]) {
        std::cout << "Error comparing block " << block_idx << ":" << '\n';
        PrintFailure(sample_data.input_blocks[block_idx],
                     xls_plaintext[block_idx], byte_idx,
                     /*ciphertext=*/false);
        return false;
      }
    }
  }

  return true;
}

static absl::StatusOr<JitData> CreateProcJit(std::string_view ir_path) {
  JitData jit_data;

  XLS_ASSIGN_OR_RETURN(std::filesystem::path full_ir_path,
                       GetXlsRunfilePath(ir_path));
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(full_ir_path));
  VLOG(1) << "Parsing IR.";
  XLS_ASSIGN_OR_RETURN(jit_data.package, Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(jit_data.proc,
                       jit_data.package->GetProc("__aes_ctr__aes_ctr_0_next"));

  VLOG(1) << "JIT compiling.";
  XLS_ASSIGN_OR_RETURN(jit_data.proc_runtime,
                       CreateJitSerialProcRuntime(jit_data.package.get()));
  VLOG(1) << "Created JIT!";

  return jit_data;
}

static absl::Status RunTest(int32_t num_samples, int32_t key_bits) {
  int key_bytes = key_bits / 8;
  SampleData sample_data;
  sample_data.key_bytes = key_bytes;
  memset(sample_data.iv.data(), 0, sizeof(sample_data.iv));

  XLS_ASSIGN_OR_RETURN(JitData encrypt_jit_data,
                       CreateProcJit(kEncrypterIrPath));

  absl::BitGen bitgen;
  absl::Duration xls_encrypt_dur;
  for (int32_t i = 0; i < num_samples; i++) {
    for (int32_t j = 0; j < key_bytes; j++) {
      sample_data.key[j] = absl::Uniform(bitgen, 0, 256);
    }

    // TODO(rspringer): Support zero blocks!
    // int num_blocks = absl::Uniform(bitgen, 1, 16);
    int num_blocks = 1;
    sample_data.input_blocks.resize(num_blocks);
    for (int32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
      for (int32_t byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
        sample_data.input_blocks[block_idx][byte_idx] =
            absl::Uniform(bitgen, 0, 256);
      }
    }
    for (int32_t j = 0; j < kInitVectorBytes; j++) {
      sample_data.iv[j] = absl::Uniform(bitgen, 0, 256);
    }

    XLS_ASSIGN_OR_RETURN(bool proceed, RunSample(&encrypt_jit_data, sample_data,
                                                 &xls_encrypt_dur));
    if (!proceed) {
      std::cout << "Key      : " << FormatKey(sample_data.key) << '\n';
      std::cout << "IV       : " << FormatInitVector(sample_data.iv) << '\n';
      std::cout << "Plaintext: " << '\n'
                << FormatBlocks(sample_data.input_blocks, /*indent=*/4) << '\n';
      return absl::InternalError(
          absl::StrCat("Testing failed at sample ", i, "."));
    }
  }

  std::cout << "AES-CTR: Successfully ran " << num_samples << " samples."
            << '\n';
  std::cout << "AES-CTR: Avg. XLS encryption time: "
            << xls_encrypt_dur / num_samples << '\n';

  return absl::OkStatus();
}

static absl::Status RealMain(int num_samples) {
  XLS_RETURN_IF_ERROR(RunTest(num_samples, /*key_bits=*/128));
  return RunTest(num_samples, /*key_bits=*/256);
}

}  // namespace xls::aes

int32_t main(int32_t argc, char** argv) {
  std::vector<std::string_view> args = xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::aes::RealMain(absl::GetFlag(FLAGS_num_samples)));
}

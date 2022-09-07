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
#include <arpa/inet.h>

#include <array>
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
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_parser.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_jit.h"
#include "xls/modules/aes/aes_test_common.h"

constexpr absl::string_view kEncrypterIrPath = "xls/modules/aes/aes_128_ctr.ir";

ABSL_FLAG(int32_t, num_samples, 1000,
          "The number of (randomly-generated) blocks to test.");
ABSL_FLAG(bool, print_traces, false,
          "If true, print any trace! or trace_fmt! messages.");

namespace xls::aes {

constexpr int kKeyBits = 128;
constexpr int kKeyBytes = kKeyBits / 8;
using Key = std::array<uint8_t, kKeyBytes>;

struct SampleData {
  Key key;
  // Held as a uN[96] in the DSLX.
  InitVector iv;
  std::vector<Block> input_blocks;
};

// Holds together all the data needed for ProcJit management.
struct JitData {
  std::unique_ptr<Package> package;
  std::unique_ptr<ProcJit> jit;
  std::unique_ptr<JitChannelQueueManager> queue_mgr;
};

// In the DSLX, the IV is treated as a uN[96], so we potentially need to swap
// its byte ordering as well.
void IvToBuffer(const InitVector& iv,
                std::array<uint8_t, kInitVectorBytes>* buffer) {
  for (int i = kInitVectorBytes - 1; i >= 0; i--) {
    buffer->data()[i] = iv[kInitVectorBytes - 1 - i];
  }
}

void PrintTraceMessages(const InterpreterEvents& events) {
  if (absl::GetFlag(FLAGS_print_traces) && !events.trace_msgs.empty()) {
    std::cout << "Trace messages:" << std::endl;
    for (const auto& tm : events.trace_msgs) {
      std::cout << " - " << tm << std::endl;
    }
  }
}

absl::StatusOr<std::vector<Block>> XlsEncrypt(const SampleData& sample_data,
                                              JitData* jit_data) {
  const absl::string_view kCmdChannel = "aes_128_ctr__command_in";
  const absl::string_view kInputDataChannel = "aes_128_ctr__ptxt_in";
  const absl::string_view kOutputDataChannel = "aes_128_ctr__ctxt_out";

  // Set initial state: step, command, ctr, and blocks_left.
  // TODO(rspringer): Get these sizes from the DSLX C++ type transpiler, then
  // consider using ConvertToXlsValue() to create the aggregate Value.
  std::vector<Value> state;
  // Step.
  state.push_back(Value(UBits(0, 1)));
  // Command.
  std::vector<Value> initial_command_elements;
  {
    // msg_bytes, key, and init_vector.
    initial_command_elements.push_back(Value(UBits(0, 32)));
    std::vector<Value> key_elements(4, Value(UBits(0, 32)));
    XLS_ASSIGN_OR_RETURN(Value key_value, Value::Array(key_elements));
    initial_command_elements.push_back(key_value);
    initial_command_elements.push_back(Value(UBits(0, 96)));
    initial_command_elements.push_back(Value(UBits(0, 32)));
  }
  state.push_back(Value::Tuple(initial_command_elements));
  // Ctr.
  state.push_back(Value(UBits(0, 32)));
  // Blocks left.
  state.push_back(Value(UBits(0, 32)));

  // TODO(rspringer): Find a better way to collect queue IDs than IR inspection:
  // numbering is not guaranteed! Perhaps GetQueueByName?
  // TODO(rspringer): This is not safe! We're mapping between native and XLS
  // types! Find a better way to do this! We're manually inserting padding to
  // anticipate Clang/LLVM data layout! Bad!
  struct CtrCommand {
    uint32_t msg_bytes;
    std::array<uint32_t, 4> key;
    uint32_t padding;
    std::array<uint8_t, kInitVectorBytes> init_vector;
    uint32_t padding_2;
    uint32_t initial_ctr;
    uint32_t padding_3;
  };
  uint32_t num_blocks = sample_data.input_blocks.size();
  uint32_t num_bytes = num_blocks * kBlockBytes;
  CtrCommand command;
  command.msg_bytes = num_bytes;
  KeyToBuffer<kKeyBytes, 4>(sample_data.key, &command.key);
  IvToBuffer(sample_data.iv, &command.init_vector);
  command.initial_ctr = 0;

  XLS_ASSIGN_OR_RETURN(Channel * cmd_channel,
                       jit_data->package->GetChannel(kCmdChannel));
  int cmd_channel_id = cmd_channel->id();
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * cmd_queue,
                       jit_data->queue_mgr->GetQueueById(cmd_channel_id));
  cmd_queue->Send(reinterpret_cast<uint8_t*>(&command), sizeof(CtrCommand));

  XLS_ASSIGN_OR_RETURN(Channel * input_data_channel,
                       jit_data->package->GetChannel(kInputDataChannel));
  int input_data_channel_id = input_data_channel->id();
  XLS_ASSIGN_OR_RETURN(
      JitChannelQueue * input_data_queue,
      jit_data->queue_mgr->GetQueueById(input_data_channel_id));
  input_data_queue->Send(sample_data.input_blocks[0].data(), kBlockBytes);

  // TODO(rspringer): Can we eliminate the need for this tuple wrap?
  Value bar = Value::Tuple({state});
  XLS_ASSIGN_OR_RETURN(InterpreterResult<std::vector<Value>> run_result,
                       jit_data->jit->Run({bar}, jit_data));
  state = run_result.value;
  PrintTraceMessages(run_result.events);

  // TODO(rspringer): Set this up to handle partial blocks.
  for (int i = 1; i < num_blocks; i++) {
    input_data_queue->Send(sample_data.input_blocks[i].data(), kBlockBytes);
    XLS_ASSIGN_OR_RETURN(InterpreterResult<std::vector<Value>> run_result,
                         jit_data->jit->Run(state, jit_data));
    state = run_result.value;
    PrintTraceMessages(run_result.events);
  }

  // Finally, read out the ciphertext.
  XLS_ASSIGN_OR_RETURN(Channel * output_data_channel,
                       jit_data->package->GetChannel(kOutputDataChannel));
  int output_data_channel_id = output_data_channel->id();
  XLS_ASSIGN_OR_RETURN(
      JitChannelQueue * output_data_queue,
      jit_data->queue_mgr->GetQueueById(output_data_channel_id));
  std::vector<Block> blocks;
  blocks.resize(num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    XLS_QCHECK(output_data_queue->Recv(blocks[i].data(), kBlockBytes));
  }

  return blocks;
}

std::vector<Block> ReferenceEncrypt(const SampleData& sample_data) {
  // Needed because the key and iv are modified during operation.
  uint8_t local_key[kKeyBytes];
  memcpy(local_key, sample_data.key.data(), kKeyBytes);
  uint8_t local_iv[kKeyBytes];
  memcpy(local_iv, sample_data.iv.data(), kInitVectorBytes);
  // The BoringSSL implementation expects a 128-bit IV, instead of the 96-bit
  // one that XLS uses, so we pad it out with zeroes.
  for (int i = 0; i < 4; i++) {
    local_iv[kInitVectorBytes + i] = 0;
  }

  uint8_t ecount[kKeyBytes] = {0};

  AES_KEY aes_key;
  uint32_t num = 0;
  XLS_QCHECK_EQ(AES_set_encrypt_key(local_key, kKeyBits, &aes_key), 0);

  int num_blocks = sample_data.input_blocks.size();
  auto input_buffer = std::make_unique<uint8_t[]>(num_blocks * kBlockBytes);
  for (int i = 0; i < num_blocks; i++) {
    memcpy(&input_buffer[i * kBlockBytes], sample_data.input_blocks[i].data(),
           kBlockBytes);
  }

  auto output_buffer = std::make_unique<uint8_t[]>(num_blocks * kBlockBytes);
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
absl::StatusOr<bool> RunSample(JitData* jit_data, const SampleData& sample_data,
                               absl::Duration* xls_encrypt_dur) {
  std::vector<Block> reference_ciphertext = ReferenceEncrypt(sample_data);

  absl::Time start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(std::vector<Block> xls_ciphertext,
                       XlsEncrypt(sample_data, jit_data));
  *xls_encrypt_dur += absl::Now() - start_time;

  XLS_VLOG(1) << "Input plaintext:\n"
              << FormatBlocks(sample_data.input_blocks, /*indent=*/4);
  XLS_VLOG(1) << "Reference ciphertext:\n"
              << FormatBlocks(reference_ciphertext, /*indent=*/4);
  XLS_VLOG(1) << "XLS ciphertext:\n"
              << FormatBlocks(xls_ciphertext, /*indent=*/4);

  if (reference_ciphertext.size() != xls_ciphertext.size()) {
    std::cout << "Error: XLS and reference ciphertexts differ in num blocks: "
              << "XLS: " << xls_ciphertext.size()
              << ", ref: " << reference_ciphertext.size() << std::endl;
    return false;
  }

  for (int32_t block_idx = 0; block_idx < reference_ciphertext.size();
       block_idx++) {
    for (int32_t byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
      if (reference_ciphertext[block_idx][byte_idx] !=
          xls_ciphertext[block_idx][byte_idx]) {
        std::cout << "Error comparing block " << block_idx << ":" << std::endl;
        PrintFailure(reference_ciphertext[block_idx], xls_ciphertext[block_idx],
                     byte_idx, /*ciphertext=*/true);
        return false;
      }
    }
  }

  // Now decryption!
  SampleData decryption_data;
  decryption_data.key = sample_data.key;
  decryption_data.iv = sample_data.iv;
  decryption_data.input_blocks = xls_ciphertext;
  start_time = absl::Now();
  XLS_ASSIGN_OR_RETURN(std::vector<Block> xls_plaintext,
                       XlsEncrypt(decryption_data, jit_data));

  XLS_VLOG(1) << "XLS plaintext:\n"
              << FormatBlocks(xls_plaintext, /*indent=*/4);
  *xls_encrypt_dur += absl::Now() - start_time;
  if (sample_data.input_blocks.size() != xls_plaintext.size()) {
    std::cout << "Error: XLS decrypted plaintext and input plaintext differ "
              << "in num blocks XLS: " << xls_ciphertext.size()
              << ", ref: " << sample_data.input_blocks.size() << std::endl;
    return false;
  }

  for (int32_t block_idx = 0; block_idx < sample_data.input_blocks.size();
       block_idx++) {
    for (int32_t byte_idx = 0; byte_idx < kBlockBytes; byte_idx++) {
      if (sample_data.input_blocks[block_idx][byte_idx] !=
          xls_plaintext[block_idx][byte_idx]) {
        std::cout << "Error comparing block " << block_idx << ":" << std::endl;
        PrintFailure(sample_data.input_blocks[block_idx],
                     xls_ciphertext[block_idx], byte_idx,
                     /*ciphertext=*/false);
        return false;
      }
    }
  }

  return true;
}

absl::StatusOr<JitData> CreateProcJit(absl::string_view ir_path,
                                      ProcJit::RecvFnT recv_fn,
                                      ProcJit::SendFnT send_fn) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path full_ir_path,
                       GetXlsRunfilePath(ir_path));
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(full_ir_path));
  XLS_VLOG(1) << "Parsing IR.";
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(Proc * proc,
                       package->GetProc("__aes_128_ctr__aes_128_ctr_0_next"));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<JitChannelQueueManager> queue_mgr,
                       JitChannelQueueManager::Create(package.get()));

  XLS_VLOG(1) << "JIT compiling.";
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(proc, queue_mgr.get(), recv_fn, send_fn));
  XLS_VLOG(1) << "Created JIT!";
  JitData jit_data;
  jit_data.jit = std::move(jit);
  jit_data.package = std::move(package);
  jit_data.queue_mgr = std::move(queue_mgr);
  return jit_data;
}

bool EncoderJitRecvFn(JitChannelQueue* queue, Receive* recv, uint8_t* buffer,
                      int64_t buf_sz, void* user_data) {
  return queue->Recv(buffer, buf_sz);
}

void EncoderJitSendFn(JitChannelQueue* queue, Send* send, uint8_t* buffer,
                      int64_t buf_sz, void* user_data) {
  return queue->Send(buffer, buf_sz);
}

absl::Status RealMain(int32_t num_samples) {
  SampleData sample_data;
  memset(sample_data.iv.data(), 0, sizeof(sample_data.iv));

  XLS_ASSIGN_OR_RETURN(
      JitData encrypt_jit_data,
      CreateProcJit(kEncrypterIrPath, &EncoderJitRecvFn, &EncoderJitSendFn));

  absl::BitGen bitgen;
  absl::Duration xls_encrypt_dur;
  for (int32_t i = 0; i < num_samples; i++) {
    for (int32_t j = 0; j < kKeyBytes; j++) {
      sample_data.key[j] = absl::Uniform(bitgen, 0, 256);
    }

    // TODO(rspringer): Support zero blocks!
    int num_blocks = absl::Uniform(bitgen, 1, 16);
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
      std::cout << "Key      : " << FormatKey<kKeyBytes>(sample_data.key)
                << std::endl;
      std::cout << "IV       : " << FormatInitVector(sample_data.iv)
                << std::endl;
      std::cout << "Plaintext: " << std::endl
                << FormatBlocks(sample_data.input_blocks, /*indent=*/4)
                << std::endl;
      return absl::InternalError(
          absl::StrCat("Testing failed at sample ", i, "."));
    }
  }

  std::cout << "AES-CTR: Successfully ran " << num_samples << " samples."
            << std::endl;
  std::cout << "AES-CTR: Avg. XLS encryption time: "
            << xls_encrypt_dur / num_samples << std::endl;

  return absl::OkStatus();
}

}  // namespace xls::aes

int32_t main(int32_t argc, char** argv) {
  std::vector<absl::string_view> args = xls::InitXls(argv[0], argc, argv);
  XLS_QCHECK_OK(xls::aes::RealMain(absl::GetFlag(FLAGS_num_samples)));
  return 0;
}

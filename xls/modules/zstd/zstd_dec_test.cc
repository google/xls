// Copyright 2024 The XLS Authors
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
//

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "external/zstd/lib/zstd.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/modules/zstd/data_generator.h"

namespace xls {
namespace {

class ZstdDecodedPacket {
 public:
  static absl::StatusOr<ZstdDecodedPacket> MakeZstdDecodedPacket(
      const Value& packet) {
    // Expect tuple
    XLS_RET_CHECK(packet.IsTuple());
    // Expect exactly 3 fields
    XLS_RET_CHECK(packet.size() == 3);
    for (int i = 0; i < 3; i++) {
      // Expect fields to be Bits
      XLS_RET_CHECK(packet.element(i).IsBits());
      // All fields must fit in 64 bits
      XLS_RET_CHECK(packet.element(i).bits().FitsInUint64());
    }

    std::vector<uint8_t> data = packet.element(0).bits().ToBytes();
    absl::StatusOr<uint64_t> len = packet.element(1).bits().ToUint64();
    XLS_RET_CHECK(len.ok());
    uint64_t length = *len;
    bool last = packet.element(2).bits().IsOne();

    return ZstdDecodedPacket(data, length, last);
  }

  std::vector<uint8_t>& GetData() { return data; }

  uint64_t GetLength() { return length; }

  bool IsLast() { return last; }

  std::string ToString() const {
    std::stringstream s;
    for (int j = 0; j < sizeof(uint64_t) && j < data.size(); j++) {
      s << "0x" << std::setw(2) << std::setfill('0') << std::right << std::hex
        << static_cast<unsigned int>(data[j]) << std::dec << ", ";
    }
    return s.str();
  }

 private:
  ZstdDecodedPacket(std::vector<uint8_t> data, uint64_t length, bool last)
      : data(std::move(data)), length(length), last(last) {}

  std::vector<uint8_t> data;
  uint64_t length;
  bool last;
};

class ZstdDecoderTest : public ::testing::Test {
 public:
  void SetUp() override {
    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                             xls::GetXlsRunfilePath(this->kIrFile));
    XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text,
                             xls::GetFileContents(ir_path));
    XLS_ASSERT_OK_AND_ASSIGN(this->package, xls::Parser::ParsePackage(ir_text));
    XLS_ASSERT_OK_AND_ASSIGN(this->interpreter,
                             CreateJitSerialProcRuntime(this->package.get()));

    auto& queue_manager = this->interpreter->queue_manager();
    XLS_ASSERT_OK_AND_ASSIGN(
        this->recv_queue, queue_manager.GetQueueByName(this->kRecvChannelName));
    XLS_ASSERT_OK_AND_ASSIGN(
        this->send_queue, queue_manager.GetQueueByName(this->kSendChannelName));
  }

  void PrintTraceMessages(const std::string& pname) {
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, this->package->GetProc(pname));
    const InterpreterEvents& events =
        this->interpreter->GetInterpreterEvents(proc);

    if (!events.trace_msgs.empty()) {
      for (const auto& tm : events.trace_msgs) {
        LOG(INFO) << "[TRACE] " << tm.message << "\n";
      }
    }
  }

  const std::string_view kProcName = "__zstd_dec__ZstdDecoderTest_0_next";
  const std::string_view kRecvChannelName = "zstd_dec__output_s";
  const std::string_view kSendChannelName = "zstd_dec__input_r";

  const std::string_view kIrFile = "xls/modules/zstd/zstd_dec_test.ir";

  std::unique_ptr<Package> package;
  std::unique_ptr<SerialProcRuntime> interpreter;
  ChannelQueue *recv_queue, *send_queue;

  void PrintVector(absl::Span<uint8_t> vec) {
    for (int i = 0; i < vec.size(); i += 8) {
      LOG(INFO) << "0x" << std::hex << std::setw(3) << std::left << i
                << std::dec << ": ";
      for (int j = 0; j < sizeof(uint64_t) && (i + j) < vec.size(); j++) {
        LOG(INFO) << std::setfill('0') << std::setw(2) << std::hex
                  << static_cast<unsigned int>(vec[i + j]) << std::dec << " ";
      }
      LOG(INFO) << "\n";
    }
  }

  void DecompressWithLibZSTD(std::vector<uint8_t> encoded_frame,
                             std::vector<uint8_t>& decoded_frame) {
    size_t buff_out_size = ZSTD_DStreamOutSize();
    uint8_t* const buff_out = new uint8_t[buff_out_size];

    ZSTD_DCtx* const dctx = ZSTD_createDCtx();
    EXPECT_FALSE(dctx == nullptr);

    void* const frame = static_cast<void*>(encoded_frame.data());
    size_t const frame_size = encoded_frame.size();
    // Put the whole frame in the buffer
    ZSTD_inBuffer input_buffer = {frame, frame_size, 0};

    while (input_buffer.pos < input_buffer.size) {
      ZSTD_outBuffer output_buffer = {buff_out, buff_out_size, 0};
      size_t decomp_result =
          ZSTD_decompressStream(dctx, &output_buffer, &input_buffer);
      bool decomp_success = ZSTD_isError(decomp_result) != 0u;
      EXPECT_FALSE(decomp_success);

      // Append output buffer contents to output vector
      decoded_frame.insert(
          decoded_frame.end(), static_cast<uint8_t*>(output_buffer.dst),
          (static_cast<uint8_t*>(output_buffer.dst) + output_buffer.pos));

      EXPECT_TRUE(decomp_result == 0 && output_buffer.pos < output_buffer.size);
    }

    ZSTD_freeDCtx(dctx);
    delete[] buff_out;
  }

  void ParseAndCompareWithZstd(std::vector<uint8_t> frame) {
    std::vector<uint8_t> lib_decomp;
    DecompressWithLibZSTD(frame, lib_decomp);
    size_t lib_decomp_size = lib_decomp.size();
    std::cerr << "lib_decomp_size: " << lib_decomp_size << "\n";

    std::vector<uint8_t> sim_decomp;
    size_t sim_decomp_size_words =
        (lib_decomp_size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    size_t sim_decomp_size_bytes =
        (lib_decomp_size + sizeof(uint64_t) - 1) * sizeof(uint64_t);
    sim_decomp.reserve(sim_decomp_size_bytes);

    // Send compressed frame to decoder simulation
    for (int i = 0; i < frame.size(); i += 8) {
      // Pad packet w/ zeros to match the frame size expected by the design.
      std::array<uint8_t, 8> packet_data = {};
      auto frame_packet_begin = frame.begin() + i;
      auto frame_packet_end = frame_packet_begin + 8 < frame.end()
                                  ? frame_packet_begin + 8
                                  : frame.end();
      std::copy(frame_packet_begin, frame_packet_end, packet_data.begin());
      auto span = absl::MakeSpan(packet_data.data(), 8);
      auto value = Value(Bits::FromBytes(span, 64));
      XLS_EXPECT_OK(this->send_queue->Write(value));
      XLS_EXPECT_OK(this->interpreter->Tick());
    }
    PrintTraceMessages("__zstd_dec__ZstdDecoderTest_0_next");

    // Tick decoder simulation until we get expected amount of output data
    // batches on output channel queue
    std::optional<int64_t> ticks_timeout = std::nullopt;
    absl::flat_hash_map<Channel*, int64_t> output_counts = {
        {this->recv_queue->channel(), sim_decomp_size_words}};
    XLS_EXPECT_OK(
        this->interpreter->TickUntilOutput(output_counts, ticks_timeout));

    // Read decompressed data from output channel queue
    for (int i = 0; i < sim_decomp_size_words; i++) {
      auto read_value = this->recv_queue->Read();
      EXPECT_EQ(read_value.has_value(), true);
      auto packet =
          ZstdDecodedPacket::MakeZstdDecodedPacket(read_value.value());
      XLS_EXPECT_OK(packet);
      auto word_vec = packet->GetData();
      auto valid_length = packet->GetLength() / CHAR_BIT;
      std::copy(begin(word_vec), begin(word_vec) + valid_length,
                back_inserter(sim_decomp));
    }

    EXPECT_EQ(lib_decomp_size, sim_decomp.size());
    for (int i = 0; i < lib_decomp_size; i++) {
      EXPECT_EQ(lib_decomp[i], sim_decomp[i]);
    }
  }
};

/* TESTS */

TEST(ZstdLib, Version) { ASSERT_EQ(ZSTD_VERSION_STRING, "1.5.6"); }

TEST_F(ZstdDecoderTest, ParseFrameWithRawBlocks) {
  int seed = 3;  // Arbitrary seed value for small ZSTD frame
  auto frame = zstd::GenerateFrame(seed, zstd::BlockType::RAW);
  EXPECT_TRUE(frame.ok());
  this->ParseAndCompareWithZstd(frame.value());
}

TEST_F(ZstdDecoderTest, ParseFrameWithRleBlocks) {
  int seed = 3;  // Arbitrary seed value for small ZSTD frame
  auto frame = zstd::GenerateFrame(seed, zstd::BlockType::RLE);
  EXPECT_TRUE(frame.ok());
  this->ParseAndCompareWithZstd(frame.value());
}

class ZstdDecoderSeededTest : public ZstdDecoderTest,
                              public ::testing::WithParamInterface<uint32_t> {
 public:
  static const uint32_t seed_generator_start = 0;
  static const uint32_t random_frames_count = 100;
};

// Test `random_frames_count` instances of randomly generated valid
// frames, generated with `decodecorpus` tool.

TEST_P(ZstdDecoderSeededTest, ParseMultipleFramesWithRawBlocks) {
  auto seed = GetParam();
  auto frame = zstd::GenerateFrame(seed, zstd::BlockType::RAW);
  EXPECT_TRUE(frame.ok());
  this->ParseAndCompareWithZstd(frame.value());
}

TEST_P(ZstdDecoderSeededTest, ParseMultipleFramesWithRleBlocks) {
  auto seed = GetParam();
  auto frame = zstd::GenerateFrame(seed, zstd::BlockType::RLE);
  EXPECT_TRUE(frame.ok());
  this->ParseAndCompareWithZstd(frame.value());
}

INSTANTIATE_TEST_SUITE_P(
    ZstdDecoderSeededTest, ZstdDecoderSeededTest,
    ::testing::Range<uint32_t>(ZstdDecoderSeededTest::seed_generator_start,
                               ZstdDecoderSeededTest::seed_generator_start +
                                   ZstdDecoderSeededTest::random_frames_count));

}  // namespace
}  // namespace xls

// Copyright 2023 The XLS Authors
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

#include "xls/simulation/generic/streammanager.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/bits_util.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/generic/common.h"
#include "xls/simulation/generic/istream.h"
#include "xls/simulation/generic/istream_mock.h"

namespace xls::simulation::generic {

namespace {

std::string ToHex(uint64_t value) {
  return "0x" + absl::StrCat(absl::Hex(value, absl::kZeroPad16));
}

uint64_t BitsToBytesFloor(uint64_t bits) { return bits >> 3; }

class StreamManagerTestParams {
 public:
  uint64_t base;
  uint64_t offset1;
  uint64_t offset2;
  uint64_t payload64;
};

class StreamManagerTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<StreamManagerTestParams> {
 protected:
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(StreamManagerTest, AddStreams) {
  // DESCRIPTION: Test registering streams. See if overlaps are reported
  // correctly.

  const uint64_t s1_width_bits = 128;  // 16 bytes
  const uint64_t s1_offset = 0;
  const uint64_t s2_width_bits = 128;  // 16 bytes
  const uint64_t s2_offset = 24;
  const uint64_t s3_width_bits = 128;  // 16 bytes
  const uint64_t s3_offset = 32;

  auto CheckRangeEnd = [](const StreamManager& sm, uint64_t last_offset,
                          uint64_t last_width_bytes) {
    uint64_t end =
        last_offset + last_width_bytes + StreamManager::ctrl_reg_width_bytes_;
    EXPECT_EQ(sm.InRange(end), false);
    EXPECT_EQ(sm.InRange(end - 1), true);
  };

  auto sm1 = StreamManager::Build(0x0, [&](const auto& RegisterStream) {
    RegisterStream(s1_offset, new IStreamMock<s1_width_bits, true>());
    RegisterStream(s2_offset, new IStreamMock<s2_width_bits, true>());
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm1);
  CheckRangeEnd(*sm1, s2_offset, BitsToBytesFloor(s2_width_bits));

  auto sm2 = StreamManager::Build(0x0, [&](const auto& RegisterStream) {
    RegisterStream(s1_offset, new IStreamMock<s1_width_bits, true>());
    RegisterStream(s2_offset, new IStreamMock<s2_width_bits, true>());
    RegisterStream(s3_offset, new IStreamMock<s3_width_bits, true>());
    return absl::OkStatus();
  });
  EXPECT_THAT(
      sm2,
      xls::status_testing::StatusIs(
          absl::StatusCode::kInternal,
          testing::HasSubstr(
              "Overlapping streams. Channel to be added at 0x0000000000000020 "
              "overlaps with a channel added at 0x0000000000000018")));
}

TEST_F(StreamManagerTest, StreamAccess) {
  const uint64_t s_width_bits = 32;
  const uint8_t stream_count = 255;

  // Register 256 4-byte streams (reverse order, just to make sure that
  // insertion works fine)
  auto sm_status = StreamManager::Build(0x0, [=](const auto& RegisterStream) {
    for (int64_t i = stream_count; i != -1; --i) {
      uint64_t offset = i * (StreamManager::ctrl_reg_width_bytes_ +
                             BitsToBytesFloor(s_width_bits));
      IStreamMock<s_width_bits, true>* stream =
          new IStreamMock<s_width_bits, true>();
      for (int j = 0; j < 4; ++j) {
        // Make those streams identifiable
        XLS_EXPECT_OK(stream->SetPayloadData8(j, static_cast<uint8_t>(i)));
      }
      RegisterStream(offset, stream);
    }
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm_status);
  StreamManager sm = std::move(sm_status.value());

  absl::StatusOr<uint8_t> status;

  uint64_t stream_bytes =
      BitsToBytesFloor(s_width_bits) + StreamManager::ctrl_reg_width_bytes_;

  for (uint64_t i = 0; i < stream_count * stream_bytes; ++i) {
    XLS_LOG(INFO) << "Checking access to stream " << (i / stream_bytes)
                  << ", byte " << (i % stream_bytes);
    XLS_EXPECT_OK(status = sm.ReadU8AtAddress(i));
    if (i % 5 < StreamManager::ctrl_reg_width_bytes_) {
      continue;
    }  // Skip the control register
    EXPECT_EQ(status.value(), i / stream_bytes);
  }
}

// 64 bits of test data for reads and writes
constexpr uint64_t Payload64(const char* str) {
  uint64_t payload = 0;
  for (int i = 0; i < 8; ++i) {
    payload |= static_cast<uint64_t>(str[i]) << BytesToBits(i);
  }
  return payload;
}

// Masks `msbs` most-significant-bytes
uint64_t Payload64Mask(int msbs) {
  if (msbs == 0) {
    return 0x0;
  }
  if (msbs == 64) {
    return -0x1;
  }
  return ~xls::Mask(BytesToBits(8 - msbs));
}
// Returns `msbs` most-significant-bytes
uint64_t Payload64Tail(uint64_t payload64, int msbs) {
  uint64_t pl = (payload64 & Payload64Mask(msbs)) >>
                (BytesToBits(sizeof(uint64_t) - msbs));
  return pl;
}

TEST_P(StreamManagerTest, ReadFromASingleStream) {
  // DESCRIPTION: Test reading from a stream. Offset must be within the streams
  // range.

  const StreamManagerTestParams params = GetParam();

  const uint64_t s1_width = 128;  // 16 bytes
  const uint64_t s1_offset = params.offset1;
  const uint64_t payload64 = params.payload64;
  const uint64_t payload_offset = 8;

  absl::StatusOr<uint64_t> status;

  IStreamMock<s1_width, true>* stream1 = new IStreamMock<s1_width, true>();
  auto sm = StreamManager::Build(params.base, [&](const auto& RegisterStream) {
    RegisterStream(s1_offset, stream1);
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm);

  // Setup stream content ({64'h0, Payload64})
  XLS_EXPECT_OK(stream1->SetPayloadData64(0, 0));
  XLS_EXPECT_OK(stream1->SetPayloadData64(payload_offset, payload64));

  // Read at all possible different offsets
  for (int i = 0; i < 8; ++i) {
    uint64_t read_offset = params.base + s1_offset + payload_offset +
                           StreamManager::ctrl_reg_width_bytes_ - i;
    uint64_t shifted_payload = payload64 << BytesToBits(i);

    XLS_EXPECT_OK(status = sm->ReadU8AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload & 0xff);

    XLS_EXPECT_OK(status = sm->ReadU16AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload & 0xffff);

    XLS_EXPECT_OK(status = sm->ReadU32AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload & 0xffffffff);

    XLS_EXPECT_OK(status = sm->ReadU64AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload);
  }
  for (int i = 1; i < 8; ++i) {
    uint64_t read_offset = params.base + s1_offset + payload_offset +
                           StreamManager::ctrl_reg_width_bytes_ + i;
    uint64_t shifted_payload = payload64 >> BytesToBits(i);

    XLS_EXPECT_OK(status = sm->ReadU8AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload & 0xff);

    XLS_EXPECT_OK(status = sm->ReadU16AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload & 0xffff);

    XLS_EXPECT_OK(status = sm->ReadU32AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload & 0xffffffff);

    XLS_EXPECT_OK(status = sm->ReadU64AtAddress(read_offset));
    EXPECT_EQ(status.value(), shifted_payload);
  }
}

TEST_F(StreamManagerTest, ReadCtrl) {
  // DESCRIPTION: Test reading the control register.

  const uint64_t r_width = 128;  // 16 bytes
  const uint64_t r_offset = 0;
  const uint64_t w_width = 128;  // 16 bytes
  const uint64_t w_offset = 24;

  auto sm = StreamManager::Build(0x0, [&](const auto& RegisterStream) {
    RegisterStream(r_offset, new IStreamMock<r_width, true>);
    RegisterStream(w_offset, new IStreamMock<w_width, false>);
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm);

  absl::StatusOr<uint8_t> res;

  // Test direction
  XLS_EXPECT_OK(res = sm->ReadU8AtAddress(r_offset));
  uint8_t expect = StreamManager::ctrl_reg_DIR_;
  EXPECT_EQ(res.value(), expect);

  // Test ready
  XLS_EXPECT_OK(res = sm->ReadU8AtAddress(w_offset));
  expect = StreamManager::ctrl_reg_READY_;
  EXPECT_EQ(res.value(), expect);
}

uint64_t AdjacentStreamOffset(uint64_t offset, uint64_t width_bits) {
  return offset + StreamManager::ctrl_reg_width_bytes_ +
         BitsToBytesFloor(width_bits);
}

TEST_P(StreamManagerTest, DoubleReadAdjacentStreams) {
  // DESCRIPTION: Test an edge case: Read offset starts in one stream, but the
  // range of the read includes another, adjacent stream.

  const StreamManagerTestParams params = GetParam();

  const uint64_t s1_width_bits = 128;  // 16 bytes
  const uint64_t s1_offset = params.offset1;
  const uint64_t s2_width_bits = 128;  // 16 bytes
  const uint64_t s2_offset = AdjacentStreamOffset(s1_offset, s1_width_bits);
  const uint64_t payload64 = params.payload64;
  const uint64_t payload_offset = 8;

  IStreamMock<s1_width_bits, true>* stream1 =
      new IStreamMock<s1_width_bits, true>();
  IStreamMock<s2_width_bits, true>* stream2 =
      new IStreamMock<s2_width_bits, true>();

  auto sm = StreamManager::Build(params.base, [&](const auto& RegisterStream) {
    RegisterStream(s1_offset, stream1);
    RegisterStream(s2_offset, stream2);
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm);

  // Setup stream content ({64'h0, Payload64})
  XLS_EXPECT_OK(stream1->SetPayloadData64(0, 0));
  XLS_EXPECT_OK(stream1->SetPayloadData64(payload_offset, payload64));

  absl::StatusOr<uint64_t> res;

  for (int i = 1; i < 8; ++i) {
    if (i < sizeof(uint16_t)) {
      XLS_LOG(INFO) << "Reading 16 bits at offset -" << i;
      XLS_EXPECT_OK(res = sm->ReadU16AtAddress(params.base + s2_offset - i));
      uint16_t expected16 = Payload64Tail(payload64, i) |
                            (StreamManager::ctrl_reg_DIR_ << BytesToBits(i));
      XLS_LOG(INFO) << "  expected: " << ToHex(expected16);
      EXPECT_EQ(res.value(), expected16);
    }
    if (i < sizeof(uint32_t)) {
      XLS_LOG(INFO) << "Reading 32 bits at offset -" << i;
      XLS_EXPECT_OK(res = sm->ReadU32AtAddress(params.base + s2_offset - i));
      uint32_t expected32 = Payload64Tail(payload64, i) |
                            (StreamManager::ctrl_reg_DIR_ << BytesToBits(i));
      XLS_LOG(INFO) << "  expected: " << ToHex(expected32);
      EXPECT_EQ(res.value(), expected32);
    }
    if (i < sizeof(uint64_t)) {
      XLS_LOG(INFO) << "Reading 64 bits at offset -" << i;
      XLS_EXPECT_OK(res = sm->ReadU64AtAddress(params.base + s2_offset - i));
      uint64_t expected64 = Payload64Tail(payload64, i) |
                            (StreamManager::ctrl_reg_DIR_ << BytesToBits(i));
      XLS_LOG(INFO) << "  expected: " << ToHex(expected64);
      EXPECT_EQ(res.value(), expected64);
    }
  }
}

TEST_P(StreamManagerTest, WriteToASingleStream) {
  // DESCRIPTION: Test writing to a stream. Offset must be within the streams
  // range.

  const StreamManagerTestParams params = GetParam();

  const uint64_t s1_width_bits = 128;  // 16 bytes
  const uint64_t s1_offset = params.offset1;
  const uint64_t payload64 = params.payload64;
  const uint64_t payload_offset = 8;

  absl::StatusOr<uint64_t> status;

  IStreamMock<s1_width_bits, false>* stream1 =
      new IStreamMock<s1_width_bits, false>();

  auto sm = StreamManager::Build(params.base, [&](const auto& RegisterStream) {
    RegisterStream(s1_offset, stream1);
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm);

  uint64_t expected64;

  for (int i = 0; i < 8; ++i) {
    uint64_t write_offset = params.base + s1_offset +
                            StreamManager::ctrl_reg_width_bytes_ +
                            payload_offset + i;

    auto Prepare = [&](uint8_t bytes) {
      XLS_EXPECT_OK(stream1->SetPayloadData64(0, 0));
      XLS_EXPECT_OK(stream1->SetPayloadData64(payload_offset, 0));
      int64_t tail_len_bytes = payload_offset + i + sizeof(uint64_t) -
                               BitsToBytesFloor(s1_width_bits);
      expected64 = payload64 & ~Payload64Mask(tail_len_bytes) &
                   xls::Mask(BytesToBits(bytes));
    };

    auto Check = [&]() {
      XLS_EXPECT_OK(status = stream1->GetPayloadData64(payload_offset + i));
      EXPECT_EQ(status.value(), expected64);
    };

    XLS_LOG(INFO) << "Writing 8 bits at offset " << i;
    Prepare(1);
    XLS_EXPECT_OK(sm->WriteU8AtAddress(write_offset, payload64));
    Check();

    XLS_LOG(INFO) << "Writing 16 bits at offset " << i;
    Prepare(2);
    XLS_EXPECT_OK(sm->WriteU16AtAddress(write_offset, payload64));
    Check();

    XLS_LOG(INFO) << "Writing 32 bits at offset " << i;
    Prepare(4);
    XLS_EXPECT_OK(sm->WriteU32AtAddress(write_offset, payload64));
    Check();

    XLS_LOG(INFO) << "Writing 64 bits at offset " << i;
    Prepare(8);
    XLS_EXPECT_OK(sm->WriteU64AtAddress(write_offset, payload64));
    Check();
  }
}

template <std::size_t WidthBytes>
uint64_t ExtractU64FromArray(const std::array<uint8_t, WidthBytes>& bytes) {
  EXPECT_GE(bytes.size(), 8);
  uint64_t res = 0;
  for (int i = 0; i < 8; ++i) {
    res |= static_cast<uint64_t>(bytes[i]) << BytesToBits(i);
  }
  return res;
}

template <std::size_t WidthBytes>
std::array<uint8_t, WidthBytes> MakeArrayFromU64(uint64_t data) {
  static_assert(WidthBytes >= sizeof(uint64_t), "Too short array");
  std::array<uint8_t, WidthBytes> res;
  for (int i = 0; i < 8; ++i) {
    res[i] = data & 0xff;
    data >>= 8;
  }
  return res;
}

TEST_P(StreamManagerTest, WriteCtrl) {
  // DESCRIPTION: Control register write test

  const StreamManagerTestParams params = GetParam();

  const uint64_t w_offset = params.offset1;
  const uint64_t w_width_bits = 64;  // 8 bytes
  const uint64_t r_offset = params.offset2;
  const uint64_t r_width_bits = 64;  // 8 bytes
  const uint64_t payload64 = params.payload64;

  absl::StatusOr<uint64_t> status;

  IStreamMock<w_width_bits, false>* wstream =
      new IStreamMock<w_width_bits, false>();
  IStreamMock<r_width_bits, true>* rstream =
      new IStreamMock<r_width_bits, true>();

  auto sm = StreamManager::Build(params.base, [&](const auto& RegisterStream) {
    RegisterStream(w_offset, wstream);
    RegisterStream(r_offset, rstream);
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm);

  // Test valid transfer for write stream
  XLS_EXPECT_OK(wstream->SetPayloadData64(0, payload64));
  EXPECT_EQ(wstream->fifo().size(), 0);
  XLS_EXPECT_OK(sm->WriteU8AtAddress(params.base + w_offset,
                                     StreamManager::ctrl_reg_DOXFER_));
  XLS_EXPECT_OK(sm->LastTransferStatus());
  EXPECT_EQ(wstream->fifo().size(), 1);
  EXPECT_EQ(ExtractU64FromArray(wstream->fifo().front()), payload64);

  // Test valid transfer for read stream
  EXPECT_EQ(rstream->fifo().size(), 0);
  rstream->fifo().push(MakeArrayFromU64<8>(payload64));
  XLS_EXPECT_OK(sm->WriteU8AtAddress(params.base + r_offset,
                                     StreamManager::ctrl_reg_DOXFER_));
  XLS_EXPECT_OK(status = rstream->GetPayloadData64(0));
  XLS_EXPECT_OK(sm->LastTransferStatus());
  EXPECT_EQ(status.value(), payload64);

  // Test invalid transfer for read stream
  EXPECT_EQ(rstream->fifo().empty(), true);
  XLS_EXPECT_OK(sm->WriteU8AtAddress(params.base + r_offset,
                                     StreamManager::ctrl_reg_DOXFER_));
  EXPECT_THAT(
      sm->LastTransferStatus(),
      xls::status_testing::StatusIs(absl::StatusCode::kInternal,
                                    testing::HasSubstr("FIFO is empty")));

  // Test status clearing
  XLS_EXPECT_OK(sm->WriteU8AtAddress(params.base + r_offset,
                                     StreamManager::ctrl_reg_ERRXFER_));
  XLS_EXPECT_OK(sm->LastTransferStatus());
}

TEST_P(StreamManagerTest, DoubleWriteAdjacentStreams) {
  // DESCRIPTION: Test an edge case: Write offset starts in one stream, but the
  // range of the read includes another, adjacent stream.

  const StreamManagerTestParams params = GetParam();

  const uint64_t s1_width_bits = 128;  // 16 bytes
  const uint64_t s1_offset = params.offset1;
  const uint64_t s2_width_bits = 128;  // 16 bytes
  const uint64_t s2_offset = AdjacentStreamOffset(s1_offset, s1_width_bits);
  const uint64_t payload64 = params.payload64;
  const uint64_t payload_offset = 8;

  absl::StatusOr<uint64_t> status;

  IStreamMock<s1_width_bits, false>* stream1 =
      new IStreamMock<s1_width_bits, false>();
  IStreamMock<s2_width_bits, false>* stream2 =
      new IStreamMock<s2_width_bits, false>();

  auto sm = StreamManager::Build(params.base, [&](const auto& RegisterStream) {
    RegisterStream(s1_offset, stream1);
    RegisterStream(s2_offset, stream2);
    return absl::OkStatus();
  });
  XLS_EXPECT_OK(sm);

  bool will_transfer;
  uint64_t expected64, wr_offset;

  auto ResetStreams = [&]() {
    XLS_EXPECT_OK(stream1->SetPayloadData64(0, 0));
    XLS_EXPECT_OK(stream1->SetPayloadData64(payload_offset, 0));
    while (!stream2->fifo().empty()) {
      stream2->fifo().pop();
    }
    // Clear error status. Requires WriteCtrl test to pass first
    XLS_EXPECT_OK(sm->WriteU8AtAddress(params.base + s2_offset,
                                       StreamManager::ctrl_reg_ERRXFER_));
  };

  auto Prepare = [&](uint8_t width_bytes, int i) {
    ResetStreams();
    will_transfer = static_cast<bool>(Payload64Tail(payload64, i) &
                                      StreamManager::ctrl_reg_DOXFER_);
    expected64 = payload64 >> BytesToBits(i);
    wr_offset =
        s1_offset + StreamManager::ctrl_reg_width_bytes_ + payload_offset + i;
    if (will_transfer) {
      XLS_EXPECT_OK(stream2->SetPayloadData64(0, payload64));
    }
  };

  auto Check = [&]() {
    if (will_transfer) {
      XLS_LOG(INFO) << "    expecting transfer";
      ASSERT_EQ(stream2->fifo().size(), 1);
      ASSERT_EQ(ExtractU64FromArray(stream2->fifo().front()), payload64);
    } else {
      ASSERT_EQ(stream2->fifo().empty(), true);
    }
  };

  for (int i = 1; i < 8; ++i) {
    if (i > 6) {
      XLS_LOG(INFO) << "Writing 16 bits at offset " << payload_offset + i;
      Prepare(2, i);
      XLS_EXPECT_OK(sm->WriteU16AtAddress(params.base + wr_offset,
                                          static_cast<uint16_t>(payload64)));
      Check();
    }
    if (i > 4) {
      XLS_LOG(INFO) << "Writing 32 bits at offset " << payload_offset + i;
      Prepare(4, i);
      XLS_EXPECT_OK(sm->WriteU32AtAddress(params.base + wr_offset,
                                          static_cast<uint32_t>(payload64)));
      Check();
    }
    XLS_LOG(INFO) << "Writing 64 bits at offset " << payload_offset + i;
    Prepare(8, i);
    XLS_EXPECT_OK(sm->WriteU64AtAddress(params.base + wr_offset, payload64));
    Check();
  }
}

INSTANTIATE_TEST_SUITE_P(
    StreamManagerTestInstantiation, StreamManagerTest,
    testing::Values(
        StreamManagerTestParams{
            .base = 0x0,
            .offset1 = 0x0,
            .offset2 = 0x18,
            .payload64 = Payload64("01 Bipp"),
        },
        StreamManagerTestParams{
            .base = 0x1,
            .offset1 = 0x0,
            .offset2 = 0x18,
            .payload64 = Payload64("02 Elle"),
        },
        StreamManagerTestParams{
            .base = 0x100,
            .offset1 = 0x10,
            .offset2 = 0x28,
            .payload64 = Payload64("03 Lemonade"),
        },
        StreamManagerTestParams{
            .base = 0x100,
            .offset1 = 0x10,
            .offset2 = 0x38,
            .payload64 = Payload64("04 Hard"),
        },
        StreamManagerTestParams{
            .base = 0x420,
            .offset1 = 0x10,
            .offset2 = 0x308,
            .payload64 = Payload64("05 Msmsmsmsm"),
        },
        StreamManagerTestParams{
            .base = 0x12,
            .offset1 = 0x0a,
            .offset2 = 0xa0,
            .payload64 = Payload64("06 Vyzee"),
        },
        StreamManagerTestParams{
            .base = 0x0,
            .offset1 = 0x60,
            .offset2 = 0x00,
            .payload64 = Payload64("07 L.o.v.e."),
        },
        StreamManagerTestParams{
            .base = 0xff,
            .offset1 = 0x100a,
            .offset2 = 0xa0,
            .payload64 = Payload64("08 Just like we never said goodbye"),
        }));

}  // namespace

}  // namespace xls::simulation::generic

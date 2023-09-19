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

#include "xls/simulation/generic/streamdmachannel.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/generic/imasterport.h"
#include "xls/simulation/generic/istream_mock.h"
#include "xls/simulation/generic/stream_dma_endpoint.h"

namespace xls::simulation::generic {

// This class has to be in generic namespace as it's
// a friend class with StreamDmaChannel
class StreamDmaChannelPrivateAccess {
 public:
  explicit StreamDmaChannelPrivateAccess(StreamDmaChannel& channel)
      : dma_channel_(channel) {}
  void SetIrq(uint64_t new_irq) { dma_channel_.irq_ = new_irq; }
  void SetIrqMask(uint64_t new_irq_mask) {
    dma_channel_.irq_mask_ = new_irq_mask;
  }
  void SetDmaRun(bool new_dma_run) { dma_channel_.dma_run_ = new_dma_run; }
  void SetDmaDiscard(bool new_dma_discard) {
    dma_channel_.dma_discard_ = new_dma_discard;
  }

  absl::Status InternalUpdate() {
    if (dma_channel_.endpoint_->IsReadStream())
      return dma_channel_.UpdateWriteToEmulator();
    return dma_channel_.UpdateReadFromEmulator();
  }

  IStream* GetChannel() {
    return dynamic_cast<StreamDmaEndpoint&>(*dma_channel_.endpoint_)
        .GetStream();
  }

 private:
  StreamDmaChannel& dma_channel_;
};

namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::Return;

class StreamDmaChannelIMasterPortMock : public IMasterPort {
 public:
  MOCK_METHOD(absl::Status, RequestWrite,
              (uint64_t address, uint64_t payload, AccessWidth width),
              (override));
  MOCK_METHOD(absl::StatusOr<uint64_t>, RequestRead,
              (uint64_t address, AccessWidth width), (override));
};

static std::unique_ptr<IDmaEndpoint> MakeIStreamMockEndpoint(
    bool is_read_stream) {
  if (is_read_stream) {
    return std::make_unique<StreamDmaEndpoint>(
        std::make_unique<IStreamMock<120, true>>());
  }
  return std::make_unique<StreamDmaEndpoint>(
      std::make_unique<IStreamMock<120, false>>());
}

static void MakeStreamReady(IStream* stream) {
  if (stream->IsReadStream()) {
    dynamic_cast<IStreamMock<120, true>*>(stream)->fifo().push(
        std::array<uint8_t, 15>{0xEF, 0xBE, 0xAD, 0xDE, 0x0D, 0xF0, 0xAD, 0x8B,
                                0xEF, 0xBE, 0xAD, 0xDE, 0x0D, 0xF0, 0xAD});
  }
}

class StreamDmaChannelTest : public ::testing::Test,
                             public ::testing::WithParamInterface<bool> {
 protected:
  StreamDmaChannelTest()
      : bus_master_port_(),
        dma_channel_(MakeIStreamMockEndpoint(GetParam()), &bus_master_port_),
        proxy_(dma_channel_) {}
  StreamDmaChannelIMasterPortMock bus_master_port_;
  StreamDmaChannel dma_channel_;
  StreamDmaChannelPrivateAccess proxy_;
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_P(StreamDmaChannelTest, GetIRQRegTest) {
  this->proxy_.SetIrq(0xDEADBEEF);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0xDEADBEEF);
}

TEST_P(StreamDmaChannelTest, ClearIRQRegTest) {
  this->proxy_.SetIrq(0x3);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x3);

  this->dma_channel_.ClearIRQReg(0);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x3);

  this->dma_channel_.ClearIRQReg(4);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x3);

  this->dma_channel_.ClearIRQReg(1);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x2);

  this->dma_channel_.ClearIRQReg(2);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x0);
}

TEST_P(StreamDmaChannelTest, BaseAddressTest) {
  uint64_t base_address = 0x4000;
  this->dma_channel_.SetTransferBaseAddress(base_address);
  EXPECT_EQ(this->dma_channel_.GetTransferBaseAddress(), base_address);

  this->proxy_.SetDmaRun(true);
  ::testing::internal::CaptureStderr();
  this->dma_channel_.SetTransferBaseAddress(base_address + 0x100);
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr(
                  "Base address can't be modified while DMA is running"));
  EXPECT_EQ(this->dma_channel_.GetTransferBaseAddress(), base_address);

  this->proxy_.SetDmaRun(false);
  this->dma_channel_.SetTransferBaseAddress(base_address + 0x100);
  EXPECT_EQ(this->dma_channel_.GetTransferBaseAddress(), base_address + 0x100);
}

TEST_P(StreamDmaChannelTest, MaxTransferLength) {
  uint64_t transfer_length = 0x3FFC;
  this->dma_channel_.SetMaxTransferLength(transfer_length);
  EXPECT_EQ(this->dma_channel_.GetMaxTransferLength(), transfer_length);

  this->proxy_.SetDmaRun(true);
  ::testing::internal::CaptureStderr();
  this->dma_channel_.SetMaxTransferLength(transfer_length + 0x100);
  EXPECT_THAT(
      ::testing::internal::GetCapturedStderr(),
      ::testing::HasSubstr(
          "Max transfer length can't be modified while DMA is running"));
  EXPECT_EQ(this->dma_channel_.GetMaxTransferLength(), transfer_length);

  this->proxy_.SetDmaRun(false);
  ::testing::internal::CaptureStderr();
  this->dma_channel_.SetMaxTransferLength(transfer_length + 0x100);
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("Max transfer length will be clipped to:"));
  EXPECT_EQ(this->dma_channel_.GetMaxTransferLength(), 0x40FB);
}

TEST_P(StreamDmaChannelTest, IRQMaskAndGetIRQTest) {
  this->proxy_.SetIrq(0x2);
  EXPECT_EQ(this->dma_channel_.GetIRQ(), false);

  this->proxy_.SetIrqMask(0x2);
  EXPECT_EQ(this->dma_channel_.GetIRQ(), true);

  this->proxy_.SetIrqMask(0x1);
  EXPECT_EQ(this->dma_channel_.GetIRQ(), false);

  this->proxy_.SetIrq(0x3);
  this->proxy_.SetIrqMask(0x2);
  EXPECT_EQ(this->dma_channel_.GetIRQ(), true);

  this->proxy_.SetIrqMask(0x1);
  EXPECT_EQ(this->dma_channel_.GetIRQ(), true);
}

TEST_P(StreamDmaChannelTest, GetTransferredLengthTest) {
  MakeStreamReady(this->proxy_.GetChannel());
  uint64_t transfer_length = 0x3FFC;
  this->dma_channel_.SetMaxTransferLength(transfer_length);
  EXPECT_CALL(this->bus_master_port_, RequestRead(_, _))
      .WillRepeatedly(Return(0));
  XLS_EXPECT_OK(this->proxy_.InternalUpdate());
  // Stream has channel width of 120bits which is exactly 15 bytes
  // Transfer ends after whole channel is empty (when reading)
  // and after all the bytes have been transferred (when writing)
  EXPECT_EQ(this->dma_channel_.GetTransferredLength(),
            proxy_.GetChannel()->IsReadStream() ? 15 : transfer_length);
}

TEST_P(StreamDmaChannelTest, UpdateTest) {
  MakeStreamReady(this->proxy_.GetChannel());
  // DMA disabled
  this->dma_channel_.SetMaxTransferLength(15);
  EXPECT_EQ(this->dma_channel_.GetTransferredLength(), 0);
  XLS_EXPECT_OK(this->dma_channel_.Update());
  EXPECT_EQ(this->dma_channel_.GetTransferredLength(), 0);

  // Empty DMA transfer
  this->dma_channel_.SetMaxTransferLength(0);
  this->dma_channel_.SetControlRegister(0x4);
  XLS_EXPECT_OK(this->dma_channel_.Update());
  EXPECT_EQ(this->dma_channel_.GetControlRegister(),
            0xC | GetParam() << 4 | 1 << 6);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x1);

  // Check that IRQ is not re triggered
  this->dma_channel_.ClearIRQReg(0x1);
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x0);
  XLS_EXPECT_OK(this->dma_channel_.Update());
  EXPECT_EQ(this->dma_channel_.GetIRQReg(), 0x0);
}

TEST_P(StreamDmaChannelTest, SetGetControlRegisterTest) {
  MakeStreamReady(this->proxy_.GetChannel());
  // CR[4] = IsReadStream()
  // CR[6] = Stream.Ready()
  EXPECT_EQ(this->dma_channel_.GetControlRegister(), GetParam() << 4 | 1 << 6);
  this->dma_channel_.SetControlRegister(0xFF);
  // CR[3] is set only by Update
  EXPECT_EQ(this->dma_channel_.GetControlRegister(),
            0x7 | GetParam() << 4 | 1 << 5 | 1 << 6);
  // Sets CR[3]
  XLS_EXPECT_OK(this->dma_channel_.Update());
  EXPECT_EQ(this->dma_channel_.GetControlRegister(),
            0xf | GetParam() << 4 | 1 << 5 | 1 << 6);

  // CR[3] must not be changed by write
  this->dma_channel_.SetControlRegister(0xF7);
  EXPECT_EQ(this->dma_channel_.GetControlRegister(),
            0xf | GetParam() << 4 | 1 << 5 | 1 << 6);

  // CR[3] must be cleared when CR[2] is cleared
  this->dma_channel_.SetControlRegister(0xFB);
  EXPECT_EQ(this->dma_channel_.GetControlRegister(),
            0x3 | GetParam() << 4 | 1 << 5 | 1 << 6);
}

INSTANTIATE_TEST_SUITE_P(StreamDmaChannelTestInstantiation,
                         StreamDmaChannelTest, testing::Values(true, false));

class StreamDmaChannelReadTest : public ::testing::Test {
 protected:
  StreamDmaChannelReadTest()
      : bus_master_port_(),
        dma_channel_(MakeIStreamMockEndpoint(true), &bus_master_port_),
        proxy_(dma_channel_) {}
  StreamDmaChannelIMasterPortMock bus_master_port_;
  StreamDmaChannel dma_channel_;
  StreamDmaChannelPrivateAccess proxy_;

  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(StreamDmaChannelReadTest, PeripheralAccessTest) {
  MakeStreamReady(this->proxy_.GetChannel());
  uint64_t transfer_length = 0x3FFC;
  this->dma_channel_.SetMaxTransferLength(transfer_length);
  this->proxy_.SetDmaDiscard(false);
  this->proxy_.SetDmaRun(true);
  {
    InSequence s;
    EXPECT_CALL(
        this->bus_master_port_,
        RequestWrite(Eq(0), Eq(0x8BADF00DDEADBEEFULL), Eq(AccessWidth::QWORD)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(8), Eq(0xEFULL), Eq(AccessWidth::BYTE)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(9), Eq(0xBEULL), Eq(AccessWidth::BYTE)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(10), Eq(0xADULL), Eq(AccessWidth::BYTE)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(11), Eq(0xDEULL), Eq(AccessWidth::BYTE)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(12), Eq(0x0DULL), Eq(AccessWidth::BYTE)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(13), Eq(0xF0ULL), Eq(AccessWidth::BYTE)))
        .Times(1);
    EXPECT_CALL(this->bus_master_port_,
                RequestWrite(Eq(14), Eq(0xADULL), Eq(AccessWidth::BYTE)))
        .Times(1);
  }
  XLS_EXPECT_OK(this->dma_channel_.Update());
}

TEST_F(StreamDmaChannelReadTest, DiscardTest) {
  MakeStreamReady(this->proxy_.GetChannel());
  uint64_t transfer_length = 0x3FFC;
  this->dma_channel_.SetMaxTransferLength(transfer_length);
  this->proxy_.SetDmaDiscard(true);
  this->proxy_.SetDmaRun(true);
  {
    InSequence s;
    EXPECT_CALL(this->bus_master_port_, RequestWrite(_, _, _)).Times(0);
    EXPECT_CALL(this->bus_master_port_, RequestRead(_, _)).Times(0);
  }
  XLS_EXPECT_OK(this->dma_channel_.Update());
}

class StreamDmaChannelWriteTest : public ::testing::Test {
 protected:
  StreamDmaChannelWriteTest()
      : bus_master_port_(),
        dma_channel_(MakeIStreamMockEndpoint(false), &bus_master_port_),
        proxy_(dma_channel_) {}
  StreamDmaChannelIMasterPortMock bus_master_port_;
  StreamDmaChannel dma_channel_;
  StreamDmaChannelPrivateAccess proxy_;

  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(StreamDmaChannelWriteTest, PeripheralAccessTest) {
  MakeStreamReady(this->proxy_.GetChannel());
  uint64_t transfer_length = 15;
  this->dma_channel_.SetMaxTransferLength(transfer_length);
  this->proxy_.SetDmaRun(true);
  {
    InSequence s;
    EXPECT_CALL(this->bus_master_port_,
                RequestRead(Eq(0), Eq(AccessWidth::QWORD)))
        .Times(1)
        .WillOnce(Return(0x8BADF00DDEADBEEFULL));
    for (int i = 8; i < 15; ++i) {
      EXPECT_CALL(this->bus_master_port_,
                  RequestRead(Eq(i), Eq(AccessWidth::BYTE)))
          .Times(1)
          .WillOnce(Return(0));
    }
  }
  XLS_EXPECT_OK(this->dma_channel_.Update());
  IStreamMock<120, false>* channel =
      dynamic_cast<IStreamMock<120, false>*>(this->proxy_.GetChannel());
  EXPECT_NE(channel, nullptr);
  EXPECT_EQ(channel->fifo().size(), 1);
  EXPECT_EQ(channel->fifo().front(),
            (std::array<uint8_t, 15>{0xEF, 0xBE, 0xAD, 0xDE, 0x0D, 0xF0, 0xAD,
                                     0x8B}));
}

}  // namespace
}  // namespace xls::simulation::generic

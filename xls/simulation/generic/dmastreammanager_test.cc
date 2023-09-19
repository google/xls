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

#include "xls/simulation/generic/dmastreammanager.h"

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
#include "xls/simulation/generic/streamdmachannel.h"

namespace xls::simulation::generic {
namespace {

using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::_;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::Return;

class DmaStreamManagerIMasterPortMock : public IMasterPort {
 public:
  MOCK_METHOD(absl::Status, RequestWrite,
              (uint64_t address, uint64_t payload, AccessWidth width));
  MOCK_METHOD(absl::StatusOr<uint64_t>, RequestRead,
              (uint64_t address, AccessWidth width));
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

class DmaStreamManagerTest : public ::testing::Test {
 protected:
  DmaStreamManagerTest() : bus_master_port_(), manager_(0x10000) {}

  DmaStreamManagerIMasterPortMock bus_master_port_;
  DmaStreamManager manager_;
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(DmaStreamManagerTest, EmptyManager) {
  EXPECT_FALSE(this->manager_.InRange(0x100));
  EXPECT_FALSE(this->manager_.InRange(0xFFFF));
  EXPECT_TRUE(this->manager_.InRange(0x10000));
  EXPECT_TRUE(this->manager_.InRange(0x10001));
  EXPECT_TRUE(this->manager_.InRange(0x1003F));
  EXPECT_FALSE(this->manager_.InRange(0x10040));

  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10000), IsOkAndHolds(Eq(0ULL)));
  ::testing::internal::CaptureStderr();
  EXPECT_THAT(this->manager_.WriteU64AtAddress(0x10000, 0xDEADBEEFULL), IsOk());
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr(
                  "Write access to RO register: NumberOfDMAChannels!"));

  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10008), IsOkAndHolds(Eq(64)));
  ::testing::internal::CaptureStderr();
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10008, 0xDEADBEEFULL)),
              IsOk());
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr(
                  "Write access to RO register: OffsetToFirstDMAChannel!"));

  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10010), IsOkAndHolds(Eq(0)));
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10010, 0xDEADBEEFULL)),
              IsOk());
  // Empty Manager can't store any IRQ mask
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10010), IsOkAndHolds(Eq(0)));

  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10018), IsOkAndHolds(Eq(0)));
  ::testing::internal::CaptureStderr();
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10018, 0xDEADBEEFULL)),
              IsOk());
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("Write access to RO register: ActiveIRQs!"));

  ::testing::internal::CaptureStderr();
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10020), IsOkAndHolds(Eq(0)));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("Read access to not existing register"));
  ::testing::internal::CaptureStderr();
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10020, 0xDEADBEEFULL)),
              IsOk());
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("Write access to not existing register"));

  ::testing::internal::CaptureStderr();
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10040), IsOkAndHolds(Eq(0)));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("DMA channel with id:0 does not exist."));
  ::testing::internal::CaptureStderr();
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10040, 0xDEADBEEFULL)),
              IsOk());
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("DMA channel with id:0 does not exist."));
}

TEST_F(DmaStreamManagerTest, UnalignedAccess) {
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10004),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unaligned access, address:")));
  EXPECT_THAT(this->manager_.ReadU32AtAddress(0x10002),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unaligned access, address:")));
  EXPECT_THAT(this->manager_.ReadU16AtAddress(0x10001),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unaligned access, address:")));

  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10004, 0xDEADBEEF)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unaligned access, address:")));
  EXPECT_THAT((this->manager_.WriteU32AtAddress(0x10002, 0xDEADBEEF)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unaligned access, address:")));
  EXPECT_THAT((this->manager_.WriteU16AtAddress(0x10001, 0xDEAD)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unaligned access, address:")));
}

TEST_F(DmaStreamManagerTest, RegisterEndpoint) {
  EXPECT_THAT(this->manager_.RegisterEndpoint(MakeIStreamMockEndpoint(true), 0,
                                              &this->bus_master_port_),
              IsOk());
  EXPECT_THAT(this->manager_.RegisterEndpoint(MakeIStreamMockEndpoint(true), 0,
                                              &this->bus_master_port_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Id: 0 already in use")));
}

TEST_F(DmaStreamManagerTest, AccessDMAChannel) {
  EXPECT_THAT(this->manager_.RegisterEndpoint(MakeIStreamMockEndpoint(true), 0,
                                              &this->bus_master_port_),
              IsOk());
  // Common for Read and Write DMA channels
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10040), IsOkAndHolds(Eq(0)));
  EXPECT_THAT(
      (this->manager_.WriteU64AtAddress(0x10040, 0xDEADBEEF8BADF00DULL)),
      IsOk());
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10040),
              IsOkAndHolds(Eq(0xDEADBEEF8BADF00DULL)));

  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10048), IsOkAndHolds(Eq(0)));
  EXPECT_THAT(
      (this->manager_.WriteU64AtAddress(0x10048, 0xDEADBEEF8BADF004ULL)),
      IsOk());
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10048),
              IsOkAndHolds(Eq(0xDEADBEEF8BADF004ULL)));

  ::testing::internal::CaptureStderr();
  EXPECT_THAT(
      (this->manager_.WriteU64AtAddress(0x10050, 0xDEADBEEF8BADF004ULL)),
      IsOk());
  EXPECT_THAT(
      ::testing::internal::GetCapturedStderr(),
      ::testing::HasSubstr(
          "Write access to DMA channel RO register: TransferredLength!"));
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10050), IsOkAndHolds(Eq(0)));

  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10060), IsOkAndHolds(Eq(0)));
  EXPECT_THAT(
      (this->manager_.WriteU64AtAddress(0x10060, 0xDEADBEEF8BADF004ULL)),
      IsOk());
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10060), IsOkAndHolds(Eq(0)));

  ::testing::internal::CaptureStderr();
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10068), IsOk());
  EXPECT_THAT(
      ::testing::internal::GetCapturedStderr(),
      ::testing::HasSubstr("Offset:40 doesn't map to any DMA register!"));
  ::testing::internal::CaptureStderr();
  EXPECT_THAT(
      (this->manager_.WriteU64AtAddress(0x10068, 0xDEADBEEF8BADF004ULL)),
      IsOk());
  EXPECT_THAT(
      ::testing::internal::GetCapturedStderr(),
      ::testing::HasSubstr("Offset:40 doesn't map to any DMA register!"));

  // Check Read DMA channel
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10058), IsOkAndHolds(Eq(0x10)));
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10058, 0x27ULL)), IsOk());
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10058),
              IsOkAndHolds(Eq(0x37LL)));

  // Check Write DMA channel
  EXPECT_THAT(this->manager_.RegisterEndpoint(MakeIStreamMockEndpoint(false), 1,
                                              &this->bus_master_port_),
              IsOk());
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10098), IsOkAndHolds(Eq(0x40)));
  EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10098, 0x27ULL)), IsOk());
  EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10098),
              IsOkAndHolds(Eq(0x67LL)));
}

TEST_F(DmaStreamManagerTest, TestUpdateAndIRQClearing) {
  const int kChannelCount = 2;
  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_THAT(this->manager_.RegisterEndpoint(MakeIStreamMockEndpoint(false),
                                                i, &this->bus_master_port_),
                IsOk());
    EXPECT_THAT(
        (this->manager_.WriteU64AtAddress(0x10040 + 0x40 * i, 0x100 << i * 4)),
        IsOk());
    EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10048 + 0x40 * i, 0xF)),
                IsOk());
    EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10058 + 0x40 * i, 0x4)),
                IsOk());
  }
  {
    InSequence s;
    for (int i = 0; i < kChannelCount; ++i) {
      EXPECT_CALL(this->bus_master_port_,
                  RequestRead(Eq(0x100 << 4 * i), Eq(AccessWidth::QWORD)))
          .Times(1)
          .WillOnce(Return(0));
      for (int j = 8; j < 15; ++j) {
        EXPECT_CALL(
            this->bus_master_port_,
            RequestRead(Eq((0x100 << 4 * i) + j), Eq(AccessWidth::BYTE)))
            .Times(1)
            .WillOnce(Return(0));
      }
    }
  }
  XLS_EXPECT_OK(this->manager_.Update());
  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_THAT(this->manager_.ReadU64AtAddress(0x10050 + 0x40 * i),
                IsOkAndHolds(Eq(15)));
    EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10058 + 0x40 * i)),
                IsOkAndHolds(Eq(0x4C)));
    EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10060 + 0x40 * i)),
                IsOkAndHolds(Eq(0x1)));
    EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10060 + 0x40 * i, 0x1)),
                IsOk());
    EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10060 + 0x40 * i)),
                IsOkAndHolds(Eq(0)));
  }
}

TEST_F(DmaStreamManagerTest, TestUpdateIRQAndGetIRQ) {
  const int kChannelCount = 4;
  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_THAT(this->manager_.RegisterEndpoint(MakeIStreamMockEndpoint(false),
                                                i, &this->bus_master_port_),
                IsOk());
    EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10058 + 0x40 * i, 0x4)),
                IsOk());
  }
  XLS_EXPECT_OK(this->manager_.Update());
  // All channels should have reached end and have IRQ ready, but masked
  EXPECT_FALSE(this->manager_.GetIRQ());
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_FALSE(this->manager_.GetIRQ());
  // Unmask IRQ in DMA manager, but not in DMA channels
  XLS_EXPECT_OK((this->manager_.WriteU64AtAddress(0x10010, 0xf)));
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_FALSE(this->manager_.GetIRQ());
  // Unmask IRQ in DMA channels
  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_THAT((this->manager_.WriteU64AtAddress(0x10058 + 0x40 * i, 0x5)),
                IsOk());
  }
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10018)),
              IsOkAndHolds(Eq(0xf)));
  EXPECT_TRUE(this->manager_.GetIRQ());
  XLS_EXPECT_OK((this->manager_.WriteU64AtAddress(0x10010, 0x5)));
  EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10018)),
              IsOkAndHolds(Eq(0xf)));
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10018)),
              IsOkAndHolds(Eq(0x5)));
  // Mask IRQ in DMA manager
  XLS_EXPECT_OK((this->manager_.WriteU64AtAddress(0x10010, 0x0)));
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_FALSE(this->manager_.GetIRQ());
  // Unmask IRQ in DMA channels
  XLS_EXPECT_OK((this->manager_.WriteU64AtAddress(0x10010, 0xf)));
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10018)),
              IsOkAndHolds(Eq(0xf)));
  XLS_EXPECT_OK((this->manager_.WriteU64AtAddress(0x10060, 0x1)));
  EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10018)),
              IsOkAndHolds(Eq(0xf)));
  XLS_EXPECT_OK(this->manager_.UpdateIRQ());
  EXPECT_THAT((this->manager_.ReadU64AtAddress(0x10018)),
              IsOkAndHolds(Eq(0xe)));
}

}  // namespace
}  // namespace xls::simulation::generic

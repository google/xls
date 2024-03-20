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

#include "xls/simulation/generic/singlevaluemanager.h"

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
#include "xls/simulation/generic/iregister_mock.h"
#include "xls/simulation/generic/iregister_stub.h"

namespace xls::simulation::generic {

// This class has to be in generic namespace as it's
// a friend class with SingleValueManager
class SingleValueManagerTest : public ::testing::Test {
 protected:
  static const uint64_t base_address = 0x10000;

  std::unique_ptr<SingleValueManager> manager_;

  uint64_t GetMaxOffset() { return manager_->GetMaxOffset(); }
  SingleValueManagerTest() {
    manager_ = std::unique_ptr<SingleValueManager>(
        new SingleValueManager(base_address));
  }
  ~SingleValueManagerTest() {}
};

namespace {
// Registering on valid offsets should succeed
TEST_F(SingleValueManagerTest, RegisterIRegisterValidOffsets) {
  std::unique_ptr<IRegister> channel_1bit =
      std::make_unique<IRegisterMock<1>>();
  std::unique_ptr<IRegister> channel_8bit =
      std::make_unique<IRegisterMock<8>>();
  std::unique_ptr<IRegister> channel_16bit =
      std::make_unique<IRegisterMock<16>>();
  std::unique_ptr<IRegister> channel_32bit =
      std::make_unique<IRegisterMock<32>>();
  std::unique_ptr<IRegister> channel_63bit =
      std::make_unique<IRegisterMock<63>>();
  std::unique_ptr<IRegister> channel_64bit =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_65bit =
      std::make_unique<IRegisterMock<65>>();
  std::unique_ptr<IRegister> channel_128bit =
      std::make_unique<IRegisterMock<128>>();
  std::unique_ptr<IRegister> channel_1024bit =
      std::make_unique<IRegisterMock<1024>>();

  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_1bit), 0x0));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_8bit), 0x40));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_16bit), 0x80));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_32bit), 0xC0));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_63bit), 0x100));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_64bit), 0x140));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_65bit), 0x180));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_128bit), 0x200));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_1024bit), 0x280));
}

// Registering on taken offsets should fail
TEST_F(SingleValueManagerTest, RegisterIRegisterOnTakenOffset) {
  std::unique_ptr<IRegister> channel_1bit =
      std::make_unique<IRegisterMock<1>>();
  std::unique_ptr<IRegister> channel_8bit =
      std::make_unique<IRegisterMock<8>>();
  std::unique_ptr<IRegister> channel_16bit =
      std::make_unique<IRegisterMock<16>>();
  std::unique_ptr<IRegister> channel_32bit =
      std::make_unique<IRegisterMock<32>>();
  std::unique_ptr<IRegister> channel_63bit =
      std::make_unique<IRegisterMock<63>>();
  std::unique_ptr<IRegister> channel_64bit =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_65bit =
      std::make_unique<IRegisterMock<65>>();
  std::unique_ptr<IRegister> channel_128bit =
      std::make_unique<IRegisterMock<128>>();
  std::unique_ptr<IRegister> channel_1024bit =
      std::make_unique<IRegisterMock<1024>>();
  std::unique_ptr<IRegister> channel_taken_offset =
      std::make_unique<IRegisterMock<64>>();

  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_1bit), 0x0));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_8bit), 0x8));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_16bit), 0x10));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_32bit), 0x18));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_63bit), 0x20));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_64bit), 0x28));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_65bit), 0x30));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_128bit), 0x40));
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_1024bit), 0x50));

  for (uint64_t offset = 0x0; offset < 0x130; offset += 0x8) {
    EXPECT_TRUE(absl::IsInvalidArgument(
        manager_->RegisterIRegister(std::move(channel_taken_offset), offset)));
  }
}

// Registering the same channel twice on different valid offsets should fail
TEST_F(SingleValueManagerTest, RegisterIRegisterDouble) {
  std::unique_ptr<IRegister> channel_1bit =
      std::make_unique<IRegisterMock<1>>();

  // Test valid channel registrations
  XLS_EXPECT_OK(manager_->RegisterIRegister(std::move(channel_1bit), 0x0));
  EXPECT_THAT(manager_->RegisterIRegister(std::move(channel_1bit), 0x8),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Invalid unique pointer!")));
}

// Registering unaligned offsets should fail
TEST_F(SingleValueManagerTest, RegisterIRegisterUnaligned) {
  std::unique_ptr<IRegister> channel_unaligned_offset =
      std::make_unique<IRegisterMock<64>>();

  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x1)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x3F)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x41)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x13F)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x181)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x1BF)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x1C1)));
  EXPECT_TRUE(absl::IsInvalidArgument(
      manager_->RegisterIRegister(std::move(channel_unaligned_offset), 0x1FF)));
}

// Channels smaller or equal than space available between already registered
// channels should be registered
TEST_F(SingleValueManagerTest, RegisterIRegisterBetweenWillFit) {
  std::unique_ptr<IRegister> channel_64bit_first =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_64bit_second =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_64bit_between =
      std::make_unique<IRegisterMock<64>>();

  XLS_EXPECT_OK(
      manager_->RegisterIRegister(std::move(channel_64bit_first), 0x0));
  // Leave space for channel of width <= 64 at offset 0x40
  XLS_EXPECT_OK(
      manager_->RegisterIRegister(std::move(channel_64bit_second), 0x80));

  XLS_EXPECT_OK(
      manager_->RegisterIRegister(std::move(channel_64bit_between), 0x40));
}

// Channels wider than space available between already registered channels
// should not be registered
TEST_F(SingleValueManagerTest, RegisterIRegisterBetweenWontFit) {
  std::unique_ptr<IRegister> channel_64bit_first =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_64bit_second =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_65bit =
      std::make_unique<IRegisterMock<65>>();
  std::unique_ptr<IRegister> channel_128bit =
      std::make_unique<IRegisterMock<128>>();

  XLS_EXPECT_OK(
      manager_->RegisterIRegister(std::move(channel_64bit_first), 0x0));
  // Leave space for channel of width <= 64 at offset 0x40
  XLS_EXPECT_OK(
      manager_->RegisterIRegister(std::move(channel_64bit_second), 0x10));

  EXPECT_THAT(manager_->RegisterIRegister(std::move(channel_65bit), 0x8),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Channel to big or bad offset")));
  EXPECT_THAT(manager_->RegisterIRegister(std::move(channel_128bit), 0x8),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Channel to big or bad offset")));
}

// Channels should move mapping upper limit
TEST_F(SingleValueManagerTest, RegisterIRegisterPastAddressRange) {
  std::unique_ptr<IRegister> channel_64bit =
      std::make_unique<IRegisterMock<64>>();
  std::unique_ptr<IRegister> channel_128bit =
      std::make_unique<IRegisterMock<128>>();

  EXPECT_THAT(manager_->RegisterIRegister(std::move(channel_64bit), 0x1000),
              xls::status_testing::IsOk());
  EXPECT_EQ(GetMaxOffset(), 0x1008);
  EXPECT_THAT(
      manager_->RegisterIRegister(std::move(channel_128bit), 0x1000 - 0x40),
      xls::status_testing::IsOk());
  EXPECT_EQ(GetMaxOffset(), 0x1008);
}

class SingleValueManagerTestWithStub : public ::testing::Test {
 protected:
  static const uint64_t base_address = 0x0;

  std::unique_ptr<SingleValueManager> manager_;

  SingleValueManagerTestWithStub() {
    manager_ = std::unique_ptr<SingleValueManager>(
        new SingleValueManager(base_address));
  }
  void SetUp() {
    std::unique_ptr<IRegister> channel_64bit =
        std::make_unique<IRegisterStub>();
    XLS_ASSERT_OK(manager_->RegisterIRegister(std::move(channel_64bit), 0));
  }
  ~SingleValueManagerTestWithStub() {}
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(SingleValueManagerTestWithStub, WriteU8AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->WriteU8AtAddress(0x0, 0xDE));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::SetPayloadData8()"));
}

TEST_F(SingleValueManagerTestWithStub, WriteU16AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->WriteU16AtAddress(0x0, 0xDEAD));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::SetPayloadData16()"));
}

TEST_F(SingleValueManagerTestWithStub, WriteU32AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->WriteU32AtAddress(0x0, 0xDEADBEEF));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::SetPayloadData32()"));
}

TEST_F(SingleValueManagerTestWithStub, WriteU64AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->WriteU64AtAddress(0x0, 0xDEADBEEFFEEBDAED));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::SetPayloadData64()"));
}

TEST_F(SingleValueManagerTestWithStub, ReadU8AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->ReadU8AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::GetPayloadData8()"));
}

TEST_F(SingleValueManagerTestWithStub, ReadU16AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->ReadU16AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::GetPayloadData16()"));
}

TEST_F(SingleValueManagerTestWithStub, ReadU32AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->ReadU32AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::GetPayloadData32()"));
}

TEST_F(SingleValueManagerTestWithStub, ReadU64AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(manager_->ReadU64AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IRegisterStub::GetPayloadData64()"));
}

class SingleValueManagerTestWithMocks
    : public ::testing::Test,
      public ::testing::WithParamInterface<uint64_t> {
 protected:
  static const uint64_t base_address = 0x0;

  std::unique_ptr<SingleValueManager> manager_;
  std::array<uint8_t, 8> first_mock_{0xDE, 0xAD, 0xBE, 0xEF,
                                     0xDE, 0xAD, 0xBE, 0xEF};
  std::array<uint8_t, 1> second_mock_{0xAA};

  SingleValueManagerTestWithMocks() {
    manager_ = std::unique_ptr<SingleValueManager>(
        new SingleValueManager(base_address));
  }
  void SetUp() {
    std::unique_ptr<IRegister> channel_64bit =
        std::make_unique<IRegisterMock<64>>(first_mock_);
    std::unique_ptr<IRegister> channel_7bit =
        std::make_unique<IRegisterMock<7>>(second_mock_);
    XLS_ASSERT_OK(manager_->RegisterIRegister(std::move(channel_64bit), 0));
    XLS_ASSERT_OK(manager_->RegisterIRegister(std::move(channel_7bit), 0x8));
  }
  ~SingleValueManagerTestWithMocks() {}
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_P(SingleValueManagerTestWithMocks, ReadU8) {
  EXPECT_THAT(this->manager_->ReadU8AtAddress(GetParam()),
              xls::status_testing::IsOkAndHolds(first_mock_[GetParam()]));
}

TEST_P(SingleValueManagerTestWithMocks, ReadU16) {
  std::array<uint8_t, 9> first_second_concatenation;
  for (int i = 0; i < first_mock_.size(); ++i)
    first_second_concatenation[i] = first_mock_[i];
  first_second_concatenation[8] = second_mock_[0] & 0x7f;
  uint16_t expected_output = first_second_concatenation[GetParam()] |
                             (first_second_concatenation[GetParam() + 1] << 8);
  EXPECT_THAT(this->manager_->ReadU16AtAddress(GetParam()),
              xls::status_testing::IsOkAndHolds(expected_output));
}

TEST_P(SingleValueManagerTestWithMocks, ReadU32) {
  std::array<uint8_t, 11> first_second_concatenation;
  first_second_concatenation.fill(0);
  for (int i = 0; i < first_mock_.size(); ++i)
    first_second_concatenation[i] = first_mock_[i];
  first_second_concatenation[8] = second_mock_[0] & 0x7f;
  uint32_t expected_output = 0;
  for (int i = 0; i < 4; ++i) {
    expected_output |= first_second_concatenation[GetParam() + i] << (8 * i);
  }
  EXPECT_THAT(this->manager_->ReadU32AtAddress(GetParam()),
              xls::status_testing::IsOkAndHolds(expected_output));
}

TEST_P(SingleValueManagerTestWithMocks, ReadU64) {
  std::array<uint8_t, 15> first_second_concatenation;
  first_second_concatenation.fill(0);
  for (int i = 0; i < first_mock_.size(); ++i)
    first_second_concatenation[i] = first_mock_[i];
  first_second_concatenation[8] = second_mock_[0] & 0x7f;
  uint64_t expected_output = 0;
  for (int i = 0; i < 8; ++i) {
    expected_output |= ((uint64_t)first_second_concatenation[GetParam() + i])
                       << (8 * i);
  }
  EXPECT_THAT(this->manager_->ReadU64AtAddress(GetParam()),
              xls::status_testing::IsOkAndHolds(expected_output));
}

INSTANTIATE_TEST_SUITE_P(RuntimeManagerTestInstantiation,
                         SingleValueManagerTestWithMocks,
                         testing::Values(0, 1, 2, 3, 4, 5, 6, 7));

}  // namespace
}  // namespace xls::simulation::generic

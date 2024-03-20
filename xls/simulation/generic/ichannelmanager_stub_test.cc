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

#include "xls/simulation/generic/ichannelmanager_stub.h"

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
#include "xls/simulation/generic/istream_stub.h"

namespace xls::simulation::generic {
namespace {

class IChannelManagerStubTest : public ::testing::Test {
 protected:
  IChannelManagerStub* stub;

  IChannelManagerStubTest() { stub = new IChannelManagerStub(0x0, 0x200); }
  ~IChannelManagerStubTest() { delete stub; }
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(IChannelManagerStubTest, GetBaseAddress) {
  EXPECT_EQ(stub->GetBaseAddress(), 0x0);
}

TEST_F(IChannelManagerStubTest, InRange) {
  EXPECT_FALSE(stub->InRange(-1));
  EXPECT_TRUE(stub->InRange(0x0));
  EXPECT_TRUE(stub->InRange(0x1));
  EXPECT_TRUE(stub->InRange(0x8));
  EXPECT_FALSE(stub->InRange(0x200));
  EXPECT_FALSE(stub->InRange(0x201));
}

TEST_F(IChannelManagerStubTest, WriteU8AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->WriteU8AtAddress(0x0, 0xDE));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::WriteU8AtAddress()"));
}

TEST_F(IChannelManagerStubTest, WriteU16AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->WriteU16AtAddress(0x0, 0xDEAD));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::WriteU16AtAddress()"));
}

TEST_F(IChannelManagerStubTest, WriteU32AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->WriteU32AtAddress(0x0, 0xDEADBEEF));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::WriteU32AtAddress()"));
}

TEST_F(IChannelManagerStubTest, WriteU64AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->WriteU64AtAddress(0x0, 0xDEADBEEFFEEBDAED));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::WriteU64AtAddress()"));
}

TEST_F(IChannelManagerStubTest, ReadU8AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->ReadU8AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::ReadU8AtAddress()"));
}

TEST_F(IChannelManagerStubTest, ReadU16AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->ReadU16AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::ReadU16AtAddress()"));
}

TEST_F(IChannelManagerStubTest, ReadU32AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->ReadU32AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::ReadU32AtAddress()"));
}

TEST_F(IChannelManagerStubTest, ReadU64AtAddress) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub->ReadU64AtAddress(0x0));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("IChannelManagerStub::ReadU64AtAddress()"));
}

}  // namespace
}  // namespace xls::simulation::generic

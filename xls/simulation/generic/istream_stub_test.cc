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

#include "xls/simulation/generic/istream_stub.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic {
namespace {

class IStreamStubTest : public ::testing::Test {
 protected:
  IStreamStub stub;

  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(IStreamStubTest, IsReadStream) {
  ::testing::internal::CaptureStderr();
  stub.IsReadStream();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::IsReadStream"));
}

TEST_F(IStreamStubTest, IsReady) {
  ::testing::internal::CaptureStderr();
  stub.IsReady();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::IsReady"));
}

TEST_F(IStreamStubTest, Transfer) {
  ::testing::internal::CaptureStderr();
  auto result = stub.Transfer();
  auto output = ::testing::internal::GetCapturedStderr();

  XLS_EXPECT_OK(result);
  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::Transfer"));
}

TEST_F(IStreamStubTest, GetChannelWidth) {
  ::testing::internal::CaptureStderr();
  stub.GetChannelWidth();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::GetChannelWidth"));
}

TEST_F(IStreamStubTest, GetPayloadData8) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData8(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::GetPayloadData8"));
}

TEST_F(IStreamStubTest, GetPayloadData16) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData16(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::GetPayloadData16"));
}

TEST_F(IStreamStubTest, GetPayloadData32) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData32(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::GetPayloadData32"));
}

TEST_F(IStreamStubTest, GetPayloadData64) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData64(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::GetPayloadData64"));
}

TEST_F(IStreamStubTest, SetPayloadData8) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData8(42, 0x12));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::SetPayloadData8"));
}

TEST_F(IStreamStubTest, SetPayloadData16) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData16(42, 0x1234));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::SetPayloadData16"));
}

TEST_F(IStreamStubTest, SetPayloadData32) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData32(42, 0x12345678));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::SetPayloadData32"));
}

TEST_F(IStreamStubTest, SetPayloadData64) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData64(42, 0x1234567890ABCDEFll));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IStreamStub::SetPayloadData64"));
}

}  // namespace
}  // namespace xls::simulation::generic

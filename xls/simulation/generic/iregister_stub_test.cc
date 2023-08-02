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

#include "xls/simulation/generic/iregister_stub.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic {
namespace {

class IRegisterStubTest : public ::testing::Test {
 protected:
  IRegisterStub stub;

  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(IRegisterStubTest, GetChannelWidth) {
  ::testing::internal::CaptureStderr();
  stub.GetChannelWidth();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::GetChannelWidth"));
}

TEST_F(IRegisterStubTest, GetPayloadData8) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData8(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::GetPayloadData8"));
}

TEST_F(IRegisterStubTest, GetPayloadData16) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData16(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::GetPayloadData16"));
}

TEST_F(IRegisterStubTest, GetPayloadData32) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData32(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::GetPayloadData32"));
}

TEST_F(IRegisterStubTest, GetPayloadData64) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData64(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::GetPayloadData64"));
}

TEST_F(IRegisterStubTest, SetPayloadData8) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData8(42, 0x12));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::SetPayloadData8"));
}

TEST_F(IRegisterStubTest, SetPayloadData16) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData16(42, 0x1234));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::SetPayloadData16"));
}

TEST_F(IRegisterStubTest, SetPayloadData32) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData32(42, 0x12345678));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::SetPayloadData32"));
}

TEST_F(IRegisterStubTest, SetPayloadData64) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData64(42, 0x1234567890ABCDEFll));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IRegisterStub::SetPayloadData64"));
}

}  // namespace
}  // namespace xls::simulation::generic

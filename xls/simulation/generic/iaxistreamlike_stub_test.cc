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

#include "xls/simulation/generic/iaxistreamlike_stub.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic {
namespace {

class IAxiStreamLikeStubTest : public ::testing::Test {
 protected:
  IAxiStreamLikeStub stub;

  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_F(IAxiStreamLikeStubTest, GetSymbolWidth) {
  ::testing::internal::CaptureStderr();
  stub.GetSymbolWidth();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::GetSymbolWidth"));
}

TEST_F(IAxiStreamLikeStubTest, SetDataValid) {
  ::testing::internal::CaptureStderr();
  stub.SetDataValid({true, false});
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::SetDataValid"));
}

TEST_F(IAxiStreamLikeStubTest, GetDataValid) {
  ::testing::internal::CaptureStderr();
  stub.GetDataValid();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::GetDataValid"));
}

TEST_F(IAxiStreamLikeStubTest, SetLast) {
  ::testing::internal::CaptureStderr();
  stub.SetLast(true);
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::SetLast"));
}

TEST_F(IAxiStreamLikeStubTest, GetLast) {
  ::testing::internal::CaptureStderr();
  stub.GetLast();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::GetLast"));
}

TEST_F(IAxiStreamLikeStubTest, IsReadStream) {
  ::testing::internal::CaptureStderr();
  stub.IsReadStream();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::IsReadStream"));
}

TEST_F(IAxiStreamLikeStubTest, IsReady) {
  ::testing::internal::CaptureStderr();
  stub.IsReady();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::IsReady"));
}

TEST_F(IAxiStreamLikeStubTest, Transfer) {
  ::testing::internal::CaptureStderr();
  auto result = stub.Transfer();
  auto output = ::testing::internal::GetCapturedStderr();

  XLS_EXPECT_OK(result);
  EXPECT_THAT(output, ::testing::HasSubstr("IAxiStreamLikeStub::Transfer"));
}

TEST_F(IAxiStreamLikeStubTest, GetChannelWidth) {
  ::testing::internal::CaptureStderr();
  stub.GetChannelWidth();
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::GetChannelWidth"));
}

TEST_F(IAxiStreamLikeStubTest, GetPayloadData8) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData8(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::GetPayloadData8"));
}

TEST_F(IAxiStreamLikeStubTest, GetPayloadData16) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData16(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::GetPayloadData16"));
}

TEST_F(IAxiStreamLikeStubTest, GetPayloadData32) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData32(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::GetPayloadData32"));
}

TEST_F(IAxiStreamLikeStubTest, GetPayloadData64) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.GetPayloadData64(42));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::GetPayloadData64"));
}

TEST_F(IAxiStreamLikeStubTest, SetPayloadData8) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData8(42, 0x12));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::SetPayloadData8"));
}

TEST_F(IAxiStreamLikeStubTest, SetPayloadData16) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData16(42, 0x1234));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::SetPayloadData16"));
}

TEST_F(IAxiStreamLikeStubTest, SetPayloadData32) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData32(42, 0x12345678));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::SetPayloadData32"));
}

TEST_F(IAxiStreamLikeStubTest, SetPayloadData64) {
  ::testing::internal::CaptureStderr();
  XLS_EXPECT_OK(stub.SetPayloadData64(42, 0x1234567890ABCDEFll));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              ::testing::HasSubstr("IAxiStreamLikeStub::SetPayloadData64"));
}

}  // namespace
}  // namespace xls::simulation::generic

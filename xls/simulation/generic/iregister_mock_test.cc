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

#include "xls/simulation/generic/iregister_mock.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic {
namespace {

using testing::ElementsAre;

class IRegisterMockTest : public ::testing::Test {};

TEST_F(IRegisterMockTest, Init1) {
  IRegisterMock<17> mock;
  ASSERT_THAT(mock.bytes(), ElementsAre(0, 0, 0));
}

TEST_F(IRegisterMockTest, Init2) {
  IRegisterMock<31> mock{{1, 2, 3, 4}};
  ASSERT_THAT(mock.bytes(), ElementsAre(1, 2, 3, 4));
}

TEST_F(IRegisterMockTest, SetBytes) {
  IRegisterMock<31> mock{};
  mock.bytes() = {3, 4, 5, 6};
  ASSERT_THAT(mock.bytes(), ElementsAre(3, 4, 5, 6));
}

TEST_F(IRegisterMockTest, GetChannelWidth) {
  IRegisterMock<31> mock{};
  ASSERT_EQ(mock.GetChannelWidth(), 31);
}

TEST_F(IRegisterMockTest, GetPayloadData8) {
  IRegisterMock<17> mock{{1, 2, 3}};
  XLS_ASSERT_OK_AND_ASSIGN(auto data, mock.GetPayloadData8(1));
  EXPECT_EQ(data, 0x02);
  EXPECT_THAT(mock.GetPayloadData8(3),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, GetPayloadData16) {
  IRegisterMock<35> mock{{1, 2, 3, 4, 5}};
  XLS_ASSERT_OK_AND_ASSIGN(auto data, mock.GetPayloadData16(3));
  EXPECT_EQ(data, 0x0504);
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData16(4));
  EXPECT_EQ(data, 0x05);
  EXPECT_THAT(mock.GetPayloadData16(5),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, GetPayloadData32) {
  IRegisterMock<35> mock{{1, 2, 3, 4, 5}};
  XLS_ASSERT_OK_AND_ASSIGN(auto data, mock.GetPayloadData32(1));
  EXPECT_EQ(data, 0x05040302);
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData32(2));
  EXPECT_EQ(data, 0x050403);
  EXPECT_THAT(mock.GetPayloadData32(5),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, GetPayloadData64) {
  IRegisterMock<70> mock{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
  XLS_ASSERT_OK_AND_ASSIGN(auto data, mock.GetPayloadData64(1));
  EXPECT_EQ(data, 0x0908070605040302LL);
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData64(4));
  EXPECT_EQ(data, 0x0908070605LL);
  EXPECT_THAT(mock.GetPayloadData64(9),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, CheckBitmasking) {
  IRegisterMock<3> mock{};
  XLS_EXPECT_OK(mock.SetPayloadData8(0, 1));
  ASSERT_THAT(mock.bytes(), ElementsAre(1));
  XLS_EXPECT_OK(mock.SetPayloadData8(0, 7));
  ASSERT_THAT(mock.bytes(), ElementsAre(7));
  XLS_EXPECT_OK(mock.SetPayloadData8(0, 8));
  ASSERT_THAT(mock.bytes(), ElementsAre(0));
  XLS_EXPECT_OK(mock.SetPayloadData8(0, 9));
  ASSERT_THAT(mock.bytes(), ElementsAre(1));
}

TEST_F(IRegisterMockTest, SetPayloadData8) {
  IRegisterMock<17> mock{};
  XLS_EXPECT_OK(mock.SetPayloadData8(0, 1));
  XLS_EXPECT_OK(mock.SetPayloadData8(1, 2));
  XLS_EXPECT_OK(mock.SetPayloadData8(2, 3));
  ASSERT_THAT(mock.bytes(), ElementsAre(1, 2, 1));
  XLS_EXPECT_OK(mock.SetPayloadData8(1, 4));
  ASSERT_THAT(mock.bytes(), ElementsAre(1, 4, 1));
  EXPECT_THAT(mock.SetPayloadData8(3, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, SetPayloadData16) {
  IRegisterMock<17> mock{};
  XLS_EXPECT_OK(mock.SetPayloadData16(1, 0x0503));
  XLS_EXPECT_OK(mock.SetPayloadData16(0, 0x0506));
  ASSERT_THAT(mock.bytes(), ElementsAre(6, 5, 1));
  XLS_EXPECT_OK(mock.SetPayloadData16(2, 0));
  ASSERT_THAT(mock.bytes(), ElementsAre(6, 5, 0));
  EXPECT_THAT(mock.SetPayloadData16(3, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, SetPayloadData32) {
  IRegisterMock<33> mock{};
  XLS_EXPECT_OK(mock.SetPayloadData32(0, 0x03020100));
  XLS_EXPECT_OK(mock.SetPayloadData32(1, 0x07060504));
  ASSERT_THAT(mock.bytes(), ElementsAre(0, 4, 5, 6, 1));
  EXPECT_THAT(mock.SetPayloadData32(5, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(IRegisterMockTest, SetPayloadData64) {
  IRegisterMock<70> mock{};
  XLS_EXPECT_OK(mock.SetPayloadData64(0, 0x0102030405060708LL));
  XLS_EXPECT_OK(mock.SetPayloadData64(1, 0x0102030405060708LL));
  ASSERT_THAT(mock.bytes(), ElementsAre(8, 8, 7, 6, 5, 4, 3, 2, 1));
  EXPECT_THAT(mock.SetPayloadData64(9, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xls::simulation::generic

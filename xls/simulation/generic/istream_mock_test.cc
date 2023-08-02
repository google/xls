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

#include "xls/simulation/generic/istream_mock.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic {
namespace {

using absl::StatusCode;
using ::testing::_;
using testing::ElementsAre;
using ::xls::status_testing::StatusIs;

class IStreamMockTest : public ::testing::Test {};

TEST_F(IStreamMockTest, InitRead) {
  IStreamMock<17, true> mock;
  ASSERT_EQ(mock.IsReadStream(), true);
  ASSERT_EQ(mock.GetChannelWidth(), 17);
  ASSERT_EQ(mock.IsReady(), false);
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(0, 0, 0));
  ASSERT_EQ(mock.fifo().empty(), true);
}

TEST_F(IStreamMockTest, InitWrite) {
  IStreamMock<17, false> mock;
  ASSERT_EQ(mock.IsReadStream(), false);
  ASSERT_EQ(mock.GetChannelWidth(), 17);
  ASSERT_EQ(mock.IsReady(), true);
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(0, 0, 0));
  ASSERT_EQ(mock.fifo().empty(), true);
}

TEST_F(IStreamMockTest, HoldingRegSet8) {
  IStreamMock<17, true> mock;
  XLS_EXPECT_OK(mock.SetPayloadData8(1, 5));
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(0, 5, 0));
  XLS_EXPECT_OK(mock.SetPayloadData8(0, 6));
  XLS_EXPECT_OK(mock.SetPayloadData8(2, 7));
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(6, 5, 1));
}

TEST_F(IStreamMockTest, HoldingRegGet8) {
  IStreamMock<17, true> mock;
  mock.holding_reg().bytes() = {1, 2, 3};
  uint8_t data;
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData8(0));
  EXPECT_EQ(data, 1);
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData8(1));
  EXPECT_EQ(data, 2);
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData8(2));
  EXPECT_EQ(data, 3);
}

TEST_F(IStreamMockTest, HoldingRegSet16) {
  IStreamMock<17, true> mock;
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(0, 0, 0));
  XLS_EXPECT_OK(mock.SetPayloadData16(0, 0x0203));
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(3, 2, 0));
  XLS_EXPECT_OK(mock.SetPayloadData16(1, 0x0505));
  ASSERT_THAT(mock.holding_reg().bytes(), ElementsAre(3, 5, 1));
}

TEST_F(IStreamMockTest, HoldingRegGet16) {
  IStreamMock<17, true> mock;
  mock.holding_reg().bytes() = {1, 2, 3};
  uint16_t data;
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData16(0));
  EXPECT_EQ(data, 0x0201);
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData16(1));
  EXPECT_EQ(data, 0x0302);
}

TEST_F(IStreamMockTest, Write) {
  IStreamMock<16, false> mock;
  ASSERT_EQ(mock.IsReady(), true);
  XLS_EXPECT_OK(mock.SetPayloadData16(0, 0x0302));
  ASSERT_THAT(mock.Transfer(), StatusIs(StatusCode::kOk));
  XLS_EXPECT_OK(mock.SetPayloadData16(0, 0x0504));
  ASSERT_THAT(mock.Transfer(), StatusIs(StatusCode::kOk));
  ASSERT_EQ(mock.fifo().size(), 2);
  ASSERT_THAT(mock.fifo().front(), ElementsAre(2, 3));
  mock.fifo().pop();
  ASSERT_THAT(mock.fifo().front(), ElementsAre(4, 5));
}

TEST_F(IStreamMockTest, Read) {
  IStreamMock<16, true> mock;
  ASSERT_EQ(mock.IsReady(), false);
  mock.fifo().push({{2, 3}});
  mock.fifo().push({{4, 5}});
  ASSERT_EQ(mock.IsReady(), true);
  ASSERT_THAT(mock.Transfer(), StatusIs(StatusCode::kOk));
  uint16_t data;
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData16(0));
  EXPECT_EQ(data, 0x0302);
  ASSERT_EQ(mock.IsReady(), true);
  ASSERT_THAT(mock.Transfer(), StatusIs(StatusCode::kOk));
  XLS_ASSERT_OK_AND_ASSIGN(data, mock.GetPayloadData16(0));
  EXPECT_EQ(data, 0x0504);
  ASSERT_EQ(mock.IsReady(), false);
  ASSERT_THAT(mock.Transfer(), StatusIs(StatusCode::kInternal));
}

}  // namespace
}  // namespace xls::simulation::generic

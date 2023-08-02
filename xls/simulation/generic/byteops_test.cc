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

#include "xls/simulation/generic/byteops.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic::byteops {
namespace {

using testing::ElementsAre;

class ByteopsTest : public ::testing::Test {};

TEST_F(ByteopsTest, BytesReadWord8) {
  std::vector<uint8_t> data{{1, 2, 3}};
  EXPECT_THAT(bytes_read_word<uint8_t>(data, 0),
              xls::status_testing::IsOkAndHolds(0x01));
  EXPECT_THAT(bytes_read_word<uint8_t>(data, 1),
              xls::status_testing::IsOkAndHolds(0x02));
  EXPECT_THAT(bytes_read_word<uint8_t>(data, 2),
              xls::status_testing::IsOkAndHolds(0x03));
  EXPECT_THAT(bytes_read_word<uint8_t>(data, 3),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(bytes_read_word<uint8_t>(data, 4),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesReadWord16) {
  std::vector<uint8_t> data{{1, 2, 3, 4, 5}};
  EXPECT_THAT(bytes_read_word<uint16_t>(data, 0),
              xls::status_testing::IsOkAndHolds(0x0201));
  EXPECT_THAT(bytes_read_word<uint16_t>(data, 3),
              xls::status_testing::IsOkAndHolds(0x0504));
  EXPECT_THAT(bytes_read_word<uint16_t>(data, 4),
              xls::status_testing::IsOkAndHolds(0x05));
  EXPECT_THAT(bytes_read_word<uint16_t>(data, 5),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesReadWord32) {
  std::vector<uint8_t> data{{1, 2, 3, 4, 5}};
  EXPECT_THAT(bytes_read_word<uint32_t>(data, 0),
              xls::status_testing::IsOkAndHolds(0x04030201));
  EXPECT_THAT(bytes_read_word<uint32_t>(data, 1),
              xls::status_testing::IsOkAndHolds(0x05040302));
  EXPECT_THAT(bytes_read_word<uint32_t>(data, 2),
              xls::status_testing::IsOkAndHolds(0x050403));
  EXPECT_THAT(bytes_read_word<uint32_t>(data, 5),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesReadWord64) {
  std::vector<uint8_t> data{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
  EXPECT_THAT(bytes_read_word<uint64_t>(data, 0),
              xls::status_testing::IsOkAndHolds(0x0807060504030201ULL));
  EXPECT_THAT(bytes_read_word<uint64_t>(data, 1),
              xls::status_testing::IsOkAndHolds(0x0908070605040302ULL));
  EXPECT_THAT(bytes_read_word<uint64_t>(data, 2),
              xls::status_testing::IsOkAndHolds(0x09080706050403ULL));
  EXPECT_THAT(bytes_read_word<uint64_t>(data, 9),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesWriteWord8) {
  std::vector<uint8_t> data(3);
  EXPECT_THAT(bytes_write_word<uint8_t>(absl::MakeSpan(data), 0, 1),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(1, 0, 0));
  EXPECT_THAT(bytes_write_word<uint8_t>(absl::MakeSpan(data), 1, 2),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(1, 2, 0));
  EXPECT_THAT(bytes_write_word<uint8_t>(absl::MakeSpan(data), 2, 3),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(1, 2, 3));
  EXPECT_THAT(bytes_write_word<uint8_t>(absl::MakeSpan(data), 1, 4),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(1, 4, 3));
  EXPECT_THAT(bytes_write_word<uint8_t>(absl::MakeSpan(data), 3, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesWriteWord16) {
  std::vector<uint8_t> data(3);
  EXPECT_THAT(bytes_write_word<uint16_t>(absl::MakeSpan(data), 0, 0x0201),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(1, 2, 0));
  EXPECT_THAT(bytes_write_word<uint16_t>(absl::MakeSpan(data), 1, 0x0403),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(1, 3, 4));
  EXPECT_THAT(bytes_write_word<uint16_t>(absl::MakeSpan(data), 0, 0x0506),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(6, 5, 4));
  EXPECT_THAT(bytes_write_word<uint16_t>(absl::MakeSpan(data), 2, 0),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(6, 5, 0));
  EXPECT_THAT(bytes_write_word<uint16_t>(absl::MakeSpan(data), 3, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesWriteWord32) {
  std::vector<uint8_t> data(5);
  EXPECT_THAT(bytes_write_word<uint32_t>(absl::MakeSpan(data), 0, 0x03020100),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(0, 1, 2, 3, 0));
  EXPECT_THAT(bytes_write_word<uint32_t>(absl::MakeSpan(data), 1, 0x07060504),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(0, 4, 5, 6, 7));
  EXPECT_THAT(bytes_write_word<uint32_t>(absl::MakeSpan(data), 3, 0),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(0, 4, 5, 0, 0));
  EXPECT_THAT(bytes_write_word<uint32_t>(absl::MakeSpan(data), 5, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ByteopsTest, BytesWriteWord64) {
  std::vector<uint8_t> data(9);
  EXPECT_THAT(
      bytes_write_word<uint64_t>(absl::MakeSpan(data), 0, 0x0102030405060708LL),
      xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(8, 7, 6, 5, 4, 3, 2, 1, 0));
  EXPECT_THAT(
      bytes_write_word<uint64_t>(absl::MakeSpan(data), 1, 0x0102030405060708LL),
      xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(8, 8, 7, 6, 5, 4, 3, 2, 1));
  EXPECT_THAT(bytes_write_word<uint64_t>(absl::MakeSpan(data), 4, 0),
              xls::status_testing::IsOk());
  ASSERT_THAT(data, ElementsAre(8, 8, 7, 6, 0, 0, 0, 0, 0));
  EXPECT_THAT(bytes_write_word<uint64_t>(absl::MakeSpan(data), 9, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xls::simulation::generic::byteops

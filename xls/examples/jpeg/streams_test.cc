// Copyright 2021 The XLS Authors
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

#include "xls/examples/jpeg/streams.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/jpeg/constants.h"

namespace xls::jpeg {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::HasSubstr;

using Popper = std::function<absl::StatusOr<std::optional<uint8_t>>()>;

Popper EmptyPopper() {
  return []() { return std::nullopt; };
}

TEST(ByteStreamTest, ByteStreamEmpty) {
  ByteStream bs(EmptyPopper());
  EXPECT_EQ(bs.popped_index(), 0);
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(true));
  EXPECT_THAT(bs.PeekEof(), IsOkAndHolds(std::nullopt));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(true));
  EXPECT_EQ(bs.popped_index(), 0);
}

TEST(ByteStreamTest, PopHiLo) {
  const std::vector<uint8_t> bytes = {0xaa, 0xbb};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_EQ(bs.popped_index(), 0);
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(false));
  EXPECT_THAT(bs.PopHiLo(), IsOkAndHolds(0xaabb));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(true));
  EXPECT_EQ(bs.popped_index(), 2);
}

TEST(ByteStreamTest, PopHiLoEof) {
  const std::vector<uint8_t> bytes = {0xaa};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_THAT(bs.PopHiLo(), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Unexpected end")));
}

TEST(ByteStreamTest, PeekPopToEof) {
  const std::vector<uint8_t> bytes = {0xaa};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(false));
  EXPECT_THAT(bs.PeekEof(), IsOkAndHolds(0xaa));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(false));
  EXPECT_THAT(bs.Pop(), IsOkAndHolds(0xaa));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(true));
  EXPECT_THAT(bs.PeekEof(), IsOkAndHolds(std::nullopt));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(true));
}

TEST(ByteStreamTest, Pop2xU4) {
  const std::vector<uint8_t> bytes = {0xab};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_THAT(bs.Pop2xU4(), IsOkAndHolds(std::make_pair(0xa, 0xb)));
}

TEST(ByteStreamTest, DropExpectedEmpty) {
  ByteStream bs(EmptyPopper());
  EXPECT_THAT(bs.DropExpected(0xaa, "message here"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected end")));
  EXPECT_THAT(bs.DropExpected(0xaa, "another"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected end")));
}

TEST(ByteStreamTest, DropExpectedThenEmpty) {
  const std::vector<uint8_t> bytes = {0xaa};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  XLS_EXPECT_OK(bs.DropExpected(0xaa, "whoo"));
  EXPECT_THAT(bs.DropExpected(0xaa, "another"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected end")));
}

TEST(ByteStreamTest, DropExpectedMulti) {
  const std::vector<uint8_t> bytes = {'a', 'b', 'c', 0xff};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  XLS_EXPECT_OK(bs.DropExpectedMulti({'a', 'b', 'c'}, "should be ok"));
  XLS_EXPECT_OK(bs.DropExpected(0xff, "another"));
}

TEST(ByteStreamTest, Pop) {
  const std::vector<uint8_t> bytes = {'a', 'b', 'c'};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_THAT(bs.Pop(), IsOkAndHolds('a'));
  EXPECT_THAT(bs.Pop(), IsOkAndHolds('b'));
  EXPECT_THAT(bs.Pop(), IsOkAndHolds('c'));
  EXPECT_THAT(bs.Pop(), StatusIs(absl::StatusCode::kInvalidArgument,
                                 HasSubstr("Unexpected end")));
}

TEST(ByteStreamTest, PopNInBounds) {
  const std::vector<uint8_t> bytes = {'a', 'b', 'c'};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_THAT(bs.PopN<3>(),
              IsOkAndHolds(std::array<uint8_t, 3>{'a', 'b', 'c'}));
}

TEST(ByteStreamTest, PopNInBoundsDynamic) {
  const std::vector<uint8_t> bytes = {'a', 'b', 'c'};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  std::vector<uint8_t> want = {'a', 'b', 'c'};
  EXPECT_THAT(bs.PopN(3), IsOkAndHolds(want));
}

TEST(ByteStreamTest, PopNOutOfBounds) {
  const std::vector<uint8_t> bytes = {'a', 'b', 'c'};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  EXPECT_THAT(bs.PopN<4>(), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Unexpected end")));
}

// -- BitStream

TEST(BitStreamTest, PeekAndPopNBitsToEof) {
  const std::vector<uint8_t> bytes = {0b10101111, 0b0000'0101, 0b0101'0011,
                                      0b1010'1011};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  BitStream bits(&bs);
  EXPECT_THAT(bits.PeekN(3), IsOkAndHolds(0b101));
  EXPECT_THAT(bits.PopN(3), IsOkAndHolds(0b101));
  EXPECT_THAT(bits.PeekN(5), IsOkAndHolds(0b0'1111));
  EXPECT_THAT(bits.PopN(5), IsOkAndHolds(0b0'1111));

  EXPECT_THAT(bits.PopN(16), IsOkAndHolds(0b0000'0101'0101'0011));

  // Final byte.
  EXPECT_THAT(bits.AtEof(), IsOkAndHolds(false));
  EXPECT_THAT(bits.PopN(8), IsOkAndHolds(0b10101011));
  EXPECT_THAT(bits.AtEof(), IsOkAndHolds(true));
}

// Note for the following coefficient-popping tests: when the high bit is set,
// the coefficient being popped is in the positive regime. When the high bit is
// not set, it is in the negative regime.

TEST(BitStreamTest, PopCoeffs) {
  const std::vector<uint8_t> bytes = {0b10'1'01010, 0b1'0000000};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  BitStream bits(&bs);
  EXPECT_THAT(bits.PopCoeff(2), IsOkAndHolds(0b10));
  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(0b1));
  EXPECT_THAT(bits.PopCoeff(5), IsOkAndHolds(-21));
  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(1));
  EXPECT_THAT(bits.PopCoeff(0), IsOkAndHolds(0));
  EXPECT_THAT(bits.PopCoeff(7), IsOkAndHolds(-127));
}

TEST(BitStreamTest, PopCoeffsPositive) {
  const std::vector<uint8_t> bytes = {0b1'10'100'10, 0b00'10000'1, 0x80, 0x01};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  BitStream bits(&bs);
  EXPECT_THAT(bits.PopCoeff(0), IsOkAndHolds(0));

  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(0b1));
  EXPECT_THAT(bits.PopCoeff(2), IsOkAndHolds(0b10));
  EXPECT_THAT(bits.PopCoeff(3), IsOkAndHolds(0b100));
  EXPECT_THAT(bits.PopCoeff(4), IsOkAndHolds(0b1000));
  EXPECT_THAT(bits.PopCoeff(5), IsOkAndHolds(0b10000));
  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(0b1));

  EXPECT_THAT(bits.PopCoeff(15), IsOkAndHolds(0x4000));
  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(0b1));
}

TEST(BitStreamTest, PopCoeffsNegative) {
  const std::vector<uint8_t> bytes = {0b0000'0000, 0b0000'0000, 0x00, 0x00};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  BitStream bits(&bs);
  EXPECT_THAT(bits.PopCoeff(0), IsOkAndHolds(0));
  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(-1));
  EXPECT_THAT(bits.PopCoeff(2), IsOkAndHolds(-3));
  EXPECT_THAT(bits.PopCoeff(3), IsOkAndHolds(-7));
  EXPECT_THAT(bits.PopCoeff(4), IsOkAndHolds(-15));
  EXPECT_THAT(bits.PopCoeff(5), IsOkAndHolds(-31));
  EXPECT_THAT(bits.PopCoeff(1), IsOkAndHolds(-1));
  EXPECT_THAT(bits.PopCoeff(15), IsOkAndHolds(-32767));
}

// The EOI marker being present at the head of the bit stream is special -- we
// pop the marker after the marker indicator byte and see that it's an end of
// image indicator.
TEST(BitStreamTest, PoppingThroughEoiMarker) {
  const std::vector<uint8_t> bytes = {0xab, 0xff, kEoiMarker};
  SpanPopper sp(bytes);
  ByteStream bs(sp);
  BitStream bits(&bs);
  EXPECT_THAT(bits.PopN(4), IsOkAndHolds(0xa));
  EXPECT_THAT(bits.PopN(4), IsOkAndHolds(0xb));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(false));
  EXPECT_THAT(bits.PopN(8), IsOkAndHolds(0xff));
  EXPECT_THAT(bs.AtEof(), IsOkAndHolds(true));
}

}  // namespace
}  // namespace xls::jpeg

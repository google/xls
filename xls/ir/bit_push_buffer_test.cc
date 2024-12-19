// Copyright 2020 The XLS Authors
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

#include "xls/ir/bit_push_buffer.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using ::testing::IsEmpty;

TEST(BitPushBufferTest, IsEmptyAfterConstruction) {
  BitPushBuffer buffer;

  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size_in_bytes(), 0);
  EXPECT_EQ(buffer.size_in_bits(), 0);
  EXPECT_THAT(buffer.GetUint8DataWithMsbPadding(), IsEmpty());
  EXPECT_THAT(buffer.GetUint8DataWithLsbPadding(), IsEmpty());
}

TEST(BitPushBufferTest, HasSingle0AfterPushingFalse) {
  BitPushBuffer buffer;

  buffer.PushBit(false);

  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size_in_bytes(), 1);
  EXPECT_EQ(buffer.GetUint8DataWithMsbPadding(), std::vector<uint8_t>{0});
  EXPECT_EQ(buffer.GetUint8DataWithLsbPadding(), std::vector<uint8_t>{0});
  EXPECT_EQ(buffer.size_in_bits(), 1);
}

TEST(BitPushBufferTest, HasSingle1InMsbAfterPushingTrue) {
  BitPushBuffer buffer;

  buffer.PushBit(true);

  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size_in_bytes(), 1);
  EXPECT_EQ(buffer.GetUint8DataWithLsbPadding(), std::vector<uint8_t>{1 << 7});
  EXPECT_EQ(buffer.GetUint8DataWithMsbPadding(), std::vector<uint8_t>{1});
  EXPECT_EQ(buffer.size_in_bits(), 1);
}

TEST(BitPushBufferTest, Has1InSecondMsbAfterPushingFalseTrue) {
  BitPushBuffer buffer;

  buffer.PushBit(false);
  buffer.PushBit(true);

  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size_in_bytes(), 1);
  EXPECT_EQ(buffer.GetUint8DataWithLsbPadding(),
            std::vector<uint8_t>{0b01 << 6});
  EXPECT_EQ(buffer.GetUint8DataWithMsbPadding(), std::vector<uint8_t>{0b01});
  EXPECT_EQ(buffer.size_in_bits(), 2);
}

TEST(BitPushBufferTest, IsOneByteAfterPushing8Values) {
  BitPushBuffer buffer;

  for (int i = 0; i < 8; i++) {
    buffer.PushBit(false);
  }

  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size_in_bytes(), 1);
  EXPECT_EQ(buffer.size_in_bits(), 8);
}

TEST(BitPushBufferTest, Has1SecondBytesMsbAfterPushing8False1True) {
  BitPushBuffer buffer;

  for (int i = 0; i < 8; i++) {
    buffer.PushBit(false);
  }
  buffer.PushBit(true);

  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size_in_bytes(), 2);
  EXPECT_EQ(buffer.GetUint8DataWithLsbPadding(),
            std::vector<uint8_t>({0, 1 << 7}));
  EXPECT_EQ(buffer.GetUint8DataWithMsbPadding(), std::vector<uint8_t>({0, 1}));
  EXPECT_EQ(buffer.size_in_bits(), 9);
}

TEST(BitPushBufferTest, ToString) {
  BitPushBuffer buffer;
  buffer.PushBit(true);
  buffer.PushBit(false);
  EXPECT_EQ(buffer.ToString(), "0b10");
}

}  // namespace
}  // namespace xls

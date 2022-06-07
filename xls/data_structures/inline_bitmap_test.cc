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

#include "xls/data_structures/inline_bitmap.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"

namespace xls {
namespace {

TEST(InlineBitmapTest, FromWord) {
  {
    auto b =
        InlineBitmap::FromWord(/*word=*/0, /*bit_count=*/0, /*fill=*/false);
    EXPECT_TRUE(b.IsAllZeroes());
  }

  {
    auto b = InlineBitmap::FromWord(/*word=*/0xFFFFFFFFFFFFFFFFULL,
                                    /*bit_count=*/0, /*fill=*/false);
    EXPECT_TRUE(b.IsAllZeroes());
  }

  {
    auto b = InlineBitmap::FromWord(/*word=*/0xFFFFFFFFFFFFFFFFULL,
                                    /*bit_count=*/1, /*fill=*/false);
    EXPECT_EQ(b.Get(0), 1);
    EXPECT_TRUE(b.IsAllOnes());
  }

  {
    auto b = InlineBitmap::FromWord(/*word=*/0x123456789abcdef0,
                                    /*bit_count=*/64, /*fill=*/false);
    EXPECT_EQ(b.GetWord(0), 0x123456789abcdef0);
  }
}

TEST(InlineBitmapTest, OneBitBitmap) {
  InlineBitmap b(/*bit_count=*/1);

  // Initialized with zeros.
  EXPECT_EQ(b.Get(0), 0);
  EXPECT_TRUE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());

  b.Set(0, false);
  EXPECT_TRUE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());
  EXPECT_EQ(b.Get(0), 0);

  b.Set(0, true);
  EXPECT_EQ(b.Get(0), 1);
  EXPECT_TRUE(b.IsAllOnes());
  EXPECT_FALSE(b.IsAllZeroes());

  b.Set(0, false);
  EXPECT_EQ(b.Get(0), 0);
  EXPECT_TRUE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());

  InlineBitmap b1(/*bit_count=*/1);
  EXPECT_EQ(b, b1);
  EXPECT_EQ(b1, b);
  b1.Set(0, true);
  EXPECT_NE(b1, b);
  b1.Set(0, false);
  EXPECT_EQ(b1, b);

  InlineBitmap b2(/*bit_count=*/2);
  EXPECT_NE(b2, b);
  EXPECT_NE(b, b2);
}

TEST(InlineBitmapTest, TwoBitBitmap) {
  InlineBitmap b(/*bit_count=*/2);
  EXPECT_TRUE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());
  EXPECT_EQ(2, b.bit_count());

  b.Set(0, true);
  EXPECT_FALSE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());
  EXPECT_EQ(b.Get(0), true);
  EXPECT_EQ(b.Get(1), false);

  b.Set(1, true);
  EXPECT_FALSE(b.IsAllZeroes());
  EXPECT_TRUE(b.IsAllOnes());
  EXPECT_EQ(b.Get(0), true);
  EXPECT_EQ(b.Get(1), true);

  EXPECT_EQ(b, b);
}

TEST(InlineBitmapTest, SixtyFiveBitBitmap) {
  InlineBitmap b(/*bit_count=*/65);
  EXPECT_TRUE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());
  EXPECT_EQ(65, b.bit_count());

  b.Set(0, true);
  EXPECT_FALSE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());
  EXPECT_EQ(b.Get(0), true);
  EXPECT_EQ(b.Get(1), false);
  EXPECT_EQ(b.Get(64), false);
  EXPECT_EQ(b, b);

  b.Set(0, false);
  b.Set(64, true);
  EXPECT_FALSE(b.IsAllZeroes());
  EXPECT_FALSE(b.IsAllOnes());
  EXPECT_EQ(b.Get(0), false);
  EXPECT_EQ(b.Get(1), false);
  EXPECT_EQ(b.Get(64), true);
  EXPECT_EQ(b, b);

  InlineBitmap empty(/*bit_count=*/65);
  EXPECT_NE(b, empty);
}

TEST(InlineBitmapTest, BytesAndBits) {
  InlineBitmap b(/*bit_count=*/16);
  b.SetByte(0, 0x80);  // Bit 7
  EXPECT_TRUE(b.Get(7));
  EXPECT_FALSE(b.Get(0));
  EXPECT_FALSE(b.Get(8));
  b.SetByte(1, 0x01);  // Bit 8
  EXPECT_TRUE(b.Get(8));
  EXPECT_FALSE(b.Get(15));
}

TEST(InlineBitmapTest, BytesAndWords) {
  {
    InlineBitmap b16(/*bit_count=*/16);
    b16.SetByte(0, 0xaa);
    b16.SetByte(1, 0xbb);
    EXPECT_EQ(b16.GetWord(0), 0xbbaa) << std::hex << b16.GetWord(0);
  }

  {
    InlineBitmap b9(/*bit_count=*/9);
    b9.SetByte(0, 0xaa);
    b9.SetByte(1, 0xbb);
    EXPECT_EQ(b9.GetWord(0), 0x1aa) << std::hex << b9.GetWord(0);
  }

  {
    InlineBitmap b(/*bit_count=*/64);
    b.SetByte(0, 0xf0);
    b.SetByte(1, 0xde);
    b.SetByte(2, 0xbc);
    b.SetByte(3, 0x9a);
    b.SetByte(4, 0x78);
    b.SetByte(5, 0x56);
    b.SetByte(6, 0x34);
    b.SetByte(7, 0x12);
    EXPECT_EQ(b.GetWord(0), 0x123456789abcdef0) << std::hex << b.GetWord(0);
  }

  {
    InlineBitmap b(/*bit_count=*/16);
    b.SetByte(0, 0xf0);
    b.SetByte(1, 0xde);
    EXPECT_EQ(b.GetWord(0), 0xdef0) << std::hex << b.GetWord(0);
  }

  {
    InlineBitmap b(/*bit_count=*/65);
    b.SetByte(7, 0xff);
    b.SetByte(8, 0x1);
    EXPECT_EQ(b.GetWord(0), 0xff00000000000000) << std::hex << b.GetWord(0);
    EXPECT_EQ(b.GetWord(1), 0x1) << std::hex << b.GetWord(1);
  }
}

TEST(InlineBitmapTest, UnsignedComparisons) {
  {
    InlineBitmap a(/*bit_count=*/0);
    InlineBitmap b(/*bit_count=*/65);
    // a == b
    EXPECT_EQ(a.UCmp(a), 0);
    EXPECT_EQ(a.UCmp(b), 0);
    EXPECT_EQ(b.UCmp(a), 0);
    EXPECT_EQ(b.UCmp(b), 0);
  }

  {
    auto a = InlineBitmap::FromWord(/*word=*/0,
                                    /*bit_count=*/0, /*fill=*/false);
    auto b = InlineBitmap::FromWord(/*word=*/0x123456789abcdef0,
                                    /*bit_count=*/64, /*fill=*/false);
    // a < b
    EXPECT_EQ(a.UCmp(b), -1);
    EXPECT_EQ(b.UCmp(a), 1);
  }

  {
    auto a = InlineBitmap::FromWord(/*word=*/0x123456789abcdef0,
                                    /*bit_count=*/64, /*fill=*/false);
    auto b = InlineBitmap::FromWord(/*word=*/0x123456789abcdef1,
                                    /*bit_count=*/64, /*fill=*/false);
    // a < b
    EXPECT_EQ(a.UCmp(b), -1);
    EXPECT_EQ(b.UCmp(a), 1);
  }

  {
    auto a = InlineBitmap::FromWord(/*word=*/0x123456789abcdef0,
                                    /*bit_count=*/64, /*fill=*/false);
    auto b = InlineBitmap::FromWord(/*word=*/0x123456789abcdef0,
                                    /*bit_count=*/65, /*fill=*/false);
    // a == b
    EXPECT_EQ(a.UCmp(b), 0);
    EXPECT_EQ(a.UCmp(a), 0);
    EXPECT_EQ(b.UCmp(a), 0);
    EXPECT_EQ(b.UCmp(b), 0);
  }
}

TEST(InlineBitmapTest, Union) {
  {
    InlineBitmap b(0);
    b.Union(InlineBitmap(0));
  }

  {
    InlineBitmap b(1);
    EXPECT_FALSE(b.Get(0));
    b.Union(InlineBitmap(1));
    EXPECT_FALSE(b.Get(0));
    b.Union(InlineBitmap::FromWord(1, 1));
    EXPECT_TRUE(b.Get(0));
  }

  {
    InlineBitmap b(2);
    EXPECT_FALSE(b.Get(0));
    b.Union(InlineBitmap(2));
    EXPECT_FALSE(b.Get(0));
    EXPECT_FALSE(b.Get(1));
    b.Union(InlineBitmap::FromWord(2, 2));
    EXPECT_FALSE(b.Get(0));
    EXPECT_TRUE(b.Get(1));
  }

  {
    InlineBitmap b = InlineBitmap::FromWord(0b00001100, 8);
    b.Union(InlineBitmap(InlineBitmap::FromWord(0b10001001, 8)));
    EXPECT_EQ(b.GetWord(0), 0b10001101);
    b.Union(InlineBitmap(InlineBitmap::FromWord(0b11111111, 8)));
    EXPECT_EQ(b.GetWord(0), 0b11111111);
  }

  {
    InlineBitmap b1(80);
    b1.SetByte(0, 0xab);
    b1.SetByte(1, 0xcd);
    b1.SetByte(2, 0xa5);
    b1.SetByte(9, 0x84);

    InlineBitmap b2(80);
    b2.SetByte(0, 0xfb);
    b2.SetByte(1, 0xee);
    b2.SetByte(5, 0x42);
    b2.SetByte(9, 0x31);

    b1.Union(b2);
    EXPECT_EQ(b1.GetByte(0), 0xfb);
    EXPECT_EQ(b1.GetByte(1), 0xef);
    EXPECT_EQ(b1.GetByte(2), 0xa5);
    EXPECT_EQ(b1.GetByte(3), 0);
    EXPECT_EQ(b1.GetByte(4), 0);
    EXPECT_EQ(b1.GetByte(5), 0x42);
    EXPECT_EQ(b1.GetByte(6), 0);
    EXPECT_EQ(b1.GetByte(7), 0);
    EXPECT_EQ(b1.GetByte(8), 0);
    EXPECT_EQ(b1.GetByte(9), 0xb5);
  }
}

}  // namespace
}  // namespace xls

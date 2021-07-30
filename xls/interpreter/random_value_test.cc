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

#include "xls/interpreter/random_value.h"

#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

TEST(RandomValueTest, RandomBits) {
  Package p("test_package");
  std::minstd_rand rng_engine;

  Value b0 = RandomValue(p.GetBitsType(0), &rng_engine);
  EXPECT_TRUE(b0.IsBits());
  EXPECT_EQ(b0.bits().bit_count(), 0);

  Value b1 = RandomValue(p.GetBitsType(1), &rng_engine);
  EXPECT_TRUE(b1.IsBits());
  EXPECT_EQ(b1.bits().bit_count(), 1);

  Value b1234 = RandomValue(p.GetBitsType(1234), &rng_engine);
  EXPECT_TRUE(b1234.IsBits());
  EXPECT_EQ(b1234.bits().bit_count(), 1234);

  // Do simple tests of a moderately sized sample of 64-bit random Bits values.
  // With overwhelming probability:
  // (1) generated values should all be distinct.
  // (2) deltas between consecutive generated values should all be distinct.
  // (3) every bit should be set to 0 and 1 at least once.
  const int64_t kSampleCount = 1024;
  const int64_t kBitWidth = 64;
  absl::flat_hash_set<uint64_t> samples;
  std::vector<int64_t> bit_set_count(kBitWidth);
  uint64_t previous_sample = 1;
  for (int64_t i = 0; i < kSampleCount; ++i) {
    Value b = RandomValue(p.GetBitsType(kBitWidth), &rng_engine);
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t as_uint64, b.bits().ToUint64());
    uint64_t delta = as_uint64 - previous_sample;
    EXPECT_FALSE(samples.contains(as_uint64));
    EXPECT_FALSE(samples.contains(delta));
    for (int64_t j = 0; j < kBitWidth; ++j) {
      if (b.bits().Get(j)) {
        ++bit_set_count[j];
      }
    }
    samples.insert(as_uint64);
    samples.insert(delta);
    previous_sample = as_uint64;
  }

  for (int64_t i = 0; i < kBitWidth; ++i) {
    EXPECT_GT(bit_set_count[i], 0);
    EXPECT_LT(bit_set_count[i], kSampleCount);
  }
}

TEST(RandomValueTest, Determinism) {
  Package p("test_package");
  std::minstd_rand rng_engine0;
  std::minstd_rand rng_engine1;
  EXPECT_EQ(RandomValue(p.GetBitsType(42), &rng_engine0),
            RandomValue(p.GetBitsType(42), &rng_engine1));
  EXPECT_EQ(RandomValue(p.GetBitsType(42), &rng_engine0),
            RandomValue(p.GetBitsType(42), &rng_engine1));
}

TEST(RandomValueTest, RandomOtherTypes) {
  Package p("test_package");
  std::minstd_rand rng_engine;

  Value empty_tuple = RandomValue(p.GetTupleType({}), &rng_engine);
  EXPECT_TRUE(empty_tuple.IsTuple());
  EXPECT_EQ(empty_tuple.size(), 0);

  Value tuple = RandomValue(
      p.GetTupleType({p.GetBitsType(12345), p.GetBitsType(32)}), &rng_engine);
  EXPECT_TRUE(tuple.IsTuple());
  EXPECT_EQ(tuple.size(), 2);
  // Overwhelmingly likely that the 12345-bit number is larger than the 32-bit
  // number.
  EXPECT_TRUE(
      bits_ops::UGreaterThan(tuple.element(0).bits(), tuple.element(1).bits()));

  Value array =
      RandomValue(p.GetArrayType(123, p.GetBitsType(57)), &rng_engine);
  EXPECT_TRUE(array.IsArray());
  EXPECT_EQ(array.size(), 123);
  for (int64_t i = 0; i < 123; ++i) {
    // Overwhelmingly likely that the elements are non-zero.
    EXPECT_NE(array.element(i).bits().ToInt64().value(), 0);
  }
}

}  // namespace
}  // namespace xls

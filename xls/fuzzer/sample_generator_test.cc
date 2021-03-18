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

#include "xls/fuzzer/sample_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

constexpr int64_t kIterations = 32 * 1024;

TEST(SampleGeneratorTest, RngRandRangeBiasedTowardsZero) {
  xls::RngState rng(std::mt19937{});
  constexpr int64_t kLimit = 3;
  std::vector<int64_t> histo(kLimit, 0);
  for (int64_t i = 0; i < kIterations; ++i) {
    histo[rng.RandRangeBiasedTowardsZero(kLimit)]++;
  }

  for (int64_t i = 0; i < kLimit; ++i) {
    XLS_LOG(INFO) << i << ": " << histo[i];
    EXPECT_GT(histo[i], 0);
    EXPECT_LT(histo[i], kIterations);
  }

  EXPECT_LT(histo[2], histo[1]);
  EXPECT_LT(histo[1], histo[0]);
}

TEST(SampleGeneratorTest, RngRandRange) {
  xls::RngState rng(std::mt19937{});
  constexpr int64_t kLimit = 3;
  std::vector<int64_t> histo(kLimit, 0);
  for (int64_t i = 0; i < kIterations; ++i) {
    histo[rng.RandRange(kLimit)]++;
  }

  for (int64_t i = 0; i < kLimit; ++i) {
    XLS_LOG(INFO) << i << ": " << histo[i];
    EXPECT_GT(histo[i], 0);
    EXPECT_LT(histo[i], kIterations);
  }
}

TEST(SampleGeneratorTest, RngRandomDouble) {
  xls::RngState rng(std::mt19937{});
  for (int64_t i = 0; i < kIterations; ++i) {
    double d = rng.RandomDouble();
    EXPECT_GE(d, 0.0);
    EXPECT_LT(d, 1.0);
  }
}

}  // namespace

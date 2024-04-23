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

#include "xls/noc/simulation/random_number_interface.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::noc {
namespace {

TEST(RandomNumberInterfaceTest, SmokeTest) {
  RandomNumberInterface rnd0;
  RandomNumberInterface rnd1;
  RandomNumberInterface rnd2;

  rnd0.SetSeed(100);
  rnd1.SetSeed(100);
  rnd2.SetSeed(1034);

  std::vector<double> p_samples = {0.5, 0.25, 0.01};
  int64_t trails = 100;

  for (double p : p_samples) {
    bool different_seeds_same = true;

    for (int64_t i = 0; i < trails; ++i) {
      bool b0 = rnd0.BernoulliDistribution(p);
      bool b1 = rnd1.BernoulliDistribution(p);
      bool b2 = rnd2.BernoulliDistribution(p);

      int64_t g0 = rnd0.GeometricDistribution(p);
      int64_t g1 = rnd1.GeometricDistribution(p);
      int64_t g2 = rnd2.GeometricDistribution(p);

      EXPECT_EQ(b0, b1);
      different_seeds_same &= (b1 == b2);

      EXPECT_EQ(g0, g1);
      different_seeds_same &= (g1 == g2);
    }

    EXPECT_FALSE(different_seeds_same);
  }

  std::vector<double> l_samples = {0.5, 0.25, 0.01};

  for (double p : p_samples) {
    for (double l : l_samples) {
      bool different_seeds_same = true;

      for (int64_t i = 0; i < trails; ++i) {
        int64_t gg0 = rnd0.GeneralizedGeometric(l, p);
        int64_t gg1 = rnd1.GeneralizedGeometric(l, p);
        int64_t gg2 = rnd2.GeneralizedGeometric(l, p);

        EXPECT_EQ(gg0, gg1);

        different_seeds_same &= (gg1 == gg2);
      }

      EXPECT_FALSE(different_seeds_same);
    }
  }
}

}  // namespace
}  // namespace xls::noc

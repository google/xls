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

#include "xls/common/stopwatch.h"

#include <algorithm>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/time/time.h"

namespace xls {
namespace {

TEST(SteadyTimeTest, RoundtripAddition) {
  SteadyTime t;

  EXPECT_EQ((t + absl::Nanoseconds(500)) - t, absl::Nanoseconds(500));
  EXPECT_EQ((t + absl::Nanoseconds(500)) - absl::Nanoseconds(500), t);
  EXPECT_EQ((t + absl::Microseconds(500)) - t, absl::Microseconds(500));
  EXPECT_EQ((t + absl::Microseconds(500)) - absl::Microseconds(500), t);
  EXPECT_EQ((t + absl::Milliseconds(500)) - t, absl::Milliseconds(500));
  EXPECT_EQ((t + absl::Milliseconds(500)) - absl::Milliseconds(500), t);
  EXPECT_EQ((t + absl::Seconds(500)) - t, absl::Seconds(500));
  EXPECT_EQ((t + absl::Seconds(500)) - absl::Seconds(500), t);

  EXPECT_EQ(t + absl::InfiniteDuration(), SteadyTime::Max());
  EXPECT_EQ(t - absl::InfiniteDuration(), SteadyTime::Min());
}

static constexpr absl::Duration kMaxSteadyDuration =
    absl::FromChrono(steady_clock_t::duration::max());
static constexpr absl::Duration kMinSteadyDuration =
    absl::FromChrono(steady_clock_t::duration::min());

void AdditionRoundtrips(SteadyTime t, absl::Duration d) {
  d = std::clamp(d, kMinSteadyDuration, kMaxSteadyDuration);

  if (SteadyTime::Min() - t > d) {
    // Saturates; should stay saturated.
    EXPECT_EQ((t + d) - d, SteadyTime::Min());
    return;
  }
  if (SteadyTime::Max() - t < d) {
    // Saturates; should stay saturated.
    EXPECT_EQ((t + d) - d, SteadyTime::Max());
    return;
  }

  // Not saturated; should roundtrip successfully (to steady-clock precision)
  static const absl::Duration kSteadyClockPrecision =
      absl::Seconds(1) / steady_clock_t::period::type::den;
  EXPECT_GT((t + d) - t, d - kSteadyClockPrecision);
  EXPECT_LT((t + d) - t, d + kSteadyClockPrecision);
}
FUZZ_TEST(SteadyTimeFuzzTest, AdditionRoundtrips)
    .WithDomains(fuzztest::Just(SteadyTime()),
                 fuzztest::Arbitrary<absl::Duration>());

}  // namespace
}  // namespace xls

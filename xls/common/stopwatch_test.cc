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
#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/time.h"
#include "rapidcheck/gtest.h"
#include "rapidcheck.h"

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

void AdditionRoundtrips(SteadyTime t, absl::Duration d) {
  if (SteadyTime::Min() - t > d) {
    // Saturated; should stay saturated.
    RC_ASSERT((t + d) - d == SteadyTime::Min());
  } else if (SteadyTime::Max() - t < d) {
    // Saturated; should stay saturated.
    RC_ASSERT((t + d) - d == SteadyTime::Max());
  } else {
    // Not saturated; should roundtrip successfully.
    RC_ASSERT((t + d) - t == d);
  }
}

RC_GTEST_PROP(SteadyTimeRapidCheck, RoundtripAddition, (int64_t v)) {
  static constexpr absl::Duration kMaxSteadyDuration =
      absl::FromChrono(steady_clock_t::duration::max());
  static constexpr absl::Duration kMinSteadyDuration =
      absl::FromChrono(steady_clock_t::duration::min());

  SteadyTime t;

  absl::Duration s =
      std::clamp(absl::Seconds(v), kMinSteadyDuration, kMaxSteadyDuration);
  AdditionRoundtrips(t, s);

  absl::Duration ms =
      std::clamp(absl::Milliseconds(v), kMinSteadyDuration, kMaxSteadyDuration);
  AdditionRoundtrips(t, ms);

  absl::Duration us =
      std::clamp(absl::Microseconds(v), kMinSteadyDuration, kMaxSteadyDuration);
  AdditionRoundtrips(t, us);

  absl::Duration ns =
      std::clamp(absl::Nanoseconds(v), kMinSteadyDuration, kMaxSteadyDuration);
  AdditionRoundtrips(t, ns);
}

}  // namespace
}  // namespace xls

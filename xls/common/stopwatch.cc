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

#include <chrono>  // NOLINT

#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xls/common/logging/logging.h"

namespace xls {
namespace {

constexpr steady_clock_t::duration kDurationZero =
    steady_clock_t::duration::zero();

constexpr steady_clock_t::time_point kTimePointMin =
    steady_clock_t::time_point::min();
constexpr steady_clock_t::time_point kTimePointMax =
    steady_clock_t::time_point::max();

steady_clock_t::duration ToSteadyDuration(absl::Duration d) {
  if constexpr (steady_clock_t::period::type::den > 1'000'000) {
    return absl::ToChronoNanoseconds(d);
  } else if constexpr (steady_clock_t::period::type::den > 1'000) {
    return absl::ToChronoMicroseconds(d);
  } else if constexpr (steady_clock_t::period::type::den > 1) {
    return absl::ToChronoMilliseconds(d);
  } else {
    return absl::ToChronoSeconds(d);
  }
}

class RealTimeSteadyClock final : public SteadyClock {
 public:
  ~RealTimeSteadyClock() override {
    LOG(FATAL) << "RealTimeSteadyClock should never be destroyed";
  }

  SteadyTime Now() override { return SteadyTime::Now(); }
};

}  // namespace

// static
SteadyTime SteadyTime::Now() { return SteadyTime(steady_clock_t::now()); }

// static
SteadyTime SteadyTime::Max() { return SteadyTime(kTimePointMax); }

// static
SteadyTime SteadyTime::Min() { return SteadyTime(kTimePointMin); }

SteadyTime& SteadyTime::operator+=(const absl::Duration d) {
  if (time_ == kTimePointMax || time_ == kTimePointMin) {
    // We've saturated; we can't make any change.
    return *this;
  }

  if (d == absl::InfiniteDuration()) {
    time_ = kTimePointMax;
    return *this;
  }
  if (d == -absl::InfiniteDuration()) {
    time_ = kTimePointMin;
    return *this;
  }

  const auto chrono_duration = ToSteadyDuration(d);
  if (chrono_duration >= kDurationZero) {
    if (time_ < kTimePointMax - chrono_duration) {
      time_ += chrono_duration;
    } else {
      time_ = kTimePointMax;
    }
  } else {
    if (time_ > kTimePointMin - chrono_duration) {
      time_ += chrono_duration;
    } else {
      time_ = kTimePointMin;
    }
  }
  return *this;
}
SteadyTime& SteadyTime::operator-=(const absl::Duration d) {
  *this += (-d);
  return *this;
}

SteadyTime operator+(SteadyTime t, const absl::Duration d) {
  t += d;
  return t;
}
SteadyTime operator-(SteadyTime t, const absl::Duration d) { return t + (-d); }

absl::Duration operator-(const SteadyTime& t1, const SteadyTime& t2) {
  return absl::FromChrono(t1.time_ - t2.time_);
}

// static
SteadyClock* SteadyClock::RealClock() {
  static absl::NoDestructor<RealTimeSteadyClock> real_clock;
  return real_clock.get();
}

}  // namespace xls

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

#ifndef XLS_COMMON_STOPWATCH_H_
#define XLS_COMMON_STOPWATCH_H_

#include <chrono>  // NOLINT

#include "absl/time/time.h"

namespace xls {

namespace internal {

template <bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct ChooseSteadyClock {
  using type = std::chrono::high_resolution_clock;
};

template <>
struct ChooseSteadyClock<false> {
  using type = std::chrono::steady_clock;
};

}  // namespace internal

using steady_clock_t = internal::ChooseSteadyClock<>::type;

class SteadyTime {
 public:
  // Construct a default time. Note that the value is not specified, but it
  // should be relatively distant from both Max and Min. (On many systems, it
  // will represent a time near the start of the process, or last reboot.)
  SteadyTime() = default;

  // Construct the current time
  static SteadyTime Now();

  // Construct the maximum representable time
  static SteadyTime Max();

  // Construct the minimum representable time
  static SteadyTime Min();

  SteadyTime& operator+=(absl::Duration d);
  SteadyTime& operator-=(absl::Duration d);

  friend SteadyTime operator+(SteadyTime t, absl::Duration d);
  friend SteadyTime operator+(absl::Duration d, const SteadyTime& t) {
    return t + d;
  }

  friend SteadyTime operator-(SteadyTime t, absl::Duration d);
  friend absl::Duration operator-(const SteadyTime& t1, const SteadyTime& t2);

  friend bool operator<(const SteadyTime& t1, const SteadyTime& t2) {
    return t1.time_ < t2.time_;
  }
  friend bool operator<=(const SteadyTime& t1, const SteadyTime& t2) {
    return t1.time_ <= t2.time_;
  }
  friend bool operator>(const SteadyTime& t1, const SteadyTime& t2) {
    return t1.time_ > t2.time_;
  }
  friend bool operator>=(const SteadyTime& t1, const SteadyTime& t2) {
    return t1.time_ >= t2.time_;
  }
  friend bool operator==(const SteadyTime& t1, const SteadyTime& t2) {
    return t1.time_ == t2.time_;
  }
  friend bool operator!=(const SteadyTime& t1, const SteadyTime& t2) {
    return t1.time_ != t2.time_;
  }

 private:
  explicit SteadyTime(std::chrono::time_point<steady_clock_t> time)
      : time_(time) {}

  std::chrono::time_point<steady_clock_t> time_;
};

// An abstract interface representing a SteadyClock, which is an object that can
// tell you the current steady and absolute times.
//
// This interface allows decoupling code that uses time from the code that
// creates a point in time. You can use this to your advantage by injecting
// SteadyClocks into interfaces rather than having implementations call
// SteadyTime::Now() directly.
//
// The SteadyClock::RealClock() function returns a pointer (that you do not own)
// to the global real-time clock.
//
// Example:
//
//   bool OneSecondSince(SteadyTime time, SteadyClock* clock) {
//     return (clock->Now() - time) >= absl::Seconds(1);
//   }
//
//   // Production code.
//   OneSecondSince(start_time, Clock::RealClock());
//
//   // Test code:
//   MyTestClock test_clock(<TIME>);
//   OneSecondSince(start_time, &test_clock);
//
class SteadyClock {
 public:
  // Returns a pointer to the global real-time clock. The caller does not own
  // the returned pointer and should not delete it. The returned clock is
  // thread-safe.
  static SteadyClock* RealClock();

  virtual SteadyTime Now() = 0;

  virtual ~SteadyClock() = default;
};

// Helper class that allows simple wall time measurements.
//
// Many projects consider absl::Time() to be sufficient, but as it uses an
// absolute (non-steady) clock, it's not appropriate for measuring durations of
// elapsed time. In practice, most consumer machines with good time syncing see
// absolute time move backwards by non-negligible amounts multiple times per
// year.
class Stopwatch {
 public:
  // Starts the timer.
  explicit Stopwatch(SteadyClock* clock)
      : clock_(clock), start_time_(clock_->Now()) {}
  Stopwatch() : Stopwatch(SteadyClock::RealClock()) {}

  // Resets the start time.
  void Reset() { start_time_ = clock_->Now(); }

  // Returns the time this was last started.
  SteadyTime GetStartTime() const { return start_time_; }

  // Returns `now() - start time` as an absl::Duration.
  absl::Duration GetElapsedTime() const { return clock_->Now() - start_time_; }

 private:
  SteadyClock* clock_;
  SteadyTime start_time_;
};

}  // namespace xls

#endif  // XLS_COMMON_STOPWATCH_H_

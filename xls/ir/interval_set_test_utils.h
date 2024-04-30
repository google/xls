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

#ifndef XLS_IR_INTERVAL_SET_TEST_UTILS_H_
#define XLS_IR_INTERVAL_SET_TEST_UTILS_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/interval_test_utils.h"

namespace xls {

// GoogleFuzzTest "Arbitrary" domain for IntervalSets with known bit count.
inline auto ArbitraryIntervalSet(int64_t bit_count) {
  CHECK_GE(bit_count, 0);
  return fuzztest::ReversibleMap(
      [bit_count](const std::vector<Interval>& intervals) {
        IntervalSet set(bit_count);
        for (const Interval& interval : intervals) {
          set.AddInterval(interval);
        }
        return set;
      },
      [bit_count](const IntervalSet& interval_set)
          -> std::optional<std::tuple<std::vector<Interval>>> {
        if (interval_set.BitCount() != bit_count) {
          return std::nullopt;
        }
        absl::Span<const Interval> interval_span = interval_set.Intervals();
        std::vector<Interval> intervals;
        intervals.insert(intervals.end(), interval_span.begin(),
                         interval_span.end());
        return std::make_tuple(intervals);
      },
      fuzztest::VectorOf(ArbitraryInterval(bit_count)));
}

// GoogleFuzzTest "Arbitrary" domain for IntervalSets with unknown bit count.
inline auto ArbitraryIntervalSet() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) { return ArbitraryIntervalSet(bit_count); },
      fuzztest::NonNegative<int64_t>());
}

// GoogleFuzzTest domain for nonempty IntervalSets with known bit count.
inline auto NonemptyIntervalSet(int64_t bit_count) {
  CHECK_GE(bit_count, 0);
  return fuzztest::ReversibleMap(
      [bit_count](const std::vector<Interval>& intervals) {
        IntervalSet set(bit_count);
        for (const Interval& interval : intervals) {
          set.AddInterval(interval);
        }
        return set;
      },
      [bit_count](const IntervalSet& interval_set)
          -> std::optional<std::tuple<std::vector<Interval>>> {
        if (interval_set.BitCount() != bit_count) {
          return std::nullopt;
        }
        absl::Span<const Interval> interval_span = interval_set.Intervals();
        std::vector<Interval> intervals;
        intervals.insert(intervals.end(), interval_span.begin(),
                         interval_span.end());
        return std::make_tuple(intervals);
      },
      fuzztest::NonEmpty(fuzztest::VectorOf(ArbitraryInterval(bit_count))));
}

// GoogleFuzzTest domain for nonempty IntervalSets with unknown bit count.
inline auto NonemptyIntervalSet() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) { return NonemptyIntervalSet(bit_count); },
      fuzztest::NonNegative<int64_t>());
}

inline auto ArbitraryNormalizedIntervalSet(int64_t bit_count) {
  return fuzztest::ReversibleMap(
      [](IntervalSet interval_set) {
        interval_set.Normalize();
        return interval_set;
      },
      [bit_count](const IntervalSet& interval_set)
          -> std::optional<std::tuple<IntervalSet>> {
        if (interval_set.BitCount() != bit_count ||
            !interval_set.IsNormalized()) {
          return std::nullopt;
        }
        return std::make_tuple(interval_set);
      },
      ArbitraryIntervalSet(bit_count));
}
inline auto ArbitraryNormalizedIntervalSet() {
  return fuzztest::ReversibleMap(
      [](IntervalSet interval_set) {
        interval_set.Normalize();
        return interval_set;
      },
      [](const IntervalSet& interval_set)
          -> std::optional<std::tuple<IntervalSet>> {
        if (!interval_set.IsNormalized()) {
          return std::nullopt;
        }
        return std::make_tuple(interval_set);
      },
      ArbitraryIntervalSet());
}
inline auto NonemptyNormalizedIntervalSet(int64_t bit_count) {
  return fuzztest::ReversibleMap(
      [](IntervalSet interval_set) {
        interval_set.Normalize();
        return interval_set;
      },
      [bit_count](const IntervalSet& interval_set)
          -> std::optional<std::tuple<IntervalSet>> {
        if (interval_set.BitCount() != bit_count ||
            !interval_set.IsNormalized()) {
          return std::nullopt;
        }
        return std::make_tuple(interval_set);
      },
      NonemptyIntervalSet(bit_count));
}
inline auto NonemptyNormalizedIntervalSet() {
  return fuzztest::ReversibleMap(
      [](IntervalSet interval_set) {
        interval_set.Normalize();
        return interval_set;
      },
      [](const IntervalSet& interval_set)
          -> std::optional<std::tuple<IntervalSet>> {
        if (!interval_set.IsNormalized()) {
          return std::nullopt;
        }
        return std::make_tuple(interval_set);
      },
      NonemptyIntervalSet());
}

}  // namespace xls

#endif  // XLS_IR_INTERVAL_SET_TEST_UTILS_H_

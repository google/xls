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

#ifndef XLS_IR_INTERVAL_TEST_UTILS_H_
#define XLS_IR_INTERVAL_TEST_UTILS_H_

#include <cstdint>
#include <optional>

#include "fuzztest/fuzztest.h"
#include "absl/log/check.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/interval.h"

namespace xls {

// GoogleFuzzTest "Arbitrary" domain for Intervals with known bit count.
inline auto ArbitraryInterval(int64_t bit_count) {
  CHECK_GE(bit_count, 0);
  return fuzztest::ReversibleMap(
      [bit_count](const Bits& lb, const Bits& ub) {
        CHECK_EQ(lb.bit_count(), bit_count);
        CHECK_EQ(ub.bit_count(), bit_count);
        return Interval(lb, ub);
      },
      [bit_count](
          const Interval& interval) -> std::optional<std::tuple<Bits, Bits>> {
        if (interval.BitCount() != bit_count) {
          return std::nullopt;
        }
        return std::make_tuple(interval.LowerBound(), interval.UpperBound());
      },
      ArbitraryBits(bit_count), ArbitraryBits(bit_count));
}

// GoogleFuzzTest "Arbitrary" domain for Intervals with unknown bit count.
inline auto ArbitraryInterval() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) { return ArbitraryInterval(bit_count); },
      fuzztest::NonNegative<int64_t>());
}

// GoogleFuzzTest domain for proper Intervals with known bit count.
inline auto ProperInterval(int64_t bit_count) {
  CHECK_GE(bit_count, 0);
  return fuzztest::ReversibleMap(
      [bit_count](const Bits& a, const Bits& b) {
        CHECK_EQ(a.bit_count(), bit_count);
        CHECK_EQ(b.bit_count(), bit_count);
        if (bits_ops::UGreaterThan(a, b)) {
          return Interval(b, a);
        }
        return Interval(a, b);
      },
      [](const Interval& interval) {
        return std::make_optional(
            std::make_tuple(interval.LowerBound(), interval.UpperBound()));
      },
      ArbitraryBits(bit_count), ArbitraryBits(bit_count));
}

// GoogleFuzzTest domain for proper Intervals with unknown bit count.
inline auto ProperInterval() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) { return ProperInterval(bit_count); },
      fuzztest::NonNegative<int64_t>());
}

}  // namespace xls

#endif  // XLS_IR_INTERVAL_TEST_UTILS_H_

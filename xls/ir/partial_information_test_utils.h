// Copyright 2025 The XLS Authors
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

#ifndef XLS_IR_PARTIAL_INFORMATION_TEST_UTILS_H_
#define XLS_IR_PARTIAL_INFORMATION_TEST_UTILS_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/interval_set_test_utils.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/ternary.h"

namespace xls {

// GoogleFuzzTest "Arbitrary" domain for PartialInformation with known bit
// count.
inline auto ArbitraryPartialInformation(int64_t bit_count) {
  return fuzztest::ReversibleMap(
      [bit_count](std::optional<IntervalSet> interval_set,
                  std::optional<TernaryVector> ternary) {
        return PartialInformation(bit_count, std::move(ternary),
                                  std::move(interval_set));
      },
      [](const PartialInformation& partial_information) {
        return std::make_optional(std::make_tuple(
            partial_information.Range(), partial_information.Ternary()));
      },
      fuzztest::OptionalOf(ArbitraryIntervalSet(bit_count)),
      fuzztest::OptionalOf(
          fuzztest::VectorOf(fuzztest::ElementOf({TernaryValue::kKnownZero,
                                                  TernaryValue::kKnownOne,
                                                  TernaryValue::kUnknown}))
              .WithSize(bit_count)));
}

// GoogleFuzzTest "Arbitrary" domain for PartialInformation with unknown bit
// count.
inline auto ArbitraryPartialInformation() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) { return ArbitraryPartialInformation(bit_count); },
      fuzztest::NonNegative<int64_t>());
}

struct PartiallyDescribedBits {
  Bits bits;
  PartialInformation partial;
};

// GoogleFuzzTest "Arbitrary" domain for PartiallyDescribedBits with known bit
// count.
inline auto ArbitraryPartiallyDescribedBits(int64_t bit_count) {
  return fuzztest::ReversibleMap(
      [](const Bits& bits, PartialInformation partial) {
        PartiallyDescribedBits result = {
            .bits = bits,
            .partial =
                std::move(partial.MeetWith(PartialInformation::Precise(bits))),
        };
        DCHECK(result.partial.IsCompatibleWith(result.bits));
        return result;
      },
      [](const PartiallyDescribedBits& partially_described_bits) {
        return std::make_optional(std::make_tuple(
            partially_described_bits.bits, partially_described_bits.partial));
      },
      ArbitraryBits(bit_count), ArbitraryPartialInformation(bit_count));
}

// GoogleFuzzTest "Arbitrary" domain for PartiallyDescribedBits with unknown bit
// count.
inline auto ArbitraryPartiallyDescribedBits() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) {
        return ArbitraryPartiallyDescribedBits(bit_count);
      },
      fuzztest::NonNegative<int64_t>());
}

}  // namespace xls

#endif  // XLS_IR_PARTIAL_INFORMATION_TEST_UTILS_H_

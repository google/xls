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

#include "xls/ir/interval_ops.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"

namespace xls::interval_ops {

namespace {
TernaryVector ExtractTernaryInterval(const Interval& interval) {
  Bits lcp = bits_ops::LongestCommonPrefixMSB(
      {interval.LowerBound(), interval.UpperBound()});
  int64_t size = interval.BitCount();
  TernaryVector result(size, TernaryValue::kUnknown);
  for (int64_t i = size - lcp.bit_count(), j = 0; i < size; ++i, ++j) {
    result[i] = lcp.Get(j) ? TernaryValue::kKnownOne : TernaryValue::kKnownZero;
  }
  return result;
}
}  // namespace

TernaryVector ExtractTernaryVector(const IntervalSet& intervals,
                                   std::optional<Node*> source) {
  CHECK(intervals.IsNormalized())
      << (source.has_value() ? source.value()->ToString() : "");
  CHECK(!intervals.Intervals().empty())
      << (source.has_value() ? source.value()->ToString() : "");
  TernaryVector result = ExtractTernaryInterval(intervals.Intervals().front());
  for (const Interval& i : intervals.Intervals().subspan(1)) {
    TernaryVector t = ExtractTernaryInterval(i);
    ternary_ops::UpdateWithIntersection(result, t);
  }
  return result;
}

KnownBits ExtractKnownBits(const IntervalSet& intervals,
                           std::optional<Node*> source) {
  TernaryVector result = ExtractTernaryVector(intervals, source);
  return KnownBits{.known_bits = ternary_ops::ToKnownBits(result),
                   .known_bit_values = ternary_ops::ToKnownBitsValues(result)};
}

IntervalSet FromTernary(TernarySpan tern, int64_t max_interval_bits) {
  CHECK_GE(max_interval_bits, 0);
  if (ternary_ops::IsFullyKnown(tern)) {
    return IntervalSet::Precise(ternary_ops::ToKnownBitsValues(tern));
  }
  // How many trailing bits are unknown. This defines the size of each group.
  int64_t lsb_xs = absl::c_find_if(tern, ternary_ops::IsKnown) - tern.cbegin();
  // Find where we need to extend the unknown region to.
  std::deque<TernarySpan::const_iterator> x_locations;
  for (auto it = tern.cbegin() + lsb_xs; it != tern.cend(); ++it) {
    if (ternary_ops::IsUnknown(*it)) {
      x_locations.push_back(it);
      if (x_locations.size() > max_interval_bits + 1) {
        x_locations.pop_front();
      }
    }
  }
  if (x_locations.size() > max_interval_bits) {
    // Need to extend the x-s to avoid creating too many intervals.
    lsb_xs = (x_locations.front() - tern.cbegin()) + 1;
    x_locations.pop_front();
  }

  IntervalSet is(tern.size());
  if (x_locations.empty()) {
    // All bits from 0 -> lsb_xs are unknown.
    Bits high_bits = ternary_ops::ToKnownBitsValues(tern.subspan(lsb_xs));
    is.AddInterval(
        Interval::Closed(bits_ops::Concat({high_bits, Bits(lsb_xs)}),
                         bits_ops::Concat({high_bits, Bits::AllOnes(lsb_xs)})));
    is.Normalize();
    return is;
  }

  TernaryVector vec(tern.size() - lsb_xs, TernaryValue::kKnownZero);
  // Copy input ternary from after the last lsb_x.
  std::copy(tern.cbegin() + lsb_xs, tern.cend(), vec.begin());

  Bits high_lsb = Bits::AllOnes(lsb_xs);
  Bits low_lsb(lsb_xs);
  for (const Bits& v : ternary_ops::AllBitsValues(vec)) {
    is.AddInterval(Interval::Closed(bits_ops::Concat({v, low_lsb}),
                                    bits_ops::Concat({v, high_lsb})));
  }
  is.Normalize();
  return is;
}

}  // namespace xls::interval_ops

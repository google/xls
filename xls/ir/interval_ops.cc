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
#include <compare>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

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

namespace {
// An intrusive list node of an interval list
struct MergeInterval {
  Interval final_interval;
  Bits gap_with_previous;
  // Intrusive list links. Next & previous lexicographic interval.
  MergeInterval* prev = nullptr;
  MergeInterval* next = nullptr;

  friend std::strong_ordering operator<=>(const MergeInterval& l,
                                          const MergeInterval& r) {
    auto cmp_bits = [](const Bits& l, const Bits& r) {
      return bits_ops::UCmp(l, r) <=> 0;
    };
    std::strong_ordering gap_order =
        cmp_bits(l.gap_with_previous, r.gap_with_previous);
    if (gap_order != std::strong_ordering::equal) {
      return gap_order;
    }
    return cmp_bits(l.final_interval.LowerBound(),
                    r.final_interval.LowerBound());
  }
};
}  // namespace

// Minimize interval set to 'size' by merging some intervals together. Intervals
// are chosen with a greedy algorithm that minimizes the number of additional
// values the overall interval set contains. That is first it will add the
// smallest components posible. In cases where multiple gaps are the same size
// it will prioritize earlier gaps over later ones.
IntervalSet MinimizeIntervals(IntervalSet interval_set, int64_t size) {
  interval_set.Normalize();

  // Check for easy cases (already small enough and convex hull)
  if (interval_set.NumberOfIntervals() <= size) {
    return interval_set;
  }
  if (size == 1) {
    IntervalSet res(interval_set.BitCount());
    res.AddInterval(*interval_set.ConvexHull());
    res.Normalize();
    return res;
  }

  std::vector<std::unique_ptr<MergeInterval>> merge_list;
  merge_list.reserve(interval_set.NumberOfIntervals() - 1);
  // The first one will never get merged with the previous since that wouldn't
  // actually remove an interval segment so we don't include it on the merge
  // list. Things can get merged into it however.
  DCHECK(absl::c_is_sorted(interval_set.Intervals()));
  MergeInterval first{.final_interval = interval_set.Intervals().front()};
  for (auto it = interval_set.Intervals().begin() + 1;
       it != interval_set.Intervals().end(); ++it) {
    MergeInterval* prev = merge_list.empty() ? &first : merge_list.back().get();
    Bits distance = bits_ops::Sub(it->LowerBound(), (it - 1)->UpperBound());
    // Generate a list with an intrusive list containing the original
    // ordering.
    merge_list.push_back(std::make_unique<MergeInterval>(
        MergeInterval{.final_interval = *it,
                      .gap_with_previous = std::move(distance),
                      .prev = prev}));
    prev->next = merge_list.back().get();
  }

  // We want a min-heap so cmp is greater-than.
  auto heap_cmp = [](const std::unique_ptr<MergeInterval>& l,
                     const std::unique_ptr<MergeInterval>& r) {
    return *l > *r;
  };
  // make the merge_list a heap.
  absl::c_make_heap(merge_list, heap_cmp);

  // Remove the minimum element from the merge_list heap.
  auto pop_min_element = [&]() -> std::unique_ptr<MergeInterval> {
    absl::c_pop_heap(merge_list, heap_cmp);
    std::unique_ptr<MergeInterval> minimum = std::move(merge_list.back());
    merge_list.pop_back();
    return minimum;
  };
  // Merge elements until we are the appropriate size.
  // NB Since the first interval isn't in the heap (since it can't get merged
  // from) we need to continue until the heap is one element shorter than
  // requested.
  while (merge_list.size() > size - 1) {
    // Pull the item with the smallest distance
    std::unique_ptr<MergeInterval> min_interval = pop_min_element();
    // Merge with the prior element.

    // extend the previous interval.
    min_interval->prev->final_interval =
        Interval(min_interval->prev->final_interval.LowerBound(),
                 min_interval->final_interval.UpperBound());
    // Update the intrusive list of active merges.
    min_interval->prev->next = min_interval->next;
    if (min_interval->next != nullptr) {
      min_interval->next->prev = min_interval->prev;
    }
  }

  // Now 'first, ...merge_list' is `size` elements.
  IntervalSet result;
  std::vector<Interval> final_intervals{std::move(first.final_interval)};
  final_intervals.reserve(size);
  for (std::unique_ptr<MergeInterval>& mi : merge_list) {
    final_intervals.push_back(std::move(mi->final_interval));
  }
  result.SetIntervals(final_intervals);
  result.Normalize();
  return result;
}

}  // namespace xls::interval_ops

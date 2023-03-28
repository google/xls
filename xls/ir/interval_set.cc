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

#include "xls/ir/interval_set.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_message.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"

namespace xls {

IntervalSet IntervalSet::Maximal(int64_t bit_count) {
  IntervalSet result(bit_count);
  result.AddInterval(Interval::Maximal(bit_count));
  result.Normalize();
  return result;
}

IntervalSet IntervalSet::Precise(const Bits& bits) {
  IntervalSet result(bits.bit_count());
  result.AddInterval(Interval::Precise(bits));
  result.Normalize();
  return result;
}

void IntervalSet::SetIntervals(absl::Span<const Interval> intervals) {
  is_normalized_ = false;
  intervals_.clear();
  if (intervals.empty()) {
    bit_count_ = -1;
  } else {
    bit_count_ = intervals[0].BitCount();
  }

  for (const Interval& interval : intervals) {
    AddInterval(interval);
  }
}

void IntervalSet::Normalize() {
  Bits zero(BitCount());
  Bits max = Bits::AllOnes(BitCount());
  std::vector<Interval> expand_improper;
  for (const Interval& interval : intervals_) {
    if (interval.IsImproper()) {
      expand_improper.push_back(Interval(zero, interval.UpperBound()));
      expand_improper.push_back(Interval(interval.LowerBound(), max));
    } else {
      expand_improper.push_back(interval);
    }
  }

  std::sort(expand_improper.begin(), expand_improper.end());

  intervals_.clear();
  for (int32_t i = 0; i < expand_improper.size();) {
    Interval interval = expand_improper[i];
    while ((i < expand_improper.size()) &&
           (Interval::Overlaps(interval, expand_improper[i]) ||
            Interval::Abuts(interval, expand_improper[i]))) {
      interval = Interval::ConvexHull(interval, expand_improper[i]);
      ++i;
    }
    intervals_.push_back(interval);
  }

  is_normalized_ = true;
}

std::optional<Interval> IntervalSet::ConvexHull() const {
  XLS_CHECK_GE(bit_count_, 0);
  std::optional<Bits> lower = LowerBound();
  std::optional<Bits> upper = UpperBound();
  if (!lower.has_value() || !upper.has_value()) {
    return std::nullopt;
  }
  return Interval(lower.value(), upper.value());
}

std::optional<Bits> IntervalSet::LowerBound() const {
  // TODO(taktoa): optimize case where is_normalized_ is true
  XLS_CHECK_GE(bit_count_, 0);
  std::optional<Bits> lower;
  for (const Interval& interval : intervals_) {
    if (lower.has_value()) {
      if (bits_ops::ULessThan(interval.LowerBound(), lower.value())) {
        lower = interval.LowerBound();
      }
    } else {
      lower = interval.LowerBound();
    }
  }
  return lower;
}

std::optional<Bits> IntervalSet::UpperBound() const {
  // TODO(taktoa): optimize case where is_normalized_ is true
  XLS_CHECK_GE(bit_count_, 0);
  std::optional<Bits> upper;
  for (const Interval& interval : intervals_) {
    if (upper.has_value()) {
      if (bits_ops::UGreaterThan(interval.UpperBound(), upper.value())) {
        upper = interval.UpperBound();
      }
    } else {
      upper = interval.UpperBound();
    }
  }
  return upper;
}

IntervalSet IntervalSet::ZeroExtend(int64_t bit_width) const {
  XLS_CHECK_GE(bit_width, BitCount());
  IntervalSet result(bit_width);
  for (const Interval& interval : Intervals()) {
    result.AddInterval(interval.ZeroExtend(bit_width));
  }
  result.Normalize();
  return result;
}

bool IntervalSet::ForEachElement(
    std::function<bool(const Bits&)> callback) const {
  XLS_CHECK(is_normalized_);
  for (const Interval& interval : intervals_) {
    if (interval.ForEachElement(callback)) {
      return true;
    }
  }
  return false;
}

IntervalSet IntervalSet::Combine(const IntervalSet& lhs,
                                 const IntervalSet& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  IntervalSet combined(lhs.BitCount());
  for (const Interval& interval : lhs.intervals_) {
    combined.AddInterval(interval);
  }
  for (const Interval& interval : rhs.intervals_) {
    combined.AddInterval(interval);
  }
  combined.Normalize();
  return combined;
}

IntervalSet IntervalSet::Intersect(const IntervalSet& lhs,
                                   const IntervalSet& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(lhs.is_normalized_);
  XLS_CHECK(rhs.is_normalized_);
  IntervalSet result(lhs.BitCount());
  std::list<Interval> lhs_intervals(lhs.Intervals().begin(),
                                    lhs.Intervals().end());
  std::list<Interval> rhs_intervals(rhs.Intervals().begin(),
                                    rhs.Intervals().end());
  auto left = lhs_intervals.begin();
  auto right = rhs_intervals.begin();
  // lhs/rhs_intervals should be sorted in increasing lexicographic order
  // (i.e.: compare on lower bound, then upper bound if lower bounds are equal)
  // at this point, since we CHECK that they are normalized.
  while ((left != lhs_intervals.end()) && (right != rhs_intervals.end())) {
    if (bits_ops::ULessThan(left->UpperBound(), right->UpperBound())) {
      if (std::optional<Interval> intersection =
              Interval::Intersect(*left, *right)) {
        result.AddInterval(*intersection);
        // The difference should only ever contain 0 or 1 interval.
        std::vector<Interval> difference = Interval::Difference(*right, *left);
        std::reverse(difference.begin(), difference.end());
        for (const Interval& remainder : difference) {
          right = rhs_intervals.insert(right, remainder);
        }
      }
      ++left;
      continue;
    }
    if (bits_ops::ULessThan(right->UpperBound(), left->UpperBound())) {
      if (std::optional<Interval> intersection =
              Interval::Intersect(*left, *right)) {
        result.AddInterval(*intersection);
        // The difference should only ever contain 0 or 1 interval.
        std::vector<Interval> difference = Interval::Difference(*left, *right);
        std::reverse(difference.begin(), difference.end());
        for (const Interval& remainder : difference) {
          left = lhs_intervals.insert(left, remainder);
        }
      }
      ++right;
      continue;
    }
    if (bits_ops::UEqual(left->UpperBound(), right->UpperBound())) {
      if (std::optional<Interval> intersection =
              Interval::Intersect(*left, *right)) {
        result.AddInterval(*intersection);
      }
      ++left;
      ++right;
      continue;
    }
  }
  result.Normalize();
  return result;
}

IntervalSet IntervalSet::Complement(const IntervalSet& set) {
  // The complement of an interval set is the intersection of the complements of
  // the component intervals.
  IntervalSet result = Maximal(set.BitCount());
  for (const Interval& interval : set.Intervals()) {
    IntervalSet complement_set(set.BitCount());
    for (const Interval& complement : Interval::Complement(interval)) {
      complement_set.AddInterval(complement);
    }
    complement_set.Normalize();
    result = Intersect(result, complement_set);
  }
  return result;
}

std::optional<int64_t> IntervalSet::Size() const {
  XLS_CHECK(is_normalized_);
  int64_t total_size = 0;
  for (const Interval& interval : intervals_) {
    if (auto size = interval.Size()) {
      total_size += size.value();
    } else {
      return std::nullopt;
    }
  }
  return total_size;
}

std::optional<Bits> IntervalSet::Index(const Bits& index) const {
  XLS_CHECK(is_normalized_);
  XLS_CHECK_EQ(index.bit_count(), BitCount());
  Bits so_far = bits_ops::ZeroExtend(index, BitCount() + 1);
  std::optional<Interval> to_index;
  for (const Interval& interval : Intervals()) {
    Bits size = interval.SizeBits();
    if (bits_ops::ULessThan(so_far, size)) {
      to_index = interval;
      break;
    }
    so_far = bits_ops::Sub(so_far, size);
  }
  if (!to_index.has_value()) {
    return std::nullopt;
  }
  Bits result = bits_ops::Add(
      so_far,
      bits_ops::ZeroExtend(to_index.value().LowerBound(), BitCount() + 1));
  XLS_CHECK(!result.msb());
  return result.Slice(0, BitCount());
}

bool IntervalSet::IsTrueWhenMaskWith(const Bits& value) const {
  XLS_CHECK_EQ(value.bit_count(), BitCount());
  for (const Interval& interval : intervals_) {
    if (interval.IsTrueWhenAndWith(value)) {
      return true;
    }
  }
  return false;
}

bool IntervalSet::Covers(const Bits& bits) const {
  XLS_CHECK_EQ(bits.bit_count(), BitCount());
  for (const Interval& interval : intervals_) {
    if (interval.Covers(bits)) {
      return true;
    }
  }
  return false;
}

bool IntervalSet::IsPrecise() const {
  XLS_CHECK_GE(bit_count_, 0);
  std::optional<Interval> precisely;
  for (const Interval& interval : intervals_) {
    if (precisely.has_value() && !(precisely.value() == interval)) {
      return false;
    }
    if (!interval.IsPrecise()) {
      return false;
    }
    precisely = interval;
  }
  return true;
}

std::optional<Bits> IntervalSet::GetPreciseValue() const {
  XLS_CHECK(is_normalized_);
  if (!IsPrecise()) {
    return std::nullopt;
  }
  XLS_CHECK_EQ(intervals_.size(), 1);
  return intervals_.front().GetPreciseValue();
}

bool IntervalSet::IsMaximal() const {
  XLS_CHECK(is_normalized_);
  XLS_CHECK_GE(bit_count_, 0);
  for (const Interval& interval : intervals_) {
    if (interval.IsMaximal()) {
      return true;
    }
  }
  return false;
}

bool IntervalSet::IsNormalized() const { return is_normalized_; }

bool IntervalSet::IsEmpty() const {
  XLS_CHECK(IsNormalized());
  return intervals_.empty();
}

std::string IntervalSet::ToString() const {
  XLS_CHECK_GE(bit_count_, 0);
  std::vector<std::string> strings;
  for (const auto& interval : intervals_) {
    strings.push_back(interval.ToString());
  }
  return absl::StrFormat("[%s]", absl::StrJoin(strings, ", "));
}

IntervalSet IntervalSet::Random(uint64_t seed, int64_t bit_count,
                                int64_t max_intervals) {
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int64_t> distrib(0, max_intervals);
  int64_t num_intervals = distrib(gen);
  IntervalSet result(bit_count);
  for (int64_t i = 0; i < num_intervals; ++i) {
    result.AddInterval(Interval::Random(gen(), bit_count));
  }
  result.Normalize();
  return result;
}

}  // namespace xls

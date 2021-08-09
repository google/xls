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

absl::optional<Interval> IntervalSet::ConvexHull() const {
  // TODO(taktoa): optimize case where is_normalized_ is true
  XLS_CHECK_GE(bit_count_, 0);
  absl::optional<Bits> lower;
  absl::optional<Bits> upper;
  for (const Interval& interval : intervals_) {
    if (lower.has_value()) {
      if (bits_ops::ULessThan(interval.LowerBound(), lower.value())) {
        lower = interval.LowerBound();
      }
    } else {
      lower = interval.LowerBound();
    }
    if (upper.has_value()) {
      if (bits_ops::UGreaterThan(interval.UpperBound(), upper.value())) {
        upper = interval.UpperBound();
      }
    } else {
      upper = interval.UpperBound();
    }
  }
  if (!lower.has_value() || !upper.has_value()) {
    return absl::nullopt;
  }
  return Interval(lower.value(), upper.value());
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

absl::optional<int64_t> IntervalSet::Size() const {
  XLS_CHECK(is_normalized_);
  int64_t total_size = 0;
  for (const Interval& interval : intervals_) {
    if (auto size = interval.Size()) {
      total_size += size.value();
    } else {
      return absl::nullopt;
    }
  }
  return total_size;
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
  absl::optional<Interval> precisely;
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

std::string IntervalSet::ToString() const {
  XLS_CHECK_GE(bit_count_, 0);
  std::vector<std::string> strings;
  for (const auto& interval : intervals_) {
    strings.push_back(interval.ToString());
  }
  return absl::StrFormat("[%s]", absl::StrJoin(strings, ", "));
}

}  // namespace xls

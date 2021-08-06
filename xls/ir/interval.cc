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

#include "xls/ir/interval.h"

namespace xls {

void Interval::EnsureValid() const {
  XLS_CHECK_EQ(lower_bound_.bit_count(), upper_bound_.bit_count());
  XLS_CHECK_GT(lower_bound_.bit_count(), 0);
}

int64_t Interval::BitCount() const {
  EnsureValid();
  return lower_bound_.bit_count();
}

Interval Interval::Maximal(int64_t bit_width) {
  XLS_CHECK_GT(bit_width, 0);
  return Interval(UBits(0, bit_width), Bits::AllOnes(bit_width));
}

bool Interval::Overlaps(const Interval& lhs, const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  if (rhs < lhs) {
    return Overlaps(rhs, lhs);
  }
  return bits_ops::UGreaterThanOrEqual(lhs.upper_bound_, rhs.lower_bound_);
}

bool Interval::Disjoint(const Interval& lhs, const Interval& rhs) {
  return !Overlaps(lhs, rhs);
}

bool Interval::Abuts(const Interval& lhs, const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  if (rhs < lhs) {
    return Abuts(rhs, lhs);
  }
  // If the two intervals overlap, they definitely don't abut.
  // This takes care of cases like
  // `Interval::Abuts(Interval::Maximal(n), Interval::Maximal(n))`
  // and any others I haven't thought of.
  if (Interval::Overlaps(lhs, rhs)) {
    return false;
  }
  Bits one = UBits(1, lhs.BitCount());
  return bits_ops::UEqual(bits_ops::Add(lhs.upper_bound_, one),
                          rhs.lower_bound_);
}

Interval Interval::ConvexHull(const Interval& lhs, const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  Interval result;
  result.lower_bound_ = lhs.lower_bound_;
  if (bits_ops::ULessThan(rhs.lower_bound_, result.lower_bound_)) {
    result.lower_bound_ = rhs.lower_bound_;
  }
  result.upper_bound_ = lhs.upper_bound_;
  if (bits_ops::UGreaterThan(rhs.upper_bound_, result.upper_bound_)) {
    result.upper_bound_ = rhs.upper_bound_;
  }
  return result;
}

bool Interval::ForEachElement(std::function<bool(const Bits&)> callback) const {
  EnsureValid();
  if (bits_ops::UEqual(lower_bound_, upper_bound_)) {
    return callback(lower_bound_);
  }
  Bits value = lower_bound_;
  Bits zero = UBits(0, BitCount());
  Bits one = UBits(1, BitCount());
  Bits max = Bits::AllOnes(BitCount());
  if (bits_ops::UGreaterThan(lower_bound_, upper_bound_)) {
    while (true) {
      if (callback(value)) {
        return true;
      }
      if (bits_ops::UEqual(value, max)) {
        break;
      }
      value = bits_ops::Add(value, one);
    }
    value = zero;
  }
  while (true) {
    if (callback(value)) {
      return true;
    }
    if (bits_ops::UEqual(value, upper_bound_)) {
      break;
    }
    value = bits_ops::Add(value, one);
  }
  return false;
}

std::vector<Bits> Interval::Elements() const {
  std::vector<Bits> result;
  ForEachElement([&result](const Bits& value) {
    result.push_back(value);
    return false;
  });
  return result;
}

Bits Interval::SizeBits() const {
  EnsureValid();
  if (bits_ops::UGreaterThan(lower_bound_, upper_bound_)) {
    Bits zero = UBits(0, BitCount());
    Bits max = Bits::AllOnes(BitCount());
    Bits x = Interval(lower_bound_, max).SizeBits();
    Bits y = Interval(zero, upper_bound_).SizeBits();
    return bits_ops::Add(x, y);
  }

  int64_t padded_size = BitCount() + 1;
  Bits difference = bits_ops::Sub(upper_bound_, lower_bound_);
  return bits_ops::Add(UBits(1, padded_size),
                       bits_ops::ZeroExtend(difference, padded_size));
}

absl::optional<int64_t> Interval::Size() const {
  absl::StatusOr<uint64_t> size_status = SizeBits().ToUint64();
  if (size_status.ok()) {
    uint64_t size = *size_status;
    if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return absl::nullopt;
    }
    return static_cast<int64_t>(size);
  }
  return absl::nullopt;
}

bool Interval::IsImproper() const {
  EnsureValid();
  return bits_ops::ULessThan(upper_bound_, lower_bound_);
}

bool Interval::IsPrecise() const {
  EnsureValid();
  return lower_bound_ == upper_bound_;
}

bool Interval::IsMaximal() const {
  EnsureValid();
  // This works because `bits_ops::Add` overflows, and correctly handles
  // improper intervals.
  Bits upper_plus_one = bits_ops::Add(upper_bound_, UBits(1, BitCount()));
  return bits_ops::UEqual(upper_plus_one, lower_bound_);
}

bool Interval::Covers(const Bits& point) const {
  EnsureValid();
  XLS_CHECK_EQ(BitCount(), point.bit_count());
  if (IsImproper()) {
    return bits_ops::ULessThanOrEqual(lower_bound_, point) ||
           bits_ops::ULessThanOrEqual(point, upper_bound_);
  } else {
    return bits_ops::ULessThanOrEqual(lower_bound_, point) &&
           bits_ops::ULessThanOrEqual(point, upper_bound_);
  }
}

bool Interval::CoversZero() const { return Covers(UBits(0, BitCount())); }

bool Interval::CoversOne() const { return Covers(UBits(1, BitCount())); }

bool Interval::CoversMax() const { return Covers(Bits::AllOnes(BitCount())); }

std::string Interval::ToString() const {
  FormatPreference pref = FormatPreference::kDefault;
  return absl::StrFormat("[%s, %s]", lower_bound_.ToString(pref, false),
                         upper_bound_.ToString(pref, false));
}

}  // namespace xls

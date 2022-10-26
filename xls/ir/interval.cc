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

#include <algorithm>
#include <random>

#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls {

void Interval::EnsureValid() const {
  XLS_CHECK(is_valid_);
  XLS_CHECK_EQ(lower_bound_.bit_count(), upper_bound_.bit_count());
}

int64_t Interval::BitCount() const {
  EnsureValid();
  return lower_bound_.bit_count();
}

Interval Interval::Maximal(int64_t bit_width) {
  return Interval(UBits(0, bit_width), Bits::AllOnes(bit_width));
}

Interval Interval::Precise(const Bits& bits) { return Interval(bits, bits); }

Interval Interval::ZeroExtend(int64_t bit_width) const {
  XLS_CHECK_GE(bit_width, BitCount());
  return Interval(bits_ops::ZeroExtend(LowerBound(), bit_width),
                  bits_ops::ZeroExtend(UpperBound(), bit_width));
}

bool Interval::Overlaps(const Interval& lhs, const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  if (lhs.BitCount() == 0) {
    // The unique zero-width interval overlaps with itself.
    return true;
  }
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
  if (lhs.BitCount() == 0) {
    // The unique zero-width interval does not abut itself.
    return false;
  }
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
  if (lhs.BitCount() == 0) {
    // The convex hull of any set of zero-width intervals is of course the
    // unique zero-width interval.
    return Interval(Bits(), Bits());
  }
  Interval result = lhs;
  if (bits_ops::ULessThan(rhs.lower_bound_, result.lower_bound_)) {
    result.lower_bound_ = rhs.lower_bound_;
  }
  if (bits_ops::UGreaterThan(rhs.upper_bound_, result.upper_bound_)) {
    result.upper_bound_ = rhs.upper_bound_;
  }
  return result;
}

std::optional<Interval> Interval::Intersect(const Interval& lhs,
                                             const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  if (!Interval::Overlaps(lhs, rhs)) {
    return absl::nullopt;
  }
  Bits lower = lhs.LowerBound();
  if (bits_ops::UGreaterThan(rhs.LowerBound(), lower)) {
    lower = rhs.LowerBound();
  }
  Bits upper = lhs.UpperBound();
  if (bits_ops::ULessThan(rhs.UpperBound(), upper)) {
    upper = rhs.UpperBound();
  }
  return Interval(lower, upper);
}

std::vector<Interval> Interval::Difference(const Interval& lhs,
                                           const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  if (!Interval::Overlaps(lhs, rhs)) {
    // X - Y = X when X ∩ Y = ø
    return {lhs};
  }
  std::vector<Interval> result;
  if (bits_ops::ULessThan(lhs.LowerBound(), rhs.LowerBound())) {
    result.push_back(
        Interval(lhs.LowerBound(),
                 bits_ops::Sub(rhs.LowerBound(), UBits(1, rhs.BitCount()))));
  }
  if (bits_ops::ULessThan(rhs.UpperBound(), lhs.UpperBound())) {
    result.push_back(
        Interval(bits_ops::Add(rhs.UpperBound(), UBits(1, rhs.BitCount())),
                 lhs.UpperBound()));
  }
  return result;
}

std::vector<Interval> Interval::Complement(const Interval& interval) {
  return Difference(Maximal(interval.BitCount()), interval);
}

bool Interval::IsSubsetOf(const Interval& lhs, const Interval& rhs) {
  XLS_CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  XLS_CHECK(!lhs.IsImproper());
  XLS_CHECK(!rhs.IsImproper());
  return bits_ops::ULessThanOrEqual(rhs.LowerBound(), lhs.LowerBound()) &&
         bits_ops::UGreaterThanOrEqual(rhs.UpperBound(), lhs.UpperBound());
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

std::optional<int64_t> Interval::Size() const {
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
  if (BitCount() == 0) {
    return false;
  }
  return bits_ops::ULessThan(upper_bound_, lower_bound_);
}

bool Interval::IsPrecise() const {
  EnsureValid();
  if (BitCount() == 0) {
    return true;
  }
  return lower_bound_ == upper_bound_;
}

std::optional<Bits> Interval::GetPreciseValue() const {
  if (!IsPrecise()) {
    return absl::nullopt;
  }
  return lower_bound_;
}

bool Interval::IsMaximal() const {
  EnsureValid();
  if (BitCount() == 0) {
    return true;
  }
  // This works because `bits_ops::Add` overflows, and correctly handles
  // improper intervals.
  Bits upper_plus_one = bits_ops::Add(upper_bound_, UBits(1, BitCount()));
  return bits_ops::UEqual(upper_plus_one, lower_bound_);
}

bool Interval::IsTrueWhenAndWith(const Bits& value) const {
  XLS_CHECK_EQ(value.bit_count(), BitCount());
  int64_t right_index = std::min(LowerBound().CountTrailingZeros(),
                                 UpperBound().CountTrailingZeros());
  int64_t left_index = BitCount() - UpperBound().CountLeadingZeros();
  Bits interval_mask_value(BitCount());
  interval_mask_value.SetRange(right_index, left_index);
  return !bits_ops::And(interval_mask_value, value).IsZero();
}

bool Interval::Covers(const Bits& point) const {
  EnsureValid();
  XLS_CHECK_EQ(BitCount(), point.bit_count());
  if (BitCount() == 0) {
    return true;
  }
  if (IsImproper()) {
    return bits_ops::ULessThanOrEqual(lower_bound_, point) ||
           bits_ops::ULessThanOrEqual(point, upper_bound_);
  } else {
    return bits_ops::ULessThanOrEqual(lower_bound_, point) &&
           bits_ops::ULessThanOrEqual(point, upper_bound_);
  }
}

bool Interval::CoversZero() const { return Covers(UBits(0, BitCount())); }

bool Interval::CoversOne() const {
  return (BitCount() > 0) && Covers(UBits(1, BitCount()));
}

bool Interval::CoversMax() const { return Covers(Bits::AllOnes(BitCount())); }

std::string Interval::ToString() const {
  FormatPreference pref = FormatPreference::kDefault;
  return absl::StrFormat("[%s, %s]", lower_bound_.ToString(pref, false),
                         upper_bound_.ToString(pref, false));
}

Interval Interval::Random(uint32_t seed, int64_t bit_count) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int32_t> distrib(0, 255);
  int64_t num_bytes = (bit_count / 8) + ((bit_count % 8 == 0) ? 0 : 1);
  std::vector<uint8_t> start_bytes(num_bytes);
  for (int64_t i = 0; i < num_bytes; ++i) {
    start_bytes[i] = distrib(gen);
  }
  std::vector<uint8_t> end_bytes(num_bytes);
  for (int64_t i = 0; i < num_bytes; ++i) {
    end_bytes[i] = distrib(gen);
  }
  return Interval(Bits::FromBytes(start_bytes, bit_count),
                  Bits::FromBytes(end_bytes, bit_count));
}

}  // namespace xls

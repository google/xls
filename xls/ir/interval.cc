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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls {

int64_t Interval::BitCount() const {
  EnsureValid();
  return lower_bound_.bit_count();
}

Interval Interval::Maximal(int64_t bit_width) {
  return Interval(UBits(0, bit_width), Bits::AllOnes(bit_width));
}

Interval Interval::Precise(const Bits& bits) { return Interval(bits, bits); }

Interval Interval::ZeroExtend(int64_t bit_width) const {
  CHECK_GE(bit_width, BitCount());
  return Interval(bits_ops::ZeroExtend(LowerBound(), bit_width),
                  bits_ops::ZeroExtend(UpperBound(), bit_width));
}

bool Interval::Overlaps(const Interval& lhs, const Interval& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(!lhs.IsImproper());
  CHECK(!rhs.IsImproper());
  if (lhs.BitCount() == 0) {
    // The unique zero-width interval overlaps itself.
    return true;
  }

  // Suppose lhs = [a, b] and rhs = [c, d].
  const Bits& a = lhs.LowerBound();
  const Bits& b = lhs.UpperBound();
  const Bits& c = rhs.LowerBound();
  const Bits& d = rhs.UpperBound();
  // Since the intervals are proper, we know that a <= b and c <= d.

  // Suppose b < c; we would then have a <= b < c <= d, so no overlap.
  // Similarly, suppose d < a; we then have c <= d < a <= b, so no overlap.
  // On the other hand, if b >= c and d >= a, then:
  // - a <= b and a <= d, so a <= min(b, d), and
  // - c <= b and c <= d, so c <= min(b, d).
  // By the definition of min(x, y):
  // - a <= min(b, d) <= b, and
  // - c <= min(b, d) <= d.
  // Therefore, min(b, d) ∈ [a, b] ∩ [c, d], so the intervals overlap.

  // In other words, the intervals overlap if and only if b >= c and d >= a.
  return bits_ops::UGreaterThanOrEqual(b, c) &&
         bits_ops::UGreaterThanOrEqual(d, a);
}

bool Interval::Disjoint(const Interval& lhs, const Interval& rhs) {
  return !Overlaps(lhs, rhs);
}

bool Interval::Abuts(const Interval& lhs, const Interval& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(!lhs.IsImproper());
  CHECK(!rhs.IsImproper());
  if (lhs.BitCount() == 0) {
    // The unique zero-width interval does not abut itself.
    return false;
  }

  // Suppose lhs = [a, b] and rhs = [c, d].
  const Bits& a = lhs.LowerBound();
  const Bits& b = lhs.UpperBound();
  const Bits& c = rhs.LowerBound();
  const Bits& d = rhs.UpperBound();
  // Since the intervals are proper, we know that a <= b and c <= d.

  // If b is all ones, then it's the maximum value; we must have b >= c, so they
  // can't abut in that direction.
  if (!b.IsAllOnes()) {
    int64_t bp_vs_c = bits_ops::UCmp(bits_ops::Increment(b), c);
    if (bp_vs_c == 0) {
      // b + 1 == c, so [a, b] and [c, d] abut.
      return true;
    }
    if (bp_vs_c < 0) {
      // b + 1 < c.
      // Since a <= b < b + 1 < c <= d, there's an element between [a, b] and
      // [c, d], so they don't abut.
      return false;
    }
  }

  // b >= c, so c <= b. The only remaining way these intervals can abut is:
  // if d is not the maximum value and a == d + 1.
  return !d.IsAllOnes() && bits_ops::UEqual(bits_ops::Increment(d), a);
}

Interval Interval::ConvexHull(const Interval& lhs, const Interval& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(!lhs.IsImproper());
  CHECK(!rhs.IsImproper());
  if (lhs.BitCount() == 0) {
    // The convex hull of any set of zero-width intervals is of course the
    // unique zero-width interval.
    return Interval(Bits(), Bits());
  }
  return Interval(bits_ops::UMin(lhs.LowerBound(), rhs.LowerBound()),
                  bits_ops::UMax(lhs.UpperBound(), rhs.UpperBound()));
}

std::optional<Interval> Interval::Intersect(const Interval& lhs,
                                            const Interval& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(!lhs.IsImproper());
  CHECK(!rhs.IsImproper());

  Interval intersection(bits_ops::UMax(lhs.LowerBound(), rhs.LowerBound()),
                        bits_ops::UMin(lhs.UpperBound(), rhs.UpperBound()));
  if (intersection.IsImproper()) {
    return std::nullopt;
  }
  return std::move(intersection);
}

std::vector<Interval> Interval::Difference(const Interval& lhs,
                                           const Interval& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(!lhs.IsImproper());
  CHECK(!rhs.IsImproper());

  // X - Y = X - (X ∩ Y)
  std::optional<Interval> intersection = Intersect(lhs, rhs);
  if (!intersection.has_value()) {
    // X ∩ Y = ø, so X - Y = X - ø = X
    return {lhs};
  }

  // Suppose lhs = [a, d] and intersection = [b, c].
  const Bits& a = lhs.LowerBound();
  const Bits& b = intersection->LowerBound();
  const Bits& c = intersection->UpperBound();
  const Bits& d = lhs.UpperBound();
  // Since intersection is a subset of lhs, a <= b <= c <= d.

  // [a, d] - [b, c] = [a, b) ∪ (c, d]
  std::vector<Interval> result;
  result.reserve(2);
  if (bits_ops::ULessThan(a, b)) {
    result.push_back(Interval::RightOpen(a, b));
  }
  if (bits_ops::ULessThan(c, d)) {
    result.push_back(Interval::LeftOpen(c, d));
  }
  return result;
}

std::vector<Interval> Interval::Complement(const Interval& interval) {
  return Difference(Maximal(interval.BitCount()), interval);
}

bool Interval::IsSubsetOf(const Interval& lhs, const Interval& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(!lhs.IsImproper());
  CHECK(!rhs.IsImproper());
  return bits_ops::ULessThanOrEqual(rhs.LowerBound(), lhs.LowerBound()) &&
         bits_ops::UGreaterThanOrEqual(rhs.UpperBound(), lhs.UpperBound());
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
  return bits_ops::Increment(bits_ops::ZeroExtend(difference, padded_size));
}

std::optional<int64_t> Interval::Size() const {
  return bits_ops::TryUnsignedBitsToInt64(SizeBits());
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
    return std::nullopt;
  }
  return lower_bound_;
}

bool Interval::IsMaximal() const {
  EnsureValid();
  if (BitCount() == 0) {
    return true;
  }
  if (IsImproper()) {
    // Means upper is less than lower.
    return bits_ops::UEqual(bits_ops::Increment(upper_bound_), lower_bound_);
  }
  return lower_bound_.IsZero() && upper_bound_.IsAllOnes();
}

bool Interval::IsTrueWhenAndWith(const Bits& value) const {
  CHECK_EQ(value.bit_count(), BitCount());
  BitsRope interval_mask_value(BitCount());
  Bits common_prefix =
      bits_ops::LongestCommonPrefixMSB({LowerBound(), UpperBound()});
  interval_mask_value.push_back(
      Bits::AllOnes(BitCount() - common_prefix.bit_count()));
  interval_mask_value.push_back(common_prefix);
  return !bits_ops::And(interval_mask_value.Build(), value).IsZero();
}

bool Interval::Covers(const Bits& point) const {
  EnsureValid();
  CHECK_EQ(BitCount(), point.bit_count());
  if (BitCount() == 0) {
    return true;
  }
  if (IsImproper()) {
    return bits_ops::ULessThanOrEqual(lower_bound_, point) ||
           bits_ops::ULessThanOrEqual(point, upper_bound_);
  }
  return bits_ops::ULessThanOrEqual(lower_bound_, point) &&
         bits_ops::ULessThanOrEqual(point, upper_bound_);
}

bool Interval::CoversZero() const { return Covers(UBits(0, BitCount())); }

bool Interval::CoversOne() const {
  return (BitCount() > 0) && Covers(UBits(1, BitCount()));
}

bool Interval::CoversMax() const { return Covers(Bits::AllOnes(BitCount())); }

std::string Interval::ToString() const {
  FormatPreference pref = FormatPreference::kDefault;
  return absl::StrFormat("[%s, %s]", BitsToString(lower_bound_, pref, false),
                         BitsToString(upper_bound_, pref, false));
}

}  // namespace xls

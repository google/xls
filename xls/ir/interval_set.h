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

#ifndef XLS_IR_INTERVAL_SET_H_
#define XLS_IR_INTERVAL_SET_H_

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
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

// This type represents a set of intervals.
class IntervalSet {
 public:
  // Create an empty `IntervalSet` with a `BitCount()` of -1. Every method in
  // this class fails if called on an `IntervalSet` with bit count -1, so you
  // must assign to a default constructed interval set before calling any method
  // on it.
  IntervalSet() : is_normalized_(true), bit_count_(-1), intervals_() {}

  // Create an empty `IntervalSet` with the given bit count.
  explicit IntervalSet(int64_t bit_count)
      : is_normalized_(true), bit_count_(bit_count), intervals_() {}

  // Returns an interval set that covers every bit pattern with the given width.
  static IntervalSet Maximal(int64_t bit_count);

  // Returns an interval set that covers exactly the given bit pattern.
  static IntervalSet Precise(const Bits& bits);

  // Returns the number of intervals in the set.
  // Does not check for normalization, as this function can be used to check if
  // normalization is required (e.g.: to prevent blowup in memory usage while
  // building a large set of intervals).
  int64_t NumberOfIntervals() const { return intervals_.size(); }

  // Get all the intervals contained within this interval set.
  // The set must be normalized prior to calling this.
  absl::Span<const Interval> Intervals() const {
    XLS_CHECK(is_normalized_);
    return intervals_;
  }

  // Returns the `BitCount()` of all intervals in the interval set.
  int64_t BitCount() const {
    XLS_CHECK_GE(bit_count_, 0);
    return bit_count_;
  }

  // Add an interval to this interval set.
  void AddInterval(const Interval& interval) {
    is_normalized_ = false;
    XLS_CHECK_EQ(BitCount(), interval.BitCount());
    intervals_.push_back(interval);
  }

  // Modify the set of intervals in this to be exactly the given set.
  // If the set of intervals is empty, then the `BitCount()` is set to -1.
  // Otherwise, the `BitCount()` is set to the `BitCount()` of the
  // given intervals.
  void SetIntervals(absl::Span<Interval const> intervals);

  // Normalize the set of intervals so that the following statements are true:
  //
  // 1. The union of the set of points contained within all intervals after
  //    normalization is the same as that before normalization
  //    (i.e.: normalization does not affect the smeantics of a set of
  //    intervals).
  // 2. After normalization, the set contains no improper intervals.
  // 3. After normalization, no two intervals in the set will overlap or abut.
  // 4. After normalization, the result of a call to `Intervals()` will be
  //    sorted in lexicographic order (with the underlying ordering given by
  //    interpreting each `Bits` as an unsigned integer).
  // 5. The result of a call to `Intervals()` has the smallest possible size
  //    of any set of intervals representing the same set of points that
  //    contains no improper intervals (hence the name "normalization").
  void Normalize();

  // Return the smallest single proper interval that contains all points in this
  // interval set. If the set of points is empty, returns `std::nullopt`.
  std::optional<Interval> ConvexHull() const;

  // Returns an `IntervalSet` the bounds of which have been zero-extended to the
  // given bit width.
  // CHECK fails if the bit width is less than the current `BitCount`.
  IntervalSet ZeroExtend(int64_t bit_width) const;

  // Call the given function on each point contained within this set of
  // intervals. The function returns a `bool` that, if true, ends the iteration
  // early and results in `ForEachElement` returning true. If the iteration does
  // not end early, false is returned.
  //
  // CHECK fails if this interval set is not normalized, as that can lead to
  // unexpectedly calling the callback on the same point twice.
  bool ForEachElement(std::function<bool(const Bits&)> callback) const;

  // Returns a normalized set of intervals comprising the union of the two given
  // interval sets.
  static IntervalSet Combine(const IntervalSet& lhs, const IntervalSet& rhs);

  // Returns a normalized set of intervals comprising the intersection of the
  // two given interval sets.
  static IntervalSet Intersect(const IntervalSet& lhs, const IntervalSet& rhs);

  // Returns the normalized set of intervals comprising the complemet of the
  // given interval set.
  static IntervalSet Complement(const IntervalSet& set);

  // Returns the number of points covered by the intervals in this interval set,
  // if that is expressible as an `int64_t`. Otherwise, returns `std::nullopt`.
  // CHECK fails if the interval set is not normalized.
  std::optional<int64_t> Size() const;

  // If all the intervals in this interval set were put next to each other and
  // treated as a map from `[0, Size() - 1]` to `Bits`, apply this mapping to
  // the given integer. Returns `std::nullopt` iff the given integer is out of
  // range for this mapping. The returned `Bits` has bitwidth equal to that of
  // this interval set.
  //
  // CHECK fails if this interval set is not normalized, or if the given `index`
  // has bitwidth not equal to the bitwidth of this interval set.
  //
  // For example, if the interval set was `{[2, 3], [5, 7]}`, then this method
  // would be like indexing the array `[2, 3, 5, 6, 7]`.
  std::optional<Bits> Index(const Bits& index) const;

  // Do any of the values within the interval ranges when bit mask with the
  // given value produce a non-zero result?
  //
  // In other words, `intervals.IsTrueWhenMaskWith(m)` is the same as `there
  // exists a value x contained in the interval ranges such that or_reduce(x &
  // m) is true`.
  //
  // For example, `intervals.IsTrueWhenMaskWith(4)` where intervals is the set
  // `{[2, 3], [5, 7]}`, the result would return true, since the values 4, 5, 6
  // and 7 bit masked with 4 returns true. However, if intervals is the set
  // `{[2, 3]}`, the result would return false, since no values bit masked with
  // 4 returns true.
  bool IsTrueWhenMaskWith(const Bits& value) const;

  // Do any of the intervals cover the given point?
  bool Covers(const Bits& bits) const;

  // Do any of the intervals cover zero?
  bool CoversZero() const { return Covers(UBits(0, BitCount())); }

  // Do any of the intervals cover one?
  bool CoversOne() const { return Covers(UBits(1, BitCount())); }

  // Do any of the intervals cover `Bits::AllOnes(BitCount())`?
  bool CoversMax() const { return Covers(Bits::AllOnes(BitCount())); }

  // Do the intervals only cover one point?
  bool IsPrecise() const;

  // Returns the unique member of the interval set if the interval set is
  // precise. If the interval is not precise, returns std::nullopt.
  // CHECK fails if the interval set is not normalized.
  std::optional<Bits> GetPreciseValue() const;

  // Do the intervals cover every point?
  // `Normalize()` must be called prior to calling this method.
  bool IsMaximal() const;

  // Returns true iff this set of intervals is normalized.
  bool IsNormalized() const;

  // Returns true iff this set of intervals is empty.
  bool IsEmpty() const;

  // Returns the inclusive lower/upper bound of the interval set.
  std::optional<Bits> LowerBound() const;
  std::optional<Bits> UpperBound() const;

  // Print this set of intervals as a string.
  std::string ToString() const;

  // Generates a random set of intervals with the given seed and bit count.
  // This is useful for tests.
  //
  // The `max_intervals` parameter controls the maximum number of intervals
  // that can be present in the returned set, which is useful for limiting
  // the amount of computation done by anything consuming the output of this.
  static IntervalSet Random(uint64_t seed, int64_t bit_count,
                            int64_t max_intervals);

  friend bool operator==(IntervalSet lhs, IntervalSet rhs) {
    lhs.Normalize();
    rhs.Normalize();
    if (lhs.bit_count_ != rhs.bit_count_) {
      return false;
    }
    return lhs.intervals_ == rhs.intervals_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const IntervalSet& set) {
    return H::combine(std::move(h), set.bit_count_, set.intervals_);
  }

 private:
  bool is_normalized_;
  int64_t bit_count_;
  std::vector<Interval> intervals_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const IntervalSet& interval_set) {
  os << interval_set.ToString();
  return os;
}

}  // namespace xls

#endif  // XLS_IR_INTERVAL_SET_H_

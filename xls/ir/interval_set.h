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

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/iterator_range.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"

namespace xls {

// This type represents a set of intervals.
class IntervalSet {
 public:
  // Create an empty `IntervalSet` with a `BitCount()` of -1. Every method in
  // this class fails if called on an `IntervalSet` with bit count -1, so you
  // must assign to a default constructed interval set before calling any method
  // on it.
  IntervalSet() : is_normalized_(true), bit_count_(-1) {}

  // Returns an interval set of all the positive intervals of this set.
  IntervalSet PositiveIntervals(bool with_zero = true) const;

  // Returns an interval set of the absolute values of all the negative
  // intervals of this set. The resulting set has the same bit-count as the
  // current set.
  IntervalSet NegativeAbsoluteIntervals() const;

  // Create an empty `IntervalSet` with the given bit count.
  explicit IntervalSet(int64_t bit_count)
      : is_normalized_(true), bit_count_(bit_count) {}

  // Returns true if the intersection of the two interval sets would be empty
  // (without constructing the intersection).
  static bool Disjoint(const IntervalSet& lhs, const IntervalSet& rhs);

  // Returns an interval set that covers every bit pattern with the given width.
  static IntervalSet Maximal(int64_t bit_count);

  // Returns an interval set that covers every bit pattern except zero with the
  // given width.
  static IntervalSet NonZero(int64_t bit_count);

  // Returns an interval set that covers exactly the given bit pattern.
  static IntervalSet Precise(const Bits& bits);

  // Returns an interval set that covers every bit pattern except the given bit
  // pattern.
  static IntervalSet Punctured(const Bits& bits);

  // Returns the number of intervals in the set.
  // Does not check for normalization, as this function can be used to check if
  // normalization is required (e.g.: to prevent blowup in memory usage while
  // building a large set of intervals).
  int64_t NumberOfIntervals() const { return intervals_.size(); }

  // Returns the number of intervals in the set assuming that there is a cut
  // between INT_MAX and INT_MIN. This is either exactly NumberOfIntervals() or
  // one more than it.
  int64_t NumberOfSignedIntervals() const {
    if (Covers(Bits::MaxSigned(BitCount())) &&
        Covers(Bits::MinSigned(BitCount()))) {
      return NumberOfIntervals() + 1;
    }
    return NumberOfIntervals();
  }

  // Get all the intervals contained within this interval set.
  // The set must be normalized prior to calling this.
  absl::Span<const Interval> Intervals() const& {
    CHECK(is_normalized_);
    return intervals_;
  }

  std::vector<Interval> Intervals() && {
    CHECK(is_normalized_);
    return intervals_;
  }

  class SignedIntervalIterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = const Interval&;

    SignedIntervalIterator(SignedIntervalIterator&&) = default;
    SignedIntervalIterator(const SignedIntervalIterator&) = default;
    SignedIntervalIterator& operator=(SignedIntervalIterator&&) = default;
    SignedIntervalIterator& operator=(const SignedIntervalIterator&) = default;

    const Interval& operator*() const;
    SignedIntervalIterator& operator++();
    SignedIntervalIterator operator++(int) {
      SignedIntervalIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const SignedIntervalIterator& o) const {
      if (InSplit() && o.InSplit() && in_negatives_ != o.in_negatives_) {
        return false;
      }
      return cur_ == o.cur_;
    }

   private:
    explicit SignedIntervalIterator(
        absl::Span<const Interval>::const_iterator cur,
        absl::Span<const Interval>::const_iterator end, bool in_negatives)
        : cur_(cur), end_(end), in_negatives_(in_negatives) {}
    bool InSplit() const;

    absl::Span<const Interval>::const_iterator cur_;
    absl::Span<const Interval>::const_iterator end_;

    // If this is 'InSplit' what side of it we are on.
    bool in_negatives_;

    // Holder for the split interval.
    mutable std::optional<Interval> sign_swap_interval_;

    friend class IntervalSet;
  };

  // Get all the intervals contained in the interval_set. The set must be
  // normalized prior to calling this. The iteration order is all positive
  // intervals followed by all negative intervals. Every interval will contain
  // only positive or negative numbers.
  xabsl::iterator_range<SignedIntervalIterator> SignedIntervals() const;

  // A lazy iterator of the bits values in this interval set.
  class ValuesIterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = Interval::ValuesIterator::value_type;
    using reference = Interval::ValuesIterator::reference;
    using pointer = Interval::ValuesIterator::pointer;
    using iterator_category = std::forward_iterator_tag;

    ValuesIterator(ValuesIterator&&) = default;
    ValuesIterator(const ValuesIterator&) = default;
    ValuesIterator& operator=(ValuesIterator&&) = default;
    ValuesIterator& operator=(const ValuesIterator&) = default;

    const Bits& operator*() const { return *cur_value_; }
    const Bits* operator->() const { return cur_value_.operator->(); }
    ValuesIterator& operator++() {
      CHECK(cur_interval_ != end_interval_) << "Incr after end!";
      CHECK(cur_value_ != cur_end_) << "Incr after end val!";
      ++cur_value_;
      if (cur_value_ == cur_end_) {
        ++cur_interval_;
        if (cur_interval_ != end_interval_) {
          cur_value_ = cur_interval_->begin();
          cur_end_ = cur_interval_->end();
        }
      }
      return *this;
    }
    ValuesIterator operator++(int) {
      ValuesIterator tmp = *this;
      ++*this;
      return tmp;
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const ValuesIterator& it) {
      if (it.cur_interval_ == it.end_interval_) {
        absl::Format(&sink, "END");
      } else {
        absl::Format(&sink, "%v", *it);
      }
    }

    bool operator==(const ValuesIterator& o) const {
      return (cur_interval_ == o.cur_interval_ &&
              cur_interval_ == end_interval_) ||
             (cur_interval_ == o.cur_interval_ && cur_value_ == o.cur_value_);
    }

   private:
    explicit ValuesIterator(absl::Span<const Interval>::const_iterator cur,
                            absl::Span<const Interval>::const_iterator end,
                            Interval::ValuesIterator&& cur_value,
                            Interval::ValuesIterator&& cur_end)
        : cur_interval_(cur),
          end_interval_(end),
          cur_value_(std::move(cur_value)),
          cur_end_(std::move(cur_end)) {}
    bool InSplit() const;

    absl::Span<const Interval>::const_iterator cur_interval_;
    absl::Span<const Interval>::const_iterator end_interval_;
    Interval::ValuesIterator cur_value_;
    Interval::ValuesIterator cur_end_;

    friend class IntervalSet;
  };
  // Get all the values contained in the interval_set. The set must be
  // normalized prior to calling this. The iteration order is unsigned values
  // from the lowest to the highest.
  xabsl::iterator_range<ValuesIterator> Values() const {
    CHECK(is_normalized_);
    if (intervals_.empty()) {
      return xabsl::make_range(
          ValuesIterator(Intervals().begin(), Intervals().end(),
                         Interval::ValuesIterator::None(),
                         Interval::ValuesIterator::None()),
          ValuesIterator(Intervals().begin(), Intervals().end(),
                         Interval::ValuesIterator::None(),
                         Interval::ValuesIterator::None()));
    }
    return xabsl::make_range(
        ValuesIterator(Intervals().begin(), Intervals().end(),
                       Intervals().front().begin(), Intervals().front().end()),
        ValuesIterator(Intervals().end(), Intervals().end(),
                       Intervals().back().end(), Intervals().back().end()));
  }

  // Returns the `BitCount()` of all intervals in the interval set.
  int64_t BitCount() const {
    CHECK_GE(bit_count_, 0);
    return bit_count_;
  }

  // Add an interval to this interval set.
  void AddInterval(const Interval& interval) {
    is_normalized_ = false;
    CHECK_EQ(BitCount(), interval.BitCount());
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
  // not end early, false is returned. Generally you should prefer the Values
  // interface.
  //
  // CHECK fails if this interval set is not normalized, as that can lead to
  // unexpectedly calling the callback on the same point twice.
  template <typename Func>
    requires(std::is_invocable_r_v<bool, Func, const Bits&>)
  ABSL_DEPRECATE_AND_INLINE()
  bool ForEachElement(Func callback) const {
    CHECK(is_normalized_);
    for (const Bits& b : Values()) {
      if (callback(b)) {
        return true;
      }
    }
    return false;
  }

  // Returns a normalized set of intervals comprising the union of the two given
  // interval sets.
  static IntervalSet Combine(const IntervalSet& lhs, const IntervalSet& rhs);

  // Returns a normalized set of intervals comprising the intersection of the
  // two given interval sets.
  static IntervalSet Intersect(const IntervalSet& lhs, const IntervalSet& rhs);

  // Returns the normalized set of intervals comprising the complemet of the
  // given interval set.
  static IntervalSet Complement(const IntervalSet& set);

  static IntervalSet Of(absl::Span<Interval const> intervals);

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

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const IntervalSet& set) {
    absl::Format(&sink, "%s", set.ToString());
  }

  // Actually check exhaustively for normalization instead of trusting the flag.
  absl::Status CheckIsNormalizedForTesting() const;

 private:
  IntervalSet(std::in_place_t in_place, bool is_normalized, int64_t bit_count,
              std::vector<Interval> intervals)
      : is_normalized_(is_normalized),
        bit_count_(bit_count),
        intervals_(intervals) {}

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

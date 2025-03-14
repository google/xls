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

#ifndef XLS_IR_INTERVAL_H_
#define XLS_IR_INTERVAL_H_

#include <stdbool.h>

#include <compare>
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
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls {

// This is a type representing intervals in the set of `Bits` of a given
// bit width. It allows improper intervals (i.e.: ones where the lower bound
// is greater than the upper bound, so the interval wraps around the end),
// though some methods do not support them (and check to ensure that they are
// not called on them).
//
// NOTE: The interval bounds are always treated as unsigned.
class Interval {
 public:
  // No argument constructor for `Interval`. This returns the interval from a
  // zero-bit 0 to another zero-bit 0. However, the interval contains metadata
  // stating that it is an invalid interval, to prevent accidental use of
  // default-constructed intervals. If you want the zero-bit interval, use the
  // other `Interval` constructor.
  Interval() : is_valid_(false) {}

  // Create an `Interval`. The `bit_count()` of the lower bound must be equal to
  // that of the upper bound.
  //
  // The upper/lower bound are both considered inclusive.
  Interval(const Bits& lower_bound, const Bits& upper_bound)
      : is_valid_(true), lower_bound_(lower_bound), upper_bound_(upper_bound) {
    CHECK_EQ(lower_bound_.bit_count(), upper_bound_.bit_count());
  }

  // Returns the interval [lower_bound, upper_bound].
  static Interval Closed(const Bits& lower_bound, const Bits& upper_bound) {
    return Interval(lower_bound, upper_bound);
  }

  // Returns the interval (lower_bound, upper_bound].
  static Interval LeftOpen(const Bits& lower_bound, const Bits& upper_bound) {
    return Interval(bits_ops::Increment(lower_bound), upper_bound);
  }

  // Returns the interval [lower_bound, upper_bound).
  static Interval RightOpen(const Bits& lower_bound, const Bits& upper_bound) {
    return Interval(lower_bound, bits_ops::Decrement(upper_bound));
  }

  // Returns the interval (lower_bound, upper_bound).
  static Interval Open(const Bits& lower_bound, const Bits& upper_bound) {
    return Interval(bits_ops::Increment(lower_bound),
                    bits_ops::Decrement(upper_bound));
  }

  // The inclusive lower bound of the interval.
  const Bits& LowerBound() const { return lower_bound_; }

  // The inclusive upper bound of the interval.
  const Bits& UpperBound() const { return upper_bound_; }

  // Returns the number of bits in the lower/upper bound of the interval
  int64_t BitCount() const;

  // Returns an `Interval` that covers every bit pattern of a given width.
  static Interval Maximal(int64_t bit_width);

  // Returns an `Interval` that covers precisely the given bit pattern.
  static Interval Precise(const Bits& bits);

  // Returns an `Interval` the bounds of which have been zero-extended to the
  // given bit width.
  // CHECK fails if the bit width is less than the current `BitCount`.
  Interval ZeroExtend(int64_t bit_width) const;

  // Given two `Interval`s, return whether they overlap.
  // Does not accept improper intervals.
  static bool Overlaps(const Interval& lhs, const Interval& rhs);

  // Given two `Interval`s, return whether they are disjoint.
  // Does not accept improper intervals.
  static bool Disjoint(const Interval& lhs, const Interval& rhs);

  // Interval (a, b) "abuts" interval (x, y) if b + 1 = x or y + 1 = a
  // In other words, they abut iff they do not overlap but their union is itself
  // an interval.
  // For example, (5, 7) and (8, 12) do not overlap but their union is (5, 12).
  // Does not accept improper intervals.
  static bool Abuts(const Interval& lhs, const Interval& rhs);

  // Given two `Interval`s, return an `Interval` representing their convex hull.
  // Does not accept improper intervals.
  static Interval ConvexHull(const Interval& lhs, const Interval& rhs);

  // Given two `Interval`s, return an `Interval` representing their
  // intersection, if one exists. Otherwise, returns `std::nullopt`.
  // Does not accept improper intervals.
  static std::optional<Interval> Intersect(const Interval& lhs,
                                           const Interval& rhs);

  // Given two `Interval`s, return a set of `Interval`s representing their
  // set difference.
  // Does not accept improper intervals.
  static std::vector<Interval> Difference(const Interval& lhs,
                                          const Interval& rhs);

  // Returns the set of intervals which represent the complement of `interval`.
  static std::vector<Interval> Complement(const Interval& interval);

  // Given two `Interval`s, return a boolean that is true iff `lhs` is a subset
  // of `rhs`. If the same interval is given twice, returns true (i.e.: improper
  // subsets are allowed). Does not accept improper intervals.
  static bool IsSubsetOf(const Interval& lhs, const Interval& rhs);

  // A lazy iterator of the values in this interval.
  //
  // TODO(allight): Implement the random-access iterator interface. NB
  // random-access iterator requires constant-time indexing.
  class ValuesIterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = Bits;
    using reference = const Bits&;
    using pointer = const Bits*;
    using iterator_category = std::forward_iterator_tag;
    static ValuesIterator None() { return ValuesIterator(Bits(0), false); }

    ValuesIterator(ValuesIterator&&) = default;
    ValuesIterator(const ValuesIterator&) = default;
    ValuesIterator& operator=(ValuesIterator&&) = default;
    ValuesIterator& operator=(const ValuesIterator&) = default;

    const Bits& operator*() const { return cur_value_; }
    const Bits* operator->() const { return &cur_value_; }
    ValuesIterator& operator++() {
      cur_value_ = bits_ops::Increment(std::move(cur_value_));
      is_first_and_maximal = false;
      return *this;
    }
    ValuesIterator operator++(int) {
      ValuesIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const ValuesIterator& o) const {
      return cur_value_ == o.cur_value_ &&
             is_first_and_maximal == o.is_first_and_maximal;
    }

   private:
    explicit ValuesIterator(const Bits& cur, bool is_first_and_maximal)
        : cur_value_(cur), is_first_and_maximal(is_first_and_maximal) {}
    Bits cur_value_;
    bool is_first_and_maximal;

    friend class Interval;
  };

  // Begin lazy iterator fo the elements of the interval.
  ValuesIterator begin() const {
    EnsureValid();
    // If we are a maximal interval we need to be sure to not consider the first
    // element the end.
    return ValuesIterator(lower_bound_, /*is_first_and_maximal=*/IsMaximal());
  }
  ValuesIterator cbegin() const { return begin(); }

  // End Iterator of the elements of the interval.
  ValuesIterator end() const {
    EnsureValid();
    return ValuesIterator(bits_ops::Increment(upper_bound_),
                          /*is_first_and_maximal=*/false);
  }
  ValuesIterator cend() const { return end(); }

  // Iterate over every point in the interval, calling the given callback for
  // each point. If the callback returns `true`, terminate the iteration early
  // and return `true`. Otherwise, continue the iteration until all points have
  // been visited and return `false`. The iterator interface should be prefered
  // to this.
  template <typename Func>
    requires(std::is_invocable_r_v<bool, Func, const Bits&>)
  ABSL_DEPRECATE_AND_INLINE()
  bool ForEachElement(Func callback) const {
    EnsureValid();
    for (const Bits& b : *this) {
      if (callback(b)) {
        return true;
      }
    }
    return false;
  }

  // This simply collects the results of iterating over this interval into a
  // `std::vector<Bits>` for convenience. This is often impractical as it will
  // use a lot of memory, but can be useful temporarily for debugging. Prefer
  // using the iterator interface above which generates the elements lazily.
  std::vector<Bits> Elements() const {
    std::vector<Bits> elems;
    elems.reserve(Size().value_or(1));
    for (const Bits& b : *this) {
      elems.push_back(b);
    }
    return elems;
  };

  // Returns the number of points contained within the interval as a `Bits`.
  //
  // The returned `Bits` has a bitwidth that is one greater than
  // the `BitCount()` of this interval.
  Bits SizeBits() const;

  // Returns the number of points contained within the interval, assuming that
  // number fits within a `uint64_t`. If it doesn't, `std::nullopt` is
  // returned.
  std::optional<int64_t> Size() const;

  // Returns `true` if this is an improper interval, `false` otherwise.
  // An improper interval is one where the upper bound is strictly less than
  // the lower bound.
  // The zero-width interval is not considered improper.
  bool IsImproper() const;

  // Returns `true` if this is a precise interval, `false` otherwise.
  // A precise interval is one that covers exactly one point.
  // The zero-width interval is considered precise.
  bool IsPrecise() const;

  // Returns the unique member of the interval if the interval is precise. If
  // the interval is not precise, returns std::nullopt.
  std::optional<Bits> GetPreciseValue() const;

  // Returns `true` if this is a maximal interval, `false` otherwise.
  // A maximal interval is one that covers every point of a given bitwidth.
  // The zero-width interval is considered maximal.
  bool IsMaximal() const;

  // Returns `true` when any values of the interval range AND'ed with the given
  // value is true.
  bool IsTrueWhenAndWith(const Bits& value) const;

  // Returns `true` if this interval covers the given point, `false` otherwise.
  bool Covers(const Bits& point) const;

  // Returns `true` if this interval covers zero, `false` otherwise.
  bool CoversZero() const;

  // Returns `true` if this interval covers one, `false` otherwise.
  bool CoversOne() const;

  // Returns `true` if this interval covers `Bits::AllOnes(this->BitCount())`,
  // `false` otherwise.
  bool CoversMax() const;

  // Prints the interval as a string.
  std::string ToString() const;

  // Lexicographic ordering of intervals.
  friend std::strong_ordering operator<=>(const Interval& lhs,
                                          const Interval& rhs) {
    auto cmp_bits = [](const Bits& l, const Bits& r) {
      return bits_ops::UCmp(l, r) <=> 0;
    };
    std::strong_ordering lower_cmp =
        cmp_bits(lhs.LowerBound(), rhs.LowerBound());
    if (lower_cmp == std::strong_ordering::equal) {
      return cmp_bits(lhs.UpperBound(), rhs.UpperBound());
    }
    return lower_cmp;
  }

  // Equality of intervals.
  friend bool operator==(const Interval& lhs, const Interval& rhs) {
    return (lhs <=> rhs) == std::strong_ordering::equal;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Interval& interval) {
    return H::combine(std::move(h), interval.lower_bound_,
                      interval.upper_bound_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Interval& interval) {
    absl::Format(&sink, "%s", interval.ToString());
  }

 private:
  void EnsureValid() const { CHECK(is_valid_); }

  bool is_valid_;
  Bits lower_bound_;
  Bits upper_bound_;
};

inline std::ostream& operator<<(std::ostream& os, const Interval& interval) {
  os << interval.ToString();
  return os;
}

}  // namespace xls

#endif  // XLS_IR_INTERVAL_H_

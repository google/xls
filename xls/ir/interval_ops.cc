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
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/common/iter_util.h"
#include "xls/common/iterator_range.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/passes/ternary_evaluator.h"

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
  CHECK(!intervals.IsEmpty())
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

  Bits lb = ternary_ops::ToKnownBitsValues(tern, /*default_set=*/false);
  Bits ub = ternary_ops::ToKnownBitsValues(tern, /*default_set=*/true);

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
    is.AddInterval(Interval::Closed(
        bits_ops::UMax(lb, bits_ops::Concat({high_bits, Bits(lsb_xs)})),
        bits_ops::UMin(ub,
                       bits_ops::Concat({high_bits, Bits::AllOnes(lsb_xs)}))));
    is.Normalize();
    return is;
  }

  TernaryVector vec(tern.size() - lsb_xs, TernaryValue::kKnownZero);
  // Copy input ternary from after the last lsb_x.
  std::copy(tern.cbegin() + lsb_xs, tern.cend(), vec.begin());

  Bits high_lsb = Bits::AllOnes(lsb_xs);
  Bits low_lsb(lsb_xs);
  for (const Bits& v : ternary_ops::AllBitsValues(vec)) {
    is.AddInterval(
        Interval::Closed(bits_ops::UMax(lb, bits_ops::Concat({v, low_lsb})),
                         bits_ops::UMin(ub, bits_ops::Concat({v, high_lsb}))));
  }
  is.Normalize();
  return is;
}

bool CoversTernary(const Interval& interval, TernarySpan ternary) {
  if (interval.BitCount() != ternary.size()) {
    return false;
  }
  if (ternary_ops::IsFullyKnown(ternary)) {
    return interval.Covers(ternary_ops::ToKnownBitsValues(ternary));
  }
  if (interval.IsPrecise()) {
    return ternary_ops::IsCompatible(ternary, interval.LowerBound());
  }

  Bits lcp = bits_ops::LongestCommonPrefixMSB(
      {interval.LowerBound(), interval.UpperBound()});

  // We know the next bit of the bounds of `interval` differs, and the interval
  // is proper iff the upper bound has a 1 there.
  const bool proper = interval.UpperBound().GetFromMsb(lcp.bit_count());

  TernarySpan prefix = ternary.subspan(ternary.size() - lcp.bit_count());

  // If the interval is proper, then the interval only contains things with
  // this least-common prefix.
  if (proper && !ternary_ops::IsCompatible(prefix, lcp)) {
    return false;
  }

  // If the interval is improper, then it contains everything that doesn't share
  // this prefix. Therefore, unless `prefix` is fully-known and matches the
  // least-common prefix, `ternary` can definitely represent something in the
  // interval.
  if (!proper && !(ternary_ops::IsFullyKnown(prefix) &&
                   ternary_ops::ToKnownBitsValues(prefix) == lcp)) {
    return true;
  }

  // Take the leading value in `ternary`.
  TernaryValue x = ternary[ternary.size() - lcp.bit_count() - 1];

  // Drop all the bits we've already confirmed match, plus one more.
  Bits L = interval.LowerBound().Slice(0, ternary.size() - lcp.bit_count() - 1);
  Bits U = interval.UpperBound().Slice(0, ternary.size() - lcp.bit_count() - 1);
  TernarySpan t = ternary.subspan(0, ternary.size() - lcp.bit_count() - 1);

  auto could_be_le = [](TernarySpan t, const Bits& L) {
    for (int64_t i = t.size() - 1; i >= 0; --i) {
      if (L.Get(i)) {
        if (t[i] != TernaryValue::kKnownOne) {
          // If this bit is zero, it will make t < L.
          return true;
        }
      } else if (t[i] == TernaryValue::kKnownOne) {
        // We know t > L.
        return false;
      }
    }
    return true;
  };
  auto could_be_ge = [](TernarySpan t, const Bits& U) {
    for (int64_t i = t.size() - 1; i >= 0; --i) {
      if (U.Get(i)) {
        if (t[i] == TernaryValue::kKnownZero) {
          // We know t < U.
          return false;
        }
      } else if (t[i] != TernaryValue::kKnownZero) {
        // If this bit is one, it will make t > L.
        return true;
      }
    }
    return true;
  };

  // NOTE: At this point, we want to know:
  //
  //   if improper, whether it's possible to have:
  //     xt <= 0U || 1L <= xt, which is true iff
  //     (x == 0 && t <= U) || (x == 1 && L <= t).
  //
  //   if proper, whether it's possible to have:
  //     0L <= xt && xt <= 1U, which is true iff
  //     (x == 1 || L <= t) && (x == 0 || t <= U).
  //
  // If x is known, then this is easy:
  //   if x == 0 && proper: check if it's possible to have L <= t.
  //   if x == 1 && improper: check if it's possible to have L <= t.
  //   if x == 0 && improper: check if it's possible to have t <= U.
  //   if x == 1 && proper: check if it's possible to have t <= U.
  // In other words:
  //   if (x == 0) == proper, check if it's possible to have L <= t.
  //               Otherwise, check if it's possible to have t <= U.
  if (ternary_ops::IsKnown(x)) {
    if ((x == TernaryValue::kKnownZero) == proper) {
      return could_be_ge(t, L);
    }
    return could_be_le(t, U);
  }

  // If x is unknown, then we can choose whichever value we want. Therefore, we
  // just need to know:
  //   if improper, whether it's possible to have... well.
  //     if we take x == 0, then we just need to check if we can have t <= U.
  //     if we take x == 1, then we just need to check if we can have L <= t.
  //     Therefore, we just need to check whether it's possible to have:
  //       t <= U || L <= t.
  //   if proper, whether it's possible to have... well.
  //     If we take x == 1, then we just need to check if we can have t <= U.
  //     If we take x == 0, then we just need to check if we can have L <= t.
  //     Therefore, we just need to check whether it's possible to have:
  //       t <= U || L <= t.
  // The conclusion is the same whether the interval is proper or improper, so
  // we check this and we're done.
  return could_be_le(t, U) || could_be_ge(t, L);
}

bool CoversTernary(const IntervalSet& intervals, TernarySpan ternary) {
  if (intervals.BitCount() != ternary.size()) {
    return false;
  }
  return absl::c_any_of(intervals.Intervals(),
                        [&ternary](const Interval& interval) {
                          return CoversTernary(interval, ternary);
                        });
}

namespace {

enum class Tonicity : bool { Monotone, Antitone };

Tonicity Opposite(Tonicity tonicity) {
  return tonicity == Tonicity::Monotone ? Tonicity::Antitone
                                        : Tonicity::Monotone;
}

// What sort of behavior the argument exhibits
struct ArgumentBehavior {
  // Whether increasing the value of this argument causes the output value to
  // increase (monotone) or decrease (antitone). NB This ignores overflows.
  Tonicity tonicity;

  // Whether the argument is size-preserving; i.e., does changing the argument
  // by 1 cause a change in the output by 1 (either up or down depending on
  // tonicity).
  bool size_preserving;

  // Whether the operation has non-linear, discontinuous or otherwise unusual
  // behavior near the 'INT_MAX/INT_MIN' split point. For example the
  // sign-extend operation adjacent values in the pre-image remain adjacent
  // after the sign extend except for the (INT_MAX,INT_MIN) pair where the
  // results end up far away from each other.
  bool sign_sensitive;

  // The set of other arguments whose sign can flip this argument's tonicity.
  // For example, arithmetic shift-right is antitone in its second operand if
  // the first operand is positive, but monotone if the first operand is
  // negative.
  std::vector<int64_t> sign_sensitive_tonicity = {};

  constexpr ArgumentBehavior SignSensitive() const {
    return {
        .tonicity = tonicity,
        .size_preserving = sign_sensitive,
        .sign_sensitive = true,
        .sign_sensitive_tonicity = sign_sensitive_tonicity,
    };
  }

  constexpr ArgumentBehavior AddSignSensitiveTonicity(int64_t i) const {
    ArgumentBehavior res = *this;
    res.sign_sensitive_tonicity.push_back(i);
    return res;
  }
};

static constexpr ArgumentBehavior kMonotoneSizePreserving{
    .tonicity = Tonicity::Monotone,
    .size_preserving = true,
    .sign_sensitive = false};
static constexpr ArgumentBehavior kMonotoneNonSizePreserving{
    .tonicity = Tonicity::Monotone,
    .size_preserving = false,
    .sign_sensitive = false};
static constexpr ArgumentBehavior kAntitoneSizePreserving{
    .tonicity = Tonicity::Antitone,
    .size_preserving = true,
    .sign_sensitive = false};
static constexpr ArgumentBehavior kAntitoneNonSizePreserving{
    .tonicity = Tonicity::Antitone,
    .size_preserving = false,
    .sign_sensitive = false};

TernaryValue OneBitRangeToTernary(const IntervalSet& is) {
  CHECK_EQ(is.BitCount(), 1);
  if (is.IsPrecise()) {
    return is.CoversZero() ? TernaryValue::kKnownZero : TernaryValue::kKnownOne;
  }
  return TernaryValue::kUnknown;
}

IntervalSet TernaryToOneBitRange(TernaryValue v) {
  switch (v) {
    case TernaryValue::kKnownZero:
      return IntervalSet::Precise(UBits(0, 1));
    case TernaryValue::kKnownOne:
      return IntervalSet::Precise(UBits(1, 1));
    case TernaryValue::kUnknown:
      return IntervalSet::Maximal(1);
  }
}

struct OverflowResult {
  Bits result;
  // Set if overflowed to 'inputs + 1' bits
  bool first_overflow_bit = false;
  // Set if overflowed to 'inputs + 2' bits
  bool second_overflow_bit = false;
};

template <typename Calculate>
  requires(
      std::is_invocable_r_v<OverflowResult, Calculate, absl::Span<Bits const>>)
std::optional<IntervalSet> MaybePerformExactCalculation(
    Calculate calc, absl::Span<ArgumentBehavior const> behaviors,
    absl::Span<IntervalSet const> input_operands, int64_t result_bit_size) {
  // If all arguments are size-preserving we don't need to do anything at all
  // since the preserving property ensures that precision is maintained.
  if (absl::c_all_of(behaviors, [](const ArgumentBehavior& b) {
        return b.size_preserving;
      })) {
    return std::nullopt;
  }
  // How many exact calculations we are willing to perform.
  static constexpr int64_t kMaxExactCalculations = 16;
  int64_t required_calculations = 1;
  for (const IntervalSet& is : input_operands) {
    // required_calculations *= is.Size();
    auto size = is.Size();
    if (!size || *size > kMaxExactCalculations || *size <= 0 ||
        required_calculations * (*size) > kMaxExactCalculations) {
      return std::nullopt;
    }
    required_calculations *= *size;
  }

  VLOG(3) << "Doing " << required_calculations << " exact calculations";

  // We have a small number of actual values to try, just try all of them for
  // best precision.
  using ValueIter = xabsl::iterator_range<IntervalSet::ValuesIterator>;
  std::vector<ValueIter> intervals;
  intervals.reserve(input_operands.size());
  for (const IntervalSet& i : input_operands) {
    intervals.push_back(i.Values());
  }
  IntervalSet results(result_bit_size);
  auto handle_combo =
      [&](absl::Span<const IntervalSet::ValuesIterator> values_ptrs) -> bool {
    std::vector<Bits> values;
    values.reserve(values_ptrs.size());
    for (const auto& value_ptr : values_ptrs) {
      values.push_back(*value_ptr);
    }
    results.AddInterval(Interval::Precise(calc(values).result));
    return false;
  };
  IteratorProduct<ValueIter>(intervals, handle_combo);
  results.Normalize();
  return std::move(results);
}

template <typename Calculate>
  requires(
      std::is_invocable_r_v<OverflowResult, Calculate, absl::Span<Bits const>>)
IntervalSet PerformVariadicOp(Calculate calc,
                              absl::Span<ArgumentBehavior const> behaviors,
                              absl::Span<IntervalSet const> input_operands,
                              int64_t result_bit_size) {
  CHECK_EQ(input_operands.size(), behaviors.size());

  std::optional<IntervalSet> exact_result = MaybePerformExactCalculation(
      calc, behaviors, input_operands, result_bit_size);
  if (exact_result) {
    // VLOG(2) << "Got exact results: " << exact_result->ToString();
    return *exact_result;
  }

  std::vector<IntervalSet> operands;
  operands.reserve(input_operands.size());

  {
    int64_t i = 0;
    for (IntervalSet interval_set : input_operands) {
      // TODO(taktoa): we could choose the minimized interval sets more
      // carefully, since `MinimizeIntervals` is minimizing optimally for each
      // interval set without the knowledge that other interval sets exist.
      // For example, we could call `ConvexHull` greedily on the sets
      // that have the smallest difference between convex hull size and size.

      // TODO(allight): We might want to distribute the intervals more evenly
      // then just giving the first 12 operands 5 segments and the rest 1.
      // Limit exponential growth after 12 parameters. 5^12 = 244 million
      interval_set = MinimizeIntervals(interval_set, (i < 12) ? 5 : 1);
      operands.push_back(interval_set);
      ++i;
    }
  }

  if (absl::c_all_of(operands,
                     [](const IntervalSet& i) { return i.IsPrecise(); })) {
    // All inputs are fully known. The result is the one result of applying the
    // calculation. Overflow doesn't matter since the operation only occurs with
    // one set of values.
    std::vector<Bits> real_values;
    real_values.reserve(operands.size());
    for (const IntervalSet& i : operands) {
      real_values.push_back(*i.GetPreciseValue());
    }
    return IntervalSet::Precise(calc(real_values).result);
  }

  IntervalSet result_intervals(result_bit_size);
  bool overflow_is_size_preserving = true;
  bool any_sign_sensitive = absl::c_any_of(
      behaviors, [](const ArgumentBehavior& b) { return b.sign_sensitive; });
  int64_t count_non_precise = 0;
  for (int64_t i = 0; overflow_is_size_preserving && i < behaviors.size();
       ++i) {
    overflow_is_size_preserving =
        overflow_is_size_preserving &&
        (behaviors[i].size_preserving || operands[i].IsPrecise());
    if (!operands[i].IsPrecise()) {
      ++count_non_precise;
    }
  }
  // If there's only one non-precise argument and overflow caused by it is
  // size-preserving then overflow of the high-side (or low-side on antitone
  // operation) can't "catch-up" to the low-side meaning that '[f(low) %
  // (1<<bit_count), f(high) % (1 << bit_count)]' is always a valid range.
  overflow_is_size_preserving =
      overflow_is_size_preserving && count_non_precise == 1;

  // Each iteration of this do-while loop explores a different choice of
  // intervals from each interval set associated with a parameter.
  auto handle_combo =
      [&](auto /*absl::Span<Iterator of Interval>*/ values_ptrs) -> bool {
    enum class Sign : uint8_t { kPositive, kNegative, kUnknown };
    std::vector<Sign> signs;
    signs.reserve(values_ptrs.size());
    for (int64_t i = 0; i < values_ptrs.size(); ++i) {
      if (behaviors[i].sign_sensitive) {
        const Interval& interval = *values_ptrs[i];
        CHECK_EQ(interval.LowerBound().msb(), interval.UpperBound().msb());
        signs.push_back(interval.LowerBound().msb() ? Sign::kNegative
                                                    : Sign::kPositive);
      } else {
        signs.push_back(Sign::kUnknown);
      }
    }

    std::vector<Bits> lower_bounds;
    lower_bounds.reserve(values_ptrs.size());
    std::vector<Bits> upper_bounds;
    upper_bounds.reserve(values_ptrs.size());
    for (int64_t i = 0; i < values_ptrs.size(); ++i) {
      Interval interval = *values_ptrs[i];
      Tonicity tonicity = behaviors[i].tonicity;
      for (int64_t j : behaviors[i].sign_sensitive_tonicity) {
        CHECK(signs[j] != Sign::kUnknown);
        if (signs[j] == Sign::kNegative) {
          tonicity = Opposite(tonicity);
        }
      }
      switch (tonicity) {
        case Tonicity::Monotone: {
          // The essential property of a unary monotone function `f` is that
          // the codomain of `f` applied to `[x, y]` is `[f(x), f(y)]`.
          // For example, the cubing function applied to `[5, 8]` gives a
          // codomain of `[125, 512]`.
          lower_bounds.push_back(interval.LowerBound());
          upper_bounds.push_back(interval.UpperBound());
          break;
        }
        case Tonicity::Antitone: {
          // The essential property of a unary antitone function `f` is that
          // the codomain of `f` applied to `[x, y]` is `[f(y), f(x)]`.
          // For example, the negation function applied to `[10, 20]` gives
          // a codomain of `[-20, -10]`.
          lower_bounds.push_back(interval.UpperBound());
          upper_bounds.push_back(interval.LowerBound());
          break;
        }
      }
    }
    OverflowResult lower = calc(lower_bounds);
    OverflowResult upper = calc(upper_bounds);
    if (!lower.first_overflow_bit && !upper.first_overflow_bit) {
      // No overflow at all.
      result_intervals.AddInterval(Interval(lower.result, upper.result));
      return false;
    }
    // Size-preserving here means that only a single input is varying. In this
    // case the fact that it's size-preserving means that to overflow twice is
    // impossible since that would mean that there would need to be elements in
    // the output that are not mapped to. Since we are size-preserving the
    // difference between the input and output must remain the same so we can
    // just add the interval.
    if (overflow_is_size_preserving) {
      // Possibly improper interval.
      result_intervals.AddInterval(Interval(lower.result, upper.result));
      return false;
    }
    // Check for overflows that cover the entire output space.
    // If both sides overflowed then its unconstrained.
    if ((lower.first_overflow_bit && upper.first_overflow_bit) ||
        // If either overflowed twice it must be unconstrained.
        lower.second_overflow_bit || upper.second_overflow_bit ||
        // If neither of the other two are true but upper is still larger
        // then lower then one of them must have gone all the way around.
        bits_ops::UGreaterThan(upper.result, lower.result)) {
      // We're unconstrained so no need to continue searching the output
      // space.
      result_intervals.AddInterval(Interval::Maximal(result_bit_size));
      return true;
    }
    // We overflowed on one end of the intervals but not all the way past
    // the other bound so we have an intervals with the inverse of [high,
    // low].
    result_intervals.AddInterval(
        Interval(lower.result, Bits::AllOnes(result_bit_size)));
    result_intervals.AddInterval(Interval(Bits(result_bit_size), upper.result));
    return false;
  };
  if (any_sign_sensitive) {
    using SignedIntervalIter =
        decltype(std::declval<IntervalSet>().SignedIntervals());
    std::vector<SignedIntervalIter> intervals;
    intervals.reserve(operands.size());
    for (const IntervalSet& i : operands) {
      intervals.push_back(i.SignedIntervals());
    }
    IteratorProduct<SignedIntervalIter>(intervals, handle_combo);
  } else {
    using IntervalIter = absl::Span<Interval const>;
    std::vector<IntervalIter> intervals;
    intervals.reserve(operands.size());
    for (const IntervalSet& i : operands) {
      intervals.push_back(i.Intervals());
    }
    IteratorProduct<IntervalIter>(intervals, handle_combo);
  }

  result_intervals.Normalize();
  return MinimizeIntervals(result_intervals, /*size=*/16);
}

template <typename Calculate>
  requires(std::is_invocable_r_v<Bits, Calculate, absl::Span<Bits const>>)
IntervalSet PerformVariadicOp(Calculate calc,
                              absl::Span<ArgumentBehavior const> behaviors,
                              absl::Span<IntervalSet const> input_operands,
                              int64_t result_bit_size) {
  return PerformVariadicOp(
      [&](absl::Span<Bits const> a) -> OverflowResult {
        return {.result = calc(a)};
      },
      behaviors, input_operands, result_bit_size);
}

template <typename Calculate>
  requires(std::is_invocable_r_v<OverflowResult, Calculate, const Bits&,
                                 const Bits&>)
IntervalSet PerformBinOp(Calculate calc, const IntervalSet& lhs,
                         ArgumentBehavior lhs_behavior, const IntervalSet& rhs,
                         ArgumentBehavior rhs_behavior,
                         int64_t result_bit_size) {
  return PerformVariadicOp(
      [&](absl::Span<Bits const> bits) -> OverflowResult {
        CHECK_EQ(bits.size(), 2);
        return calc(bits[0], bits[1]);
      },
      {lhs_behavior, rhs_behavior}, {lhs, rhs}, result_bit_size);
}
template <typename Calculate>
  requires(std::is_invocable_r_v<Bits, Calculate, const Bits&, const Bits&>)
IntervalSet PerformBinOp(Calculate calc, const IntervalSet& lhs,
                         ArgumentBehavior lhs_behavior, const IntervalSet& rhs,
                         ArgumentBehavior rhs_behavior,
                         int64_t result_bit_size) {
  return PerformBinOp(
      [&calc](const Bits& l, const Bits& r) -> OverflowResult {
        return {.result = calc(l, r)};
      },
      lhs, lhs_behavior, rhs, rhs_behavior, result_bit_size);
}

template <typename Calculate>
  requires(std::is_invocable_r_v<OverflowResult, Calculate, const Bits&>)
IntervalSet PerformUnaryOp(Calculate calc, const IntervalSet& arg,
                           ArgumentBehavior behavior, int64_t result_bit_size) {
  return PerformVariadicOp(
      [&](absl::Span<Bits const> bits) -> OverflowResult {
        CHECK_EQ(bits.size(), 1);
        return calc(bits[0]);
      },
      {behavior}, {arg}, result_bit_size);
}

template <typename Calculate>
  requires(std::is_invocable_r_v<Bits, Calculate, const Bits&>)
IntervalSet PerformUnaryOp(Calculate calc, const IntervalSet& arg,
                           ArgumentBehavior behavior, int64_t result_bit_size) {
  return PerformUnaryOp(
      [&calc](const Bits& b) -> OverflowResult {
        return OverflowResult{.result = calc(b)};
      },
      arg, behavior, result_bit_size);
}

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

IntervalSet Add(const IntervalSet& a, const IntervalSet& b) {
  return PerformBinOp(
      [](const Bits& lhs, const Bits& rhs) -> OverflowResult {
        int64_t padded_size = std::max(lhs.bit_count(), rhs.bit_count()) + 1;
        Bits padded_lhs = bits_ops::ZeroExtend(lhs, padded_size);
        Bits padded_rhs = bits_ops::ZeroExtend(rhs, padded_size);
        Bits padded_result = bits_ops::Add(padded_lhs, padded_rhs);
        // If the MSB is 1, then we overflowed.
        bool overflow = padded_result.GetFromMsb(0);
        return OverflowResult{
            .result =
                bits_ops::Truncate(std::move(padded_result), padded_size - 1),
            .first_overflow_bit = overflow,
            .second_overflow_bit = false,
        };
      },
      a, kMonotoneSizePreserving, b, kMonotoneSizePreserving, a.BitCount());
}
IntervalSet Sub(const IntervalSet& a, const IntervalSet& b) {
  return PerformBinOp(
      [](const Bits& lhs, const Bits& rhs) -> OverflowResult {
        // x - y overflows if x < y
        return {.result = bits_ops::Sub(lhs, rhs),
                .first_overflow_bit = bits_ops::ULessThan(lhs, rhs)};
      },
      a, kMonotoneSizePreserving, b, kAntitoneSizePreserving, a.BitCount());
}
IntervalSet Neg(const IntervalSet& a) {
  return PerformUnaryOp(bits_ops::Negate, a, kAntitoneSizePreserving,
                        a.BitCount());
}
IntervalSet UMul(const IntervalSet& a, const IntervalSet& b,
                 int64_t output_bitwidth) {
  return PerformBinOp(
      [output_bitwidth](const Bits& lhs, const Bits& rhs) -> OverflowResult {
        Bits result = bits_ops::UMul(lhs, rhs);
        int64_t msb_set_bit =
            result.bit_count() - result.CountLeadingZeros() - 1;
        return {.result = Bits::FromBitmap(std::move(result).bitmap().WithSize(
                    output_bitwidth, /*new_data=*/false)),
                .first_overflow_bit = msb_set_bit >= output_bitwidth,
                .second_overflow_bit = msb_set_bit >= output_bitwidth + 1};
      },
      a, kMonotoneNonSizePreserving, b, kMonotoneNonSizePreserving,
      output_bitwidth);
}
IntervalSet UDiv(const IntervalSet& a, const IntervalSet& b) {
  // Integer division is antitone on the second argument since
  // `\forall x,y \in \real: y > 1 \implies x / y <= x`. The one unsigned
  // integer value for which this implication does not hold is `0`. Our UDiv
  // implementation is defined such that UDiv(x, 0) == MAX_int so in cases where
  // zero is possible we add that in.
  if (!b.CoversZero()) {
    return PerformBinOp(bits_ops::UDiv, a, kMonotoneNonSizePreserving, b,
                        kAntitoneNonSizePreserving, a.BitCount());
  }
  IntervalSet nonzero_divisor =
      IntervalSet::Intersect(b, IntervalSet::NonZero(b.BitCount()));
  IntervalSet results(a.BitCount());
  if (!nonzero_divisor.IsEmpty()) {
    // We aren't *only* dividing by zero. Get the non-zero divisor ranges.
    results =
        PerformBinOp(bits_ops::UDiv, a, kMonotoneNonSizePreserving,
                     nonzero_divisor, kAntitoneNonSizePreserving, a.BitCount());
  }
  // Stick in the single value that division by zero yields.
  results.AddInterval(Interval::Precise(Bits::AllOnes(a.BitCount())));
  results.Normalize();
  return results;
}
IntervalSet UMod(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  // Integer modulus is a complex case. The result is restricted to be less than
  // `b` (and no larger than `a`), but we also can get more information out of
  // intervals in `a` if (e.g.) `b` is precise and `a` contains intervals with
  // fewer than `b` values in them.
  if (b.UpperBound()->IsZero()) {
    return IntervalSet::Of({Interval::Precise(Bits(a.BitCount()))});
  }
  // TODO(epastor): Extract more information when we can.
  Bits upper_bound =
      bits_ops::UMin(*a.UpperBound(), bits_ops::Decrement(*b.UpperBound()));
  return IntervalSet::Of({Interval::Closed(Bits(a.BitCount()), upper_bound)});
}
IntervalSet SMul(const IntervalSet& a, const IntervalSet& b,
                 int64_t output_bitwidth) {
  // Tonality depends on the sign of both sides so its easiest to simply
  // calculate the 4 segments and combine them.
  IntervalSet pos_a = a.PositiveIntervals(/*with_zero=*/false);
  IntervalSet pos_b = b.PositiveIntervals(/*with_zero=*/false);
  IntervalSet neg_a = a.NegativeAbsoluteIntervals();
  IntervalSet neg_b = b.NegativeAbsoluteIntervals();
  // Positive intervals
  IntervalSet pos_pos = UMul(pos_a, pos_b, output_bitwidth);
  IntervalSet neg_neg = UMul(neg_a, neg_b, output_bitwidth);
  IntervalSet positives = IntervalSet::Combine(pos_pos, neg_neg);
  if (a.CoversZero() || b.CoversZero()) {
    positives = IntervalSet::Combine(
        positives,
        IntervalSet::Of({Interval::Precise(UBits(0, output_bitwidth))}));
  }
  // Negative intervals
  IntervalSet pos_neg = UMul(pos_a, neg_b, output_bitwidth);
  IntervalSet neg_pos = UMul(neg_a, pos_b, output_bitwidth);
  IntervalSet negatives = Neg(IntervalSet::Combine(pos_neg, neg_pos));
  return IntervalSet::Combine(positives, negatives);
}

IntervalSet SDiv(const IntervalSet& a, const IntervalSet& b) {
  // Tonality depends on the sign of both sides so its easiest to simply
  // calculate the 4 segments and combine them.
  IntervalSet pos_a = a.PositiveIntervals(/*with_zero=*/false);
  IntervalSet pos_b = b.PositiveIntervals(/*with_zero=*/false);
  IntervalSet neg_a = a.NegativeAbsoluteIntervals();
  IntervalSet neg_b = b.NegativeAbsoluteIntervals();

  IntervalSet pos_pos = UDiv(pos_a, pos_b);
  IntervalSet neg_neg = UDiv(neg_a, neg_b);
  IntervalSet positives = IntervalSet::Combine(pos_pos, neg_neg);
  IntervalSet pos_neg = UDiv(pos_a, neg_b);
  IntervalSet neg_pos = UDiv(neg_a, pos_b);
  IntervalSet negatives = Neg(IntervalSet::Combine(pos_neg, neg_pos));

  IntervalSet res = IntervalSet::Combine(positives, negatives);

  // Add in the values if there are zeros
  if (a.CoversZero()) {
    res.AddInterval(Interval::Precise(UBits(0, a.BitCount())));
  }
  if (b.CoversZero()) {
    if (!a.LowerBound() || !a.LowerBound()->msb()) {
      res.AddInterval(Interval::Precise(Bits::MaxSigned(a.BitCount())));
    }
    if (!a.UpperBound() || a.UpperBound()->msb()) {
      res.AddInterval(Interval::Precise(Bits::MinSigned(a.BitCount())));
    }
  }
  res.Normalize();
  return res;
}
IntervalSet SMod(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  // Integer modulus is a complex case. The result is restricted to be less than
  // `b` and at most `a` (in absolute value), but we also can get more
  // information out of intervals in `a` if (e.g.) `b` is precise and `a`
  // contains intervals with fewer than `b` values in them.

  // TODO(epastor): Extract more information when we can.
  Bits absolute_b_bound(b.BitCount());
  for (const Interval& i : b.SignedIntervals()) {
    absolute_b_bound =
        bits_ops::UMax(absolute_b_bound, bits_ops::Abs(i.LowerBound()));
    absolute_b_bound =
        bits_ops::UMax(absolute_b_bound, bits_ops::Abs(i.UpperBound()));
  }
  if (absolute_b_bound.IsZero()) {
    return IntervalSet::Of({Interval::Precise(Bits(a.BitCount()))});
  }

  Bits absolute_a_bound(a.BitCount());
  for (const Interval& i : a.SignedIntervals()) {
    absolute_a_bound =
        bits_ops::UMax(absolute_a_bound, bits_ops::Abs(i.LowerBound()));
    absolute_a_bound =
        bits_ops::UMax(absolute_a_bound, bits_ops::Abs(i.UpperBound()));
  }

  Bits absolute_bound =
      bits_ops::UMin(absolute_a_bound, bits_ops::Decrement(absolute_b_bound));
  return IntervalSet::Of(
      {Interval::Closed(bits_ops::Negate(absolute_bound), absolute_bound)});
}

IntervalSet Shll(const IntervalSet& a, const IntervalSet& b) {
  return PerformBinOp(
      [](const Bits& lhs, const Bits& rhs) -> OverflowResult {
        if (rhs.IsZero()) {
          return {.result = lhs};
        }
        int64_t shift_amount = bits_ops::UnsignedBitsToSaturatedInt64(rhs);
        if (shift_amount >= lhs.bit_count()) {
          // We must be overshifting; the result is zero.
          return {.result = Bits(lhs.bit_count()),
                  .first_overflow_bit = !lhs.IsZero(),
                  .second_overflow_bit =
                      !lhs.IsZero() &&
                      (shift_amount > lhs.bit_count() || !lhs.IsOne())};
        }
        Bits shifted_out =
            lhs.Slice(lhs.bit_count() - shift_amount, shift_amount);
        return {.result = bits_ops::ShiftLeftLogical(lhs, shift_amount),
                .first_overflow_bit = !shifted_out.IsZero(),
                .second_overflow_bit =
                    !shifted_out.IsZero() && !shifted_out.IsOne()};
      },
      a, kMonotoneNonSizePreserving, b, kMonotoneNonSizePreserving,
      a.BitCount());
}
IntervalSet Shrl(const IntervalSet& a, const IntervalSet& b) {
  return PerformBinOp(
      [](const Bits& lhs, const Bits& rhs) -> OverflowResult {
        absl::StatusOr<uint64_t> shift_amount = rhs.ToUint64();
        if (!shift_amount.ok() ||
            *shift_amount >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          // We must be overshifting; the result is zero.
          return {.result = Bits(lhs.bit_count())};
        }
        return {.result = bits_ops::ShiftRightLogical(lhs, *shift_amount)};
      },
      a, kMonotoneNonSizePreserving, b, kAntitoneNonSizePreserving,
      a.BitCount());
}
IntervalSet Shra(const IntervalSet& a, const IntervalSet& b) {
  return PerformBinOp(
      [](const Bits& lhs, const Bits& rhs) -> OverflowResult {
        absl::StatusOr<uint64_t> shift_amount = rhs.ToUint64();
        if (!shift_amount.ok() ||
            *shift_amount >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          // We must be overshifting; the result is zero (if the sign bit was
          // zero) or all ones (if the sign bit was one).
          return {.result = lhs.msb() ? Bits::AllOnes(lhs.bit_count())
                                      : Bits(lhs.bit_count())};
        }
        return {.result = bits_ops::ShiftRightArith(lhs, *shift_amount)};
      },
      a, kMonotoneNonSizePreserving.SignSensitive(), b,
      kAntitoneNonSizePreserving.AddSignSensitiveTonicity(0), a.BitCount());
}
IntervalSet Decode(const IntervalSet& a, int64_t width) {
  IntervalSet result(width);
  // We step through the possible values `i` in `a`, up to `width`, and generate
  // the corresponding power-of-two as an interval. The good news is that we can
  // (and do) stop as soon as we reach `width` or any larger value, since all of
  // those encode the same point (0)... so this visits at most `width` elements.
  for (const Bits& b : a.Values()) {
    uint64_t bit = b.ToUint64().value_or(width);
    if (bit >= static_cast<uint64_t>(width)) {
      // We're done; no need to check later elements.
      result.AddInterval(Interval::Precise(UBits(0, width)));
      break;
    }

    result.AddInterval(
        Interval::Precise(Bits::PowerOfTwo(static_cast<int64_t>(bit), width)));
  }
  result.Normalize();
  return result;
}
IntervalSet SignExtend(const IntervalSet& a, int64_t width) {
  return PerformUnaryOp(
      [&](const Bits& b) -> Bits { return bits_ops::SignExtend(b, width); }, a,
      // NB Monotone in unsigned domain as all values either stay the same
      // (positive) or increase (2s complement negatives).
      kMonotoneNonSizePreserving.SignSensitive(), width);
}
IntervalSet ZeroExtend(const IntervalSet& a, int64_t width) {
  return PerformUnaryOp(
      [&](const Bits& b) -> Bits { return bits_ops::ZeroExtend(b, width); }, a,
      kMonotoneSizePreserving, width);
}
IntervalSet Truncate(const IntervalSet& a, int64_t width) {
  IntervalSet result(width);
  Bits output_space = Bits::AllOnes(width);
  for (const Interval& i : a.Intervals()) {
    if (bits_ops::UGreaterThan(bits_ops::Sub(i.UpperBound(), i.LowerBound()),
                               output_space)) {
      // Interval covers everything.
      return IntervalSet::Maximal(width);
    }
    Bits low = i.LowerBound().Slice(0, width);
    Bits high = i.UpperBound().Slice(0, width);
    // NB Improper intervals are automatically split once we do normalization.
    result.AddInterval(Interval(low, high));
  }
  result.Normalize();
  return result;
}
IntervalSet BitSlice(const IntervalSet& a, int64_t start, int64_t width) {
  return Truncate(Shrl(a, IntervalSet::Precise(UBits(start, 64))), width);
}

IntervalSet Concat(absl::Span<IntervalSet const> sets) {
  std::vector<ArgumentBehavior> behaviors(sets.size(),
                                          kMonotoneNonSizePreserving);
  behaviors.back() = kMonotoneSizePreserving;
  return PerformVariadicOp(
      bits_ops::Concat, behaviors, sets,
      absl::c_accumulate(
          sets, int64_t{0},
          [](int64_t v, const IntervalSet& is) { return v + is.BitCount(); }));
}

// Bit ops.
IntervalSet Not(const IntervalSet& a) {
  if (a.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  TernaryEvaluator eval;
  // Special case 1-bit version to avoid allocations.
  if (a.BitCount() == 1) {
    return TernaryToOneBitRange(eval.Not(OneBitRangeToTernary(a)));
  }
  TernaryVector vec = ExtractTernaryVector(a);
  TernaryVector res = eval.BitwiseNot(vec);
  return FromTernary(res);
}
IntervalSet And(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  // Special case 1-bit version to avoid allocations.
  TernaryEvaluator eval;
  if (a.BitCount() == 1) {
    return TernaryToOneBitRange(
        eval.And(OneBitRangeToTernary(a), OneBitRangeToTernary(b)));
  }
  // Special case AND with all ones.
  IntervalSet only_ones = IntervalSet::Precise(SBits(-1, a.BitCount()));
  if (a == only_ones) {
    return b;
  }
  if (b == only_ones) {
    return a;
  }
  // Special case a mask-select i.e. 'AND with all zeros or all ones' case since
  // this can come up a lot due to select-simp turning small selects into masks.
  // NB Size is how many values evaluate as being inside the interval set. If
  // there are two values total and both 0b1111...111 and 0b0 are in it those
  // are the only two values.
  IntervalSet ones_or_zero =
      IntervalSet::Of({Interval::Precise(UBits(0, a.BitCount())),
                       Interval::Precise(SBits(-1, a.BitCount()))});
  if (a == ones_or_zero) {
    return IntervalSet::Combine(b,
                                IntervalSet::Precise(UBits(0, b.BitCount())));
  }
  if (b == ones_or_zero) {
    return IntervalSet::Combine(a,
                                IntervalSet::Precise(UBits(0, a.BitCount())));
  }
  TernaryVector res =
      eval.BitwiseAnd(ExtractTernaryVector(a), ExtractTernaryVector(b));
  return FromTernary(res);
}
IntervalSet Or(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  TernaryEvaluator eval;
  if (a.BitCount() == 1) {
    return TernaryToOneBitRange(
        eval.Or(OneBitRangeToTernary(a), OneBitRangeToTernary(b)));
  }
  IntervalSet only_zero = IntervalSet::Precise(UBits(0, a.BitCount()));
  if (a == only_zero) {
    return b;
  }
  if (b == only_zero) {
    return a;
  }
  // Special case a mask-select i.e. 'OR with all zeros or all ones' which is
  // either the other value or all ones.
  // NB Size is how many values evaluate as being inside the interval set. If
  // there are two values total and both 0b1111...111 and 0b0 are in it those
  // are the only two values.
  IntervalSet ones_or_zero =
      IntervalSet::Of({Interval::Precise(UBits(0, a.BitCount())),
                       Interval::Precise(SBits(-1, a.BitCount()))});
  if (a == ones_or_zero) {
    return IntervalSet::Combine(b,
                                IntervalSet::Precise(SBits(-1, a.BitCount())));
  }
  if (b == ones_or_zero) {
    return IntervalSet::Combine(a,
                                IntervalSet::Precise(SBits(-1, b.BitCount())));
  }
  TernaryVector res =
      eval.BitwiseOr(ExtractTernaryVector(a), ExtractTernaryVector(b));
  return FromTernary(res);
}

IntervalSet Xor(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  TernaryEvaluator eval;
  if (a.BitCount() == 1) {
    return TernaryToOneBitRange(
        eval.Xor(OneBitRangeToTernary(a), OneBitRangeToTernary(b)));
  }
  IntervalSet only_zero = IntervalSet::Precise(UBits(0, a.BitCount()));
  if (a == only_zero) {
    return b;
  }
  if (b == only_zero) {
    return a;
  }
  TernaryVector res =
      eval.BitwiseXor(ExtractTernaryVector(a), ExtractTernaryVector(b));
  return FromTernary(res);
}

IntervalSet AndReduce(const IntervalSet& a) {
  if (a.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  // Unless the intervals cover max, the and_reduce of the input must be 0.
  if (!a.CoversMax()) {
    return TernaryToOneBitRange(TernaryValue::kKnownZero);
  }
  // If the intervals is precise and covers max it must be 1.
  if (a.IsPrecise()) {
    return TernaryToOneBitRange(TernaryValue::kKnownOne);
  }
  // Not knowable
  return TernaryToOneBitRange(TernaryValue::kUnknown);
}

IntervalSet OrReduce(const IntervalSet& a) {
  if (a.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  // Unless the intervals cover 0, the or_reduce of the input must be 1.
  if (!a.CoversZero()) {
    return TernaryToOneBitRange(TernaryValue::kKnownOne);
  }
  // If the intervals are known to only cover 0, then the result must be 0.
  if (a.IsPrecise()) {
    return TernaryToOneBitRange(TernaryValue::kKnownZero);
  }
  // Not knowable
  return TernaryToOneBitRange(TernaryValue::kUnknown);
}
IntervalSet XorReduce(const IntervalSet& a) {
  if (a.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  // XorReduce determines the parity of the number of 1s in a bitstring.
  // Incrementing a bitstring always outputs in a bitstring with a different
  // parity of 1s (since even + 1 = odd and odd + 1 = even). Therefore, this
  // analysis cannot return anything but unknown when an interval is
  // imprecise. When the given set of intervals only contains precise
  // intervals, we can check whether they all have the same parity of 1s, and
  // return 1 or 0 if they are all the same, or unknown otherwise.
  if (!a.Intervals().front().IsPrecise()) {
    return TernaryToOneBitRange(TernaryValue::kUnknown);
  }
  Bits output = bits_ops::XorReduce(*a.Intervals().front().GetPreciseValue());
  for (const Interval& interval :
       absl::MakeConstSpan(a.Intervals()).subspan(1)) {
    if (!interval.IsPrecise() ||
        bits_ops::XorReduce(*interval.GetPreciseValue()) != output) {
      return TernaryToOneBitRange(TernaryValue::kUnknown);
    }
  }
  return TernaryToOneBitRange(output.IsOne() ? TernaryValue::kKnownOne
                                             : TernaryValue::kKnownZero);
}

// Cmp
IntervalSet Eq(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }

  if (a.IsPrecise() && b.IsPrecise()) {
    return TernaryToOneBitRange(
        bits_ops::UEqual(*a.GetPreciseValue(), *b.GetPreciseValue())
            ? TernaryValue::kKnownOne
            : TernaryValue::kKnownZero);
  }

  return TernaryToOneBitRange(IntervalSet::Disjoint(a, b)
                                  ? TernaryValue::kKnownZero
                                  : TernaryValue::kUnknown);
}

IntervalSet Ne(const IntervalSet& a, const IntervalSet& b) {
  return Not(Eq(a, b));
}

IntervalSet ULt(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  if (a.IsPrecise() && a.GetPreciseValue() == Bits::AllOnes(a.BitCount())) {
    // If a is all ones, then it is not less than any value.
    return TernaryToOneBitRange(TernaryValue::kKnownZero);
  }
  if (b.IsPrecise() && b.GetPreciseValue() == Bits(b.BitCount())) {
    // If b is zero, then it is not greater than any value.
    return TernaryToOneBitRange(TernaryValue::kKnownZero);
  }
  Interval lhs_hull = *a.ConvexHull();
  Interval rhs_hull = *b.ConvexHull();
  if (bits_ops::ULessThan(lhs_hull.UpperBound(), rhs_hull.LowerBound())) {
    // The LHS is entirely below the RHS, so all possibilities are less than.
    return TernaryToOneBitRange(TernaryValue::kKnownOne);
  }
  if (bits_ops::ULessThanOrEqual(rhs_hull.UpperBound(),
                                 lhs_hull.LowerBound())) {
    // The RHS is entirely below the LHS (except for at most a single point of
    // overlap), so the LHS can't be less than.
    return TernaryToOneBitRange(TernaryValue::kKnownZero);
  }
  return TernaryToOneBitRange(TernaryValue::kUnknown);
}

IntervalSet SLt(const IntervalSet& a, const IntervalSet& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  if (a.IsEmpty() || b.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(a.BitCount());
  }
  CHECK(a.IsNormalized());
  CHECK(b.IsNormalized());
  auto is_all_negative = [](const IntervalSet& v) {
    return v.LowerBound()->GetFromMsb(0) && v.UpperBound()->GetFromMsb(0);
  };
  auto is_all_positive = [](const IntervalSet& v) {
    return !v.LowerBound()->GetFromMsb(0) && !v.UpperBound()->GetFromMsb(0);
  };
  // Avoid doing the add if possible.
  if ((is_all_positive(a) && is_all_positive(b)) ||
      (is_all_negative(a) && is_all_negative(b))) {
    // Entire range is positive or negative on both args. Can do an unsigned
    // compare.
    return ULt(a, b);
  }
  IntervalSet offset =
      IntervalSet::Precise(Bits::PowerOfTwo(a.BitCount() - 1, a.BitCount()));
  return ULt(Add(a, offset), Add(b, offset));
}

IntervalSet Gate(const IntervalSet& cond, const IntervalSet& val) {
  if (cond.IsEmpty() || val.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(val.BitCount());
  }
  if (cond.IsPrecise()) {
    if (cond.CoversZero()) {
      return IntervalSet::Precise(Bits(val.BitCount()));
    }
    return val;
  }
  if (cond.CoversZero()) {
    // has zero and some other value so mix.
    return IntervalSet::Combine(val,
                                IntervalSet::Precise(Bits(val.BitCount())));
  }
  // Definitely not zero.
  return val;
}
IntervalSet OneHot(const IntervalSet& val, LsbOrMsb lsb_or_msb,
                   int64_t max_interval_bits) {
  if (val.IsEmpty()) {
    // If the input is empty, so is the output.
    return IntervalSet(val.BitCount() + 1);
  }
  TernaryEvaluator tern;
  TernaryVector src = ExtractTernaryVector(val);
  TernaryVector res;
  switch (lsb_or_msb) {
    case LsbOrMsb::kLsb:
      res = tern.OneHotLsbToMsb(src);
      break;
    case LsbOrMsb::kMsb:
      res = tern.OneHotMsbToLsb(src);
      break;
  }
  return FromTernary(res, max_interval_bits);
}

int64_t MinimumBitCount(const IntervalSet& a) {
  if (a.IsEmpty()) {
    return 0;
  }
  return a.BitCount() - a.UpperBound()->CountLeadingZeros();
}

int64_t MinimumSignedBitCount(const IntervalSet& a) {
  // Handle the extreme cases manually. No values or only the zero value are a
  // 0-bit integer.
  if (a.IsEmpty() || (a.GetPreciseValue() && a.GetPreciseValue()->IsZero())) {
    return 0;
  }
  if (a.Covers(Bits::MinSigned(a.BitCount())) ||
      a.Covers(Bits::MaxSigned(a.BitCount()))) {
    return a.BitCount();
  }
  IntervalSet positive = IntervalSet::Intersect(
      a, IntervalSet::Of({Interval::Closed(SBits(0, a.BitCount()),
                                           Bits::MaxSigned(a.BitCount()))}));
  // NB Intervals are unsigned and sbits are represented 2s complement so -1 is
  // the highest value possible in this bit-width. MinSigned is the next value
  // after MaxSigned in 2s complement representation.
  IntervalSet negative = IntervalSet::Intersect(
      a, IntervalSet::Of({Interval::Closed(Bits::MinSigned(a.BitCount()),
                                           SBits(-1, a.BitCount()))}));
  return std::max(positive.IsEmpty() ? 0 : MinimumBitCount(positive),
                  negative.IsEmpty()
                      ? 0
                      : (negative.BitCount() -
                         negative.LowerBound()->CountLeadingOnes())) +
         1;
}

}  // namespace xls::interval_ops

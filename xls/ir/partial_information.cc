// Copyright 2025 The XLS Authors
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

#include "xls/ir/partial_information.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ternary.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {

bool PartialInformation::IsCompatibleWith(Bits value) const {
  if (IsImpossible()) {
    return false;
  }
  if (ternary_.has_value() && !ternary_ops::IsCompatible(*ternary_, value)) {
    return false;
  }
  if (range_.has_value() && !range_->Covers(value)) {
    return false;
  }
  return true;
}

bool PartialInformation::IsCompatibleWith(PartialInformation other) const {
  if (IsImpossible() || other.IsImpossible()) {
    return false;
  }
  return !other.JoinWith(*this).IsImpossible();
}

bool PartialInformation::IsCompatibleWith(const IntervalSet& other) const {
  if (IsImpossible() || other.IsEmpty()) {
    return false;
  }
  if (IsUnconstrained()) {
    return true;
  }
  if (!ternary_.has_value()) {
    return !IntervalSet::Disjoint(*range_, other);
  }
  if (!range_.has_value()) {
    return interval_ops::CoversTernary(other, *ternary_);
  }
  return interval_ops::CoversTernary(IntervalSet::Intersect(other, *range_),
                                     *ternary_);
}

bool PartialInformation::IsCompatibleWith(const Interval& other) const {
  return IsCompatibleWith(IntervalSet::Of({other}));
}

bool PartialInformation::IsCompatibleWith(TernarySpan other) const {
  if (IsImpossible() || other.empty()) {
    return false;
  }
  if (IsUnconstrained()) {
    return true;
  }
  if (!ternary_.has_value()) {
    return interval_ops::CoversTernary(*range_, other);
  }
  if (!range_.has_value()) {
    return ternary_ops::IsCompatible(*ternary_, other);
  }
  TernaryVector ternary_join = *ternary_;
  if (!ternary_ops::TryUpdateWithUnion(ternary_join, other)) {
    // The ternaries are in conflict.
    return false;
  }
  return interval_ops::CoversTernary(*range_, ternary_join);
}

std::optional<Bits> PartialInformation::GetPuncturedValue() const {
  if (!range_.has_value()) {
    return std::nullopt;
  }
  absl::Span<const Interval> intervals = range_->Intervals();
  if (intervals.empty() || intervals.size() > 2) {
    return std::nullopt;
  }
  if (intervals.size() == 1) {
    // If the interval is either (0, 2^n-1] or [0, 2^n-1) then this is a
    // punctured set (punctured at 0 or 2^n-1 respectively).
    if (intervals.front().UpperBound().IsAllOnes() &&
        intervals.front().LowerBound() == UBits(1, bit_count_)) {
      return Bits(bit_count_);
    }
    if (intervals.front().LowerBound().IsZero() &&
        intervals.front().UpperBound() ==
            bits_ops::Decrement(Bits::AllOnes(bit_count_))) {
      return Bits::AllOnes(bit_count_);
    }
    return std::nullopt;
  }
  // We have two intervals; this set is punctured at x iff they are [0, x) and
  // (x, 2^n-1].
  if (!intervals.front().LowerBound().IsZero()) {
    return std::nullopt;
  }
  if (!intervals.back().UpperBound().IsAllOnes()) {
    return std::nullopt;
  }
  Bits x = bits_ops::Increment(intervals.front().UpperBound());
  if (x == bits_ops::Decrement(intervals.back().LowerBound())) {
    return x;
  }
  return std::nullopt;
}

std::string PartialInformation::ToString() const {
  if (IsUnconstrained()) {
    return "unconstrained";
  }
  if (IsImpossible()) {
    return "impossible";
  }
  if (range_.has_value() && range_->IsPrecise()) {
    return absl::StrCat("{", BitsToString(*range_->GetPreciseValue()), "}");
  }
  absl::InlinedVector<std::string, 2> pieces;
  if (ternary_.has_value()) {
    pieces.push_back(xls::ToString(*ternary_));
  }
  if (range_.has_value()) {
    pieces.push_back(range_->ToString());
  }
  return absl::StrJoin(pieces, " ∩ ");
}

PartialInformation& PartialInformation::Not() {
  if (ternary_.has_value()) {
    ternary_ = ternary_ops::Not(*ternary_);
  }
  if (range_.has_value()) {
    range_ = interval_ops::Not(*range_);
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::And(const PartialInformation& other) {
  if (IsImpossible()) {
    return *this;
  }
  if (other.IsImpossible()) {
    this->MarkImpossible();
    return *this;
  }

  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = ternary_ops::And(*ternary_, *other.ternary_);
  } else if (ternary_.has_value()) {
    absl::c_replace(*ternary_, TernaryValue::kKnownOne, TernaryValue::kUnknown);
  } else if (other.ternary_.has_value()) {
    ternary_ = other.ternary_;
    absl::c_replace(*ternary_, TernaryValue::kKnownOne, TernaryValue::kUnknown);
  }

  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::And(*range_, *other.range_);
  } else if (range_.has_value()) {
    range_ = interval_ops::And(*range_, IntervalSet::Maximal(other.bit_count_));
  } else if (other.range_.has_value()) {
    range_ = interval_ops::And(IntervalSet::Maximal(bit_count_), *other.range_);
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Or(const PartialInformation& other) {
  if (IsImpossible()) {
    return *this;
  }
  if (other.IsImpossible()) {
    this->MarkImpossible();
    return *this;
  }

  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = ternary_ops::Or(*ternary_, *other.ternary_);
  } else if (ternary_.has_value()) {
    absl::c_replace(*ternary_, TernaryValue::kKnownZero,
                    TernaryValue::kUnknown);
  } else if (other.ternary_.has_value()) {
    ternary_ = other.ternary_;
    absl::c_replace(*ternary_, TernaryValue::kKnownZero,
                    TernaryValue::kUnknown);
  }

  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Or(*range_, *other.range_);
  } else if (range_.has_value()) {
    range_ = interval_ops::Or(*range_, IntervalSet::Maximal(other.bit_count_));
  } else if (other.range_.has_value()) {
    range_ = interval_ops::Or(IntervalSet::Maximal(bit_count_), *other.range_);
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Xor(const PartialInformation& other) {
  if (IsImpossible()) {
    return *this;
  }
  if (other.IsImpossible()) {
    this->MarkImpossible();
    return *this;
  }

  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = ternary_ops::Xor(*ternary_, *other.ternary_);
  } else {
    ternary_ = std::nullopt;
  }

  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Xor(*range_, *other.range_);
  } else {
    range_ = std::nullopt;
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Reverse() {
  if (IsImpossible()) {
    return *this;
  }

  if (ternary_.has_value()) {
    absl::c_reverse(*ternary_);
  }
  range_ = std::nullopt;
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Gate(
    const PartialInformation& control) {
  if (IsImpossible()) {
    return *this;
  }
  if (control.IsImpossible()) {
    this->MarkImpossible();
    return *this;
  }

  CHECK_EQ(control.BitCount(), 1);
  std::optional<Bits> precise_control = control.GetPreciseValue();
  if (precise_control.has_value() && precise_control->IsAllOnes()) {
    // The control is always enabled, so the result is unchanged.
    return *this;
  }
  if (precise_control.has_value() && precise_control->IsZero()) {
    // The control is always disabled, so the result is identically zero.
    if (ternary_.has_value()) {
      absl::c_fill(*ternary_, TernaryValue::kKnownZero);
    } else {
      ternary_ = TernaryVector(bit_count_, TernaryValue::kKnownZero);
    }
    range_ = IntervalSet::Precise(Bits(bit_count_));
    return *this;
  }

  // If the control is unconstrained, then the result is either unchanged or
  // zero.
  if (ternary_.has_value()) {
    absl::c_replace(*ternary_, TernaryValue::kKnownOne, TernaryValue::kUnknown);
  }
  if (range_.has_value()) {
    range_ =
        IntervalSet::Combine(*range_, IntervalSet::Precise(Bits(bit_count_)));
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Neg() {
  if (IsImpossible()) {
    return *this;
  }

  if (ternary_.has_value()) {
    ternary_ = TernaryEvaluator().Neg(*ternary_);
  }
  if (range_.has_value()) {
    range_ = interval_ops::Neg(*range_);
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Add(const PartialInformation& other) {
  CHECK_EQ(bit_count_, other.bit_count_);
  if (IsImpossible()) {
    return *this;
  }
  if (other.IsImpossible()) {
    this->MarkImpossible();
    return *this;
  }

  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = TernaryEvaluator().Add(*ternary_, *other.ternary_);
  } else {
    ternary_ = std::nullopt;
  }

  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Add(*range_, *other.range_);
  } else {
    range_ = std::nullopt;
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Sub(const PartialInformation& other) {
  CHECK_EQ(bit_count_, other.bit_count_);
  if (IsImpossible()) {
    return *this;
  }
  if (other.IsImpossible()) {
    this->MarkImpossible();
    return *this;
  }

  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = TernaryEvaluator().Sub(*ternary_, *other.ternary_);
  } else {
    ternary_ = std::nullopt;
  }

  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Sub(*range_, *other.range_);
  } else {
    range_ = std::nullopt;
  }

  ReconcileInformation();
  return *this;
}

namespace {

TernaryVector ShiftTernary(TernarySpan original, int64_t shift_amt,
                           TernaryValue fill_value) {
  TernaryVector shifted(original.size(), fill_value);
  CHECK_LT(original.size(),
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  const int64_t width = static_cast<int64_t>(original.size());
  for (int64_t i = std::max(int64_t{0}, -shift_amt);
       i < width - std::max(int64_t{0}, shift_amt); ++i) {
    shifted[i + shift_amt] = original[i];
  }
  return shifted;
}

enum class ShiftDirection { kLeft, kRight };

constexpr int64_t kMaxShiftCount = 64;

TernaryVector ShiftTernary(TernarySpan original, Interval shift_range,
                           ShiftDirection direction, TernaryValue fill_value) {
  TernaryVector result;
  for (const Bits& shift_amt_bits : shift_range) {
    int64_t shift_amt = bits_ops::UnsignedBitsToSaturatedInt64(shift_amt_bits);
    if (direction == ShiftDirection::kRight) {
      shift_amt = -shift_amt;
    }
    if (result.empty()) {
      result = ShiftTernary(original, shift_amt, fill_value);
    } else {
      ternary_ops::UpdateWithIntersection(
          result, ShiftTernary(original, shift_amt, fill_value));
    }
  }
  return result;
}

std::optional<TernaryVector> ShiftTernary(TernaryVector original,
                                          IntervalSet shift_range,
                                          ShiftDirection direction,
                                          TernaryValue fill_value) {
  const int64_t width = original.size();

  bool can_overshift = false;
  if (Bits::MinBitCountUnsigned(width) <= shift_range.BitCount()) {
    can_overshift = !IntervalSet::Disjoint(
        shift_range, IntervalSet::Of({Interval::Closed(
                         UBits(width, shift_range.BitCount()),
                         Bits::AllOnes(shift_range.BitCount()))}));

    shift_range = IntervalSet::Intersect(
        shift_range, IntervalSet::Of({Interval::RightOpen(
                         Bits(shift_range.BitCount()),
                         UBits(width, shift_range.BitCount()))}));
  }

  if (shift_range.Size().value_or(kMaxShiftCount + 1) > kMaxShiftCount) {
    // We at least know that the leading/trailing `shift_range.LowerBound()`
    // bits are equal to the fill value.
    if (fill_value == TernaryValue::kUnknown ||
        shift_range.LowerBound()->IsZero()) {
      return std::nullopt;
    }
    return ShiftTernary(
        TernaryVector(original.size(), TernaryValue::kUnknown),
        bits_ops::UnsignedBitsToSaturatedInt64(*shift_range.LowerBound()),
        fill_value);
  }

  TernaryVector result;
  if (can_overshift) {
    result = TernaryVector(width, fill_value);
  }

  for (const Interval& shift_interval : shift_range.Intervals()) {
    if (result.empty()) {
      result = ShiftTernary(original, shift_interval, direction, fill_value);
    } else {
      ternary_ops::UpdateWithIntersection(
          result,
          ShiftTernary(original, shift_interval, direction, fill_value));
    }
  }

  return result;
}

}  // namespace

PartialInformation& PartialInformation::Shll(const PartialInformation& other) {
  IntervalSet shift_range = other.RangeOrMaximal();

  if (range_.has_value()) {
    range_ = interval_ops::Shll(*range_, shift_range);
  }

  if (ternary_.has_value()) {
    ternary_ = ShiftTernary(*ternary_, shift_range, ShiftDirection::kLeft,
                            /*fill_value=*/TernaryValue::kKnownZero);
  } else if (!shift_range.LowerBound()->IsZero()) {
    ternary_ = ShiftTernary(
        TernaryVector(bit_count_, TernaryValue::kUnknown),
        bits_ops::UnsignedBitsToSaturatedInt64(*shift_range.LowerBound()),
        /*fill_value=*/TernaryValue::kKnownZero);
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Shrl(const PartialInformation& other) {
  IntervalSet shift_range = other.RangeOrMaximal();

  if (range_.has_value()) {
    range_ = interval_ops::Shrl(*range_, shift_range);
  }

  if (ternary_.has_value()) {
    ternary_ = ShiftTernary(*ternary_, shift_range, ShiftDirection::kRight,
                            /*fill_value=*/TernaryValue::kKnownZero);
  } else if (!shift_range.LowerBound()->IsZero()) {
    ternary_ = ShiftTernary(
        TernaryVector(bit_count_, TernaryValue::kUnknown),
        -bits_ops::UnsignedBitsToSaturatedInt64(*shift_range.LowerBound()),
        /*fill_value=*/TernaryValue::kKnownZero);
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Shra(const PartialInformation& other) {
  IntervalSet shift_range = other.RangeOrMaximal();

  if (range_.has_value()) {
    range_ = interval_ops::Shra(*range_, shift_range);
  }

  if (ternary_.has_value()) {
    // The leading bits will be filled with the sign bit, if it happens to be
    // known.
    ternary_ = ShiftTernary(*ternary_, shift_range, ShiftDirection::kRight,
                            /*fill_value=*/ternary_->back());
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::JoinWith(
    const PartialInformation& other) {
  if (ternary_.has_value()) {
    if (other.ternary_.has_value() &&
        !ternary_ops::TryUpdateWithUnion(*ternary_, *other.ternary_)) {
      // This is impossible; put this in the standard "impossible" state.
      MarkImpossible();
      return *this;
    }
  } else {
    ternary_ = other.ternary_;
  }

  if (range_.has_value()) {
    if (other.range_.has_value()) {
      range_ = IntervalSet::Intersect(*range_, *other.range_);

      if (range_->IsEmpty()) {
        // Put this in the standard "impossible" state.
        MarkImpossible();
        return *this;
      }
    }
  } else {
    range_ = other.range_;
  }

  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::MeetWith(
    const PartialInformation& other) {
  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ops::UpdateWithIntersection(*ternary_, *other.ternary_);
  } else {
    ternary_ = std::nullopt;
  }

  if (range_.has_value() && other.range_.has_value()) {
    range_ = IntervalSet::Combine(*range_, *other.range_);
  } else {
    range_ = std::nullopt;
  }

  ReconcileInformation();
  return *this;
}

void PartialInformation::ReconcileInformation() {
  if (range_.has_value() && range_->IsMaximal()) {
    range_ = std::nullopt;
  }
  if (ternary_.has_value() && ternary_ops::AllUnknown(*ternary_)) {
    ternary_ = std::nullopt;
  }

  if (range_.has_value() && range_->IsEmpty()) {
    // Standardize the "impossible" state.
    MarkImpossible();
    return;
  }

  // Check whether `ternary_` and `range_` are in conflict before we go any
  // further; if they are, mark the combination impossible.
  if (ternary_.has_value() && range_.has_value() &&
      !interval_ops::CoversTernary(*range_, *ternary_)) {
    MarkImpossible();
    return;
  }

  if (range_.has_value()) {
    // Transfer as much information as we can into the ternary.
    TernaryVector range_ternary = interval_ops::ExtractTernaryVector(*range_);
    if (ternary_.has_value()) {
      CHECK(ternary_ops::TryUpdateWithUnion(*ternary_, range_ternary));
    } else if (!ternary_ops::AllUnknown(range_ternary)) {
      ternary_ = std::move(range_ternary);
    }
  }

  if (ternary_.has_value()) {
    // Transfer as much information as we can into the range.
    IntervalSet ternary_range = interval_ops::FromTernary(*ternary_);
    if (range_.has_value()) {
      range_ = IntervalSet::Intersect(ternary_range, *range_);
      CHECK(!range_->IsEmpty());
    } else if (!ternary_range.IsMaximal()) {
      range_ = std::move(ternary_range);
    }
  }
}

}  // namespace xls

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

#include <cstdint>
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

namespace xls {

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
  if (IsUnrestricted()) {
    return "unrestricted";
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
  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = ternary_ops::And(*ternary_, *other.ternary_);
  }
  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::And(*range_, *other.range_);
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Or(const PartialInformation& other) {
  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = ternary_ops::Or(*ternary_, *other.ternary_);
  }
  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Or(*range_, *other.range_);
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Xor(const PartialInformation& other) {
  if (ternary_.has_value() && other.ternary_.has_value()) {
    ternary_ = ternary_ops::Xor(*ternary_, *other.ternary_);
  }
  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Xor(*range_, *other.range_);
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Neg() {
  if (range_.has_value()) {
    range_ = interval_ops::Neg(*range_);
  }
  ReconcileInformation();
  return *this;
}
PartialInformation& PartialInformation::Add(const PartialInformation& other) {
  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Add(*range_, *other.range_);
  }
  ReconcileInformation();
  return *this;
}
PartialInformation& PartialInformation::Sub(const PartialInformation& other) {
  if (range_.has_value() && other.range_.has_value()) {
    range_ = interval_ops::Sub(*range_, *other.range_);
  }
  ReconcileInformation();
  return *this;
}

PartialInformation& PartialInformation::Shrl(const PartialInformation& other) {
  const std::optional<IntervalSet>& shift_range = other.Range();
  if (!shift_range.has_value()) {
    return *this;
  };
  if (range_.has_value()) {
    range_ = interval_ops::Shrl(*range_, *shift_range);
  }
  if (ternary_.has_value()) {
    constexpr int64_t kMaxShiftCount = 64;
    if (shift_range->Size().value_or(kMaxShiftCount + 1) > kMaxShiftCount) {
      ternary_ = std::nullopt;
    } else {
      TernaryVector original = std::move(*ternary_);
      ternary_ = std::nullopt;
      shift_range->ForEachElement([&](const Bits& shift_amt_bits) -> bool {
        uint64_t shift_amt = *shift_amt_bits.ToUint64();
        TernaryVector shifted(original.size(), TernaryValue::kKnownZero);
        absl::c_copy(absl::MakeConstSpan(original).subspan(shift_amt),
                     shifted.begin());
        if (ternary_ == std::nullopt) {
          ternary_ = std::move(shifted);
        } else {
          ternary_ops::UpdateWithIntersection(*ternary_, shifted);
        }

        // Exit early if we know that the result is all unknown.
        return ternary_ops::AllUnknown(*ternary_);
      });
    }
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

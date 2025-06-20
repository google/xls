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

#include "xls/ir/partial_ops.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/ternary.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls::partial_ops {

PartialInformation Join(PartialInformation a, const PartialInformation& b) {
  a.JoinWith(b);
  return a;
}

PartialInformation Meet(PartialInformation a, const PartialInformation& b) {
  a.MeetWith(b);
  return a;
}

PartialInformation Not(PartialInformation p) { return p.Not(); }
PartialInformation And(PartialInformation a, const PartialInformation& b) {
  return a.And(b);
}
PartialInformation Or(PartialInformation a, const PartialInformation& b) {
  return a.Or(b);
}
PartialInformation Xor(PartialInformation a, const PartialInformation& b) {
  return a.Xor(b);
}
PartialInformation Nand(PartialInformation a, const PartialInformation& b) {
  return a.Nand(b);
}
PartialInformation Nor(PartialInformation a, const PartialInformation& b) {
  return a.Nor(b);
}

PartialInformation AndReduce(const PartialInformation& p) {
  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    TernaryValue result = TernaryValue::kKnownOne;
    for (TernaryValue value : *p.Ternary()) {
      result = ternary_ops::And(result, value);
    }
    if (result != TernaryValue::kUnknown) {
      ternary = TernaryVector({result});
    }
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::AndReduce(*p.Range());
  }

  return PartialInformation(1, std::move(ternary), std::move(range));
}
PartialInformation OrReduce(const PartialInformation& p) {
  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    TernaryValue result = TernaryValue::kKnownZero;
    for (TernaryValue value : *p.Ternary()) {
      result = ternary_ops::Or(result, value);
    }
    if (result != TernaryValue::kUnknown) {
      ternary = TernaryVector({result});
    }
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::OrReduce(*p.Range());
  }

  return PartialInformation(1, std::move(ternary), std::move(range));
}
PartialInformation XorReduce(const PartialInformation& p) {
  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    TernaryValue result = TernaryValue::kKnownZero;
    for (TernaryValue value : *p.Ternary()) {
      result = ternary_ops::Xor(result, value);
    }
    if (result != TernaryValue::kUnknown) {
      ternary = TernaryVector({result});
    }
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::XorReduce(*p.Range());
  }

  return PartialInformation(1, std::move(ternary), std::move(range));
}

PartialInformation Concat(absl::Span<PartialInformation const> infos) {
  int64_t bit_count = 0;
  bool has_ternary = false;
  bool has_range = false;
  bool is_impossible = false;
  for (const PartialInformation& info : infos) {
    bit_count += info.BitCount();
    has_ternary |= info.Ternary().has_value();
    has_range |= info.Range().has_value();
    is_impossible |= info.IsImpossible();
  }
  if (is_impossible) {
    return PartialInformation::Impossible(bit_count);
  }

  std::optional<TernaryVector> ternary;
  if (has_ternary) {
    ternary.emplace(bit_count, TernaryValue::kUnknown);
    auto ternary_it = ternary->begin();
    for (auto it = infos.rbegin(); it != infos.rend(); ++it) {
      if (it->Ternary().has_value()) {
        ternary_it = absl::c_copy(*it->Ternary(), ternary_it);
      } else {
        ternary_it += it->BitCount();
      }
    }
  }

  std::optional<IntervalSet> range;
  if (has_range) {
    std::vector<IntervalSet> ranges;
    ranges.reserve(infos.size());
    for (const PartialInformation& info : infos) {
      if (info.Range().has_value()) {
        ranges.push_back(*info.Range());
      } else {
        ranges.push_back(IntervalSet::Maximal(info.BitCount()));
      }
    }
    range = interval_ops::Concat(ranges);
  }

  return PartialInformation(bit_count, std::move(ternary), std::move(range));
}

PartialInformation SignExtend(const PartialInformation& p, int64_t width) {
  CHECK_GT(p.BitCount(), 0);
  CHECK_GE(width, p.BitCount());

  if (width == p.BitCount()) {
    return p;
  }

  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    ternary.emplace(width, p.Ternary()->back());
    absl::c_copy(*p.Ternary(), ternary->begin());
  }

  return PartialInformation(
      width, std::move(ternary),
      interval_ops::SignExtend(p.RangeOrMaximal(), width));
}
PartialInformation ZeroExtend(const PartialInformation& p, int64_t width) {
  CHECK_GE(width, p.BitCount());

  if (width == p.BitCount()) {
    return p;
  }

  TernaryVector ternary(width, TernaryValue::kKnownZero);
  if (p.Ternary().has_value()) {
    absl::c_copy(*p.Ternary(), ternary.begin());
  } else {
    absl::c_fill_n(ternary, p.BitCount(), TernaryValue::kUnknown);
  }

  return PartialInformation(
      width, std::move(ternary),
      interval_ops::ZeroExtend(p.RangeOrMaximal(), width));
}
PartialInformation Truncate(const PartialInformation& p, int64_t width) {
  CHECK_LE(width, p.BitCount());

  if (width == p.BitCount()) {
    return p;
  }

  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    ternary.emplace(p.Ternary()->begin(), p.Ternary()->begin() + width);
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::Truncate(*p.Range(), width);
  }

  return PartialInformation(width, std::move(ternary), std::move(range));
}
PartialInformation BitSlice(const PartialInformation& p, int64_t start,
                            int64_t width) {
  CHECK_LE(start + width, p.BitCount());
  if (width == 0) {
    return PartialInformation::Unconstrained(0);
  }

  CHECK_LT(start, p.BitCount());
  const int64_t end = start + width;

  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    ternary.emplace(p.Ternary()->begin() + start, p.Ternary()->begin() + end);
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::BitSlice(*p.Range(), start, width);
  }

  return PartialInformation(width, std::move(ternary), std::move(range));
}

PartialInformation DynamicBitSlice(const PartialInformation& p,
                                   const PartialInformation& start,
                                   int64_t width) {
  return ZeroExtend(BitSlice(Shrl(p, start), 0, std::min(width, p.BitCount())),
                    width);
}
PartialInformation BitSliceUpdate(const PartialInformation& to_update,
                                  const PartialInformation& start,
                                  const PartialInformation& update_value) {
  // Mask out the bits we're replacing; position a mask, negate it, then AND it
  // with the value we're updating.
  Bits mask_bits = Bits::AllOnes(update_value.BitCount());
  if (mask_bits.bit_count() > to_update.BitCount()) {
    mask_bits = bits_ops::Truncate(mask_bits, to_update.BitCount());
  } else if (mask_bits.bit_count() < to_update.BitCount()) {
    mask_bits = bits_ops::ZeroExtend(mask_bits, to_update.BitCount());
  }

  PartialInformation masked_to_update =
      PartialInformation::Precise(mask_bits).Shll(start).Not().And(to_update);

  // Position the update value at the correct location, then OR it with the
  // updated value.
  PartialInformation sized_update_value = update_value;
  if (sized_update_value.BitCount() > to_update.BitCount()) {
    sized_update_value = Truncate(sized_update_value, to_update.BitCount());
  } else if (sized_update_value.BitCount() < to_update.BitCount()) {
    sized_update_value = ZeroExtend(sized_update_value, to_update.BitCount());
  }
  return masked_to_update.Or(sized_update_value.Shll(start));
}

PartialInformation Decode(const PartialInformation& p, int64_t width) {
  if (p.IsImpossible()) {
    return PartialInformation::Impossible(width);
  }

  IntervalSet intervals(width);
  for (int64_t i = 0; i < width; ++i) {
    if (Bits::MinBitCountUnsigned(i) > p.BitCount()) {
      break;
    }
    if (p.IsCompatibleWith(UBits(i, p.BitCount()))) {
      intervals.AddInterval(Interval::Precise(Bits::PowerOfTwo(i, width)));
    }
  }
  if (Bits::MinBitCountUnsigned(width) <= p.BitCount() &&
      p.IsCompatibleWith(Interval::Closed(UBits(width, p.BitCount()),
                                          Bits::AllOnes(p.BitCount())))) {
    // "Overflow" is possible, so the output could be zero.
    intervals.AddInterval(Interval::Precise(Bits(width)));
  }
  intervals.Normalize();
  CHECK(!intervals.IsEmpty());

  TernaryVector ternary = interval_ops::ExtractTernaryVector(intervals);
  return PartialInformation(
      std::move(ternary),
      interval_ops::MinimizeIntervals(std::move(intervals), /*size=*/16));
}

PartialInformation Reverse(PartialInformation p) {
  return std::move(p.Reverse());
}

PartialInformation Encode(const PartialInformation& p) {
  const int64_t output_width = CeilOfLog2(p.BitCount());
  if (!p.Ternary().has_value()) {
    return PartialInformation::Unconstrained(output_width);
  }
  PartialInformation result =
      PartialInformation(ternary_ops::BitsToTernary(Bits(output_width)));
  for (int64_t i = 0; i < p.Ternary()->size(); ++i) {
    if (p.Ternary()->at(i) == TernaryValue::kKnownZero) {
      continue;
    }

    // We may be encoding `i` into the output, depending on whether the input
    // bit is set; OR all the corresponding output bits with the input bit at
    // position `i`.
    TernaryVector encoded_i =
        ternary_ops::BitsToTernary(UBits(i, output_width));
    if (p.Ternary()->at(i) == TernaryValue::kUnknown) {
      absl::c_replace(encoded_i, TernaryValue::kKnownOne,
                      TernaryValue::kUnknown);
    }
    result.Or(PartialInformation(std::move(encoded_i)));
  }
  return result;
}

PartialInformation OneHotLsbToMsb(const PartialInformation& p) {
  std::optional<TernaryVector> ternary = std::nullopt;
  if (p.Ternary().has_value()) {
    ternary = TernaryEvaluator().OneHotLsbToMsb(*p.Ternary());
  }

  IntervalSet one_hot(p.BitCount() + 1);
  for (int64_t i = 0; i <= p.BitCount(); ++i) {
    one_hot.AddInterval(
        Interval::Precise(Bits::PowerOfTwo(i, p.BitCount() + 1)));
  }

  return PartialInformation(
      p.BitCount() + 1, std::move(ternary),
      interval_ops::MinimizeIntervals(std::move(one_hot), /*size=*/16));
}
PartialInformation OneHotMsbToLsb(const PartialInformation& p) {
  std::optional<TernaryVector> ternary = std::nullopt;
  if (p.Ternary().has_value()) {
    ternary = TernaryEvaluator().OneHotMsbToLsb(*p.Ternary());
  }

  IntervalSet one_hot(p.BitCount() + 1);
  for (int64_t i = 0; i <= p.BitCount(); ++i) {
    one_hot.AddInterval(
        Interval::Precise(Bits::PowerOfTwo(i, p.BitCount() + 1)));
  }

  return PartialInformation(
      p.BitCount() + 1, std::move(ternary),
      interval_ops::MinimizeIntervals(std::move(one_hot), /*size=*/16));
}

PartialInformation OneHotSelect(const PartialInformation& selector,
                                absl::Span<PartialInformation const> cases,
                                bool selector_can_be_zero) {
  CHECK(!cases.empty());
  const int64_t bit_count = cases.front().BitCount();
  PartialInformation result = PartialInformation::Precise(Bits(bit_count));
  if (!selector_can_be_zero) {
    // If bit `i` is never low in any case, then it must be high in the result.
    PartialInformation and_of_cases = cases.front();
    for (const PartialInformation& case_info : cases.subspan(1)) {
      and_of_cases.And(case_info);
    }
    result = and_of_cases;
  }
  for (int64_t i = 0; i < cases.size(); ++i) {
    TernaryValue selector_bit = selector.Ternary().has_value()
                                    ? selector.Ternary()->at(i)
                                    : TernaryValue::kUnknown;
    if (selector_bit == TernaryValue::kKnownZero) {
      continue;
    }
    result.Or(partial_ops::Gate(PartialInformation(TernarySpan{selector_bit}),
                                cases[i]));
  }
  return result;
}

PartialInformation Gate(const PartialInformation& control,
                        PartialInformation input) {
  return input.Gate(control);
}

PartialInformation Neg(PartialInformation p) { return std::move(p.Neg()); }

PartialInformation Add(PartialInformation a, const PartialInformation& b) {
  return std::move(a.Add(b));
}

PartialInformation Sub(PartialInformation a, const PartialInformation& b) {
  return std::move(a.Sub(b));
}

namespace {

int64_t CountLeadingZeros(TernarySpan ternary) {
  auto first_nonzero = std::find_if(
      ternary.rbegin(), ternary.rend(),
      [](TernaryValue value) { return value != TernaryValue::kKnownZero; });
  return first_nonzero - ternary.rbegin();
}

int64_t CountTrailingZeros(TernarySpan ternary) {
  auto first_nonzero = absl::c_find_if(ternary, [](TernaryValue value) {
    return value != TernaryValue::kKnownZero;
  });
  return first_nonzero - ternary.begin();
}

}  // namespace

PartialInformation UMul(const PartialInformation& a,
                        const PartialInformation& b, int64_t output_bitwidth) {
  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = TernaryEvaluator().UMul(*a.Ternary(), *b.Ternary());
    ternary->resize(output_bitwidth, TernaryValue::kKnownZero);
  } else if (a.Ternary().has_value() || b.Ternary().has_value()) {
    const int64_t trailing_zeros = a.Ternary().has_value()
                                       ? CountTrailingZeros(*a.Ternary())
                                       : CountTrailingZeros(*b.Ternary());
    if (trailing_zeros > 0) {
      ternary = TernaryVector(output_bitwidth, TernaryValue::kUnknown);
      absl::c_fill_n(*ternary, std::min(output_bitwidth, trailing_zeros),
                     TernaryValue::kKnownZero);
    }
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() || b.Range().has_value()) {
    range = interval_ops::UMul(a.RangeOrMaximal(), b.RangeOrMaximal(),
                               output_bitwidth);
  }

  return PartialInformation(output_bitwidth, std::move(ternary),
                            std::move(range));
}
PartialInformation UDiv(const PartialInformation& a,
                        const PartialInformation& b) {
  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = TernaryEvaluator().UDiv(*a.Ternary(), *b.Ternary());
  } else if (a.Ternary().has_value() || b.Ternary().has_value()) {
    int64_t leading_zeros = 0;
    if (!b.Range().has_value() || b.Range()->CoversZero()) {
      // If `b` could be zero, then the result might be forced to the maximum
      // value, so we don't know if the result has any leading zeros.
    } else if (a.Ternary().has_value()) {
      leading_zeros = CountLeadingZeros(*a.Ternary());
    } else if (!ternary_ops::IsCompatible(*b.Ternary(), Bits(b.BitCount())) ||
               (b.Range().has_value() && !b.Range()->CoversZero())) {
      leading_zeros = CountTrailingZeros(*b.Ternary());
    }
    if (leading_zeros > 0) {
      ternary = TernaryVector(a.BitCount(), TernaryValue::kUnknown);
      std::fill_n(ternary->rbegin(), std::min(a.BitCount(), leading_zeros),
                  TernaryValue::kKnownZero);
    }
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() || b.Range().has_value()) {
    range = interval_ops::UDiv(a.RangeOrMaximal(), b.RangeOrMaximal());
  }

  return PartialInformation(a.BitCount(), std::move(ternary), std::move(range));
}
PartialInformation UMod(const PartialInformation& a,
                        const PartialInformation& b) {
  if (a.Range().has_value() || b.Range().has_value()) {
    return PartialInformation(
        interval_ops::UMod(a.RangeOrMaximal(), b.RangeOrMaximal()));
  }
  return PartialInformation(IntervalSet::Of(
      {Interval::RightOpen(Bits(a.BitCount()), Bits::AllOnes(a.BitCount()))}));
}

PartialInformation SMul(const PartialInformation& a,
                        const PartialInformation& b, int64_t output_bitwidth) {
  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = TernaryEvaluator().SMul(*a.Ternary(), *b.Ternary());
    ternary->resize(output_bitwidth, ternary->back());
  } else if (a.Ternary().has_value() || b.Ternary().has_value()) {
    const int64_t trailing_zeros = a.Ternary().has_value()
                                       ? CountTrailingZeros(*a.Ternary())
                                       : CountTrailingZeros(*b.Ternary());
    if (trailing_zeros > 0) {
      ternary = TernaryVector(output_bitwidth, TernaryValue::kUnknown);
      absl::c_fill_n(*ternary, std::min(output_bitwidth, trailing_zeros),
                     TernaryValue::kKnownZero);
    }
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() || b.Range().has_value()) {
    range = interval_ops::SMul(a.RangeOrMaximal(), b.RangeOrMaximal(),
                               output_bitwidth);
  }

  return PartialInformation(output_bitwidth, std::move(ternary),
                            std::move(range));
}
PartialInformation SDiv(const PartialInformation& a,
                        const PartialInformation& b) {
  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = TernaryEvaluator().SDiv(*a.Ternary(), *b.Ternary());
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() || b.Range().has_value()) {
    range = interval_ops::SDiv(a.RangeOrMaximal(), b.RangeOrMaximal());
  }

  return PartialInformation(a.BitCount(), std::move(ternary), std::move(range));
}
PartialInformation SMod(const PartialInformation& a,
                        const PartialInformation& b) {
  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = TernaryEvaluator().SMod(*a.Ternary(), *b.Ternary());
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() || b.Range().has_value()) {
    range = interval_ops::SMod(a.RangeOrMaximal(), b.RangeOrMaximal());
  }

  return PartialInformation(a.BitCount(), std::move(ternary), std::move(range));
}

PartialInformation Shll(PartialInformation a, const PartialInformation& b) {
  return a.Shll(b);
}
PartialInformation Shrl(PartialInformation a, const PartialInformation& b) {
  return a.Shrl(b);
}
PartialInformation Shra(PartialInformation a, const PartialInformation& b) {
  return a.Shra(b);
}

PartialInformation Eq(const PartialInformation& a,
                      const PartialInformation& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());

  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = {TernaryValue::kKnownOne};
    for (int64_t i = 0; i < a.BitCount(); ++i) {
      if (ternary_ops::IsUnknown(a.Ternary()->at(i)) ||
          ternary_ops::IsUnknown(b.Ternary()->at(i))) {
        ternary->at(0) = TernaryValue::kUnknown;
      } else if (a.Ternary()->at(i) != b.Ternary()->at(i)) {
        ternary->at(0) = TernaryValue::kKnownZero;
        break;
      }
    }
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() && b.Range().has_value()) {
    range = interval_ops::Eq(*a.Range(), *b.Range());
  }

  return PartialInformation(/*bit_count=*/1, std::move(ternary),
                            std::move(range));
}
PartialInformation Ne(const PartialInformation& a,
                      const PartialInformation& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());

  std::optional<TernaryVector> ternary;
  if (a.Ternary().has_value() && b.Ternary().has_value()) {
    ternary = {TernaryValue::kKnownZero};
    for (int64_t i = 0; i < a.BitCount(); ++i) {
      if (ternary_ops::IsUnknown(a.Ternary()->at(i)) ||
          ternary_ops::IsUnknown(b.Ternary()->at(i))) {
        ternary->at(0) = TernaryValue::kUnknown;
      } else if (a.Ternary()->at(i) != b.Ternary()->at(i)) {
        ternary->at(0) = TernaryValue::kKnownOne;
        break;
      }
    }
  }

  std::optional<IntervalSet> range;
  if (a.Range().has_value() && b.Range().has_value()) {
    range = interval_ops::Ne(*a.Range(), *b.Range());
  }

  return PartialInformation(/*bit_count=*/1, std::move(ternary),
                            std::move(range));
}

PartialInformation ULt(const PartialInformation& a,
                       const PartialInformation& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  return PartialInformation(
      interval_ops::ULt(a.RangeOrMaximal(), b.RangeOrMaximal()));
}

PartialInformation SLt(const PartialInformation& a,
                       const PartialInformation& b) {
  CHECK_EQ(a.BitCount(), b.BitCount());
  return PartialInformation(
      interval_ops::SLt(a.RangeOrMaximal(), b.RangeOrMaximal()));
}

}  // namespace xls::partial_ops

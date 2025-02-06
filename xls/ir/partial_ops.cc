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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/ternary.h"

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

PartialInformation Concat(absl::Span<PartialInformation const> infos) {
  int64_t bit_count = 0;
  bool has_ternary = false;
  bool has_range = false;
  for (const PartialInformation& info : infos) {
    bit_count += info.BitCount();
    has_ternary |= info.Ternary().has_value();
    has_range |= info.Range().has_value();
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
  CHECK_GE(width, p.BitCount());

  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    CHECK_GT(p.Ternary()->size(), 0);
    ternary = TernaryVector(p.Ternary()->size(), p.Ternary()->back());
    absl::c_copy(*p.Ternary(), ternary->begin());
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::SignExtend(*p.Range(), width);
  }

  return PartialInformation(width, std::move(ternary), std::move(range));
}
PartialInformation ZeroExtend(const PartialInformation& p, int64_t width) {
  CHECK_GE(width, p.BitCount());

  std::optional<TernaryVector> ternary;
  if (p.Ternary().has_value()) {
    ternary = TernaryVector(p.Ternary()->size(), TernaryValue::kKnownZero);
    absl::c_copy(*p.Ternary(), ternary->begin());
  }

  std::optional<IntervalSet> range;
  if (p.Range().has_value()) {
    range = interval_ops::ZeroExtend(*p.Range(), width);
  }

  return PartialInformation(width, std::move(ternary), std::move(range));
}
PartialInformation Truncate(const PartialInformation& p, int64_t width) {
  CHECK_LE(width, p.BitCount());

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
  CHECK_LT(start, p.BitCount());
  CHECK_LE(start + width, p.BitCount());
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

PartialInformation Neg(PartialInformation p) { return p.Neg(); }
PartialInformation Add(PartialInformation a, const PartialInformation& b) {
  return a.Add(b);
}
PartialInformation Sub(PartialInformation a, const PartialInformation& b) {
  return a.Sub(b);
}

PartialInformation UMul(const PartialInformation& a,
                        const PartialInformation& b, int64_t output_bitwidth) {
  if (a.Range().has_value() && b.Range().has_value()) {
    return PartialInformation(
        std::nullopt,
        interval_ops::UMul(*a.Range(), *b.Range(), output_bitwidth));
  }
  return PartialInformation(output_bitwidth);
}
PartialInformation UDiv(const PartialInformation& a,
                        const PartialInformation& b) {
  if (a.Range().has_value() && b.Range().has_value()) {
    return PartialInformation(std::nullopt,
                              interval_ops::UDiv(*a.Range(), *b.Range()));
  }
  return PartialInformation(a.BitCount());
}

PartialInformation Shrl(PartialInformation a, const PartialInformation& b) {
  return a.Shrl(b);
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
PartialInformation SLt(const PartialInformation& a,
                       const PartialInformation& b);
PartialInformation SGt(const PartialInformation& a,
                       const PartialInformation& b);
PartialInformation ULt(const PartialInformation& a,
                       const PartialInformation& b);
PartialInformation UGt(const PartialInformation& a,
                       const PartialInformation& b);

}  // namespace xls::partial_ops

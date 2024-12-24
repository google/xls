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

#include "xls/dslx/exhaustiveness/interp_value_interval.h"

#include "absl/log/log.h"

namespace xls::dslx {

/* static */ InterpValueInterval InterpValueInterval::MakeFull(
    bool is_signed, int64_t bit_count) {
  return InterpValueInterval(InterpValue::MakeMinValue(is_signed, bit_count),
                             InterpValue::MakeMaxValue(is_signed, bit_count));
}

InterpValueInterval::InterpValueInterval(InterpValue min, InterpValue max)
    : min_(std::move(min)), max_(std::move(max)) {
  absl::StatusOr<InterpValue> le = min_.Le(max_);
  CHECK_OK(le.status());
  CHECK(le->IsTrue()) << absl::StreamFormat(
      "InterpValueInterval; min: %s max: %s need min <= max", min_.ToString(),
      max_.ToString());
}

bool InterpValueInterval::Contains(InterpValue value) const {
  return min_.Le(value)->IsTrue() && max_.Ge(value)->IsTrue();
}

std::string InterpValueInterval::ToString(bool show_types) const {
  return absl::StrFormat("[%s, %s]", min_.ToString(/*humanize=*/!show_types),
                         max_.ToString(/*humanize=*/!show_types));
}

bool InterpValueInterval::IsSigned() const {
  CHECK_EQ(min_.IsSigned(), max_.IsSigned());
  return min_.IsSigned();
}

int64_t InterpValueInterval::GetBitCount() const {
  CHECK_EQ(min_.GetBitCount(), max_.GetBitCount());
  return min_.GetBitCount().value();
}

}  // namespace xls::dslx

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

#include "xls/noc/config_ng/arbitrary_base_number.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/status/status.h"

namespace xls::noc {

ArbitraryBaseNumber::ArbitraryBaseNumber(const int64_t digit_count,
                                         const int64_t numerical_base)
    : digits_(internal::CheckGe(digit_count, int64_t{0}), 0),
      numerical_base_(numerical_base) {
  CHECK_GE(digit_count, 1);
  CHECK_GE(numerical_base, 2);
  CHECK_LE(numerical_base, 128);
}

std::optional<int64_t> ArbitraryBaseNumber::GetValue(
    const int64_t index) const {
  if (index < 0 || index >= digits_.size()) {
    return std::nullopt;
  }
  int64_t value = 0;
  int64_t base = 1;
  for (int64_t count = index; count < digits_.size(); count++) {
    value += digits_[count] * base;
    base *= numerical_base_;
  }
  return value;
}

void ArbitraryBaseNumber::Reset() {
  std::fill(digits_.begin(), digits_.end(), 0);
}

bool ArbitraryBaseNumber::AddOne() {
  int64_t index = 0;
  bool carry = false;
  do {
    digits_[index]++;
    carry = digits_[index] == numerical_base_;
    if (carry) {
      digits_[index] = 0;
    }
    index++;
  } while (carry && index < digits_.size());
  return carry;
}

absl::Status ArbitraryBaseNumber::SwapDigits(const int64_t first_digit_index,
                                             const int64_t second_digit_index) {
  if (first_digit_index < 0 || first_digit_index >= digits_.size()) {
    return absl::OutOfRangeError("First digit index is out of range.");
  }
  if (second_digit_index < 0 || second_digit_index >= digits_.size()) {
    return absl::OutOfRangeError("Second digit index is out of range.");
  }
  std::swap(digits_[first_digit_index], digits_[second_digit_index]);
  return absl::OkStatus();
}

int64_t ArbitraryBaseNumber::GetDigitCount() const { return digits_.size(); }

int64_t ArbitraryBaseNumber::GetNumericalBase() const {
  return numerical_base_;
}

}  // namespace xls::noc

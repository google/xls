// Copyright 2020 The XLS Authors
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

#include "xls/ir/ternary.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace xls {

std::string ToString(const TernaryVector& value) {
  std::string result = "0b";
  for (int64_t i = value.size() - 1; i >= 0; --i) {
    std::string symbol;
    switch (value[i]) {
      case TernaryValue::kKnownZero:
        symbol = "0";
        break;
      case TernaryValue::kKnownOne:
        symbol = "1";
        break;
      case TernaryValue::kUnknown:
        symbol = "X";
        break;
    }
    absl::StrAppend(&result, symbol);
    if (i != 0 && i % 4 == 0) {
      absl::StrAppend(&result, "_");
    }
  }
  return result;
}

std::string ToString(const TernaryValue& value) {
  switch (value) {
    case TernaryValue::kKnownZero:
      return "TernaryValue::kKnownZero";
    case TernaryValue::kKnownOne:
      return "TernaryValue::kKnownOne";
    case TernaryValue::kUnknown:
      return "TernaryValue::kUnknown";
  }
  XLS_LOG(FATAL) << "Invalid ternary value: " << static_cast<int>(value);
}

absl::StatusOr<TernaryVector> StringToTernaryVector(std::string_view s) {
  auto invalid_input = [&]() {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid ternary string: %s", s));
  };
  if (s.substr(0, 2) != "0b") {
    return invalid_input();
  }
  TernaryVector result;
  for (char c : s.substr(2)) {
    switch (c) {
      case '0':
        result.push_back(TernaryValue::kKnownZero);
        break;
      case '1':
        result.push_back(TernaryValue::kKnownOne);
        break;
      case 'X':
      case 'x':
        result.push_back(TernaryValue::kUnknown);
        break;
      case '_':
        break;
      default:
        return invalid_input();
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

namespace ternary_ops {

Bits ToKnownBits(const TernaryVector& ternary_vector) {
  absl::InlinedVector<bool, 1> bits(ternary_vector.size());
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = (ternary_vector[i] != TernaryValue::kUnknown);
  }
  return Bits(bits);
}

Bits ToKnownBitsValues(const TernaryVector& ternary_vector) {
  absl::InlinedVector<bool, 1> bits(ternary_vector.size());
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = (ternary_vector[i] == TernaryValue::kKnownOne);
  }
  return Bits(bits);
}

}  // namespace ternary_ops

}  // namespace xls

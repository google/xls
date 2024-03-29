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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"

namespace xls {

std::string ToString(TernarySpan value) {
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
  LOG(FATAL) << "Invalid ternary value: " << static_cast<int>(value);
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

TernaryVector FromKnownBits(const Bits& known_bits,
                            const Bits& known_bits_values) {
  CHECK_EQ(known_bits.bit_count(), known_bits_values.bit_count());
  TernaryVector result;
  result.reserve(known_bits.bit_count());

  for (int64_t i = 0; i < known_bits.bit_count(); ++i) {
    if (known_bits.Get(i)) {
      result.push_back(known_bits_values.Get(i) ? TernaryValue::kKnownOne
                                                : TernaryValue::kKnownZero);
    } else {
      result.push_back(TernaryValue::kUnknown);
    }
  }

  return result;
}

Bits ToKnownBits(TernarySpan ternary_vector) {
  absl::InlinedVector<bool, 1> bits(ternary_vector.size());
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = (ternary_vector[i] != TernaryValue::kUnknown);
  }
  return Bits(bits);
}

Bits ToKnownBitsValues(TernarySpan ternary_vector) {
  absl::InlinedVector<bool, 1> bits(ternary_vector.size());
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = (ternary_vector[i] == TernaryValue::kKnownOne);
  }
  return Bits(bits);
}

std::optional<TernaryVector> Difference(TernarySpan lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  const int64_t size = lhs.size();

  TernaryVector result;
  result.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    if (lhs[i] != TernaryValue::kUnknown) {
      if (rhs[i] == TernaryValue::kUnknown) {
        result.push_back(lhs[i]);
      } else {
        if (lhs[i] != rhs[i]) {
          return std::nullopt;
        }
        result.push_back(TernaryValue::kUnknown);
      }
    } else {
      result.push_back(TernaryValue::kUnknown);
    }
  }
  return result;
}

absl::StatusOr<TernaryVector> Union(TernarySpan lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  const int64_t size = lhs.size();

  TernaryVector result;
  result.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    if (lhs[i] == TernaryValue::kUnknown) {
      result.push_back(rhs[i]);
    } else if (rhs[i] == TernaryValue::kUnknown) {
      result.push_back(lhs[i]);
    } else if (lhs[i] == rhs[i]) {
      result.push_back(lhs[i]);
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Incompatible values (mismatch at bit %d); cannot unify %s and %s", i,
          ToString(lhs), ToString(rhs)));
    }
  }

  return result;
}

absl::Status UpdateWithUnion(TernaryVector& lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());

  for (int64_t i = 0; i < lhs.size(); ++i) {
    if (rhs[i] == TernaryValue::kUnknown) {
      continue;
    }

    if (lhs[i] == TernaryValue::kUnknown) {
      lhs[i] = rhs[i];
    } else if (lhs[i] != rhs[i]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Incompatible values (mismatch at bit %d); cannot update %s with %s",
          i, ToString(lhs), ToString(rhs)));
    }
  }

  return absl::OkStatus();
}

TernaryVector Intersection(TernarySpan lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  const int64_t size = lhs.size();

  TernaryVector result;
  result.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    if (lhs[i] != rhs[i]) {
      result.push_back(TernaryValue::kUnknown);
    } else {
      result.push_back(lhs[i]);
    }
  }

  return result;
}

void UpdateWithIntersection(TernaryVector& lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());

  for (int64_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      lhs[i] = TernaryValue::kUnknown;
    }
  }
}
void UpdateWithIntersection(TernaryVector& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.size(), rhs.bit_count());

  for (int64_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] == TernaryValue::kUnknown) {
      continue;
    }
    const bool lhs_bit = lhs[i] == TernaryValue::kKnownOne;
    const bool rhs_bit = rhs.Get(i);
    if (lhs_bit != rhs_bit) {
      lhs[i] = TernaryValue::kUnknown;
    }
  }
}

int64_t NumberOfKnownBits(TernarySpan vec) {
  int64_t result = 0;
  for (TernaryValue value : vec) {
    if (value != TernaryValue::kUnknown) {
      ++result;
    }
  }
  return result;
}

TernaryVector BitsToTernary(const Bits& bits) {
  TernaryVector result;
  result.resize(bits.bit_count());
  for (int64_t i = 0; i < bits.bit_count(); ++i) {
    result[i] = static_cast<TernaryValue>(bits.Get(i));
  }
  return result;
}

/* static */ std::vector<int64_t> RealizedTernaryIterator::FindUnknownOffsets(
    TernarySpan span) {
  std::vector<int64_t> result;
  result.reserve(span.size());
  for (int64_t i = 0; i < span.size(); ++i) {
    if (span[i] == TernaryValue::kUnknown) {
      result.push_back(i);
    }
  }
  result.shrink_to_fit();
  return result;
}

namespace {
std::pair<bool, InlineBitmap> IncrementOnOffsets(
    InlineBitmap bm, absl::Span<int64_t const> offsets) {
  bool overflow = true;
  for (int64_t off : offsets) {
    if (bm.Get(off) == false) {
      bm.Set(off, true);
      overflow = false;
      break;
    }
    bm.Set(off, false);
  }
  return {overflow, std::move(bm)};
}
}  // namespace

void RealizedTernaryIterator::Advance(const Bits& amnt) {
  InlineBitmap bm = std::move(value_).bitmap();
  // Do minimal amount of single increments.
  // Advancing by (1<<V) only has a visible effect on the V'th and up X bits in
  // the ternary so we can leave lower bits alone.

  for (int64_t i = 0;
       i < amnt.bit_count() - amnt.CountLeadingZeros() && !finished_; ++i) {
    if (amnt.Get(i)) {
      std::tie(finished_, bm) = IncrementOnOffsets(
          std::move(bm), absl::MakeConstSpan(unknown_bit_offsets_).subspan(i));
    }
  }
  value_ = Bits::FromBitmap(std::move(bm));
}

void RealizedTernaryIterator::Advance(int64_t amnt) {
  CHECK_GE(amnt, 0);
  InlineBitmap bm = std::move(value_).bitmap();
  // Do minimal amount of single increments.
  // Advancing by (1<<V) only has a visible effect on the V'th and up X bits in
  // the ternary so we can leave lower bits alone.
  for (int64_t i = 0; i < 64 && amnt != 0 && !finished_; ++i, amnt >>= 1) {
    if (amnt & 0b1) {
      std::tie(finished_, bm) = IncrementOnOffsets(
          std::move(bm), absl::MakeConstSpan(unknown_bit_offsets_).subspan(i));
    }
  }
  value_ = Bits::FromBitmap(std::move(bm));
}

}  // namespace ternary_ops

}  // namespace xls

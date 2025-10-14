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
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/iter_util.h"
#include "xls/common/iterator_range.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

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

std::string ToString(TernaryValue value) {
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

  TernaryVector result(known_bits.bit_count(), TernaryValue::kUnknown);
  for (auto [result_bit, is_known, known_value] :
       iter::zip(result, known_bits, known_bits_values)) {
    // We take advantage of the cppitertools feature that `zip` iterates over
    // tuples of references for non-const containers... meaning that we can
    // modify `result` in place by writing to `result_bit`.
    static_assert(std::is_same_v<decltype(result_bit), TernaryValue&>);

    if (is_known) {
      result_bit =
          known_value ? TernaryValue::kKnownOne : TernaryValue::kKnownZero;
    }
  }

  return result;
}

Bits ToKnownBits(TernarySpan ternary_vector) {
  InlineBitmap bitmap(ternary_vector.size());
  for (int64_t i = 0; i < ternary_vector.size(); ++i) {
    if (ternary_vector[i] != TernaryValue::kUnknown) {
      bitmap.Set(i, true);
    }
  }
  return Bits::FromBitmap(std::move(bitmap));
}

Bits ToKnownBitsValues(TernarySpan ternary_vector, bool default_set) {
  InlineBitmap bitmap(ternary_vector.size());
  for (int64_t i = 0; i < ternary_vector.size(); ++i) {
    if (ternary_vector[i] == TernaryValue::kKnownOne ||
        (default_set && ternary_vector[i] == TernaryValue::kUnknown)) {
      bitmap.Set(i, true);
    }
  }
  return Bits::FromBitmap(std::move(bitmap));
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

bool TryUpdateWithUnion(TernaryVector& lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());

  for (int64_t i = 0; i < lhs.size(); ++i) {
    if (rhs[i] == TernaryValue::kUnknown) {
      continue;
    }

    if (lhs[i] == TernaryValue::kUnknown) {
      lhs[i] = rhs[i];
    } else if (lhs[i] != rhs[i]) {
      return false;
    }
  }

  return true;
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

bool IsCompatible(TernarySpan pattern, const Bits& bits) {
  if (pattern.size() != bits.bit_count()) {
    return false;
  }

  for (auto [pattern_bit, bit] : iter::zip(pattern, bits)) {
    if (pattern_bit == TernaryValue::kUnknown) {
      continue;
    }
    if (bit != (pattern_bit == TernaryValue::kKnownOne)) {
      return false;
    }
  }
  return true;
}

bool IsCompatible(TernarySpan a, TernarySpan b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int64_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i] && a[i] != TernaryValue::kUnknown &&
        b[i] != TernaryValue::kUnknown) {
      return false;
    }
  }
  return true;
}

int64_t MinimumUnsignedBitCount(TernarySpan t) {
  for (int64_t len = t.size(); len > 0; --len) {
    if (t[len - 1] != TernaryValue::kKnownZero) {
      return len;
    }
  }
  // Every element is known zero.
  return 0;
}

void UpdateWithIntersection(TernaryVector& lhs, TernarySpan rhs) {
  CHECK_EQ(lhs.size(), rhs.size());

  for (int64_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      lhs[i] = TernaryValue::kUnknown;
    }
  }
}
void UpdateWithIntersection(TernaryVector& lhs, const TernaryVector& rhs) {
  UpdateWithIntersection(lhs, absl::MakeConstSpan(rhs));
}
void UpdateWithIntersection(TernaryVector& lhs, const Bits& rhs) {
  CHECK_EQ(lhs.size(), rhs.bit_count());

  for (auto [lhs_bit, rhs_bit] : iter::zip(lhs, rhs)) {
    if (lhs_bit == TernaryValue::kUnknown) {
      continue;
    }
    if (rhs_bit != (lhs_bit == TernaryValue::kKnownOne)) {
      lhs_bit = TernaryValue::kUnknown;
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
  result.reserve(bits.bit_count());
  for (bool bit : bits) {
    result.push_back(bit ? TernaryValue::kKnownOne : TernaryValue::kKnownZero);
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
  if (value_.bit_count() == 0 && !amnt.IsZero()) {
    // To match the somewhat strange bits behavior a zero-length ternary is
    // considered to have a single value.
    finished_ = true;
    return;
  }
  InlineBitmap bm = std::move(value_).bitmap();
  // Do minimal amount of single increments.
  // Advancing by (1<<V) only has a visible effect on the V'th and up X bits in
  // the ternary so we can leave lower bits alone.

  for (int64_t i = 0;
       i < amnt.bit_count() - amnt.CountLeadingZeros() && !finished_; ++i) {
    if (amnt.Get(i)) {
      std::tie(finished_, bm) = IncrementOnOffsets(
          std::move(bm),  // NOLINT(bugprone-use-after-move)
          absl::MakeConstSpan(unknown_bit_offsets_).subspan(i));
    }
  }
  value_ = Bits::FromBitmap(std::move(bm));
}

void RealizedTernaryIterator::Advance(int64_t amnt) {
  CHECK_GE(amnt, 0);
  if (value_.bit_count() == 0 && amnt != 0) {
    // To match the somewhat strange bits behavior a zero-length ternary is
    // considered to have a single value.
    finished_ = true;
    return;
  }
  InlineBitmap bm = std::move(value_).bitmap();
  // Do minimal amount of single increments.
  // Advancing by (1<<V) only has a visible effect on the V'th and up X bits in
  // the ternary so we can leave lower bits alone.
  for (int64_t i = 0; i < 64 && amnt != 0 && !finished_; ++i, amnt >>= 1) {
    if (amnt & 0b1) {
      std::tie(finished_, bm) = IncrementOnOffsets(
          std::move(bm),  // NOLINT(bugprone-use-after-move)
          absl::MakeConstSpan(unknown_bit_offsets_).subspan(i));
    }
  }
  value_ = Bits::FromBitmap(std::move(bm));
}

absl::StatusOr<std::vector<Value>> AllValues(
    LeafTypeTreeView<TernaryVector> ltt) {
  if (ltt.leaf_types().empty()) {
    // This type has no leaf types, so it consists entirely of empty tuples or
    // arrays; as such, it has only one possible value.
    LeafTypeTree<Value> value_ltt(ltt.type(), Value());
    XLS_ASSIGN_OR_RETURN(Value value, LeafTypeTreeToValue(value_ltt.AsView()));
    return std::vector<Value>({std::move(value)});
  }

  using AccessibleRange =
      xabsl::iterator_range<ternary_ops::RealizedTernaryIterator>;
  LeafTypeTree<AccessibleRange> iterators =
      leaf_type_tree::Map<AccessibleRange, TernaryVector>(
          ltt, [](const TernaryVector& ternary) {
            return ternary_ops::AllBitsValues(ternary);
          });
  absl::Span<Type* const> types = iterators.leaf_types();

  absl::StatusOr<std::vector<Value>> values = std::vector<Value>();
  bool failed = IteratorProduct<AccessibleRange>(
      iterators.elements(),
      [&](absl::Span<ternary_ops::RealizedTernaryIterator const> its) {
        std::vector<Value> elements;
        elements.reserve(its.size());
        for (int64_t i = 0; i < its.size(); ++i) {
          if (types[i]->IsToken()) {
            if (its[i]->bit_count() != 0) {
              values = absl::InternalError("Non-empty value given for token");
              return true;
            }
            elements.push_back(Value::Token());
            continue;
          }

          elements.push_back(Value(*its[i]));
        }
        LeafTypeTree<Value> value_ltt(ltt.type(), elements);
        absl::StatusOr<Value> value = LeafTypeTreeToValue(value_ltt.AsView());
        if (!value.ok()) {
          values = std::move(value).status();
          return true;
        }
        values->push_back(*value);
        return false;
      });
  XLS_RET_CHECK(failed != values.ok());
  return values;
}

}  // namespace ternary_ops

}  // namespace xls

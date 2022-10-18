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

#ifndef XLS_IR_TERNARY_H_
#define XLS_IR_TERNARY_H_

#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/bits.h"

namespace xls {

// Ternary logic is used to propagate known and unknown bit values through the
// graph. Ternary logic allows three values for each "bit".
enum class TernaryValue : int8_t {
  kKnownZero = 0,
  kKnownOne = 1,
  kUnknown = 2
};

// A vector of ternary values. Analogous to the Bits object for concrete values.
using TernaryVector = std::vector<TernaryValue>;

// Format of the ternary vector is, for example: 0b10XX1
std::string ToString(const TernaryVector& value);
std::string ToString(const TernaryValue& value);

// Converts the given string to a TernaryVector. Expects string to be of form
// emitted by ToString (for example, 0b01X0). Underscores are ignored.
absl::StatusOr<TernaryVector> StringToTernaryVector(std::string_view s);

inline std::ostream& operator<<(std::ostream& os, TernaryValue value) {
  os << ToString(value);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, TernaryVector vector) {
  os << ToString(vector);
  return os;
}

namespace ternary_ops {

inline TernaryVector FromKnownBits(const Bits& known_bits,
                                   const Bits& known_bits_values) {
  XLS_CHECK_EQ(known_bits.bit_count(), known_bits_values.bit_count());
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

// Returns a `Bits` that contains a 1 for each element of the given ternary
// vector that is either `kKnownZero` or `kKnownOne`, and a 0 otherwise.
Bits ToKnownBits(const TernaryVector& ternary_vector);

// Returns a `Bits` that contains a 1 for each element of the given ternary
// vector that is `kKnownOne`, and a 0 otherwise.
Bits ToKnownBitsValues(const TernaryVector& ternary_vector);

// Returns a vector with known positions for each bit known in `lhs` that isn't
// known in `rhs`. If `lhs` and `rhs` conflict, returns `absl::nullopt`.
// CHECK fails if `lhs` and `rhs` have different lengths.
inline std::optional<TernaryVector> Difference(const TernaryVector& lhs,
                                                const TernaryVector& rhs) {
  XLS_CHECK_EQ(lhs.size(), rhs.size());
  int64_t size = lhs.size();

  TernaryVector result;
  for (int64_t i = 0; i < size; ++i) {
    if (lhs[i] != TernaryValue::kUnknown) {
      if (rhs[i] == TernaryValue::kUnknown) {
        result.push_back(lhs[i]);
      } else {
        if (lhs[i] != rhs[i]) {
          return absl::nullopt;
        }
        result.push_back(TernaryValue::kUnknown);
      }
    } else {
      result.push_back(TernaryValue::kUnknown);
    }
  }

  return result;
}

// Returns the number of known bits in the given `TernaryVector`.
inline int64_t NumberOfKnownBits(const TernaryVector& vec) {
  int64_t result = 0;
  for (TernaryValue value : vec) {
    if (value != TernaryValue::kUnknown) {
      ++result;
    }
  }
  return result;
}

inline bool IsKnown(TernaryValue t) { return t != TernaryValue::kUnknown; }
inline bool IsUnknown(TernaryValue t) { return t == TernaryValue::kUnknown; }

inline bool AllUnknown(const TernaryVector& v) {
  return std::all_of(v.begin(), v.end(), IsUnknown);
}

inline TernaryValue And(const TernaryValue& a, const TernaryValue& b) {
  // Truth table:
  //              rhs
  //      & |  0   1   X
  //      --+------------
  // lhs  0 |  0   0   0
  //      1 |  0   1   X
  //      X |  0   X   X
  if (a == TernaryValue::kKnownZero || b == TernaryValue::kKnownZero) {
    return TernaryValue::kKnownZero;
  }
  if (a == TernaryValue::kKnownOne && b == TernaryValue::kKnownOne) {
    return TernaryValue::kKnownOne;
  }
  return TernaryValue::kUnknown;
}

inline TernaryValue Or(const TernaryValue& a, const TernaryValue& b) {
  // Truth table:
  //              rhs
  //      | |  0   1   X
  //      --+------------
  // lhs  0 |  0   1   X
  //      1 |  1   1   1
  //      X |  X   1   X
  if (a == TernaryValue::kKnownOne || b == TernaryValue::kKnownOne) {
    return TernaryValue::kKnownOne;
  }
  if (a == TernaryValue::kKnownZero && b == TernaryValue::kKnownZero) {
    return TernaryValue::kKnownZero;
  }
  return TernaryValue::kUnknown;
}

inline TernaryVector BitsToTernary(const Bits& bits) {
  TernaryVector result;
  result.resize(bits.bit_count());
  for (int64_t i = 0; i < bits.bit_count(); ++i) {
    result[i] = static_cast<TernaryValue>(bits.Get(i));
  }
  return result;
}

}  // namespace ternary_ops
}  // namespace xls

#endif  // XLS_IR_TERNARY_H_

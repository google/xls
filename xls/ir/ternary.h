// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_IR_TERNARY_H_
#define THIRD_PARTY_XLS_IR_TERNARY_H_

#include <vector>

#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"

namespace xls {

// Ternary logic is used to propagate known and unknown bit values through the
// graph. Ternary logic allows three values for each "bit".
enum class TernaryValue : int8 { kKnownZero = 0, kKnownOne = 1, kUnknown = 2 };

// A vector of ternary values. Analogous to the Bits object for concrete values.
using TernaryVector = std::vector<TernaryValue>;

// Format of the ternary vector is, for example: 0b10XX1
std::string ToString(const TernaryVector& value);
std::string ToString(const TernaryValue& value);

// Converts the given string to a TernaryVector. Expects string to be of form
// emitted by ToString (for example, 0b01X0). Underscores are ignored.
xabsl::StatusOr<TernaryVector> StringToTernaryVector(absl::string_view s);

inline std::ostream& operator<<(std::ostream& os, TernaryValue value) {
  os << ToString(value);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, TernaryVector vector) {
  os << ToString(vector);
  return os;
}

namespace ternary_ops {

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

// Returns whether a is (definitely) equal to b.
//
// Note that this also operates as the "meet" operator as in a lattice, where
// "bottom" is represented by "X".
inline TernaryValue Equals(const TernaryValue& a, const TernaryValue& b) {
  // Truth table:
  //              rhs
  //      | |  0   1   X
  //      --+------------
  // lhs  0 |  0   X   X
  //      1 |  X   1   X
  //      X |  X   X   X
  if (IsKnown(a) && IsKnown(b) && a == b) {
    return a;
  }
  return TernaryValue::kUnknown;
}

// Returns the "equals" operator above for all bits in same-sized ternary
// vectors.
inline TernaryVector Equals(const TernaryVector& lhs,
                            const TernaryVector& rhs) {
  XLS_CHECK_EQ(lhs.size(), rhs.size());
  TernaryVector result;
  result.resize(lhs.size());
  for (int64 i = 0; i < lhs.size(); ++i) {
    result[i] = Equals(lhs[i], rhs[i]);
  }
  return result;
}

inline TernaryVector BitsToTernary(const Bits& bits) {
  TernaryVector result;
  result.resize(bits.bit_count());
  for (int64 i = 0; i < bits.bit_count(); ++i) {
    result[i] = static_cast<TernaryValue>(bits.Get(i));
  }
  return result;
}

}  // namespace ternary_ops
}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_TERNARY_H_

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

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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
using TernarySpan = absl::Span<TernaryValue const>;

// Format of the ternary vector is, for example: 0b10XX1
std::string ToString(TernarySpan value);
std::string ToString(const TernaryValue& value);

// Converts the given string to a TernaryVector. Expects string to be of form
// emitted by ToString (for example, 0b01X0). Underscores are ignored.
absl::StatusOr<TernaryVector> StringToTernaryVector(std::string_view s);

inline std::ostream& operator<<(std::ostream& os, TernaryValue value) {
  os << ToString(value);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, TernarySpan vector) {
  os << ToString(vector);
  return os;
}

namespace ternary_ops {

// Returns a vector with known bits as represented in `known_bits`, with values
// as given in `known_bits_values`.
TernaryVector FromKnownBits(const Bits& known_bits,
                            const Bits& known_bits_values);

// Returns a `Bits` that contains a 1 for each element of the given ternary
// vector that is either `kKnownZero` or `kKnownOne`, and a 0 otherwise.
Bits ToKnownBits(TernarySpan ternary_vector);

// Returns a `Bits` that contains a 1 for each element of the given ternary
// vector that is `kKnownOne`, and a 0 otherwise.
Bits ToKnownBitsValues(TernarySpan ternary_vector);

// Returns a vector with known positions for each bit known in `lhs` that isn't
// known in `rhs`. If `lhs` and `rhs` conflict, returns `std::nullopt`.
// CHECK fails if `lhs` and `rhs` have different lengths.
std::optional<TernaryVector> Difference(TernarySpan lhs,
                                        TernarySpan rhs);

// Returns a vector with known positions for each bit known to have a value in
// either `lhs` or `rhs`, or an error if `lhs` and `rhs` are incompatible (have
// known bits that disagree). CHECK fails if `lhs` and `rhs` have different
// lengths.
absl::StatusOr<TernaryVector> Union(TernarySpan lhs,
                                    TernarySpan rhs);

// Updates `lhs` to include additional known information from `rhs`, or an error
// if `lhs` and `rhs` are incompatible (have known bits that disagree). CHECK
// fails if `lhs` and `rhs` have different lengths.
absl::Status UpdateWithUnion(TernaryVector& lhs, TernarySpan rhs);

// Returns a vector with known positions for each bit known to have the same
// value in both `lhs` and `rhs`. CHECK fails if `lhs` and `rhs` have different
// lengths.
TernaryVector Intersection(TernarySpan lhs, TernarySpan rhs);

// Updates `lhs`, turning it into a vector of bits known to have the same value
// in both `lhs` and `rhs`. CHECK fails if `lhs` and `rhs` have different
// lengths.
void UpdateWithIntersection(TernaryVector& lhs, TernarySpan rhs);
void UpdateWithIntersection(TernaryVector& lhs, const Bits& rhs);

// Returns the number of known bits in the given `TernaryVector`.
int64_t NumberOfKnownBits(TernarySpan vec);

inline bool IsKnown(TernaryValue t) { return t != TernaryValue::kUnknown; }
inline bool IsUnknown(TernaryValue t) { return t == TernaryValue::kUnknown; }

inline bool IsFullyKnown(TernarySpan ternary) {
  return absl::c_all_of(ternary, IsKnown);
}
inline bool AllUnknown(TernarySpan v) {
  return absl::c_all_of(v, IsUnknown);
}

inline bool IsKnownOne(TernarySpan ternary) {
  return absl::c_all_of(
      ternary, [](TernaryValue v) { return v == TernaryValue::kKnownOne; });
}
inline bool IsKnownZero(TernarySpan ternary) {
  return absl::c_all_of(
      ternary, [](TernaryValue v) { return v == TernaryValue::kKnownZero; });
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

TernaryVector BitsToTernary(const Bits& bits);

}  // namespace ternary_ops
}  // namespace xls

#endif  // XLS_IR_TERNARY_H_

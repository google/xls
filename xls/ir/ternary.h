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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/iterator_range.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

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

std::string ToString(TernaryValue value);

// Format of the ternary vector is, for example: 0b10XX1
std::string ToString(TernarySpan value);
inline std::string ToString(const TernaryVector& value) {
  return ToString(TernarySpan(value));
}
// Converts the given string to a TernaryVector. Expects string to be of form
// emitted by ToString (for example, 0b01X0). Underscores are ignored.
absl::StatusOr<TernaryVector> StringToTernaryVector(std::string_view s);

template <typename Sink>
void AbslStringify(Sink& sink, TernarySpan e) {
  sink.Append(ToString(e));
}
template <typename Sink>
void AbslStringify(Sink& sink, const TernaryVector& e) {
  AbslStringify(sink, TernarySpan(e));
}
template <typename Sink>
void AbslStringify(Sink& sink, TernaryValue e) {
  sink.Append(ToString(e));
}

inline std::ostream& operator<<(std::ostream& os, const TernaryValue& value) {
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
// vector that is `kKnownOne`, a 0 for each element that is `kKnownZero`, and
// `default_set` otherwise.
Bits ToKnownBitsValues(TernarySpan ternary_vector, bool default_set = false);

// Returns a vector with known positions for each bit known in `lhs` that isn't
// known in `rhs`. If `lhs` and `rhs` conflict, returns `std::nullopt`.
// CHECK fails if `lhs` and `rhs` have different lengths.
std::optional<TernaryVector> Difference(TernarySpan lhs, TernarySpan rhs);

// Returns a vector with known positions for each bit known to have a value in
// either `lhs` or `rhs`, or an error if `lhs` and `rhs` are incompatible (have
// known bits that disagree). CHECK fails if `lhs` and `rhs` have different
// lengths.
absl::StatusOr<TernaryVector> Union(TernarySpan lhs, TernarySpan rhs);

// Updates `lhs` to include additional known information from `rhs`. Returns
// false if `lhs` and `rhs` are incompatible (have known bits that disagree),
// leaving `lhs` in an unspecified state. CHECK fails if `lhs` and `rhs` have
// different lengths.
bool TryUpdateWithUnion(TernaryVector& lhs, TernarySpan rhs);

// Updates `lhs` to include additional known information from `rhs`, or an error
// if `lhs` and `rhs` are incompatible (have known bits that disagree). CHECK
// fails if `lhs` and `rhs` have different lengths.
absl::Status UpdateWithUnion(TernaryVector& lhs, TernarySpan rhs);

// Returns a vector with known positions for each bit known to have the same
// value in both `lhs` and `rhs`. CHECK fails if `lhs` and `rhs` have different
// lengths.
TernaryVector Intersection(TernarySpan lhs, TernarySpan rhs);

// Returns true if `bits` is a possible value for `pattern`.
bool IsCompatible(TernarySpan pattern, const Bits& bits);

// Returns true if `a` and `b` are compatible (have no known bits that
// disagree).
bool IsCompatible(TernarySpan a, TernarySpan b);

// Updates `lhs`, turning it into a vector of bits known to have the same value
// in both `lhs` and `rhs`. CHECK fails if `lhs` and `rhs` have different
// lengths.
void UpdateWithIntersection(TernaryVector& lhs, TernarySpan rhs);
void UpdateWithIntersection(TernaryVector& lhs, const TernaryVector& rhs);
void UpdateWithIntersection(TernaryVector& lhs, const Bits& rhs);

// Returns the number of known bits in the given `TernaryVector`.
int64_t NumberOfKnownBits(TernarySpan vec);

// Returns the number of bits required to represent every possible value the
// ternary may take assuming an unsigned representation.
int64_t MinimumUnsignedBitCount(TernarySpan t);

inline bool IsKnown(TernaryValue t) { return t != TernaryValue::kUnknown; }
inline bool IsUnknown(TernaryValue t) { return t == TernaryValue::kUnknown; }

inline bool IsFullyKnown(TernarySpan ternary) {
  return absl::c_all_of(ternary, IsKnown);
}
inline bool AllUnknown(TernarySpan v) { return absl::c_all_of(v, IsUnknown); }
inline bool AnyKnown(TernarySpan v) { return absl::c_any_of(v, IsKnown); }

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
inline TernaryVector And(TernarySpan a, TernarySpan b) {
  CHECK_EQ(a.size(), b.size());
  TernaryVector result(a.size());
  for (int64_t i = 0; i < a.size(); ++i) {
    result[i] = And(a[i], b[i]);
  }
  return result;
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
inline TernaryVector Or(TernarySpan a, TernarySpan b) {
  CHECK_EQ(a.size(), b.size());
  TernaryVector result(a.size());
  for (int64_t i = 0; i < a.size(); ++i) {
    result[i] = Or(a[i], b[i]);
  }
  return result;
}

inline TernaryValue Not(const TernaryValue& a) {
  switch (a) {
    case TernaryValue::kKnownZero:
      return TernaryValue::kKnownOne;
    case TernaryValue::kKnownOne:
      return TernaryValue::kKnownZero;
    case TernaryValue::kUnknown:
      return TernaryValue::kUnknown;
  }
  ABSL_UNREACHABLE();
}
inline TernaryVector Not(TernarySpan a) {
  TernaryVector result;
  result.reserve(a.size());
  absl::c_transform(a, std::back_inserter(result),
                    [](TernaryValue v) { return Not(v); });
  return result;
}

inline TernaryValue Xor(const TernaryValue& a, const TernaryValue& b) {
  // Truth table:
  //              rhs
  //      | |  0   1   X
  //      --+------------
  // lhs  0 |  0   1   X
  //      1 |  1   0   X
  //      X |  X   X   X
  if (a == TernaryValue::kUnknown || b == TernaryValue::kUnknown) {
    return TernaryValue::kUnknown;
  }
  return a == b ? TernaryValue::kKnownZero : TernaryValue::kKnownOne;
}
inline TernaryVector Xor(TernarySpan a, TernarySpan b) {
  CHECK_EQ(a.size(), b.size());
  TernaryVector result(a.size());
  for (int64_t i = 0; i < a.size(); ++i) {
    result[i] = Xor(a[i], b[i]);
  }
  return result;
}

TernaryVector BitsToTernary(const Bits& bits);

// An iterator of possible ternary values.
class RealizedTernaryIterator {
 public:
  using value_type = Bits;
  using reference = const Bits&;
  using pointer = const Bits*;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;
  using iterator_concept = std::forward_iterator_tag;

  RealizedTernaryIterator(const RealizedTernaryIterator&) = default;
  RealizedTernaryIterator(RealizedTernaryIterator&&) = default;
  RealizedTernaryIterator& operator=(const RealizedTernaryIterator&) = default;
  RealizedTernaryIterator& operator=(RealizedTernaryIterator&&) = default;
  explicit RealizedTernaryIterator() : finished_(true) {}

  const Bits& operator*() const {
    CHECK(!finished_);
    return value_;
  }

  const Bits* operator->() const { return &**this; }

  bool operator==(const RealizedTernaryIterator& o) const {
    if (finished_ != o.finished_) {
      // One is at the end and the other isn't
      return false;
    }
    if (finished_ && o.finished_) {
      return true;
    }
    return value_ == o.value_;
  }

  // Pre increment
  RealizedTernaryIterator& operator++() {
    Advance(1);
    return *this;
  }

  RealizedTernaryIterator& operator+=(int64_t amnt) {
    CHECK_GE(amnt, 0);
    Advance(amnt);
    return *this;
  }

  RealizedTernaryIterator& operator+=(const Bits& amnt) {
    Advance(amnt);
    return *this;
  }

  // Post increment
  RealizedTernaryIterator operator++(int) {
    RealizedTernaryIterator cpy = *this;
    Advance(1);
    return cpy;
  }

  RealizedTernaryIterator operator+(const Bits& v) const {
    RealizedTernaryIterator res = *this;
    res += v;
    return res;
  }

  RealizedTernaryIterator operator+(int v) const {
    RealizedTernaryIterator res = *this;
    res += v;
    return res;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RealizedTernaryIterator& it) {
    if (it.finished_) {
      absl::Format(&sink, "(ternary-iterator @ end)");
    } else {
      absl::Format(&sink, "(ternary-iterator @ %s)", it.value_.ToDebugString());
    }
  }

 private:
  explicit RealizedTernaryIterator(TernarySpan span)
      : finished_(false),
        value_(ToKnownBitsValues(span)),
        unknown_bit_offsets_(
            RealizedTernaryIterator::FindUnknownOffsets(span)) {}

  void Advance(int64_t amnt);
  void Advance(const Bits& amnt);
  static std::vector<int64_t> FindUnknownOffsets(TernarySpan span);

  bool finished_;
  Bits value_;
  std::vector<int64_t> unknown_bit_offsets_;

  friend xabsl::iterator_range<RealizedTernaryIterator> AllBitsValues(
      TernarySpan span);
};

// Make an iterator range that enumerates all possible values which match the
// given ternary. The values are produced in order from smallest to largest.
//
// NB To match with bits convention an empty span is considered to have one
// (0-bit) value associated with it.
inline xabsl::iterator_range<RealizedTernaryIterator> AllBitsValues(
    TernarySpan span) {
  static_assert(std::forward_iterator<RealizedTernaryIterator>);
  return {RealizedTernaryIterator(span), RealizedTernaryIterator()};
}

// Make an iterator range that enumerates all possible values which match the
// given tree of ternaries. The values are produced in order from smallest to
// largest at each position, in little-endian order (i.e., the first leaf
// changes most often, and the last leaf changes least often).
absl::StatusOr<std::vector<Value>> AllValues(
    LeafTypeTreeView<TernaryVector> ltt);

}  // namespace ternary_ops
}  // namespace xls

#endif  // XLS_IR_TERNARY_H_

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

#ifndef XLS_IR_VALUE_H_
#define XLS_IR_VALUE_H_

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/ir/xls_value.pb.h"

namespace xls {

class Type;

enum class ValueKind {
  kInvalid,

  kBits,

  kTuple,

  // Arrays must be homogeneous in their elements, and may choose to use a
  // more efficient storage mechanism as a result. For now we always used the
  // boxed Value type, though.
  kArray,

  kToken
};

std::string ValueKindToString(ValueKind kind);

inline std::ostream& operator<<(std::ostream& os, ValueKind kind) {
  os << ValueKindToString(kind);
  return os;
}

// Represents a value in the XLS system; e.g. values can be "bits", tuples of
// values, or arrays or values. Arrays are represented similarly to tuples, but
// are monomorphic and potentially multi-dimensional.
//
// TODO(leary): 2019-04-04 Arrays are not currently multi-dimensional, we had
// some discussion around this, maybe they should be?
class Value {
 public:
  static Value Tuple(absl::Span<const Value> elements) {
    return Value(ValueKind::kTuple, elements);
  }
  static Value TupleOwned(std::vector<Value>&& elements) {
    return Value(ValueKind::kTuple, elements);
  }

  // All members of "elements" must be of the same type, or an error status will
  // be returned.
  static absl::StatusOr<Value> Array(absl::Span<const Value> elements);

  // Shortcut to create an array of bits from an initializer list of literals
  // ex. UBitsArray({1, 2}, 32) will create a Value of type bits[32][2]
  static absl::StatusOr<Value> UBitsArray(absl::Span<const uint64_t> elements,
                                          int64_t bit_count);
  static absl::StatusOr<Value> UBits2DArray(
      absl::Span<const absl::Span<const uint64_t>> elements, int64_t bit_count);
  static absl::StatusOr<Value> SBitsArray(absl::Span<const int64_t> elements,
                                          int64_t bit_count);
  static absl::StatusOr<Value> SBits2DArray(
      absl::Span<const absl::Span<const int64_t>> elements, int64_t bit_count);

  // As above, but as a precondition all elements must be known to be of the
  // same type.
  //
  // Prefer using ValueBuilder for construction.
  static Value ArrayOrDie(absl::Span<const Value> elements) {
    return Array(elements).value();
  }

  // Construct an array using the given elements. Ownership of the vector is
  // transferred to the Value. DCHECK fails if not all elements are the same
  // type.
  //
  // Prefer using ValueBuilder for construction.
  static Value ArrayOwned(std::vector<Value>&& elements) {
    for (int64_t i = 1; i < elements.size(); ++i) {
      DCHECK(elements[0].SameTypeAs(elements[i]));
    }
    return Value(ValueKind::kArray, std::move(elements));
  }

  static Value Token() {
    return Value(ValueKind::kToken, std::vector<Value>({}));
  }
  static Value Bool(bool enabled) {
    return Value(UBits(/*value=*/enabled, /*bit_count=*/1));
  }

  Value() : kind_(ValueKind::kInvalid), payload_(nullptr) {}

  explicit Value(Bits bits)
      : kind_(ValueKind::kBits), payload_(std::move(bits)) {}

  // Convert a ValueProto back to a value.
  //
  // If any bits element has a bit-count of more than max_bit_size the
  // conversion will fail. Default to rejecting 1GiB or larger bits values.
  // This limit is not for any correctness reason (if you want to have gigabytes
  // of binary data that's fine) but just to change weird OOM crashes when fed
  // with fuzzed/corrupted data into more debuggable Status errors. Set this to
  // std::number_limits<int64_t>::max() to disable the check.
  static absl::StatusOr<Value> FromProto(const ValueProto& proto,
                                         int64_t max_bit_size = int64_t{1}
                                                                << 33);

  // Serializes the contents of this value as bits in the buffer.
  void FlattenTo(BitPushBuffer* buffer) const;

  ValueKind kind() const { return kind_; }
  bool IsTuple() const { return kind_ == ValueKind::kTuple; }
  bool IsArray() const { return kind_ == ValueKind::kArray; }
  bool IsBits() const { return std::holds_alternative<Bits>(payload_); }
  bool IsToken() const { return kind_ == ValueKind::kToken; }
  const Bits& bits() const { return std::get<Bits>(payload_); }
  absl::StatusOr<Bits> GetBitsWithStatus() const;

  absl::StatusOr<std::vector<Value>> GetElements() const;

  absl::Span<const Value> elements() const {
    return std::get<std::vector<Value>>(payload_);
  }
  const Value& element(int64_t i) const { return elements().at(i); }
  int64_t size() const { return elements().size(); }
  bool empty() const { return elements().empty(); }

  // Returns the total number of bits in this value.
  int64_t GetFlatBitCount() const;

  // Returns whether all bits within this value are zeros/ones.
  bool IsAllZeros() const;
  bool IsAllOnes() const;

  // Converts this value to a descriptive string format for printing to
  // screen. String includes the type. Examples:
  //   bits[8]:0b01
  //   (bits[1]:0, bits[32]:42)
  //   [bits[7]:22, bits[7]:1, bits[7]:123]
  std::string ToString(
      FormatPreference preference = FormatPreference::kDefault) const;

  // Emits the Value as a human-readable string where numbers represented as
  // undecorated numbers without bit width (eg, "42" or "0xabcd"). This is used
  // when emitting the literals in the IR where the bit count is already known
  // and a more compact representation is desirable.
  std::string ToHumanString(
      FormatPreference preference = FormatPreference::kDefault) const;

  // Returns the type of the Value as a type proto.
  absl::StatusOr<TypeProto> TypeAsProto() const;

  // Returns this Value as a ValueProto.
  absl::StatusOr<ValueProto> AsProto() const;

  // Returns true if 'other' has the same type as this Value.
  bool SameTypeAs(const Value& other) const;

  bool operator==(const Value& other) const;
  bool operator!=(const Value& other) const { return !(*this == other); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Value& v) {
    absl::Format(&sink, "%s", v.ToString(FormatPreference::kDefault));
  }

 private:
  Value(ValueKind kind, absl::Span<const Value> elements)
      : kind_(kind),
        payload_(std::vector<Value>(elements.begin(), elements.end())) {}

  Value(ValueKind kind, std::vector<Value>&& elements)
      : kind_(kind), payload_(std::move(elements)) {}

  ValueKind kind_;
  std::variant<std::nullptr_t, std::vector<Value>, Bits> payload_;
};

inline std::ostream& operator<<(std::ostream& os, const Value& value) {
  os << value.ToString(FormatPreference::kDefault);
  return os;
}

}  // namespace xls

#endif  // XLS_IR_VALUE_H_

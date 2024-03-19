// Copyright 2024 The XLS Authors
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

#ifndef XLS_IR_VALUE_BUILDER_H_
#define XLS_IR_VALUE_BUILDER_H_

#include <cstdint>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {

// A helper for building complicated value objects allowing one to delay the
// status checks to the end.
//
// Usage:
// XLS_ASSIGN_OR_RETURN(
//     auto array_of_tuples,
//     ValueBuilder::Array(
//         {ValueBuilder::Tuple({ValueBuilder::Bits(UBits(1, 2)),
//                               ValueBuilder::Bits(UBits(1, 2)),
//                               ValueBuilder::Bits(UBits(1, 2))}),
//          ValueBuilder::Tuple({ValueBuilder::Bits(UBits(1, 2)),
//                               ValueBuilder::Bits(UBits(1, 2)),
//                               ValueBuilder::Bits(UBits(1, 2))}),
//          ValueBuilder::Tuple({ValueBuilder::Bits(UBits(1, 2)),
//                               ValueBuilder::Bits(UBits(1, 2)),
//                               ValueBuilder::Bits(UBits(1, 2))})})
//         .Build());
class ValueBuilder {
 public:
  using MaybeValueBuilder = std::variant<ValueBuilder, Value>;
  explicit ValueBuilder(Value real) : my_value_(real) {}
  ValueBuilder(const ValueBuilder&) = default;
  ValueBuilder(ValueBuilder&&) = default;
  ValueBuilder& operator=(const ValueBuilder&) = default;
  ValueBuilder& operator=(ValueBuilder&&) = default;

  static ValueBuilder Array(absl::Span<MaybeValueBuilder const> elements);
  static ValueBuilder ArrayB(absl::Span<ValueBuilder const> elements);
  static ValueBuilder ArrayV(absl::Span<Value const> elements);
  // Shortcut to create an array of bits from an initializer list of literals
  // ex. UBitsArray({1, 2}, 32) will create a Value of type bits[32][2]
  static ValueBuilder UBitsArray(absl::Span<uint64_t const> elements,
                                 int64_t bit_count);
  static ValueBuilder UBits2DArray(
      absl::Span<const absl::Span<const uint64_t>> elements, int64_t bit_count);
  static ValueBuilder SBitsArray(absl::Span<const int64_t> elements,
                                 int64_t bit_count);
  static ValueBuilder SBits2DArray(
      absl::Span<const absl::Span<const int64_t>> elements, int64_t bit_count);

  static ValueBuilder Tuple(absl::Span<MaybeValueBuilder const> elements);
  static ValueBuilder TupleB(absl::Span<ValueBuilder const> elements);
  static ValueBuilder TupleV(absl::Span<Value const> elements);
  static ValueBuilder Token() { return ValueBuilder(Value::Token()); }
  static ValueBuilder Bits(const Bits& b) { return ValueBuilder(Value(b)); }
  static ValueBuilder Bool(bool enabled) {
    return ValueBuilder(Value(
        UBits(/*value=*/static_cast<uint64_t>(enabled), /*bit_count=*/1)));
  }

  absl::StatusOr<Value> Build() const;

  bool IsArray() const {
    return std::holds_alternative<ArrayHolder>(my_value_) ||
           (std::holds_alternative<Value>(my_value_) &&
            std::get<Value>(my_value_).IsArray());
  }
  bool IsBits() const {
    return std::holds_alternative<Value>(my_value_) &&
           std::get<Value>(my_value_).IsBits();
  }
  bool IsToken() const {
    return std::holds_alternative<Value>(my_value_) &&
           std::get<Value>(my_value_).IsToken();
  }
  bool IsTuple() const {
    return std::holds_alternative<TupleHolder>(my_value_) ||
           (std::holds_alternative<Value>(my_value_) &&
            std::get<Value>(my_value_).IsTuple());
  }

 private:
  struct ArrayHolder {
    std::vector<MaybeValueBuilder> v;
  };
  struct TupleHolder {
    std::vector<MaybeValueBuilder> v;
  };
  explicit ValueBuilder(std::variant<ArrayHolder, TupleHolder, Value> v)
      : my_value_(v) {}

  std::variant<ArrayHolder, TupleHolder, Value> my_value_;
};

}  // namespace xls

#endif  // XLS_IR_VALUE_BUILDER_H_

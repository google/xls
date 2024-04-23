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

#ifndef XLS_IR_VALUE_UTILS_H_
#define XLS_IR_VALUE_UTILS_H_

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Returns a zero value for the given type.
//
// Note that for composite values (tuples / arrays) all member values are zero.
Value ZeroOfType(Type* type);

// As above, but fills bits with all-ones pattern.
Value AllOnesOfType(Type* type);

// Formatter for emitting values with type prefix (e.g, "42").
inline void ValueFormatter(std::string* out, const Value& value) {
  absl::StrAppend(out, value.ToString());
}

// Formatter for emitting values without type prefix (e.g, "42" for a value of
// bits[32]:42).
inline void UntypedValueFormatter(std::string* out, const Value& value) {
  absl::StrAppend(out, value.ToHumanString());
}

inline void ValueFormatterBinary(std::string* out, const Value& value) {
  absl::StrAppend(out, value.ToString(FormatPreference::kBinary));
}

inline void ValueFormatterHex(std::string* out, const Value& value) {
  absl::StrAppend(out, value.ToString(FormatPreference::kHex));
}

// Returns whether "value" conforms to type "type" -- this lets us avoid
// constructing a Type and doing an equivalence check on hot paths.
inline bool ValueConformsToType(const Value& value, Type* type) {
  switch (value.kind()) {
    case ValueKind::kBits:
      return type->IsBits() &&
             value.bits().bit_count() == type->AsBitsOrDie()->bit_count();
    case ValueKind::kArray:
      return type->IsArray() && type->AsArrayOrDie()->size() == value.size() &&
             (value.empty() ||
              ValueConformsToType(value.element(0),
                                  type->AsArrayOrDie()->element_type()));
    case ValueKind::kTuple: {
      if (!type->IsTuple()) {
        return false;
      }
      const TupleType* tuple_type = type->AsTupleOrDie();
      if (tuple_type->size() != value.size()) {
        return false;
      }
      for (int64_t i = 0; i < tuple_type->size(); ++i) {
        if (!ValueConformsToType(value.element(i),
                                 tuple_type->element_type(i))) {
          return false;
        }
      }
      return true;
    }
    case ValueKind::kToken:
      return type->IsToken();
    default:
      LOG(FATAL) << "Invalid value kind: " << value.kind();
  }
}

// Converts a float value in C++ to a 3-tuple suitable for feeding DSLX F32
// routines as the type is defined the standard library; see
// `xls/dslx/stdlib/float32.x`.
Value F32ToTuple(float value);

// Converts a 3-tuple F32 (as noted in F32ToTuple above) into a C++ float.
absl::StatusOr<float> TupleToF32(const Value& v);

// Converts a `LeafTypeTree<Value>` to a `Value`.
absl::StatusOr<Value> LeafTypeTreeToValue(LeafTypeTreeView<Value> tree);

// Converts a `Value` to a `LeafTypeTree<Value>`.
// The given `Type*` must be the type of the given `Value`.
absl::StatusOr<LeafTypeTree<Value>> ValueToLeafTypeTree(const Value& value,
                                                        Type* type);

}  // namespace xls

#endif  // XLS_IR_VALUE_UTILS_H_

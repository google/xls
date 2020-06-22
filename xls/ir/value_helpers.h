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

#ifndef XLS_IR_VALUE_HELPERS_H_
#define XLS_IR_VALUE_HELPERS_H_

#include <random>

#include "absl/strings/str_cat.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Returns a zero value for the given type.
//
// Note that for composite values (tuples / arrays) all member values are zero.
Value ZeroOfType(Type* type);

// As above, but fills bits with all-ones pattern.
Value AllOnesOfType(Type* type);

// For use in e.g. StrJoin as a lambda.
inline void ValueFormatter(std::string* out, const Value& value) {
  absl::StrAppend(out, value.ToString());
}
inline void ValueFormatterBinary(std::string* out, const Value& value) {
  absl::StrAppend(out, value.ToString(FormatPreference::kBinary));
}

// Returns a Value with random uniformly distributed bits using the given
// engine.
Value RandomValue(Type* type, std::minstd_rand* engine);

// Returns a set of argument values for the given function with random uniformly
// distributed bits using the given engine.
std::vector<Value> RandomFunctionArguments(Function* f,
                                           std::minstd_rand* engine);

// Returns whether "value" conforms to type "type" -- this lets us avoid
// constructing a Type and doing an equivalence check on hot paths.
inline bool ValueConformsToType(const Value& value, const Type* type) {
  switch (value.kind()) {
    case ValueKind::kBits:
      return type->IsBits() &&
             value.bits().bit_count() == type->AsBitsOrDie()->bit_count();
    case ValueKind::kArray:
      return type->IsArray() &&
             ValueConformsToType(value.element(0),
                                 type->AsArrayOrDie()->element_type());
    case ValueKind::kTuple: {
      if (!type->IsTuple()) {
        return false;
      }
      const TupleType* tuple_type = type->AsTupleOrDie();
      if (tuple_type->size() != value.size()) {
        return false;
      }
      for (int64 i = 0; i < tuple_type->size(); ++i) {
        if (!ValueConformsToType(value.element(i),
                                 tuple_type->element_type(i))) {
          return false;
        }
      }
      return true;
    }
    default:
      XLS_LOG(FATAL) << "Invalid value kind: " << value.kind();
  }
}

// Converts a float value in C++ to a 3-tuple suitable for feeding the XLS rsqrt
// routine.
Value F32ToTuple(float value);

// Converts a 3-tuple (XLS rsqrt routine float representation) into a C++ float.
xabsl::StatusOr<float> TupleToF32(const Value& v);

}  // namespace xls

#endif  // XLS_IR_VALUE_HELPERS_H_

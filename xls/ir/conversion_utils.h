// Copyright 2022 The XLS Authors
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

#ifndef XLS_IR_CONVERSION_UTILS_H_
#define XLS_IR_CONVERSION_UTILS_H_

#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/function.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

// Utilities for conversions between a value represented in the C++ domain to
// the IR domain (xls::Value).
//
// The conversion are performed recursively. As a result, when converting to an
// array and a tuple, the elements within the array and tuple are also
// converted.
//
// To leverage the recursive mechanism when performing conversions for
// user-defined types such as Classes and Structs, the conversion function must
// have the following signature, where 'MyClassOrStruct' is a user-defined class
// or struct:
//
//        absl::StatusOr<xls::Value> Convert(const Type& type,
//                                           const MyClassOrStruct& my_object);
//
// Moreover, the function must be placed in the 'xls' namespace.
//
// For example, for the following user-defined struct:
//     namespace userspace {
//       struct UserStruct {
//        int64_t integer_value;
//        bool boolean_value;
//       };
//     }  // namespace userspace
//
//  the conversion function would be as follows:
//
//     namespace xls {
//        absl::StatusOr<xls::Value>
//                          Convert(
//                                  const Type& type,
//                                  const userspace::UserStruct& user_struct) {
//          return Convert(type, std::make_tuple(user_struct.integer_value,
//                                               user_struct.boolean_value));
//        }
//      }  // namespace xls
//
// Note that the conversion function leverages the conversion for tuples.

namespace xls {

// Convert the int64_t 'value' from the C++ domain to the IR domain (xls::Value)
// using 'type' as the conversion type to the IR domain.
//
// An integral of 'int64_t' type boolean can only be converted to a Bits type.
// The value must fit within bit count defined by 'type'. Otherwise, an error is
// reported.
absl::StatusOr<xls::Value> Convert(const Type* type, int64_t value);

// Convert the uint64_t 'value' from the C++ domain to the IR domain
// (xls::Value) using 'type' as the conversion type to the IR domain.
//
// An integral of 'uint64_t' type boolean can only be converted to a Bits type.
// The value must fit within bit count defined by 'type'. Otherwise, an error is
// reported.
absl::StatusOr<xls::Value> Convert(const Type* type, uint64_t value);

// Convert the absl::Span 'values' from the C++ domain to the IR domain
// (xls::Value) using 'type' as the conversion type to the IR domain. Note that
// the function uses a recursive mechanism to that also converts the elements of
// the absl::Span. Please see file comment for more information.
//
// The 'type' must be of array type. The 'type' and absl::Span must define the
// same number of elements for the array. Otherwise, an error is reported.
template <typename T>
absl::StatusOr<xls::Value> Convert(const Type* type,
                                   absl::Span<const T> values) {
  XLS_CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (!type->IsArray()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for an absl::Span. An absl::Span can only be "
        "converted using an array type. Expected type (array), got (%s).",
        type->ToString()));
  }
  const ArrayType* array_type = type->AsArrayOrDie();
  if (array_type->size() != values.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Array size mismatch between conversion type and "
                        "value. Expected (%d), got (%d).",
                        array_type->size(), values.size()));
  }
  std::vector<xls::Value> xls_array(values.size(), xls::Value());
  for (int64_t i = 0; i < values.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(xls_array[i],
                         Convert(array_type->element_type(), values[i]));
  }
  return Value::Array(xls_array);
}

namespace internal {

// Base case for tuple conversion: all elements have been converted.
absl::Status ConvertTupleElements(const TupleType* type, int64_t index,
                                  std::vector<Value>& xls_tuple);

// Recursive case for tuple conversion: at least one element exist for
// conversion.
template <typename ValueType, typename... ValuesType>
absl::Status ConvertTupleElements(const TupleType* type, int64_t index,
                                  std::vector<Value>& xls_tuple,
                                  ValueType value, ValuesType... values) {
  XLS_CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (index >= xls_tuple.size()) {
    // When the user is using the Convert(..., std::tuple...), the following
    // error should not occur.
    return absl::InvalidArgumentError(
        absl::StrFormat("The current element count exceeds the element count "
                        "used for memory allocation. Expected (%d), got (%d).",
                        xls_tuple.size(), index));
  }
  XLS_ASSIGN_OR_RETURN(xls_tuple[index],
                       xls::Convert(type->element_type(index), value));
  return ConvertTupleElements(type, index + 1, xls_tuple, values...);
}

// Unpack/Flatten the tuple to perform conversion on each element.
template <typename... ValuesType, size_t... I>
absl::Status UnpackTuple(const TupleType* type, std::vector<Value>& xls_tuple,
                         const std::tuple<ValuesType...>& tuple,
                         std::index_sequence<I...>) {
  XLS_CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  return ConvertTupleElements(type, 0, xls_tuple, std::get<I>(tuple)...);
}

}  // namespace internal

// Convert the std::tuple 'tuple' from the C++ domain to the IR domain
// (xls::Value) using 'type' as the conversion type to the IR domain. Note that
// the function uses a recursive mechanism to that also converts the elements of
// the std::tuple. Please see file comment for more information.
//
// The 'type' must be of tuple type. The 'type' and std::tuple must define the
// same number of elements for the tuple. Otherwise, an error is reported.
template <typename... ValuesType>
absl::StatusOr<xls::Value> Convert(const Type* type,
                                   const std::tuple<ValuesType...>& tuple) {
  XLS_CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (!type->IsTuple()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for a std::tuple. An std::tuple can only be "
        "converted using an tuple type. Expected type (tuple), got (%s).",
        type->ToString()));
  }
  constexpr size_t num_elements = std::tuple_size_v<std::tuple<ValuesType...>>;
  const TupleType* tuple_type = type->AsTupleOrDie();
  if (tuple_type->size() != num_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Tuple size mismatch between conversion type and "
                        "value. Expected (%d), got (%d).",
                        tuple_type->size(), num_elements));
  }
  std::vector<xls::Value> xls_tuple(num_elements, xls::Value());
  XLS_RETURN_IF_ERROR(internal::UnpackTuple(
      tuple_type, xls_tuple, tuple, std::index_sequence_for<ValuesType...>{}));
  return Value::Tuple(xls_tuple);
}

// Conversion for an empty tuple.
absl::StatusOr<xls::Value> Convert(const Type* type, const std::tuple<>& tuple);

}  // namespace xls

#endif  // XLS_IR_CONVERSION_UTILS_H_

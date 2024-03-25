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

#ifndef XLS_IR_VALUE_CONVERSION_UTILS_H_
#define XLS_IR_VALUE_CONVERSION_UTILS_H_

#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

// Utilities for conversions from a value represented in the C++ domain to
// the IR domain (xls::Value), and vice-versa, from the IR domain (xls::Value)
// to a value represented in the C++ domain.

namespace xls {

// Below are utilities for conversions from a value represented in the C++
// domain to the IR domain (xls::Value).
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
//        absl::StatusOr<xls::Value> ConvertToXlsValue(const Type& type,
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
//        absl::StatusOr<xls::Value> ConvertToXlsValue(
//            const Type* type,
//            const userspace::UserStruct& user_struct) {
//          return ConvertToXlsValue(
//              type,
//              std::make_tuple(user_struct.integer_value,
//                              user_struct.boolean_value));
//        }
//      }  // namespace xls
//
// Note that the conversion function leverages the conversion for tuples.

namespace internal {
// Convert the int64_t 'value' from the C++ domain to the IR domain (xls::Value)
// using 'type' as the conversion type to the IR domain.
//
// An integral of 'int64_t' type boolean can only be converted to a Bits type.
// The value must fit within bit count defined by 'type'. Otherwise, an error is
// reported.
absl::StatusOr<xls::Value> ConvertInt64(const Type* type, int64_t value);

// Convert the uint64_t 'value' from the C++ domain to the IR domain
// (xls::Value) using 'type' as the conversion type to the IR domain.
//
// An integral of 'uint64_t' type boolean can only be converted to a Bits type.
// The value must fit within bit count defined by 'type'. Otherwise, an error is
// reported.
absl::StatusOr<xls::Value> ConvertUint64(const Type* type, uint64_t value);
}  // namespace internal

// Convert the integer 'value' from the C++ domain to the IR domain (xls::Value)
// using 'type' as the conversion type to the IR domain.
//
// The type of integer must be smaller or equal to that of an int64_t.
template <typename T,
          typename std::enable_if<std::is_unsigned_v<T>, T>::type* = nullptr>
absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type, T value) {
  static_assert(sizeof(T) <= sizeof(uint64_t));
  return internal::ConvertUint64(type, static_cast<uint64_t>(value));
}

template <typename T, typename std::enable_if<std::is_signed_v<T> &&
                                                  !std::is_floating_point_v<T>,
                                              T>::type* = nullptr>
absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type, T value) {
  static_assert(sizeof(T) <= sizeof(int64_t));
  return internal::ConvertInt64(type, static_cast<int64_t>(value));
}

// Convert the absl::Span 'values' from the C++ domain to the IR domain
// (xls::Value) using 'type' as the conversion type to the IR domain. Note that
// the function uses a recursive mechanism that also converts the elements of
// the absl::Span. Please see comment above regarding conversion to the IR
// domain for more information.
//
// The 'type' must be of array type. The 'type' and absl::Span must define the
// same number of elements for the array. Otherwise, an error is reported.
template <typename T>
absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type,
                                             absl::Span<const T> values) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
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
    XLS_ASSIGN_OR_RETURN(
        xls_array[i], ConvertToXlsValue(array_type->element_type(), values[i]));
  }
  return Value::Array(xls_array);
}

template <typename T>
absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type,
                                             const std::vector<T>& values) {
  return ConvertToXlsValue(type, absl::MakeSpan(values));
}

namespace internal {

// Base case for tuple conversion: all elements have been converted.
absl::Status ConvertTupleElements(const TupleType* type, int64_t index,
                                  std::vector<Value>& xls_tuple);

// Recursive cases for tuple conversion.
// Scenario where the value is of type xls::Value, thus no conversion is
// required.
template <typename... ValuesType>
absl::Status ConvertTupleElements(const TupleType* type, int64_t index,
                                  std::vector<Value>& xls_tuple,
                                  xls::Value value, ValuesType&... values) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (index >= xls_tuple.size()) {
    // When the user is using the ConvertToXlsValue(..., std::tuple...), the
    // following error should not occur.
    return absl::InvalidArgumentError(absl::StrFormat(
        "The current element index exceeds the element count "
        "in the tuple. The current element index cannot exceed %d, got %d.",
        xls_tuple.size() - 1, index));
  }
  Type* element_type = type->element_type(index);
  if (!ValueConformsToType(value, element_type)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Type mismatch for a xls::Value. Expected type (%s), got value (%s).",
        element_type->ToString(), value.ToString()));
  }
  xls_tuple[index] = value;
  return ConvertTupleElements(type, index + 1, xls_tuple, values...);
}
// Scenario where at least one element exist for conversion.
template <typename ValueType, typename... ValuesType>
absl::Status ConvertTupleElements(const TupleType* type, int64_t index,
                                  std::vector<Value>& xls_tuple,
                                  ValueType& value, ValuesType&... values) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (index >= xls_tuple.size()) {
    // When the user is using the ConvertToXlsValue(..., std::tuple...), the
    // following error should not occur.
    return absl::InvalidArgumentError(absl::StrFormat(
        "The current element index exceeds the element count "
        "in the tuple. The current element index cannot exceed %d, got %d.",
        xls_tuple.size() - 1, index));
  }
  XLS_ASSIGN_OR_RETURN(xls_tuple[index], xls::ConvertToXlsValue(
                                             type->element_type(index), value));
  return ConvertTupleElements(type, index + 1, xls_tuple, values...);
}

// Unpack/Flatten the tuple to perform conversion on each element.
template <typename... ValuesType, size_t... I>
absl::Status UnpackTuple(const TupleType* type, std::vector<Value>& xls_tuple,
                         const std::tuple<ValuesType...>& tuple,
                         std::index_sequence<I...>) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  return ConvertTupleElements(type, 0, xls_tuple, std::get<I>(tuple)...);
}

}  // namespace internal

// Convert the std::tuple 'tuple' from the C++ domain to the IR domain
// (xls::Value) using 'type' as the conversion type to the IR domain. Note that
// the function uses a recursive mechanism to that also converts the elements of
// the std::tuple. Please see comment above regarding conversion to the IR
// domain for more information.
//
// The 'type' must be of tuple type. The 'type' and std::tuple must define the
// same number of elements for the tuple. Otherwise, an error is reported.
template <typename... ValuesType>
absl::StatusOr<xls::Value> ConvertToXlsValue(
    const Type* type, const std::tuple<ValuesType...>& tuple) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
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
absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type,
                                             const std::tuple<>& tuple);

// Below are utilities for conversions from the IR domain (xls::Value) to a
// value represented in the C++ domain.
//
// The conversion are performed recursively. As a result, when converting from
// an array and a tuple, the elements within the array and tuple are also
// converted.
//
// To leverage the recursive mechanism when performing conversions for
// user-defined types such as Classes and Structs, the conversion function must
// have the following signature, where 'MyClassOrStruct' is a user-defined class
// or struct:
//
//        absl::Status ConvertFromXlsValue(const xls::Value& value,
//        MyClassOrStruct& my_object);
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
//
//        absl::Status ConvertFromXlsValue(const xls::Value& user_struct_value,
//                                   userspace::UserStruct& user_struct) {
//          return xls::ConvertFromXlsValue(user_struct_value,
//                                    std::tuple<int64_t&, bool&>(
//                                      user_struct.integer_value,
//                                      user_struct.boolean_value)
//                                    );
//
//      }  // namespace xls
//
// Note that the conversion function leverages the conversion for tuples.

namespace internal {
constexpr int64_t kBitPerByte = 8;
}  // namespace internal

// Convert the value from the IR domain (xls::Value) to a bool type in the C++
// domain.
//
// The function emits an error when:
//   1. the xls::Value is not of bits type, or,
//   2. the xls::Value does not fit in 1 bit.
absl::Status ConvertFromXlsValue(const xls::Value& value, bool& cpp_value);

// Convert the value from the IR domain (xls::Value) to a signed integral type
// defined by the template 'T' in the C++ domain.
//
// The function emits an error when:
//   1. the xls::Value is not of bits type, or,
//   2. the xls::Value does not fit in type T.
template <typename T, typename std::enable_if<
                          std::is_signed_v<T> && !std::is_floating_point_v<T> &&
                              !std::is_const_v<T>,
                          bool>::type = true>
absl::Status ConvertFromXlsValue(const xls::Value& value, T& cpp_value) {
  if (!value.IsBits()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid type conversion for signed integral input. A "
                        "signed integral value can only be converted using a "
                        "bits type. Expected type (bits), got value (%s).",
                        value.ToString()));
  }
  int64_t cpp_value_bit_count = sizeof(T) * internal::kBitPerByte;
  if (!value.bits().FitsInNBitsSigned(cpp_value_bit_count)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value does not fit in signed integral type. Value "
                        "requires %d bits, got %d.",
                        value.bits().bit_count(), cpp_value_bit_count));
  }
  XLS_ASSIGN_OR_RETURN(cpp_value, value.bits().ToInt64());
  return absl::OkStatus();
}

// Convert the value from the IR domain (xls::Value) to a unsigned integral type
// defined by the template 'T' in the C++ domain.
//
// The function emits an error when:
//   1. the xls::Value is not of bits type, or,
//   2. the xls::Value does not fit in type T.
template <typename T,
          typename std::enable_if<std::is_unsigned_v<T> && !std::is_const_v<T>,
                                  bool>::type = true>
absl::Status ConvertFromXlsValue(const xls::Value& value, T& cpp_value) {
  if (!value.IsBits()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for unsigned integral input. An unsigned "
        "integral value can only be converted using a bits type. Expected type "
        "(bits), got value (%s).",
        value.ToString()));
  }
  int64_t cpp_value_bit_count = sizeof(T) * internal::kBitPerByte;
  if (!value.bits().FitsInNBitsUnsigned(cpp_value_bit_count)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value does not fit in unsigned integral type. Value "
                        "requires %d bits, got %d.",
                        value.bits().bit_count(), cpp_value_bit_count));
  }
  XLS_ASSIGN_OR_RETURN(cpp_value, value.bits().ToUint64());
  return absl::OkStatus();
}

// Convert the value from the IR domain (xls::Value) to a std::vector<T> in
// the C++ domain, where 'T' is defined by the template 'T'. Note that the
// function uses a recursive mechanism that also converts the elements within
// the xls::Value. Please see comment above regarding conversion from the IR
// domain for more information.
//
// The function emits an error when:
//   1. the xls::Value is not of array type, or,
//   2. the recursive calls to ConvertFromXlsValue emit an error.
//
// Template type T needs to be default constructible to use the
// resize members function of std::vector.
template <typename T,
          typename std::enable_if<std::is_default_constructible_v<T>,
                                  bool>::type = true>
absl::Status ConvertFromXlsValue(const xls::Value& array,
                                 std::vector<T>& cpp_array) {
  if (!array.IsArray()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid type conversion for std::vector input. A "
                        "std::vector input can only be converted using an "
                        "array type. Expected type (array), got value (%s).",
                        array.ToString()));
  }
  cpp_array.resize(array.size());
  for (int64_t i = 0; i < array.size(); ++i) {
    XLS_RETURN_IF_ERROR(ConvertFromXlsValue(array.element(i), cpp_array[i]));
  }
  return absl::OkStatus();
}

namespace internal {
// Base case for tuple conversion: all elements have been converted.
absl::Status ConvertTupleElements(const xls::Value& tuple, int64_t index,
                                  int64_t num_elements);

// Recursive cases for tuple conversion.
// Sceanario where at least one element exist for conversion.
template <typename ValueType, typename... ValuesType>
absl::Status ConvertTupleElements(const xls::Value& tuple, int64_t index,
                                  int64_t num_elements, ValueType& value,
                                  ValuesType&... values) {
  if (index >= num_elements) {
    // When the user is using the ConvertFromXlsValue(..., std::tuple...), the
    // following error should not occur.
    return absl::InvalidArgumentError(absl::StrFormat(
        "The current element index exceeds the element count "
        "in the tuple. The current element index cannot exceed %d, got %d.",
        num_elements - 1, index));
  }
  XLS_RETURN_IF_ERROR(xls::ConvertFromXlsValue(tuple.element(index), value));
  return ConvertTupleElements(tuple, index + 1, num_elements, values...);
}

// Unpack/Flatten the tuple to perform conversion to each element within the
// tuple.
template <typename... ValuesType, size_t... I>
absl::Status UnpackTuple(const xls::Value& tuple,
                         std::tuple<ValuesType...>& cpp_tuple,
                         std::index_sequence<I...>) {
  constexpr size_t num_elements = std::tuple_size_v<std::tuple<ValuesType...>>;
  return ConvertTupleElements(tuple, 0, num_elements,
                              std::get<I>(cpp_tuple)...);
}
}  // namespace internal

// Convert the value from the IR domain (xls::Value) to a std::tuple<...> in
// the C++ domain. Note that the function uses a recursive mechanism that also
// converts the elements within the xls::Value. Please see comment above
// regarding conversion from the IR domain for more information.
//
// The function emits an error when:
//   1. the xls::Value is not of tuple type, or,
//   2. there is a size mismatch between the xls::Value and the std::tuple, or,
//   3. the recursive calls to ConvertFromXlsValue emit an error.
template <typename... ValuesType>
absl::Status ConvertFromXlsValue(const xls::Value& tuple,
                                 std::tuple<ValuesType...>&& cpp_tuple) {
  if (!tuple.IsTuple()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid type conversion for std::tuple input. A "
                        "std::tuple input can only be converted using an "
                        "tuple type. Expected type (tuple), got value (%s).",
                        tuple.ToString()));
  }
  constexpr size_t num_elements = std::tuple_size_v<std::tuple<ValuesType...>>;
  if (tuple.size() != num_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Tuple size mismatch between conversion type and "
                        "value. Expected (%d), got (%d).",
                        tuple.size(), num_elements));
  }
  XLS_RETURN_IF_ERROR(internal::UnpackTuple(
      tuple, cpp_tuple, std::index_sequence_for<ValuesType...>{}));
  return absl::OkStatus();
}

// Same function as the above function with an lvalue reference for the
// std::tuple.
template <typename... ValuesType>
absl::Status ConvertFromXlsValue(const xls::Value& tuple,
                                 std::tuple<ValuesType...>& cpp_tuple) {
  if (!tuple.IsTuple()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid type conversion for std::tuple input. A "
                        "std::tuple input can only be converted using an "
                        "tuple type. Expected type (tuple), got value (%s).",
                        tuple.ToString()));
  }
  constexpr size_t num_elements = std::tuple_size_v<std::tuple<ValuesType...>>;
  if (tuple.size() != num_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Tuple size mismatch between conversion type and "
                        "value. Expected (%d), got (%d).",
                        tuple.size(), num_elements));
  }
  XLS_RETURN_IF_ERROR(internal::UnpackTuple(
      tuple, cpp_tuple, std::index_sequence_for<ValuesType...>{}));
  return absl::OkStatus();
}

}  // namespace xls

#endif  // XLS_IR_VALUE_CONVERSION_UTILS_H_

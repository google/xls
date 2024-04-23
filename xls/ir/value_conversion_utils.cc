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

#include "xls/ir/value_conversion_utils.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

namespace internal {
absl::StatusOr<xls::Value> ConvertInt64(const Type* type, int64_t value) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (!type->IsBits()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for integral input value (%d). "
        "An integral value can only be converted using a bits type. "
        "Expected type (bits), got (%s).",
        value, type->ToString()));
  }
  XLS_ASSIGN_OR_RETURN(
      Bits bit_value, SBitsWithStatus(value, type->AsBitsOrDie()->bit_count()));
  return Value(bit_value);
}

absl::StatusOr<xls::Value> ConvertUint64(const Type* type, uint64_t value) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (!type->IsBits()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for integral input value (%d). "
        "An integral value can only be converted using a bits type. "
        "Expected type (bits), got (%s).",
        value, type->ToString()));
  }
  XLS_ASSIGN_OR_RETURN(
      Bits bit_value, UBitsWithStatus(value, type->AsBitsOrDie()->bit_count()));
  return Value(bit_value);
}
}  // namespace internal

absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type,
                                             const std::tuple<>& tuple) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (!type->IsTuple()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for a std::tuple. An std::tuple can only be "
        "converted using an tuple type. Expected type (tuple), got (%s).",
        type->ToString()));
  }
  return Value::Tuple({});
}

namespace internal {

absl::Status ConvertTupleElements(const TupleType* type, int64_t index,
                                  std::vector<Value>& xls_tuple) {
  CHECK_NE(type, nullptr) << "Type cannot be a nullptr.";
  if (index < xls_tuple.size()) {
    // When the user is using the ConvertToXlsValue(..., std::tuple...), the
    // following error should not occur.
    return absl::InvalidArgumentError(absl::StrFormat(
        "Insufficient tuple elements to convert. Expected (%d), got (%d).",
        xls_tuple.size(), index));
  }
  return absl::OkStatus();
}

absl::Status ConvertTupleElements(const xls::Value& tuple, int64_t index,
                                  int64_t num_elements) {
  if (index < num_elements) {
    // When the user is using the ConvertFromXlsValue(..., std::tuple...), the
    // following error should not occur.
    return absl::InvalidArgumentError(absl::StrFormat(
        "Insufficient tuple elements to convert. Expected (%d), got (%d).",
        num_elements, index));
  }
  return absl::OkStatus();
}
}  // namespace internal

absl::Status ConvertFromXlsValue(const xls::Value& value, bool& cpp_value) {
  if (!value.IsBits()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid type conversion for bool input. A bool value can only be "
        "converted using a bits type. Expected type (bits), got value (%s).",
        value.ToString()));
  }
  if (!value.bits().FitsInNBitsUnsigned(1)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Value does not fit in bool type. Value requires %d bits, got 1.",
        value.bits().bit_count()));
  }
  XLS_ASSIGN_OR_RETURN(cpp_value, value.bits().ToUint64());
  return absl::OkStatus();
}

}  // namespace xls

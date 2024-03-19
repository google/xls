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

#include "xls/ir/value.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bit_push_buffer.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/xls_value.pb.h"

namespace xls {

/* static */ absl::StatusOr<Value> Value::Array(
    absl::Span<const Value> elements) {
  if (elements.empty()) {
    return absl::UnimplementedError("Empty array Values are not supported.");
  }

  // Blow up if they're not all of the same type.
  for (int64_t i = 1; i < elements.size(); ++i) {
    XLS_RET_CHECK(elements[0].SameTypeAs(elements[i]));
  }
  return Value(ValueKind::kArray, elements);
}

/* static */ absl::StatusOr<Value> Value::UBitsArray(
    absl::Span<const uint64_t> elements, int64_t bit_count) {
  if (elements.empty()) {
    return absl::UnimplementedError("Empty array Values are not supported.");
  }

  std::vector<Value> elements_as_values;
  for (auto x : elements) {
    XLS_ASSIGN_OR_RETURN(Bits bits, UBitsWithStatus(x, bit_count));
    elements_as_values.push_back(Value(bits));
  }

  return Value::Array(elements_as_values);
}

/* static */ absl::StatusOr<Value> Value::UBits2DArray(
    absl::Span<const absl::Span<const uint64_t>> elements, int64_t bit_count) {
  if (elements.empty()) {
    return absl::UnimplementedError("Empty array Values are not supported.");
  }

  std::vector<Value> elements_as_values;
  int64_t element_size = -1;
  for (auto x : elements) {
    XLS_ASSIGN_OR_RETURN(Value x_as_value, UBitsArray(x, bit_count));

    if (element_size == -1) {
      element_size = x_as_value.size();
    } else if (element_size != x_as_value.size()) {
      return absl::InternalError(
          "UBitsArray - elements of arrays should have consistent size.");
    }

    elements_as_values.push_back(x_as_value);
  }

  return Value::Array(elements_as_values);
}

/* static */ absl::StatusOr<Value> Value::SBitsArray(
    absl::Span<const int64_t> elements, int64_t bit_count) {
  if (elements.empty()) {
    return absl::UnimplementedError("Empty array Values are not supported.");
  }

  std::vector<Value> elements_as_values;
  for (auto x : elements) {
    XLS_ASSIGN_OR_RETURN(Bits bits, SBitsWithStatus(x, bit_count));
    elements_as_values.push_back(Value(bits));
  }

  return Value::Array(elements_as_values);
}

/* static */ absl::StatusOr<Value> Value::SBits2DArray(
    absl::Span<const absl::Span<const int64_t>> elements, int64_t bit_count) {
  if (elements.empty()) {
    return absl::UnimplementedError("Empty array Values are not supported.");
  }

  std::vector<Value> elements_as_values;
  int64_t element_size = -1;

  for (auto x : elements) {
    XLS_ASSIGN_OR_RETURN(Value x_as_value, SBitsArray(x, bit_count));

    if (element_size == -1) {
      element_size = x_as_value.size();
    } else if (element_size != x_as_value.size()) {
      return absl::InternalError(
          "SBitsArray - elements of arrays should have consistent size.");
    }

    elements_as_values.push_back(x_as_value);
  }

  return Value::Array(elements_as_values);
}

std::string ValueKindToString(ValueKind kind) {
  switch (kind) {
    case ValueKind::kInvalid:
      return "invalid";
    case ValueKind::kBits:
      return "bits";
    case ValueKind::kTuple:
      return "tuple";
    case ValueKind::kArray:
      return "array";
    case ValueKind::kToken:
      return "token";
    default:
      return absl::StrCat("<invalid ValueKind(", static_cast<int>(kind), ")>");
  }
}

int64_t Value::GetFlatBitCount() const {
  if (kind() == ValueKind::kBits) {
    return bits().bit_count();
  }
  if (kind() == ValueKind::kToken) {
    return 0;
  }
  if (kind() == ValueKind::kTuple) {
    int64_t total_size = 0;
    for (const Value& e : elements()) {
      total_size += e.GetFlatBitCount();
    }
    return total_size;
  }
  if (kind() == ValueKind::kArray) {
    if (empty()) {
      return 0;
    }
    return size() * element(0).GetFlatBitCount();
  }
  LOG(FATAL) << "Invalid value kind: " << kind();
}

bool Value::IsAllZeros() const {
  if (kind() == ValueKind::kBits) {
    return bits().IsZero();
  }
  if (kind() == ValueKind::kTuple || kind() == ValueKind::kArray) {
    for (const Value& e : elements()) {
      if (!e.IsAllZeros()) {
        return false;
      }
    }
  } else {
    LOG(FATAL) << "Invalid value kind: " << kind();
  }
  return true;
}

bool Value::IsAllOnes() const {
  if (kind() == ValueKind::kBits) {
    return bits().IsAllOnes();
  }
  if (kind() == ValueKind::kTuple || kind() == ValueKind::kArray) {
    for (const Value& e : elements()) {
      if (!e.IsAllOnes()) {
        return false;
      }
    }
  } else {
    LOG(FATAL) << "Invalid value kind: " << kind();
  }
  return true;
}

std::string Value::ToString(FormatPreference preference) const {
  switch (kind_) {
    case ValueKind::kInvalid:
      return "<invalid value>";
    case ValueKind::kBits:
      return absl::StrFormat("bits[%d]:%s", bits().bit_count(),
                             BitsToString(bits(), preference));
    case ValueKind::kTuple:
      return absl::StrCat(
          "(",
          absl::StrJoin(elements(), ", ",
                        [&](std::string* out, const Value& element) {
                          absl::StrAppend(out, element.ToString(preference));
                        }),
          ")");
    case ValueKind::kArray:
      return absl::StrCat(
          "[",
          absl::StrJoin(elements(), ", ",
                        [&](std::string* out, const Value& element) {
                          absl::StrAppend(out, element.ToString(preference));
                        }),
          "]");
    case ValueKind::kToken:
      return "token";
  }
  LOG(FATAL) << "Value has invalid kind: " << static_cast<int>(kind_);
}

void Value::FlattenTo(BitPushBuffer* buffer) const {
  switch (kind_) {
    case ValueKind::kBits:
      bits().FlattenTo(buffer);
      return;
    case ValueKind::kTuple:
    case ValueKind::kArray:
      for (const Value& element : elements()) {
        element.FlattenTo(buffer);
      }
      return;
    case ValueKind::kToken:
    case ValueKind::kInvalid:
      break;
  }
  LOG(FATAL) << "Invalid value kind: " << ValueKindToString(kind_);
}

absl::StatusOr<std::vector<Value>> Value::GetElements() const {
  if (!std::holds_alternative<std::vector<Value>>(payload_)) {
    return absl::InvalidArgumentError("Value does not hold elements.");
  }
  return std::vector<Value>(elements().begin(), elements().end());
}

absl::StatusOr<Bits> Value::GetBitsWithStatus() const {
  if (!IsBits()) {
    return absl::InvalidArgumentError(
        "Attempted to convert a non-Bits Value to Bits.");
  }
  return bits();
}

std::string Value::ToHumanString(FormatPreference preference) const {
  switch (kind_) {
    case ValueKind::kInvalid:
      return "<invalid value>";
    case ValueKind::kBits:
      return BitsToString(bits(), preference);
    case ValueKind::kArray:
      return absl::StrCat("[",
                          absl::StrJoin(elements(), ", ",
                                        [&](std::string* out, const Value& v) {
                                          absl::StrAppend(
                                              out, v.ToHumanString(preference));
                                        }),
                          "]");
    case ValueKind::kTuple:
      return absl::StrCat("(",
                          absl::StrJoin(elements(), ", ",
                                        [&](std::string* out, const Value& v) {
                                          return absl::StrAppend(
                                              out, v.ToHumanString(preference));
                                        }),
                          ")");
    case ValueKind::kToken:
      return "token";
  }
  LOG(FATAL) << "Invalid value kind: " << ValueKindToString(kind_);
}

bool Value::SameTypeAs(const Value& other) const {
  if (kind() != other.kind()) {
    return false;
  }
  switch (kind()) {
    case ValueKind::kBits:
      return bits().bit_count() == other.bits().bit_count();
    case ValueKind::kTuple:
      if (size() != other.size()) {
        return false;
      }
      for (int64_t i = 0; i < size(); ++i) {
        if (!element(i).SameTypeAs(other.element(i))) {
          return false;
        }
      }
      return true;
    case ValueKind::kArray: {
      return size() == other.size() && element(0).SameTypeAs(other.element(0));
    }
    case ValueKind::kToken:
      return true;
    case ValueKind::kInvalid:
      break;
  }
  LOG(FATAL) << "Invalid value encountered: " << ValueKindToString(kind());
}

absl::StatusOr<ValueProto> Value::AsProto() const {
  ValueProto v;
  switch (kind()) {
    case ValueKind::kBits: {
      ValueProto::Bits* proto_bits = v.mutable_bits();
      proto_bits->set_bit_count(bits().bit_count());
      std::vector<uint8_t> bytes = bits().ToBytes();
      proto_bits->mutable_data()->resize(bytes.size());
      memcpy(proto_bits->mutable_data()->data(), bytes.data(), bytes.size());
      break;
    }
    case ValueKind::kTuple: {
      ValueProto::Tuple* tuple = v.mutable_tuple();
      for (const Value& e : elements()) {
        XLS_ASSIGN_OR_RETURN(*tuple->add_elements(), e.AsProto());
      }
      break;
    }
    case ValueKind::kArray: {
      ValueProto::Array* array = v.mutable_array();
      for (const Value& e : elements()) {
        XLS_ASSIGN_OR_RETURN(*array->add_elements(), e.AsProto());
      }
      break;
    }
    case ValueKind::kToken:
      v.mutable_token();
      break;
    case ValueKind::kInvalid:
      return absl::InternalError("Cannot serialize an invalid value");
  }
  return v;
}

namespace {
absl::Span<const uint8_t> StrToU8Span(const std::string& s) {
  return absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()),
                                   s.size());
}
}  // namespace

absl::StatusOr<Value> Value::FromProto(const ValueProto& proto,
                                       int64_t max_bit_size) {
  switch (proto.variant_case()) {
    case ValueProto::kBits: {
      if (proto.bits().bit_count() <= 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Bit count must be at least 1 but was %d",
                            proto.bits().bit_count()));
      }
      if (proto.bits().bit_count() >= max_bit_size) {
        return absl::InternalError(absl::StrFormat(
            "Bit value is too large with %d bits", proto.bits().bit_count()));
      }
      std::string data_copy;
      absl::Span<const uint8_t> data;
      int64_t byte_size = CeilOfRatio(proto.bits().bit_count(), int64_t{8});
      if (byte_size == proto.bits().data().size()) {
        data = StrToU8Span(proto.bits().data());
      } else {
        data_copy = proto.bits().data();
        data_copy.resize(byte_size, '\0');
        data = StrToU8Span(data_copy);
      }
      return Value(Bits::FromBytes(data, proto.bits().bit_count()));
    }
    case ValueProto::kTuple: {
      std::vector<Value> values;
      values.reserve(proto.tuple().elements_size());
      for (const ValueProto& e : proto.tuple().elements()) {
        XLS_ASSIGN_OR_RETURN(Value element, Value::FromProto(e));
        values.push_back(element);
      }
      return Value::TupleOwned(std::move(values));
    }
    case ValueProto::kToken: {
      return Value::Token();
    }
    case ValueProto::kArray: {
      if (proto.array().elements_size() == 0) {
        return absl::InvalidArgumentError("empty arrays are not supported.");
      }
      std::vector<Value> values;
      values.reserve(proto.array().elements_size());
      for (const ValueProto& e : proto.array().elements()) {
        XLS_ASSIGN_OR_RETURN(Value element, Value::FromProto(e));
        values.push_back(element);
        if (values.size() != 1 && !values.front().SameTypeAs(values.back())) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Multiple different value types in array. Value %s does not "
              "match %s",
              values.back().ToString(), values.front().ToString()));
        }
      }
      return Value::ArrayOwned(std::move(values));
    }
    case ValueProto::VARIANT_NOT_SET:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unpopulated value proto: %v", proto.DebugString()));
  }
}

absl::StatusOr<TypeProto> Value::TypeAsProto() const {
  TypeProto proto;
  switch (kind()) {
    case ValueKind::kBits:
      proto.set_type_enum(TypeProto::BITS);
      proto.set_bit_count(bits().bit_count());
      break;
    case ValueKind::kTuple:
      proto.set_type_enum(TypeProto::TUPLE);
      for (const Value& elem : elements()) {
        XLS_ASSIGN_OR_RETURN(*proto.add_tuple_elements(), elem.TypeAsProto());
      }
      break;
    case ValueKind::kArray: {
      if (elements().empty()) {
        return absl::InternalError(
            "Cannot determine type of empty array value");
      }
      proto.set_type_enum(TypeProto::ARRAY);
      proto.set_array_size(size());
      XLS_ASSIGN_OR_RETURN(*proto.mutable_array_element(),
                           elements().front().TypeAsProto());
      break;
    }
    case ValueKind::kToken:
      proto.set_type_enum(TypeProto::TOKEN);
      break;
    case ValueKind::kInvalid:
      return absl::InternalError(absl::StrCat("Invalid value kind: ", kind()));
  }
  return proto;
}

bool Value::operator==(const Value& other) const {
  if (kind() != other.kind()) {
    return false;
  }

  if (IsBits()) {
    return bits() == other.bits();
  }

  // All non-Bits types are container types -- should have a size attribute.
  if (size() != other.size()) {
    return false;
  }

  return absl::c_equal(elements(), other.elements());
}

}  // namespace xls

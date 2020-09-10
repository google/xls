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

#include "xls/ir/value.h"

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "re2/re2.h"

namespace xls {

/* static */ xabsl::StatusOr<Value> Value::Array(
    absl::Span<const Value> elements) {
  if (elements.empty()) {
    return absl::UnimplementedError("Empty array Values are not supported.");
  }

  // Blow up if they're not all of the same type.
  for (int64 i = 1; i < elements.size(); ++i) {
    XLS_RET_CHECK(elements[0].SameTypeAs(elements[i]));
  }
  return Value(ValueKind::kArray, elements);
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

int64 Value::GetFlatBitCount() const {
  if (kind() == ValueKind::kBits) {
    return bits().bit_count();
  } else if (kind() == ValueKind::kTuple) {
    int64 total_size = 0;
    for (const Value& e : elements()) {
      total_size += e.GetFlatBitCount();
    }
    return total_size;
  } else if (kind() == ValueKind::kArray) {
    if (empty()) {
      return 0;
    }
    return size() * element(0).GetFlatBitCount();
  } else {
    XLS_LOG(FATAL) << "Invalid value kind: " << kind();
  }
}

bool Value::IsAllZeros() const {
  if (kind() == ValueKind::kBits) {
    return bits().IsAllZeros();
  } else if (kind() == ValueKind::kTuple || kind() == ValueKind::kArray) {
    for (const Value& e : elements()) {
      if (!e.IsAllZeros()) {
        return false;
      }
    }
  } else {
    XLS_LOG(FATAL) << "Invalid value kind: " << kind();
  }
  return true;
}

bool Value::IsAllOnes() const {
  if (kind() == ValueKind::kBits) {
    return bits().IsAllOnes();
  } else if (kind() == ValueKind::kTuple || kind() == ValueKind::kArray) {
    for (const Value& e : elements()) {
      if (!e.IsAllOnes()) {
        return false;
      }
    }
  } else {
    XLS_LOG(FATAL) << "Invalid value kind: " << kind();
  }
  return true;
}

std::string Value::ToString(FormatPreference preference) const {
  switch (kind_) {
    case ValueKind::kInvalid:
      return "<invalid value>";
    case ValueKind::kBits:
      return absl::StrFormat("bits[%d]:%s", bits().bit_count(),
                             bits().ToString(preference));
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
  XLS_LOG(FATAL) << "Value has invalid kind: " << static_cast<int>(kind_);
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
  XLS_LOG(FATAL) << "Invalid value kind: " << ValueKindToString(kind_);
}

xabsl::StatusOr<std::vector<Value>> Value::GetElements() const {
  if (!absl::holds_alternative<std::vector<Value>>(payload_)) {
    return absl::InvalidArgumentError("Value does not hold elements.");
  }
  return std::vector<Value>(elements().begin(), elements().end());
}

xabsl::StatusOr<Bits> Value::GetBitsWithStatus() const {
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
      return bits().ToString(preference);
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
  XLS_LOG(FATAL) << "Invalid value kind: " << ValueKindToString(kind_);
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
      for (int64 i = 0; i < size(); ++i) {
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
  XLS_LOG(FATAL) << "Invalid value encountered: " << ValueKindToString(kind());
}

xabsl::StatusOr<TypeProto> Value::TypeAsProto() const {
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

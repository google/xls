// Copyright 2025 The XLS Authors
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

#include "xls/ir/value_flattening.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

// Gathers the Bits objects at the leaves of the Value.
static void GatherValueLeaves(const Value& value, std::vector<Bits>* leaves) {
  switch (value.kind()) {
    case ValueKind::kBits:
      leaves->push_back(value.bits());
      break;
    case ValueKind::kTuple:
      for (const Value& e : value.elements()) {
        GatherValueLeaves(e, leaves);
      }
      break;
    case ValueKind::kArray:
      for (int64_t i = value.size() - 1; i >= 0; --i) {
        GatherValueLeaves(value.element(i), leaves);
      }
      break;
    default:
      LOG(FATAL) << "Invalid value kind: " << value.kind();
  }
}

Bits FlattenValueToBits(const Value& value) {
  std::vector<Bits> leaves;
  GatherValueLeaves(value, &leaves);
  return bits_ops::Concat(leaves);
}

absl::StatusOr<Value> UnflattenBitsToValue(const Bits& bits, const Type* type) {
  if (bits.bit_count() != type->GetFlatBitCount()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot unflatten input. Has %d bits, expected %d bits",
                        bits.bit_count(), type->GetFlatBitCount()));
  }
  if (type->IsBits()) {
    return Value(bits);
  }
  if (type->IsTuple()) {
    std::vector<Value> elements;
    const TupleType* tuple_type = type->AsTupleOrDie();
    for (int64_t i = 0; i < tuple_type->size(); ++i) {
      Type* element_type = tuple_type->element_type(i);
      XLS_ASSIGN_OR_RETURN(
          Value element, UnflattenBitsToValue(
                             bits.Slice(GetFlatBitIndexOfElement(tuple_type, i),
                                        element_type->GetFlatBitCount()),
                             element_type));
      elements.push_back(element);
    }
    return Value::Tuple(elements);
  }
  if (type->IsArray()) {
    std::vector<Value> elements;
    const ArrayType* array_type = type->AsArrayOrDie();
    for (int64_t i = 0; i < array_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Value element,
          UnflattenBitsToValue(
              bits.Slice(GetFlatBitIndexOfElement(array_type, i),
                         array_type->element_type()->GetFlatBitCount()),
              array_type->element_type()));
      elements.push_back(element);
    }
    return Value::Array(elements);
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid type: %s", type->ToString()));
}

absl::StatusOr<Value> UnflattenBitsToValue(const Bits& bits,
                                           const TypeProto& type_proto) {
  // Create a dummy package for converting  a TypeProto into a Type*.
  Package p("unflatten_dummy");
  XLS_ASSIGN_OR_RETURN(Type * type, p.GetTypeFromProto(type_proto));
  return UnflattenBitsToValue(bits, type);
}

int64_t GetFlatBitIndexOfElement(const TupleType* tuple_type, int64_t index) {
  CHECK_GE(index, 0);
  CHECK_LT(index, tuple_type->size());
  int64_t flat_index = 0;
  for (int64_t i = tuple_type->size() - 1; i > index; --i) {
    flat_index += tuple_type->element_type(i)->GetFlatBitCount();
  }
  return flat_index;
}

int64_t GetFlatBitIndexOfElement(const ArrayType* array_type, int64_t index) {
  CHECK_GE(index, 0);
  CHECK_LT(index, array_type->size());
  return index * array_type->element_type()->GetFlatBitCount();
}

}  // namespace xls

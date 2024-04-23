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

#include "xls/codegen/flattening.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/package.h"

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

// Recursive helper for Unflatten functions.
static verilog::Expression* UnflattenArrayHelper(
    int64_t flat_index_offset, verilog::IndexableExpression* input,
    ArrayType* array_type, verilog::VerilogFile* file, const SourceInfo& loc) {
  std::vector<verilog::Expression*> elements;
  const int64_t element_width = array_type->element_type()->GetFlatBitCount();
  for (int64_t i = 0; i < array_type->size(); ++i) {
    const int64_t element_start =
        flat_index_offset + GetFlatBitIndexOfElement(array_type, i);
    if (array_type->element_type()->IsArray()) {
      elements.push_back(UnflattenArrayHelper(
          element_start, input, array_type->element_type()->AsArrayOrDie(),
          file, loc));
    } else {
      elements.push_back(file->Slice(input, element_start + element_width - 1,
                                     element_start, loc));
    }
  }
  return file->ArrayAssignmentPattern(elements, loc);
}

verilog::Expression* UnflattenArray(verilog::IndexableExpression* input,
                                    ArrayType* array_type,
                                    verilog::VerilogFile* file,
                                    const SourceInfo& loc) {
  return UnflattenArrayHelper(/*flat_index_offset=*/0, input, array_type, file,
                              loc);
}

verilog::Expression* UnflattenArrayShapedTupleElement(
    verilog::IndexableExpression* input, TupleType* tuple_type,
    int64_t tuple_index, verilog::VerilogFile* file, const SourceInfo& loc) {
  CHECK(tuple_type->element_type(tuple_index)->IsArray());
  ArrayType* array_type = tuple_type->element_type(tuple_index)->AsArrayOrDie();
  return UnflattenArrayHelper(
      /*flat_index_offset=*/GetFlatBitIndexOfElement(tuple_type, tuple_index),
      input, array_type, file, loc);
}

verilog::Expression* FlattenArray(verilog::IndexableExpression* input,
                                  ArrayType* array_type,
                                  verilog::VerilogFile* file,
                                  const SourceInfo& loc) {
  std::vector<verilog::Expression*> elements;
  for (int64_t i = array_type->size() - 1; i >= 0; --i) {
    verilog::IndexableExpression* element = file->Index(input, i, loc);
    if (array_type->element_type()->IsArray()) {
      elements.push_back(FlattenArray(
          element, array_type->element_type()->AsArrayOrDie(), file, loc));
    } else {
      elements.push_back(element);
    }
  }
  return file->Concat(elements, loc);
}

absl::StatusOr<verilog::Expression*> FlattenTuple(
    absl::Span<verilog::Expression* const> inputs, TupleType* tuple_type,
    verilog::VerilogFile* file, const SourceInfo& loc) {
  // Tuples are represented as a flat vector of bits. Flatten and concatenate
  // all operands. Only non-zero-width elements of the tuple are represented in
  // inputs.
  std::vector<Type*> nontrivial_element_types;
  for (Type* type : tuple_type->element_types()) {
    if (type->GetFlatBitCount() != 0) {
      nontrivial_element_types.push_back(type);
    }
  }
  XLS_RET_CHECK_EQ(nontrivial_element_types.size(), inputs.size());
  std::vector<verilog::Expression*> flattened_elements;
  for (int64_t i = 0; i < inputs.size(); ++i) {
    verilog::Expression* element = inputs[i];
    Type* element_type = nontrivial_element_types[i];
    if (element_type->IsArray()) {
      flattened_elements.push_back(
          FlattenArray(element->AsIndexableExpressionOrDie(),
                       element_type->AsArrayOrDie(), file, loc));
    } else {
      flattened_elements.push_back(element);
    }
  }
  return file->Concat(flattened_elements, loc);
}

}  // namespace xls

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

#include "xls/codegen/expression_flattening.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/ir/value_flattening.h"

namespace xls {

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

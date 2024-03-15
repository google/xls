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

// Utility binary to convert input XLS IR to SMTLIB2.
// Adds the handy option of converting the XLS IR into a "fundamental"
// representation, i.e., consisting of only AND/OR/NOT ops.
#include "xls/tools/booleanifier.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/topo_sort.h"

namespace xls {

// Evaluator for converting Nodes representing high-level Ops into
// single-bit AND/OR/NOT-based ones.
class BitEvaluator : public AbstractEvaluator<Node*, BitEvaluator> {
 public:
  explicit BitEvaluator(FunctionBuilder* builder)
      : builder_(builder),
        one_(builder->Literal(UBits(1, 1))),
        zero_(builder->Literal(UBits(0, 1))) {}

  Node* One() const { return one_.node(); }
  Node* Zero() const { return zero_.node(); }
  Node* Not(Node* const& input) const {
    return builder_->Not(BValue(input, builder_)).node();
  }
  Node* And(Node* const& a, Node* const& b) const {
    return builder_->And(BValue(a, builder_), BValue(b, builder_)).node();
  }
  Node* Or(Node* const& a, Node* const& b) const {
    return builder_->Or(BValue(a, builder_), BValue(b, builder_)).node();
  }

 private:
  FunctionBuilder* builder_;
  BValue one_;
  BValue zero_;
};

absl::StatusOr<Function*> Booleanifier::Booleanify(
    Function* f, std::string_view boolean_function_name) {
  Booleanifier b(f, boolean_function_name);
  return b.Run();
}

Booleanifier::Booleanifier(Function* f, std::string_view boolean_function_name)
    : input_fn_(f),
      builder_(boolean_function_name.empty()
                   ? absl::StrCat(input_fn_->name(), "_boolean")
                   : boolean_function_name,
               input_fn_->package()),
      evaluator_(std::make_unique<BitEvaluator>(&builder_)) {}

absl::StatusOr<Function*> Booleanifier::Run() {
  for (const Param* param : input_fn_->params()) {
    params_[param->name()] = builder_.Param(param->name(), param->GetType());
  }

  for (Node* node : TopoSort(input_fn_)) {
    std::vector<Vector> operands;
    // Not the most efficient way of doing this, but not an issue yet.
    for (Node* node : node->operands()) {
      operands.push_back(node_map_.at(node));
    }

    XLS_ASSIGN_OR_RETURN(
        Vector result,
        AbstractEvaluate(node, operands, evaluator_.get(), [this](Node* node) {
          return HandleSpecialOps(node);
        }));
    node_map_[node] = result;
  }

  Node* return_node = input_fn_->return_value();
  return builder_.BuildWithReturnValue(
      PackReturnValue(node_map_.at(return_node), return_node->GetType()));
}

Booleanifier::Vector Booleanifier::FlattenValue(const Value& value) {
  if (value.IsBits()) {
    return evaluator_->BitsToVector(value.bits());
  }

  Vector result;
  for (const auto& element : value.elements()) {
    Vector element_result = FlattenValue(element);
    result.insert(result.end(), element_result.begin(), element_result.end());
  }
  return result;
}

Booleanifier::Vector Booleanifier::HandleLiteralArrayIndex(
    const ArrayType* array_type, const Vector& array, const Value& index,
    int64_t start_offset) {
  const int64_t element_size = array_type->element_type()->GetFlatBitCount();

  CHECK(index.IsBits());
  int64_t concrete_index =
      bits_ops::UGreaterThanOrEqual(index.bits(), array_type->size())
          ? array_type->size() - 1
          : index.bits().ToUint64().value();
  start_offset += concrete_index * element_size;
  return evaluator_->BitSlice(array, start_offset, element_size);
}

Booleanifier::Vector Booleanifier::HandleArrayIndex(
    const ArrayType* array_type, const Vector& array,
    absl::Span<Node* const> indices, int64_t start_offset) {
  Type* element_type = array_type->element_type();
  int64_t element_size = element_type->GetFlatBitCount();

  // If the indices array is empty, just return the original array from the
  // start_offset through the end of the array.
  if (indices.empty()) {
    return evaluator_->BitSlice(array, start_offset,
                                array.size() - start_offset);
  }

  // Optimization for a common case where an array is indexed via a literal.  In
  // this case we can emit the code directly without issuing a comparison check
  // on the index, at that level of the array.
  // Note that indices.size() == 1 is a check in a potentially recursive
  // invocation on this method--this would happen if this is the last index in a
  // sequence that indexes a multi-dimensional array.
  if (indices.size() == 1 && indices[0]->Is<Literal>()) {
    return HandleLiteralArrayIndex(
        array_type, array, indices[0]->As<Literal>()->value(), start_offset);
  }

  std::vector<Vector> cases;
  for (int i = 0; i < array_type->size(); i++) {
    if (element_type->IsArray() && indices.size() > 1) {
      cases.push_back(HandleArrayIndex(element_type->AsArrayOrDie(), array,
                                       indices.subspan(1),
                                       start_offset + i * element_size));
    } else {
      cases.push_back(evaluator_->BitSlice(
          array, start_offset + i * element_size, element_size));
    }
  }
  return evaluator_->Select(node_map_.at(indices[0]), cases, cases.back());
}

Booleanifier::Vector Booleanifier::HandleArrayUpdate(
    const ArrayType* array_type, const Vector& array,
    absl::Span<Node* const> indices, const Vector& update_value,
    int64_t start_offset) {
  // If the indices array is empty, then the update value is a new value for the
  // whole array.
  if (indices.empty()) {
    return update_value;
  }

  Vector result;

  const Type* element_type = array_type->element_type();
  const int64_t element_size = element_type->GetFlatBitCount();

  // If the outermost index is literal, then we can directly construct the
  // resulting array without emitting nodes to perform checks against the index,
  // at least for the part of the array before and after the update index.  If
  // the outermost index is not literal, then we must emit gates to perform the
  // checks.  Further, if the array is multidimensional and we have more than
  // one index in the array of indices, we call this function recurisively on
  // the subarray corresponding to the element at which we wish to perform the
  // update.
  if (indices[0]->Is<Literal>()) {
    const int64_t concrete_update_index =
        indices[0]->As<Literal>()->value().bits().ToUint64().value();
    // Out-of-bounds indices results in the original array being returned.
    if (concrete_update_index < 0 ||
        concrete_update_index >= array_type->size()) {
      return array;
    }
    // Splice in the elements before the updated value, if any.
    if (concrete_update_index > 0) {
      Vector before = evaluator_->BitSlice(
          array, start_offset, concrete_update_index * element_size);
      result.insert(result.end(), before.begin(), before.end());
    }
    // Splice in the updated value.
    if (element_type->IsArray()) {
      Vector updated_subarray = HandleArrayUpdate(
          element_type->AsArrayOrDie(), array, indices.subspan(1), update_value,
          start_offset + concrete_update_index * element_size);
      result.insert(result.end(), updated_subarray.begin(),
                    updated_subarray.end());
    } else {
      // If it's not a multi-dimensional array, then there must not be any
      // leftover indices in the array of indices.
      CHECK_EQ(indices.size(), 1);
      result.insert(result.end(), update_value.begin(), update_value.end());
    }

    // Splice in the elements after the updated value, if any
    if (concrete_update_index < array_type->size() - 1) {
      start_offset += (concrete_update_index + 1) * element_size;
      Vector after = evaluator_->BitSlice(array, start_offset,
                                          array.size() - start_offset);
      result.insert(result.end(), after.begin(), after.end());
    }
  } else {
    const Vector& update_index = node_map_.at(indices[0]);
    const int64_t index_width = update_index.size();
    for (int i = 0; i < array_type->size(); i++) {
      const Value loop_index(UBits(i, index_width));
      const Element equals_index =
          evaluator_->Equals(FlattenValue(loop_index), update_index);
      Vector old_value =
          HandleLiteralArrayIndex(array_type, array, loop_index, start_offset);
      const uint64_t concrete_update_index =
          loop_index.bits().ToUint64().value();
      Vector updated_subarray = update_value;
      if (element_type->IsArray()) {
        updated_subarray = HandleArrayUpdate(
            element_type->AsArrayOrDie(), array, indices.subspan(1),
            update_value, start_offset + concrete_update_index * element_size);
      } else {
        // If it's not a milti-dimensional array, then there must not be any
        // leftover indices in the array of indices.
        CHECK_EQ(indices.size(), 1);
      }
      const Vector new_value = evaluator_->Select(
          Vector({equals_index}), {old_value, updated_subarray});
      result.insert(result.end(), new_value.begin(), new_value.end());
    }
  }

  return result;
}

Booleanifier::Vector Booleanifier::HandleSpecialOps(Node* node) {
  switch (node->op()) {
    case Op::kArray:
    case Op::kTuple: {
      // Array and tuples are held as flat bit/Node arrays.
      Vector result;
      for (const Node* operand : node->operands()) {
        Vector& v = node_map_.at(operand);
        result.insert(result.end(), v.begin(), v.end());
      }
      return result;
    }
    case Op::kArrayIndex: {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      const ArrayType* array_type =
          array_index->array()->GetType()->AsArrayOrDie();
      std::vector<Vector> indices;
      for (const auto& index : array_index->indices()) {
        indices.push_back(node_map_.at(index));
      }
      return HandleArrayIndex(array_type, node_map_.at(array_index->array()),
                              array_index->indices(), /*start_offset=*/0);
    }
    case Op::kArrayUpdate: {
      // Use the old value for each element, except for the updated element.
      ArrayUpdate* array_update = node->As<ArrayUpdate>();
      return HandleArrayUpdate(array_update->GetType()->AsArrayOrDie(),
                               node_map_.at(array_update->array_to_update()),
                               array_update->indices(),
                               node_map_.at(array_update->update_value()),
                               /*start_offset=*/0);
    }
    case Op::kLiteral: {
      Vector result;
      Literal* literal = node->As<Literal>();
      return FlattenValue(literal->value());
    }
    case Op::kParam: {
      // Params are special, as they come in as n-bit objects. They're one of
      // the interfaces to the outside world that convert N-bit items into N
      // 1-bit items.
      Param* param = node->As<Param>();
      return UnpackParam(param->GetType(), params_.at(param->name()));
    }
    case Op::kTupleIndex: {
      // Tuples are flat vectors, so we just need to extract the right
      // offset/width.
      TupleIndex* tuple_index = node->As<TupleIndex>();
      TupleType* tuple_type = node->operand(0)->GetType()->AsTupleOrDie();
      int64_t start_bit = 0;
      for (int i = 0; i < tuple_index->index(); i++) {
        start_bit += tuple_type->element_type(i)->GetFlatBitCount();
      }
      int64_t width =
          tuple_type->element_type(tuple_index->index())->GetFlatBitCount();
      return evaluator_->BitSlice(node_map_.at(node->operand(0)), start_bit,
                                  width);
    }
    default:
      XLS_LOG(FATAL) << "Unsupported/unimplemented op: " << node->op();
  }
}

Booleanifier::Vector Booleanifier::UnpackParam(Type* type, BValue bv_node) {
  int64_t bit_count = type->GetFlatBitCount();
  switch (type->kind()) {
    case TypeKind::kBits: {
      Vector result;
      for (int i = 0; i < bit_count; i++) {
        BValue y = builder_.BitSlice(bv_node, i, 1);
        result.push_back(y.node());
      }
      return result;
    }
    case TypeKind::kArray: {
      Vector result;
      ArrayType* array_type = type->AsArrayOrDie();
      for (int i = 0; i < array_type->size(); i++) {
        std::vector<BValue> indices = {
            builder_.Literal(UBits(i, CeilOfLog2(array_type->size())))};
        BValue array_index = builder_.ArrayIndex(bv_node, indices);
        Vector element = UnpackParam(array_type->element_type(), array_index);
        result.insert(result.end(), element.begin(), element.end());
      }
      return result;
    }
    case TypeKind::kTuple: {
      Vector result;
      TupleType* tuple_type = type->AsTupleOrDie();
      for (int i = 0; i < tuple_type->size(); i++) {
        BValue tuple_index = builder_.TupleIndex(bv_node, i);
        Vector element = UnpackParam(tuple_type->element_type(i), tuple_index);
        result.insert(result.end(), element.begin(), element.end());
      }
      return result;
    }
    default:
      XLS_LOG(FATAL) << "Unsupported/unimplemened param kind: " << type->kind();
  }
}

// The inverse of UnpackParam - overlays structure on top of a flat bit array.
// We take a span here, instead of a Vector, so we can easily create subspans.
BValue Booleanifier::PackReturnValue(absl::Span<const Element> bits,
                                     const Type* type) {
  switch (type->kind()) {
    case TypeKind::kBits: {
      std::vector<BValue> result;
      for (const Element& bit : bits) {
        result.push_back(BValue(bit, &builder_));
      }
      // Need to reverse to match IR/Verilog Concat semantics.
      std::reverse(result.begin(), result.end());
      return builder_.Concat(result);
    }
    case TypeKind::kArray: {
      const ArrayType* array_type = type->AsArrayOrDie();
      Type* element_type = array_type->element_type();
      std::vector<BValue> elements;
      int64_t offset = 0;
      for (int i = 0; i < array_type->size(); i++) {
        absl::Span<const Element> element =
            bits.subspan(offset, element_type->GetFlatBitCount());
        elements.push_back(PackReturnValue(element, element_type));
        offset += element_type->GetFlatBitCount();
      }
      return builder_.Array(elements, element_type);
    }
    case TypeKind::kTuple: {
      const TupleType* tuple_type = type->AsTupleOrDie();
      std::vector<BValue> elements;
      int64_t offset = 0;
      for (const Type* elem_type : tuple_type->element_types()) {
        absl::Span<const Element> elem =
            bits.subspan(offset, elem_type->GetFlatBitCount());
        elements.push_back(PackReturnValue(elem, elem_type));
        offset += elem_type->GetFlatBitCount();
      }
      return builder_.Tuple(elements);
    }
    default:
      XLS_LOG(FATAL) << "Unsupported/unimplemented type kind: " << type->kind();
  }
}

}  // namespace xls

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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_flattening.h"

namespace xls {

// Accepts a bvalue that is coerced to a type specified by target_type. The
// coerced_type contains coercion information that specifies how the bvalue
// should be coerced to the target_type. The target_type should be created based
// off of the coerced_type. Additional helper functions, like CoercedBits, are
// provided if you already know that the bvalue is of a bits type.
BValue IrFuzzHelpers::Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
                              const CoercedTypeProto& coerced_type,
                              Type* target_type) const {
  switch (coerced_type.type_case()) {
    case CoercedTypeProto::kBits:
      return CoercedBits(p, fb, bvalue, coerced_type.bits(), target_type);
    case CoercedTypeProto::kTuple:
      return CoercedTuple(p, fb, bvalue, coerced_type.tuple(), target_type);
    case CoercedTypeProto::kArray:
      return CoercedArray(p, fb, bvalue, coerced_type.array(), target_type);
    default:
      return DefaultValue(p, fb, TypeCase::UNSET_CASE);
  }
}

BValue IrFuzzHelpers::CoercedBits(Package* p, FunctionBuilder* fb,
                                  BValue bvalue,
                                  const BitsCoercedTypeProto& coerced_type,
                                  Type* target_type) const {
  Type* bvalue_type = bvalue.GetType();
  // If the bvalue is already the specified type, return it as is.
  if (bvalue_type == target_type) {
    return bvalue;
  }
  // If the bvalue differs in categorical type to bits, simply return a default
  // value of the specified type.
  if (!bvalue_type->IsBits()) {
    return DefaultValueOfBitsType(p, fb, target_type);
  }
  BitsType* target_bits_type = target_type->AsBitsOrDie();
  auto coercion_method = coerced_type.coercion_method();
  // Change the bit width to the specified bit width.
  return ChangeBitWidth(fb, bvalue, target_bits_type->bit_count(),
                        coercion_method.change_bit_width_method());
}

BValue IrFuzzHelpers::CoercedTuple(Package* p, FunctionBuilder* fb,
                                   BValue bvalue,
                                   const TupleCoercedTypeProto& coerced_type,
                                   Type* target_type) const {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == target_type) {
    return bvalue;
  }
  if (!bvalue_type->IsTuple()) {
    return DefaultValueOfTupleType(p, fb, target_type);
  }
  TupleType* target_tuple_type = target_type->AsTupleOrDie();
  auto coercion_method = coerced_type.coercion_method();
  // Change the size of the tuple to match the specified size.
  bvalue = ChangeTupleSize(fb, bvalue, target_tuple_type->size(),
                           coercion_method.change_tuple_size_method());
  std::vector<BValue> coerced_elements;
  // Coerce each tuple element and create a new tuple with the coerced
  // elements.
  for (int64_t i = 0; i < target_tuple_type->size(); i += 1) {
    BValue element = fb->TupleIndex(bvalue, i);
    BValue coerced_element =
        Coerced(p, fb, element, coerced_type.tuple_elements(i),
                target_tuple_type->element_type(i));
    coerced_elements.push_back(coerced_element);
  }
  return fb->Tuple(coerced_elements);
}

BValue IrFuzzHelpers::CoercedArray(Package* p, FunctionBuilder* fb,
                                   BValue bvalue,
                                   const ArrayCoercedTypeProto& coerced_type,
                                   Type* target_type) const {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == target_type) {
    return bvalue;
  }
  if (!bvalue_type->IsArray()) {
    return DefaultValueOfArrayType(p, fb, target_type);
  }
  ArrayType* target_array_type = target_type->AsArrayOrDie();
  auto coercion_method = coerced_type.coercion_method();
  int64_t array_size = target_array_type->size();
  // Change the size of the array to match the specified size.
  bvalue = ChangeArraySize(p, fb, bvalue, array_size,
                           coercion_method.change_array_size_method());
  // If the array elements are already of the specified type, return it as is.
  if (bvalue_type->AsArrayOrDie()->element_type() ==
      target_array_type->element_type()) {
    return bvalue;
  }
  std::vector<BValue> coerced_elements;
  // Coerce each array element and create a new array with the coerced
  // elements.
  for (int64_t i = 0; i < array_size; i += 1) {
    BValue element = fb->ArrayIndex(bvalue, {fb->Literal(UBits(i, 64))}, true);
    BValue coerced_element =
        Coerced(p, fb, element, coerced_type.array_element(),
                target_array_type->element_type());
    coerced_elements.push_back(coerced_element);
  }
  return fb->Array(coerced_elements, target_array_type->element_type());
}

// Accepts a bvalue that is coerced to a type specified by target_type. The
// coercion_method contains coercion information that specifies how the bvalue
// should be coerced to the target_type. Fitted() is different from Coerced()
// because Fitted() uses coercion_method instead of coerced_type. Additional
// helper functions, like FittedBits, are provided if you already know that the
// bvalue is of a bits type.
BValue IrFuzzHelpers::Fitted(Package* p, FunctionBuilder* fb, BValue bvalue,
                             const CoercionMethodProto& coercion_method,
                             Type* target_type) const {
  if (target_type->IsBits()) {
    return FittedBits(p, fb, bvalue, coercion_method.bits(), target_type);
  } else if (target_type->IsTuple()) {
    return FittedTuple(p, fb, bvalue, coercion_method, target_type);
  } else if (target_type->IsArray()) {
    return FittedArray(p, fb, bvalue, coercion_method, target_type);
  } else {
    return DefaultValue(p, fb, TypeCase::UNSET_CASE);
  }
}

BValue IrFuzzHelpers::FittedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                                 const BitsCoercionMethodProto& coercion_method,
                                 Type* target_type) const {
  Type* bvalue_type = bvalue.GetType();
  // If the bvalue is already the specified type, return it as is.
  if (bvalue_type == target_type) {
    return bvalue;
  }
  // If the bvalue differs in categorical type to bits, simply return a default
  // value of the specified type.
  if (!bvalue_type->IsBits()) {
    return DefaultValueOfBitsType(p, fb, target_type);
  }
  BitsType* target_bits_type = target_type->AsBitsOrDie();
  return ChangeBitWidth(fb, bvalue, target_bits_type->bit_count(),
                        coercion_method.change_bit_width_method());
}

BValue IrFuzzHelpers::FittedTuple(Package* p, FunctionBuilder* fb,
                                  BValue bvalue,
                                  const CoercionMethodProto& coercion_method,
                                  Type* target_type) const {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == target_type) {
    return bvalue;
  }
  if (!bvalue_type->IsTuple()) {
    return DefaultValueOfTupleType(p, fb, target_type);
  }
  TupleType* target_tuple_type = target_type->AsTupleOrDie();
  auto tuple_coercion_method = coercion_method.tuple();
  // Change the size of the tuple to match the specified size.
  bvalue = ChangeTupleSize(fb, bvalue, target_tuple_type->size(),
                           tuple_coercion_method.change_tuple_size_method());
  std::vector<BValue> fitted_elements;
  // Fit each tuple element and create a new tuple with the fitted elements.
  for (int64_t i = 0; i < target_tuple_type->size(); i += 1) {
    BValue element = fb->TupleIndex(bvalue, i);
    BValue fitted_element = Fitted(p, fb, element, coercion_method,
                                   target_tuple_type->element_type(i));
    fitted_elements.push_back(fitted_element);
  }
  return fb->Tuple(fitted_elements);
}

BValue IrFuzzHelpers::FittedArray(Package* p, FunctionBuilder* fb,
                                  BValue bvalue,
                                  const CoercionMethodProto& coercion_method,
                                  Type* target_type) const {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == target_type) {
    return bvalue;
  }
  if (!bvalue_type->IsArray()) {
    return DefaultValueOfArrayType(p, fb, target_type);
  }
  ArrayType* target_array_type = target_type->AsArrayOrDie();
  auto array_coercion_method = coercion_method.array();
  // Change the size of the array to match the specified size.
  bvalue = ChangeArraySize(p, fb, bvalue, target_array_type->size(),
                           array_coercion_method.change_array_size_method());
  // If the array elements are already of the specified type, return it as is.
  if (bvalue_type->AsArrayOrDie()->element_type() ==
      target_array_type->element_type()) {
    return bvalue;
  }
  std::vector<BValue> fitted_elements;
  // Fit each array element and create a new array with the fitted elements.
  for (int64_t i = 0; i < target_array_type->size(); i += 1) {
    BValue element = fb->ArrayIndex(bvalue, {fb->Literal(UBits(i, 64))}, true);
    BValue fitted_element = Fitted(p, fb, element, coercion_method,
                                   target_array_type->element_type());
    fitted_elements.push_back(fitted_element);
  }
  return fb->Array(fitted_elements, target_array_type->element_type());
}

// Changes the bit width of an inputted BValue to a specified bit width.
// Multiple bit width modification methods can be used and specified.
BValue IrFuzzHelpers::ChangeBitWidth(
    FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
    const ChangeBitWidthMethodProto& change_bit_width_method) const {
  if (bvalue.BitCountOrDie() > new_bit_width) {
    // If the bvalue is larger than the bit width, use the decrease width method
    // to reduce the bit width.
    return DecreaseBitWidth(fb, bvalue, new_bit_width,
                            change_bit_width_method.decrease_width_method());
  } else if (bvalue.BitCountOrDie() < new_bit_width) {
    // If the bvalue is smaller than the bit width, use the increase width
    // method to increase the bit width.
    return IncreaseBitWidth(fb, bvalue, new_bit_width,
                            change_bit_width_method.increase_width_method());
  } else {
    // If the bvalue is the same bit width as the specified bit width, return
    // the bvalue as is because no change is needed.
    return bvalue;
  }
}

// Overloaded version of ChangeBitWidth that uses default width fitting methods.
BValue IrFuzzHelpers::ChangeBitWidth(FunctionBuilder* fb, BValue bvalue,
                                     int64_t bit_width) const {
  ChangeBitWidthMethodProto default_method;
  default_method.set_decrease_width_method(
      DecreaseWidthMethod::UNSET_DECREASE_WIDTH_METHOD);
  default_method.set_increase_width_method(
      IncreaseWidthMethod::UNSET_INCREASE_WIDTH_METHOD);
  return ChangeBitWidth(fb, bvalue, bit_width, default_method);
}  // namespace xls

BValue IrFuzzHelpers::DecreaseBitWidth(
    FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
    const DecreaseWidthMethod& decrease_width_method) const {
  switch (decrease_width_method) {
    case BIT_SLICE_METHOD:
    default:
      return fb->BitSlice(bvalue, /*start=*/0, new_bit_width);
  }
}

BValue IrFuzzHelpers::IncreaseBitWidth(
    FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
    const IncreaseWidthMethod& increase_width_method) const {
  switch (increase_width_method) {
    case SIGN_EXTEND_METHOD:
      return fb->SignExtend(bvalue, new_bit_width);
    case ZERO_EXTEND_METHOD:
    default:
      return fb->ZeroExtend(bvalue, new_bit_width);
  }
}

// Changes the size of a tuple to match the specified size.
BValue IrFuzzHelpers::ChangeTupleSize(
    FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const ChangeTupleSizeMethodProto& change_tuple_size_method) const {
  int64_t current_size = bvalue.GetType()->AsTupleOrDie()->size();
  if (current_size > new_size) {
    return DecreaseTupleSize(fb, bvalue, new_size,
                             change_tuple_size_method.decrease_size_method());
  } else if (current_size < new_size) {
    return IncreaseTupleSize(fb, bvalue, new_size,
                             change_tuple_size_method.increase_size_method());
  } else {
    return bvalue;
  }
}

BValue IrFuzzHelpers::DecreaseTupleSize(
    FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const DecreaseTupleSizeMethod& decrease_size_method) const {
  switch (decrease_size_method) {
    case DecreaseTupleSizeMethod::SHRINK_TUPLE_METHOD:
    default:
      // Create a new tuple with only some of the elements from the original
      // tuple.
      std::vector<BValue> elements;
      for (int64_t i = 0; i < new_size; i += 1) {
        elements.push_back(fb->TupleIndex(bvalue, i));
      }
      return fb->Tuple(elements);
  }
}

BValue IrFuzzHelpers::IncreaseTupleSize(
    FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const IncreaseTupleSizeMethod& increase_size_method) const {
  switch (increase_size_method) {
    case IncreaseTupleSizeMethod::EXPAND_TUPLE_METHOD:
    default:
      // Create a new tuple with the original elements and some default elements
      // to expand the tuple to the specified size.
      int64_t current_size = bvalue.GetType()->AsTupleOrDie()->size();
      std::vector<BValue> elements;
      for (int64_t i = 0; i < current_size; i += 1) {
        elements.push_back(fb->TupleIndex(bvalue, i));
      }
      while (elements.size() < new_size) {
        elements.push_back(DefaultBitsValue(fb));
      }
      return fb->Tuple(elements);
  }
}

// Changes the size of an array to match the specified size.
BValue IrFuzzHelpers::ChangeArraySize(
    Package* p, FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const ChangeArraySizeMethodProto& change_array_size_method) const {
  ArrayType* array_type = bvalue.GetType()->AsArrayOrDie();
  if (array_type->size() > new_size) {
    return DecreaseArraySize(fb, bvalue, new_size,
                             change_array_size_method.decrease_size_method());
  } else if (array_type->size() < new_size) {
    return IncreaseArraySize(p, fb, bvalue, new_size,
                             change_array_size_method.increase_size_method());
  } else {
    return bvalue;
  }
}

BValue IrFuzzHelpers::DecreaseArraySize(
    FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const DecreaseArraySizeMethod& decrease_size_method) const {
  ArrayType* array_type = bvalue.GetType()->AsArrayOrDie();
  switch (decrease_size_method) {
    case DecreaseArraySizeMethod::ARRAY_SLICE_METHOD:
      return fb->ArraySlice(bvalue, fb->Literal(UBits(0, 64)), new_size);
    case DecreaseArraySizeMethod::SHRINK_ARRAY_METHOD:
    default:
      // Create a new array with only some of the elements from the original
      // array.
      std::vector<BValue> elements;
      for (int64_t i = 0; i < new_size; i += 1) {
        elements.push_back(
            fb->ArrayIndex(bvalue, {fb->Literal(UBits(i, 64))}, true));
      }
      return fb->Array(elements, array_type->element_type());
  }
}

BValue IrFuzzHelpers::IncreaseArraySize(
    Package* p, FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const IncreaseArraySizeMethod& increase_size_method) const {
  ArrayType* array_type = bvalue.GetType()->AsArrayOrDie();
  switch (increase_size_method) {
    case IncreaseArraySizeMethod::EXPAND_ARRAY_METHOD:
    default:
      // Create a new array with the original elements and some default elements
      // to expand the array to the specified size.
      std::vector<BValue> elements;
      for (int64_t i = 0; i < array_type->size(); i += 1) {
        elements.push_back(
            fb->ArrayIndex(bvalue, {fb->Literal(UBits(i, 64))}, true));
      }
      // Use a default value to fill the rest of the array.
      BValue default_value =
          DefaultValueOfType(p, fb, array_type->element_type());
      while (elements.size() < new_size) {
        elements.push_back(default_value);
      }
      return fb->Array(elements, array_type->element_type());
  }
}

// Returns the closest bound to the integer if the integer is not in bounds.
int64_t IrFuzzHelpers::Bounded(int64_t value, int64_t left_bound,
                               int64_t right_bound) const {
  CHECK_LE(left_bound, right_bound);
  if (fuzz_version_ < FuzzVersion::BOUND_WITH_MODULO_VERSION) {
    if (value < left_bound) {
      return left_bound;
    } else if (value > right_bound) {
      return right_bound;
    } else {
      return value;
    }
  }
  int64_t diff = right_bound - left_bound;
  return left_bound +
         (std::bit_cast<uint64_t>(value) % static_cast<uint64_t>(diff + 1));
}

// Returns a default bit width if it is not in range. Bit width cannot be
// negative, cannot be 0, and cannot be too large otherwise the test will run
// out of memory or take too long.
int64_t IrFuzzHelpers::BoundedWidth(int64_t bit_width, int64_t left_bound,
                                    int64_t right_bound) const {
  return Bounded(bit_width, left_bound, right_bound);
}

int64_t IrFuzzHelpers::BoundedTupleSize(int64_t tuple_size, int64_t left_bound,
                                        int64_t right_bound) const {
  return Bounded(tuple_size, left_bound, right_bound);
}

// Arrays must have at least one element.
int64_t IrFuzzHelpers::BoundedArraySize(int64_t array_size, int64_t left_bound,
                                        int64_t right_bound) const {
  return Bounded(array_size, left_bound, right_bound);
}

// Returns a default BValue of the specified type.
BValue IrFuzzHelpers::DefaultValue(Package* p, FunctionBuilder* fb,
                                   TypeCase type_case) const {
  switch (type_case) {
    case TypeCase::UNSET_CASE:
    case TypeCase::BITS_CASE:
      return DefaultBitsValue(fb);
    case TypeCase::TUPLE_CASE:
      return fb->Tuple({DefaultValue(p, fb, TypeCase::BITS_CASE)});
    case TypeCase::ARRAY_CASE:
      return fb->Array({DefaultValue(p, fb, TypeCase::BITS_CASE)},
                       p->GetBitsType(64));
  }
}

BValue IrFuzzHelpers::DefaultBitsValue(FunctionBuilder* fb) const {
  return fb->Literal(UBits(0, 64));
}

// Returns a default BValue of the specified type.
BValue IrFuzzHelpers::DefaultValueOfType(Package* p, FunctionBuilder* fb,
                                         Type* type) const {
  if (type->IsBits()) {
    return DefaultValueOfBitsType(p, fb, type);
  } else if (type->IsTuple()) {
    return DefaultValueOfTupleType(p, fb, type);
  } else if (type->IsArray()) {
    return DefaultValueOfArrayType(p, fb, type);
  } else {
    return DefaultValue(p, fb, TypeCase::UNSET_CASE);
  }
}

BValue IrFuzzHelpers::DefaultValueOfBitsType(Package* p, FunctionBuilder* fb,
                                             Type* type) const {
  BitsType* bits_type = type->AsBitsOrDie();
  return fb->Literal(UBits(0, bits_type->bit_count()));
}

BValue IrFuzzHelpers::DefaultValueOfTupleType(Package* p, FunctionBuilder* fb,
                                              Type* type) const {
  TupleType* tuple_type = type->AsTupleOrDie();
  std::vector<BValue> elements;
  // Retrieve the default of each element in the tuple and make a new tuple.
  for (int64_t i = 0; i < tuple_type->size(); i += 1) {
    elements.push_back(DefaultValueOfType(p, fb, tuple_type->element_type(i)));
  }
  return fb->Tuple(elements);
}

BValue IrFuzzHelpers::DefaultValueOfArrayType(Package* p, FunctionBuilder* fb,
                                              Type* type) const {
  ArrayType* array_type = type->AsArrayOrDie();
  std::vector<BValue> elements;
  // Retrieve the default element, fill the array with the same element, and
  // make a new array.
  BValue element = DefaultValueOfType(p, fb, array_type->element_type());
  for (int64_t i = 0; i < array_type->size(); i += 1) {
    elements.push_back(element);
  }
  return fb->Array(elements, array_type->element_type());
}

// These template specializations define that the ConvertTypeProtoToType
// function can be called with the following types. Use of a template to
// allow the traversal of any type proto.
template Type* IrFuzzHelpers::ConvertTypeProtoToType<FuzzTypeProto>(
    Package* p, const FuzzTypeProto&) const;
template Type* IrFuzzHelpers::ConvertTypeProtoToType<CoercedTypeProto>(
    Package* p, const CoercedTypeProto&) const;
// Returns a Type object from the specified type proto.
template <typename TypeProto>
Type* IrFuzzHelpers::ConvertTypeProtoToType(Package* p,
                                            const TypeProto& type_proto) const {
  using TypeCase = decltype(type_proto.type_case());
  switch (type_proto.type_case()) {
    case TypeCase::kBits:
      return ConvertBitsTypeProtoToType(p, type_proto.bits());
    case TypeCase::kTuple:
      return ConvertTupleTypeProtoToType(p, type_proto.tuple());
    case TypeCase::kArray:
      return ConvertArrayTypeProtoToType(p, type_proto.array());
    default:
      return p->GetBitsType(64);
  }
}

template Type* IrFuzzHelpers::ConvertBitsTypeProtoToType<BitsCoercedTypeProto>(
    Package* p, const BitsCoercedTypeProto&) const;
template <typename BitsTypeProto>
Type* IrFuzzHelpers::ConvertBitsTypeProtoToType(
    Package* p, const BitsTypeProto& bits_type) const {
  int64_t bit_width = BoundedWidth(bits_type.bit_width());
  return p->GetBitsType(bit_width);
}

template <typename TupleTypeProto>
Type* IrFuzzHelpers::ConvertTupleTypeProtoToType(
    Package* p, const TupleTypeProto& tuple_type) const {
  int64_t tuple_size = BoundedTupleSize(tuple_type.tuple_elements_size());
  std::vector<Type*> element_types;
  for (int64_t i = 0; i < tuple_size; i += 1) {
    Type* element_type =
        ConvertTypeProtoToType(p, tuple_type.tuple_elements(i));
    element_types.push_back(element_type);
  }
  return p->GetTupleType(element_types);
}

template <typename ArrayTypeProto>
Type* IrFuzzHelpers::ConvertArrayTypeProtoToType(
    Package* p, const ArrayTypeProto& array_type) const {
  int64_t array_size = BoundedArraySize(array_type.array_size());
  Type* element_type = ConvertTypeProtoToType(p, array_type.array_element());
  return p->GetArrayType(array_size, element_type);
}

// Returns arg_count number of randomly generated arguments that are compatible
// for a given parameter type.
std::vector<Value> IrFuzzHelpers::GenArgsForParam(int64_t arg_count, Type* type,
                                                  const Bits& args_bits,
                                                  int64_t& bits_idx) const {
  std::vector<Value> args;
  // Only generate the minimum number of bits needed for the argument.
  int64_t bit_width = type->GetFlatBitCount();
  for (int64_t i = 0; i < arg_count; i += 1) {
    Bits arg_bits;
    if (bits_idx + bit_width <= args_bits.bit_count()) {
      // If there are enough bits in the args_bits, use them.
      arg_bits = args_bits.Slice(bits_idx, bit_width);
      bits_idx += bit_width;
    } else {
      // If there aren't enough bits in the args_bits, use zero bits.
      arg_bits = bits_ops::ZeroExtend(arg_bits, bit_width);
    }
    // Use of UnflattenBitsToValue to convert the arg_bits into a Value of the
    // specified type.
    Value arg_value = UnflattenBitsToValue(arg_bits, type).value();
    args.push_back(arg_value);
  }
  return args;
}

// Accepts a string representing a byte array, which represents an integer. This
// byte array is converted into a Bits object, then truncated or zero extended
// to the specified bit width.
Bits IrFuzzHelpers::ChangeBytesBitWidth(std::string bytes,
                                        int64_t bit_width) const {
  Bits bits = Bits::FromBytes(
      absl::MakeSpan(reinterpret_cast<const uint8_t*>(bytes.data()),
                     bytes.size()),
      bytes.size() * 8);
  if (bits.bit_count() > bit_width) {
    bits = bits_ops::Truncate(bits, bit_width);
  } else if (bits.bit_count() < bit_width) {
    bits = bits_ops::ZeroExtend(bits, bit_width);
  }
  return bits;
}

}  // namespace xls

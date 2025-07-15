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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Returns a BValue that is coerced to the specified type.
BValue Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
               CoercedTypeProto* coerced_type, Type* type) {
  switch (coerced_type->type_case()) {
    case CoercedTypeProto::kBits:
      return CoercedBits(p, fb, bvalue, coerced_type->mutable_bits(), type);
    case CoercedTypeProto::kTuple:
      return CoercedTuple(p, fb, bvalue, coerced_type->mutable_tuple(), type);
    case CoercedTypeProto::kArray:
      return CoercedArray(p, fb, bvalue, coerced_type->mutable_array(), type);
    default:
      return DefaultValue(p, fb);
  }
}

BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   BitsCoercedTypeProto* coerced_type, Type* type) {
  Type* bvalue_type = bvalue.GetType();
  // If the bvalue is already the specified type, return it as is.
  if (bvalue_type == type) {
    return bvalue;
  }
  // If the bvalue differs in categorical type to bits, simply return a default
  // value of the specified type.
  if (!bvalue_type->IsBits()) {
    return DefaultFromBitsTypeProto(p, fb, coerced_type);
  }
  auto coercion_method = coerced_type->mutable_coercion_method();
  // If the bvalue is the same categorical type as bits, change the bit width to
  // the specified bit width.
  int64_t bit_width = BoundedWidth(coerced_type->bit_width());
  return ChangeBitWidth(fb, bvalue, bit_width,
                        coercion_method->mutable_change_bit_width_method());
}

BValue CoercedTuple(Package* p, FunctionBuilder* fb, BValue bvalue,
                    TupleCoercedTypeProto* coerced_type, Type* type) {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == type) {
    return bvalue;
  }
  if (!bvalue_type->IsTuple()) {
    return DefaultFromTupleTypeProto(p, fb, coerced_type);
  }
  auto coercion_method = coerced_type->mutable_coercion_method();
  // Change the size of the tuple to match the specified size.
  int64_t size = BoundedSize(coerced_type->tuple_elements_size());
  bvalue = ChangeTupleSize(fb, bvalue, size,
                           coercion_method->mutable_change_list_size_method());
  std::vector<BValue> coerced_elements;
  // Coerce each tuple element and create a new tuple with the coerced
  // elements.
  for (int64_t i = 0; i < coerced_type->tuple_elements_size(); i += 1) {
    BValue element = fb->TupleIndex(bvalue, i);
    auto element_coerced_type = coerced_type->mutable_tuple_elements(i);
    Type* element_type = type->AsTupleOrDie()->element_type(i);
    BValue coerced_element =
        Coerced(p, fb, element, element_coerced_type, element_type);
    coerced_elements.push_back(coerced_element);
  }
  return fb->Tuple(coerced_elements);
}

BValue CoercedArray(Package* p, FunctionBuilder* fb, BValue bvalue,
                    ArrayCoercedTypeProto* coerced_type, Type* type) {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == type) {
    return bvalue;
  }
  if (!bvalue_type->IsArray()) {
    return DefaultFromArrayTypeProto(p, fb, coerced_type);
  }
  auto coercion_method = coerced_type->mutable_coercion_method();
  // Change the size of the array to match the specified size.
  int64_t size = BoundedSize(coerced_type->array_size());
  bvalue = ChangeArraySize(fb, bvalue, size,
                           coercion_method->mutable_change_list_size_method());
  std::vector<BValue> coerced_elements;
  // Coerce each array element and create a new array with the coerced
  // elements.
  for (int64_t i = 0; i < coerced_type->array_size(); i += 1) {
    BValue element = fb->ArrayIndex(bvalue, {fb->Literal(UBits(i, 64))}, true);
    auto element_coerced_type = coerced_type->mutable_array_element();
    Type* element_type = type->AsArrayOrDie()->element_type();
    BValue coerced_element =
        Coerced(p, fb, element, element_coerced_type, element_type);
    coerced_elements.push_back(coerced_element);
  }
  return fb->Array(
      coerced_elements,
      ConvertTypeProtoToType(p, coerced_type->mutable_array_element()));
}

// Changes the bit width of an inputted BValue to a specified bit width.
// Multiple bit width modification methods can be used and specified.
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
                      ChangeBitWidthMethodProto* change_bit_width_method) {
  if (bvalue.BitCountOrDie() > new_bit_width) {
    // If the bvalue is larger than the bit width, use the decrease width method
    // to reduce the bit width.
    return DecreaseBitWidth(fb, bvalue, new_bit_width,
                            change_bit_width_method->decrease_width_method());
  } else if (bvalue.BitCountOrDie() < new_bit_width) {
    // If the bvalue is smaller than the bit width, use the increase width
    // method to increase the bit width.
    return IncreaseBitWidth(fb, bvalue, new_bit_width,
                            change_bit_width_method->increase_width_method());
  } else {
    // If the bvalue is the same bit width as the specified bit width, return
    // the bvalue as is because no change is needed.
    return bvalue;
  }
}

// Overloaded version of ChangeBitWidth that uses default width fitting methods.
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t bit_width) {
  ChangeBitWidthMethodProto default_method;
  default_method.set_decrease_width_method(
      DecreaseWidthMethod::UNSET_DECREASE_WIDTH_METHOD);
  default_method.set_increase_width_method(
      IncreaseWidthMethod::UNSET_INCREASE_WIDTH_METHOD);
  return ChangeBitWidth(fb, bvalue, bit_width, &default_method);
}

BValue DecreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        DecreaseWidthMethod decrease_width_method) {
  switch (decrease_width_method) {
    case BIT_SLICE_METHOD:
    default:
      return fb->BitSlice(bvalue, /*start=*/0, new_bit_width);
  }
}

BValue IncreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        IncreaseWidthMethod increase_width_method) {
  switch (increase_width_method) {
    case SIGN_EXTEND_METHOD:
      return fb->SignExtend(bvalue, new_bit_width);
    case ZERO_EXTEND_METHOD:
    default:
      return fb->ZeroExtend(bvalue, new_bit_width);
  }
}

// Changes the size of a tuple to match the specified size.
BValue ChangeTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                       ChangeListSizeMethodProto* change_list_size_method) {
  int64_t current_size = bvalue.GetType()->AsTupleOrDie()->size();
  if (current_size > new_size) {
    return DecreaseTupleSize(fb, bvalue, new_size,
                             change_list_size_method->decrease_size_method());
  } else if (current_size < new_size) {
    return IncreaseTupleSize(fb, bvalue, new_size,
                             change_list_size_method->increase_size_method());
  } else {
    return bvalue;
  }
}

BValue DecreaseTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         DecreaseSizeMethod decrease_size_method) {
  switch (decrease_size_method) {
    case DecreaseSizeMethod::SHRINK_METHOD:
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

BValue IncreaseTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         IncreaseSizeMethod increase_size_method) {
  switch (increase_size_method) {
    case IncreaseSizeMethod::EXPAND_METHOD:
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
BValue ChangeArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                       ChangeListSizeMethodProto* change_list_size_method) {
  ArrayType* array_type = bvalue.GetType()->AsArrayOrDie();
  if (array_type->size() > new_size) {
    return DecreaseArraySize(fb, bvalue, new_size,
                             change_list_size_method->decrease_size_method());
  } else if (array_type->size() < new_size) {
    return IncreaseArraySize(fb, bvalue, new_size,
                             change_list_size_method->increase_size_method());
  } else {
    return bvalue;
  }
}

BValue DecreaseArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         DecreaseSizeMethod decrease_size_method) {
  ArrayType* array_type = bvalue.GetType()->AsArrayOrDie();
  switch (decrease_size_method) {
    case DecreaseSizeMethod::SHRINK_METHOD:
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

BValue IncreaseArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         IncreaseSizeMethod increase_size_method) {
  ArrayType* array_type = bvalue.GetType()->AsArrayOrDie();
  switch (increase_size_method) {
    case IncreaseSizeMethod::EXPAND_METHOD:
    default:
      // Create a new array with the original elements and some default elements
      // to expand the array to the specified size.
      std::vector<BValue> elements;
      for (int64_t i = 0; i < array_type->size(); i += 1) {
        elements.push_back(
            fb->ArrayIndex(bvalue, {fb->Literal(UBits(i, 64))}, true));
      }
      // Use the last element to fill the rest of the array.
      BValue last_element = elements.back();
      while (elements.size() < new_size) {
        elements.push_back(last_element);
      }
      return fb->Array(elements, array_type->element_type());
  }
}

// Returns a default integer if the integer is not in bounds.
int64_t Bounded(int64_t value, int64_t left_bound, int64_t right_bound) {
  CHECK_LE(left_bound, right_bound);
  if (left_bound <= value && value <= right_bound) {
    return value;
  } else {
    return left_bound;
  }
}

// Returns a default bit width if it is not in range. Bit width cannot be
// negative, cannot be 0, and cannot be too large otherwise the test will run
// out of memory or take too long.
int64_t BoundedWidth(int64_t bit_width, int64_t left_bound,
                     int64_t right_bound) {
  return Bounded(bit_width, left_bound, right_bound);
}

int64_t BoundedSize(int64_t size, int64_t left_bound, int64_t right_bound) {
  return Bounded(size, left_bound, right_bound);
}

// Returns a default BValue of the specified type.
BValue DefaultValue(Package* p, FunctionBuilder* fb, TypeCase type_case) {
  switch (type_case) {
    case TypeCase::UNSET_CASE:
    case TypeCase::BITS_CASE:
      return DefaultBitsValue(fb);
    case TypeCase::TUPLE_CASE:
      return fb->Tuple({});
    case TypeCase::ARRAY_CASE:
      return fb->Array({}, p->GetBitsType(64));
  }
}

BValue DefaultBitsValue(FunctionBuilder* fb) {
  return fb->Literal(UBits(0, 64));
}

// These template specializations define that the DefaultFromTypeProto function
// can be called with the following types.
template BValue DefaultFromTypeProto<FuzzTypeProto>(Package*, FunctionBuilder*,
                                                    FuzzTypeProto*);
template BValue DefaultFromTypeProto<ValueTypeProto>(Package*, FunctionBuilder*,
                                                     ValueTypeProto*);
template BValue DefaultFromTypeProto<CoercedTypeProto>(Package*,
                                                       FunctionBuilder*,
                                                       CoercedTypeProto*);
// Returns a default BValue of the specified type proto. Use of a template to
// allow the traversal of any type proto.
template <typename TypeProto>
BValue DefaultFromTypeProto(Package* p, FunctionBuilder* fb,
                            TypeProto* type_proto) {
  using TypeCase = decltype(type_proto->type_case());
  switch (type_proto->type_case()) {
    case TypeCase::kBits:
      return DefaultFromBitsTypeProto(p, fb, type_proto->mutable_bits());
    case TypeCase::kTuple:
      return DefaultFromTupleTypeProto(p, fb, type_proto->mutable_tuple());
    case TypeCase::kArray:
      return DefaultFromArrayTypeProto(p, fb, type_proto->mutable_array());
    default:
      return DefaultValue(p, fb);
  }
}

template <typename BitsTypeProto>
BValue DefaultFromBitsTypeProto(Package* p, FunctionBuilder* fb,
                                BitsTypeProto* bits_type) {
  int64_t bit_width = BoundedWidth(bits_type->bit_width());
  return fb->Literal(UBits(0, bit_width));
}

template <typename TupleTypeProto>
BValue DefaultFromTupleTypeProto(Package* p, FunctionBuilder* fb,
                                 TupleTypeProto* tuple_type) {
  std::vector<BValue> elements;
  // Retrieve the default of each element in the tuple and make a new tuple.
  for (auto& element_type_proto : *tuple_type->mutable_tuple_elements()) {
    elements.push_back(DefaultFromTypeProto(p, fb, &element_type_proto));
  }
  return fb->Tuple(elements);
}

template <typename ArrayTypeProto>
BValue DefaultFromArrayTypeProto(Package* p, FunctionBuilder* fb,
                                 ArrayTypeProto* array_type) {
  std::vector<BValue> elements;
  // Retrieve the default element, fill the array with the same element, and
  // make a new array.
  BValue element =
      DefaultFromTypeProto(p, fb, array_type->mutable_array_element());
  for (int64_t i = 0; i < array_type->array_size(); i += 1) {
    elements.push_back(element);
  }
  Type* element_type = element.GetType();
  if (array_type->array_size() == 0) {
    element_type =
        ConvertTypeProtoToType(p, array_type->mutable_array_element());
  }
  return fb->Array(elements, element_type);
}

template Type* ConvertTypeProtoToType<FuzzTypeProto>(Package*, FuzzTypeProto*);
template Type* ConvertTypeProtoToType<ValueTypeProto>(Package*,
                                                      ValueTypeProto*);
template Type* ConvertTypeProtoToType<CoercedTypeProto>(Package*,
                                                        CoercedTypeProto*);
// Returns a Type object from the specified type proto.
template <typename TypeProto>
Type* ConvertTypeProtoToType(Package* p, TypeProto* type_proto) {
  using TypeCase = decltype(type_proto->type_case());
  switch (type_proto->type_case()) {
    case TypeCase::kBits:
      return ConvertBitsTypeProtoToType(p, type_proto->mutable_bits());
    case TypeCase::kTuple:
      return ConvertTupleTypeProtoToType(p, type_proto->mutable_tuple());
    case TypeCase::kArray:
      return ConvertArrayTypeProtoToType(p, type_proto->mutable_array());
    default:
      return p->GetBitsType(64);
  }
}

template Type* ConvertBitsTypeProtoToType<BitsCoercedTypeProto>(
    Package*, BitsCoercedTypeProto*);
template <typename BitsTypeProto>
Type* ConvertBitsTypeProtoToType(Package* p, BitsTypeProto* bits_type) {
  int64_t bit_width = BoundedWidth(bits_type->bit_width());
  return p->GetBitsType(bit_width);
}

template <typename TupleTypeProto>
Type* ConvertTupleTypeProtoToType(Package* p, TupleTypeProto* tuple_type) {
  std::vector<Type*> element_types;
  for (auto& element_type_proto : *tuple_type->mutable_tuple_elements()) {
    element_types.push_back(ConvertTypeProtoToType(p, &element_type_proto));
  }
  return p->GetTupleType(element_types);
}

template <typename ArrayTypeProto>
Type* ConvertArrayTypeProtoToType(Package* p, ArrayTypeProto* array_type) {
  Type* element_type =
      ConvertTypeProtoToType(p, array_type->mutable_array_element());
  return p->GetArrayType(array_type->array_size(), element_type);
}

// Traverses the ValueTypeProto and returns a BValue representing the
// instantiated literal.
BValue ValueFromValueTypeProto(Package* p, FunctionBuilder* fb,
                               ValueTypeProto* value_type) {
  switch (value_type->type_case()) {
    case ValueTypeProto::kBits: {
      // Take the bytes protobuf datatype and convert it to a Bits object by
      // making a const uint8_t span. Any bytes that exceed the bit width of the
      // literal will be dropped.
      auto bits_type = value_type->mutable_bits();
      int64_t bit_width = BoundedWidth(bits_type->bit_width());
      Bits value_bits =
          ChangeBytesBitWidth(bits_type->value_bytes(), bit_width);
      return fb->Literal(value_bits);
    }
    case ValueTypeProto::kTuple: {
      auto tuple_type = value_type->mutable_tuple();
      std::vector<BValue> elements;
      // Loops over each tuple element to recursively retrieve the BValue.
      for (auto& element_value_type : *tuple_type->mutable_tuple_elements()) {
        elements.push_back(ValueFromValueTypeProto(p, fb, &element_value_type));
      }
      return fb->Tuple(elements);
    }
    case ValueTypeProto::kArray: {
      auto array_type = value_type->mutable_array();
      std::vector<BValue> elements;
      // Recursively retrieves the array element once and fills the array
      // with the same element.
      BValue element =
          ValueFromValueTypeProto(p, fb, array_type->mutable_array_element());
      for (int64_t i = 0; i < array_type->array_size(); i += 1) {
        elements.push_back(element);
      }
      // Gets the array element type as it is required for the Array IR node.
      Type* element_type = element.GetType();
      if (array_type->array_size() == 0) {
        element_type =
            ConvertTypeProtoToType(p, array_type->mutable_array_element());
      }
      return fb->Array(elements, element_type);
    }
    default:
      return DefaultValue(p, fb);
  }
}

// Returns arg_count number of randomly generated arguments that are compatible
// for a given parameter type.
std::vector<Value> GenRandomArgs(int64_t arg_count, Type* type) {
  std::vector<Value> args;
  if (type->IsBits()) {
    BitsType* bits_type = type->AsBitsOrDie();
    int64_t bit_width = bits_type->bit_count();
    // Only generate the minimum number of bytes needed to hold the bits.
    int64_t byte_count = (bit_width + 7) / 8;
    for (int64_t i = 0; i < arg_count; i += 1) {
      // Randomly generate the byte string.
      std::string arg_bytes = GenRandomBytes(byte_count);
      // Truncate the bytes to the exact bit width as a Bits object.
      Bits arg_bits = ChangeBytesBitWidth(arg_bytes, bit_width);
      args.push_back(Value(arg_bits));
    }
  } else if (type->IsTuple()) {
    TupleType* tuple_type = type->AsTupleOrDie();
    std::vector<std::vector<Value>> tuple_arg_sets;
    // Recursively generate the arguments for each element in the tuple.
    for (int64_t row_idx = 0; row_idx < tuple_type->size(); row_idx += 1) {
      tuple_arg_sets.push_back(
          GenRandomArgs(arg_count, tuple_type->element_type(row_idx)));
    }
    // Combine the tuple elements into a tuple by looping column-row order.
    for (int64_t col_idx = 0; col_idx < arg_count; col_idx += 1) {
      std::vector<Value> tuple_args;
      for (int64_t row_idx = 0; row_idx < tuple_type->size(); row_idx += 1) {
        tuple_args.push_back(tuple_arg_sets[row_idx][col_idx]);
      }
      args.push_back(Value::Tuple(tuple_args));
    }
  } else if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    std::vector<std::vector<Value>> array_arg_sets;
    // Recursively generate the arguments for each element in the array.
    for (int64_t row_idx = 0; row_idx < array_type->size(); row_idx += 1) {
      array_arg_sets.push_back(
          GenRandomArgs(arg_count, array_type->element_type()));
    }
    // Combine the array elements into an array by looping column-row order.
    for (int64_t col_idx = 0; col_idx < arg_count; col_idx += 1) {
      std::vector<Value> array_args;
      for (int64_t row_idx = 0; row_idx < array_type->size(); row_idx += 1) {
        array_args.push_back(array_arg_sets[row_idx][col_idx]);
      }
      args.push_back(Value::Array(array_args).value());
    }
  }
  return args;
}

// Returns a byte array of the specified byte size with random characters.
std::string GenRandomBytes(int64_t byte_count) {
  std::string random_bytes;
  absl::BitGen gen;
  for (int64_t i = 0; i < byte_count; i += 1) {
    random_bytes.push_back(absl::Uniform<unsigned char>(gen));
  }
  return random_bytes;
}

// Accepts a string representing a byte array, which represents an integer. This
// byte array is converted into a Bits object, then truncated or zero extended
// to the specified bit width.
Bits ChangeBytesBitWidth(std::string bytes, int64_t bit_width) {
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

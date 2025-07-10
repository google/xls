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
    default:
      return DefaultValue(p, fb);
  }
}

BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   BitsCoercedTypeProto* coerced_type, Type* type) {
  Type* bvalue_type = bvalue.GetType();
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

// Returns a default BValue of the specified type.
BValue DefaultValue(Package* p, FunctionBuilder* fb, TypeCase type_case) {
  switch (type_case) {
    case TypeCase::UNSET_CASE:
    case TypeCase::BITS_CASE:
      return DefaultBitsValue(fb);
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

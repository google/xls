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

// Returns a BValue that is coerced to the specified type.
BValue Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
               const CoercedTypeProto& coerced_type, Type* target_type) {
  switch (coerced_type.type_case()) {
    case CoercedTypeProto::kBits:
      return CoercedBits(p, fb, bvalue, coerced_type.bits(), target_type);
    default:
      return DefaultValue(p, fb);
  }
}

BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   const BitsCoercedTypeProto& coerced_type,
                   Type* target_type) {
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
  BitsType* bits_type = target_type->AsBitsOrDie();
  auto coercion_method = coerced_type.coercion_method();
  // Change the bit width to the specified bit width.
  return ChangeBitWidth(fb, bvalue, bits_type->bit_count(),
                        coercion_method.change_bit_width_method());
}

// Changes the bit width of an inputted BValue to a specified bit width.
// Multiple bit width modification methods can be used and specified.
BValue ChangeBitWidth(
    FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
    const ChangeBitWidthMethodProto& change_bit_width_method) {
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
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t bit_width) {
  ChangeBitWidthMethodProto default_method;
  default_method.set_decrease_width_method(
      DecreaseWidthMethod::UNSET_DECREASE_WIDTH_METHOD);
  default_method.set_increase_width_method(
      IncreaseWidthMethod::UNSET_INCREASE_WIDTH_METHOD);
  return ChangeBitWidth(fb, bvalue, bit_width, default_method);
}

BValue DecreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        const DecreaseWidthMethod& decrease_width_method) {
  switch (decrease_width_method) {
    case BIT_SLICE_METHOD:
    default:
      return fb->BitSlice(bvalue, /*start=*/0, new_bit_width);
  }
}

BValue IncreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        const IncreaseWidthMethod& increase_width_method) {
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

// Returns a default BValue of the specified type.
BValue DefaultValueOfType(Package* p, FunctionBuilder* fb, Type* type) {
  if (type->IsBits()) {
    return DefaultValueOfBitsType(p, fb, type);
  } else {
    return DefaultValue(p, fb);
  }
}

BValue DefaultValueOfBitsType(Package* p, FunctionBuilder* fb, Type* type) {
  BitsType* bits_type = type->AsBitsOrDie();
  return fb->Literal(UBits(0, bits_type->bit_count()));
}

// These template specializations define that the ConvertTypeProtoToType
// function can be called with the following types. Use of a template to
// allow the traversal of any type proto.
template Type* ConvertTypeProtoToType<FuzzTypeProto>(Package*,
                                                     const FuzzTypeProto&);
template Type* ConvertTypeProtoToType<CoercedTypeProto>(
    Package*, const CoercedTypeProto&);
// Returns a Type object from the specified type proto.
template <typename TypeProto>
Type* ConvertTypeProtoToType(Package* p, const TypeProto& type_proto) {
  using TypeCase = decltype(type_proto.type_case());
  switch (type_proto.type_case()) {
    case TypeCase::kBits:
      return ConvertBitsTypeProtoToType(p, type_proto.bits());
    default:
      return p->GetBitsType(64);
  }
}

template Type* ConvertBitsTypeProtoToType<BitsCoercedTypeProto>(
    Package*, const BitsCoercedTypeProto&);
template <typename BitsTypeProto>
Type* ConvertBitsTypeProtoToType(Package* p, const BitsTypeProto& bits_type) {
  int64_t bit_width = BoundedWidth(bits_type.bit_width());
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
std::vector<Value> GenArgsForParam(int64_t arg_count, Type* type,
                                   const Bits& args_bits, int64_t& bits_idx) {
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

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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_HELPERS_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_HELPERS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

// Generic helper functions used in the IR Fuzzer.

namespace xls {

// A simple type enum. Used for specifying kinds of default values. Also used in
// the context list to specify which list to use.
enum TypeCase {
  UNSET_CASE = 0,
  BITS_CASE = 1,
  TUPLE_CASE = 2,
  ARRAY_CASE = 3,
};

class IrFuzzHelpers {
 public:
  constexpr IrFuzzHelpers(FuzzVersion version) : fuzz_version_(version) {}

  BValue Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
                 const CoercedTypeProto& coerced_type, Type* target_type) const;
  BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                     const BitsCoercedTypeProto& coerced_type,
                     Type* target_type) const;
  BValue CoercedTuple(Package* p, FunctionBuilder* fb, BValue bvalue,
                      const TupleCoercedTypeProto& coerced_type,
                      Type* target_type) const;
  BValue CoercedArray(Package* p, FunctionBuilder* fb, BValue bvalue,
                      const ArrayCoercedTypeProto& coerced_type,
                      Type* target_type) const;

  BValue Fitted(Package* p, FunctionBuilder* fb, BValue bvalue,
                const CoercionMethodProto& coercion_method, Type* type) const;
  BValue FittedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                    const BitsCoercionMethodProto& coercion_method,
                    Type* target_type) const;
  BValue FittedTuple(Package* p, FunctionBuilder* fb, BValue bvalue,
                     const CoercionMethodProto& coercion_method,
                     Type* target_type) const;
  BValue FittedArray(Package* p, FunctionBuilder* fb, BValue bvalue,
                     const CoercionMethodProto& coercion_method,
                     Type* target_type) const;

  BValue ChangeBitWidth(
      FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
      const ChangeBitWidthMethodProto& change_bit_width_method) const;
  BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width) const;
  BValue DecreaseBitWidth(
      FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
      const DecreaseWidthMethod& decrease_width_method) const;
  BValue IncreaseBitWidth(
      FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
      const IncreaseWidthMethod& increase_bit_width_method) const;

  BValue ChangeTupleSize(
      FunctionBuilder* fb, BValue bvalue, int64_t new_size,
      const ChangeTupleSizeMethodProto& change_tuple_size_method) const;
  BValue DecreaseTupleSize(
      FunctionBuilder* fb, BValue bvalue, int64_t new_size,
      const DecreaseTupleSizeMethod& decrease_size_method) const;
  BValue IncreaseTupleSize(
      FunctionBuilder* fb, BValue bvalue, int64_t new_size,
      const IncreaseTupleSizeMethod& increase_size_method) const;

  BValue ChangeArraySize(
      Package* p, FunctionBuilder* fb, BValue bvalue, int64_t new_size,
      const ChangeArraySizeMethodProto& change_array_size_method) const;
  BValue DecreaseArraySize(
      FunctionBuilder* fb, BValue bvalue, int64_t new_size,
      const DecreaseArraySizeMethod& decrease_size_method) const;
  BValue IncreaseArraySize(
      Package* p, FunctionBuilder* fb, BValue bvalue, int64_t new_size,
      const IncreaseArraySizeMethod& increase_size_method) const;

  static constexpr int64_t kMaxFuzzBitWidth = 1000;
  static constexpr int64_t kMaxFuzzTupleSize = 100;
  static constexpr int64_t kMaxFuzzArraySize = 100;

  int64_t Bounded(int64_t value, int64_t left_bound, int64_t right_bound) const;
  int64_t BoundedWidth(int64_t bit_width, int64_t left_bound = 1,
                       int64_t right_bound = kMaxFuzzBitWidth) const;
  int64_t BoundedTupleSize(int64_t tuple_size, int64_t left_bound = 0,
                           int64_t right_bound = kMaxFuzzTupleSize) const;
  int64_t BoundedArraySize(int64_t array_size, int64_t left_bound = 1,
                           int64_t right_bound = kMaxFuzzArraySize) const;

  BValue DefaultValue(Package* p, FunctionBuilder* fb,
                      TypeCase type_case = UNSET_CASE) const;
  BValue DefaultBitsValue(FunctionBuilder* fb) const;

  BValue DefaultValueOfType(Package* p, FunctionBuilder* fb, Type* type) const;
  BValue DefaultValueOfBitsType(Package* p, FunctionBuilder* fb,
                                Type* type) const;
  BValue DefaultValueOfTupleType(Package* p, FunctionBuilder* fb,
                                 Type* type) const;
  BValue DefaultValueOfArrayType(Package* p, FunctionBuilder* fb,
                                 Type* type) const;

  template <typename TypeProto>
  Type* ConvertTypeProtoToType(Package* p, const TypeProto& type_proto) const {
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
  template <typename BitsTypeProto>
  Type* ConvertBitsTypeProtoToType(Package* p,
                                   const BitsTypeProto& bits_type) const {
    int64_t bit_width = BoundedWidth(bits_type.bit_width());
    return p->GetBitsType(bit_width);
  }
  template <typename TupleTypeProto>
  Type* ConvertTupleTypeProtoToType(Package* p,
                                    const TupleTypeProto& tuple_type) const {
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
  Type* ConvertArrayTypeProtoToType(Package* p,
                                    const ArrayTypeProto& array_type) const {
    int64_t array_size = BoundedArraySize(array_type.array_size());
    Type* element_type = ConvertTypeProtoToType(p, array_type.array_element());
    return p->GetArrayType(array_size, element_type);
  }

  std::vector<Value> GenArgsForParam(int64_t arg_count, Type* type,
                                     const Bits& args_bits,
                                     int64_t& bits_idx) const;
  Bits ChangeBytesBitWidth(std::string bytes, int64_t bit_width) const;

 private:
  FuzzVersion fuzz_version_;
};

constexpr std::array<IrFuzzHelpers, 2> kFuzzHelpers =
    std::array<IrFuzzHelpers, 2>{
        IrFuzzHelpers(FuzzVersion::UNSET_FUZZ_VERSION),
        IrFuzzHelpers(FuzzVersion::BOUND_WITH_MODULO_VERSION),
    };

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_HELPERS_H_

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

BValue Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
               const CoercedTypeProto& coerced_type, Type* target_type);
BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   const BitsCoercedTypeProto& coerced_type, Type* target_type);
BValue CoercedTuple(Package* p, FunctionBuilder* fb, BValue bvalue,
                    const TupleCoercedTypeProto& coerced_type,
                    Type* target_type);
BValue CoercedArray(Package* p, FunctionBuilder* fb, BValue bvalue,
                    const ArrayCoercedTypeProto& coerced_type,
                    Type* target_type);

BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
                      const ChangeBitWidthMethodProto& change_bit_width_method);
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue,
                      int64_t new_bit_width);
BValue DecreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        const DecreaseWidthMethod& decrease_width_method);
BValue IncreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        const IncreaseWidthMethod& increase_width_method);

BValue ChangeTupleSize(
    FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const ChangeTupleSizeMethodProto& change_tuple_size_method);
BValue DecreaseTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         const DecreaseTupleSizeMethod& decrease_size_method);
BValue IncreaseTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         const IncreaseTupleSizeMethod& increase_size_method);

BValue ChangeArraySize(
    Package* p, FunctionBuilder* fb, BValue bvalue, int64_t new_size,
    const ChangeArraySizeMethodProto& change_array_size_method);
BValue DecreaseArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         const DecreaseArraySizeMethod& decrease_size_method);
BValue IncreaseArraySize(Package* p, FunctionBuilder* fb, BValue bvalue,
                         int64_t new_size,
                         const IncreaseArraySizeMethod& increase_size_method);

int64_t Bounded(int64_t value, int64_t left_bound, int64_t right_bound);
int64_t BoundedWidth(int64_t bit_width, int64_t left_bound = 1,
                     int64_t right_bound = 1000);
int64_t BoundedTupleSize(int64_t size, int64_t left_bound = 0,
                         int64_t right_bound = 100);
int64_t BoundedArraySize(int64_t size, int64_t left_bound = 1,
                         int64_t right_bound = 100);

BValue DefaultValue(Package* p, FunctionBuilder* fb,
                    TypeCase type_case = UNSET_CASE);
BValue DefaultBitsValue(FunctionBuilder* fb);

BValue DefaultValueOfType(Package* p, FunctionBuilder* fb, Type* type);
BValue DefaultValueOfBitsType(Package* p, FunctionBuilder* fb, Type* type);
BValue DefaultValueOfTupleType(Package* p, FunctionBuilder* fb, Type* type);
BValue DefaultValueOfArrayType(Package* p, FunctionBuilder* fb, Type* type);

template <typename TypeProto>
Type* ConvertTypeProtoToType(Package* p, const TypeProto& type_proto);
template <typename BitsTypeProto>
Type* ConvertBitsTypeProtoToType(Package* p, const BitsTypeProto& bits_type);
template <typename TupleTypeProto>
Type* ConvertTupleTypeProtoToType(Package* p, const TupleTypeProto& tuple_type);
template <typename ArrayTypeProto>
Type* ConvertArrayTypeProtoToType(Package* p, const ArrayTypeProto& array_type);

std::vector<Value> GenArgsForParam(int64_t arg_count, Type* type,
                                   const Bits& args_bits, int64_t& bits_idx);
Bits ChangeBytesBitWidth(std::string bytes, int64_t bit_width);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_HELPERS_H_

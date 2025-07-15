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
               CoercedTypeProto* coerced_type, Type* type);
BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   BitsCoercedTypeProto* bits_coerced_type, Type* type);
BValue CoercedTuple(Package* p, FunctionBuilder* fb, BValue bvalue,
                    TupleCoercedTypeProto* tuple_coerced_type, Type* type);
BValue CoercedArray(Package* p, FunctionBuilder* fb, BValue bvalue,
                    ArrayCoercedTypeProto* array_coerced_type, Type* type);

BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t new_bit_width,
                      ChangeBitWidthMethodProto* change_bit_width_method);
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue,
                      int64_t new_bit_width);
BValue DecreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        DecreaseWidthMethod decrease_width_method);
BValue IncreaseBitWidth(FunctionBuilder* fb, BValue bvalue,
                        int64_t new_bit_width,
                        IncreaseWidthMethod increase_width_method);

BValue ChangeTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                       ChangeListSizeMethodProto* change_list_size_method);
BValue DecreaseTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         DecreaseSizeMethod decrease_size_method);
BValue IncreaseTupleSize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         IncreaseSizeMethod increase_size_method);

BValue ChangeArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                       ChangeListSizeMethodProto* change_list_size_method);
BValue DecreaseArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         DecreaseSizeMethod decrease_size_method);
BValue IncreaseArraySize(FunctionBuilder* fb, BValue bvalue, int64_t new_size,
                         IncreaseSizeMethod increase_size_method);

int64_t Bounded(int64_t value, int64_t left_bound, int64_t right_bound);
int64_t BoundedWidth(int64_t bit_width, int64_t left_bound = 1,
                     int64_t right_bound = 1000);
int64_t BoundedSize(int64_t size, int64_t left_bound = 1,
                    int64_t right_bound = 100);

BValue DefaultValue(Package* p, FunctionBuilder* fb,
                    TypeCase type_case = UNSET_CASE);
BValue DefaultBitsValue(FunctionBuilder* fb);

template <typename TypeProto>
BValue DefaultFromTypeProto(Package* p, FunctionBuilder* fb,
                            TypeProto* type_proto);
template <typename BitsTypeProto>
BValue DefaultFromBitsTypeProto(Package* p, FunctionBuilder* fb,
                                BitsTypeProto* bits_type);
template <typename TupleTypeProto>
BValue DefaultFromTupleTypeProto(Package* p, FunctionBuilder* fb,
                                 TupleTypeProto* tuple_type);
template <typename ArrayTypeProto>
BValue DefaultFromArrayTypeProto(Package* p, FunctionBuilder* fb,
                                 ArrayTypeProto* array_type);

template <typename TypeProto>
Type* ConvertTypeProtoToType(Package* p, TypeProto* type_proto);
template <typename BitsTypeProto>
Type* ConvertBitsTypeProtoToType(Package* p, BitsTypeProto* bits_type);
template <typename TupleTypeProto>
Type* ConvertTupleTypeProtoToType(Package* p, TupleTypeProto* tuple_type);
template <typename ArrayTypeProto>
Type* ConvertArrayTypeProtoToType(Package* p, ArrayTypeProto* array_type);

BValue ValueFromValueTypeProto(Package* p, FunctionBuilder* fb,
                               ValueTypeProto* value_type);

std::vector<Value> GenRandomArgs(int64_t arg_count, Type* type);
std::string GenRandomBytes(int64_t byte_count);
Bits ChangeBytesBitWidth(std::string bytes, int64_t bit_width);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_HELPERS_H_

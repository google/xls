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

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

// Generic helper functions used in the IR Fuzzer.

namespace xls {

enum TypeCase {
  UNSET = 0,
  BITS = 1,
};

BValue Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
               CoercedTypeProto* coerced_type, Type* type);
BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   BitsCoercedTypeProto* bits_coerced_type, Type* type);

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

BValue DefaultValue(Package* p, FunctionBuilder* fb,
                    TypeCase type_case = UNSET);
BValue DefaultBitsValue(FunctionBuilder* fb);

template <typename TypeProto>
BValue DefaultFromTypeProto(Package* p, FunctionBuilder* fb,
                            TypeProto* type_proto);
template <typename BitsTypeProto>
BValue DefaultFromBitsTypeProto(Package* p, FunctionBuilder* fb,
                                BitsTypeProto* bits_type);

template <typename TypeProto>
Type* ConvertTypeProtoToType(Package* p, TypeProto* type_proto);
template <typename BitsTypeProto>
Type* ConvertBitsTypeProtoToType(Package* p, BitsTypeProto* bits_type);

int64_t Bounded(int64_t value, int64_t left_bound, int64_t right_bound);
int64_t BoundedWidth(int64_t bit_width, int64_t left_bound = 1,
                     int64_t right_bound = 1000);

Bits ChangeBytesBitWidth(std::string bytes, int64_t bit_width);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_HELPERS_H_

// Copyright 2020 Google LLC
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

// Library defining how arrays and tuples are lowered into vectors of bits by
// the generators.
#ifndef XLS_CODEGEN_FLATTENING_H_
#define XLS_CODEGEN_FLATTENING_H_

#include "absl/types/span.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

// Flattens this arbitrarily-typed Value to a bits type containing the same
// total number of bits. Tuples are flattened by concatenating all of the leaf
// elements. The zero-th tuple element ends up in the highest-indexed bits in
// the resulting vector. Similarly, in a flattened array the zero-th element
// ends up in the highest index bits. This is in line with the behavior of
// Verilog concatenate operation.
Bits FlattenValueToBits(const Value& value);

// Unflattens the given Bits to a Value of the given type. This is the inverse
// of FlattenValueToBits.
xabsl::StatusOr<Value> UnflattenBitsToValue(const Bits& bits, const Type* type);
xabsl::StatusOr<Value> UnflattenBitsToValue(const Bits& bits,
                                            const TypeProto& type_proto);

// Returns the index of the first bit of tuple element at 'index' where the
// tuple is flattened into a vector of bits.
int64 GetFlatBitIndexOfElement(const TupleType* tuple_type, int64 index);

// Overload which returns the index of an element for an array type.
int64 GetFlatBitIndexOfElement(const ArrayType* array_type, int64 index);

// Unflattens the given VAST expression into a unpacked array
// representation. 'array_type' is the underlying XLS type of the expression.
// Uses the SystemVerilog-only array assignment construct.
verilog::Expression* UnflattenArray(verilog::IndexableExpression* input,
                                    ArrayType* array_type,
                                    verilog::VerilogFile* file);

// Flattens the given VAST expression into a flat bit vector. 'input' must be an
// unpacked array. 'array_type' is the underlying XLS type of the expression.
verilog::Expression* FlattenArray(verilog::IndexableExpression* input,
                                  ArrayType* array_type,
                                  verilog::VerilogFile* file);

// Unflattens the array element at the given index of 'input', a flattened
// tuple. 'tuple_type' is the underlying XLS type of the tuple.
verilog::Expression* UnflattenArrayShapedTupleElement(
    verilog::IndexableExpression* input, TupleType* tuple_type,
    int64 tuple_index, verilog::VerilogFile* file);

}  // namespace xls

#endif  // XLS_CODEGEN_FLATTENING_H_

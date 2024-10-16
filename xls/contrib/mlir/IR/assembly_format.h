// Copyright 2024 The XLS Authors
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

#ifndef GDM_HW_MLIR_XLS_IR_ASSEMBLY_FORMAT_H_
#define GDM_HW_MLIR_XLS_IR_ASSEMBLY_FORMAT_H_

#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/OpImplementation.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"

namespace mlir::xls {

// Declarative `custom<SameOperandsAndResultType>(...)` implementation:
// Pretty print for ops with many operands, but one result type, simplifies
// print if all operand types match the result type.
//
// Example:
//   custom<SameOperandsAndResultType>(type($result), type($operand1),
//   type($operand2))
//
//   Generic:
//     %0 = "stablehlo.op"(%0, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
//   Custom:
//     %0 = stablehlo.op(%0, %1) : tensor<i1>
//
// Falls back to `printFunctionalType` if all operands do not match result
// type.
//
// Note that `type($result)` is the first argument, this is done because the
// behavior of trailing parameter packs is easily understandable.

namespace detail {
void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result);

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               mlir::ArrayRef<Type*> operands,
                                               Type& result);
}  // namespace detail

template <class... OpTypes>
void printSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                    OpTypes... types) {
  static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
  SmallVector<Type> typesVec{types...};
  mlir::ArrayRef<Type> typesRef = mlir::ArrayRef(typesVec);
  return detail::printSameOperandsAndResultTypeImpl(
      p, op, typesRef.drop_back(1), typesRef.back());
}

template <class... OpTypes>
ParseResult parseSameOperandsAndResultType(OpAsmParser& parser,
                                           OpTypes&... types) {
  static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
  SmallVector<Type*> typesVec{&types...};
  mlir::ArrayRef<Type*> typesRef = mlir::ArrayRef(typesVec);
  return detail::parseSameOperandsAndResultTypeImpl(
      parser, typesRef.drop_back(1), *typesRef.back());
}

void printVariadicSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                            OperandRange operands,
                                            TypeRange opTypes, Type result);

ParseResult parseVariadicSameOperandsAndResultType(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
    SmallVectorImpl<Type>& opTypes, Type& result);

void printInOutSpecifier(mlir::AsmPrinter& p, bool isInput);
ParseResult parseInOutSpecifier(mlir::AsmParser& parser, bool& isInput);

void printArrayUpdateSliceBrackets(mlir::AsmPrinter& p, Operation* op,
                                   Type arrayType, IntegerAttr width,
                                   Type sliceType);
ParseResult parseArrayUpdateSliceBrackets(mlir::AsmParser& parser,
                                          Type& arrayType, IntegerAttr& width,
                                          Type& sliceType);

void printZippedSymbols(mlir::AsmPrinter& p, Operation* op,
                        ArrayAttr globalRefs, ArrayAttr localRefs);
ParseResult parseZippedSymbols(mlir::AsmParser& parser, ArrayAttr& globalRefs,
                               ArrayAttr& localRefs);
}  // namespace mlir::xls

#endif  // GDM_HW_MLIR_XLS_IR_ASSEMBLY_FORMAT_H_

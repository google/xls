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

#include "xls/contrib/mlir/IR/assembly_format.h"

#include <cassert>
#include <string>

#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "llvm/include/llvm/Support/SMLoc.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/OpImplementation.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/Support/LLVM.h"

#define GET_TYPEDEF_CLASSES
#include "xls/contrib/mlir/IR/xls_ops_typedefs.h.inc"

namespace mlir::xls {

namespace {
// Utility function, used by printSelectOpType and
// printSameOperandsAndResultType. Given a FunctionType, assign the types
// to operands and results, erroring if any mismatch in number of operands
// or results occurs.
ParseResult assignFromFunctionType(OpAsmParser& parser, llvm::SMLoc loc,
                                   ArrayRef<Type*> operands, Type& result,
                                   FunctionType& fnType) {
  assert(fnType);
  if (fnType.getInputs().size() != operands.size()) {
    return parser.emitError(loc)
           << operands.size() << " operands present, but expected "
           << fnType.getInputs().size();
  }

  // Set operand types to function input types
  for (auto [operand, input] : llvm::zip(operands, fnType.getInputs())) {
    *operand = input;
  }

  // Set result type
  if (fnType.getResults().size() != 1) {
    return parser.emitError(loc, "expected single output");
  }
  result = fnType.getResults()[0];

  return success();
}
}  // namespace

namespace detail {
void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result) {
  // Handle zero operand types `() -> a` prints `a`
  if (operands.empty()) {
    p.printType(result);
    return;
  }

  // Handle all same type `(a,a,...) -> a` prints `a`
  bool allSameType =
      llvm::all_of(operands, [&result](auto t) { return t == result; });
  if (allSameType) {
    p.printType(result);
    return;
  }

  // Fall back to generic
  p.printFunctionalType(op);
}

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               ArrayRef<Type*> operands,
                                               Type& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type type;
  if (parser.parseType(type)) {
    return failure();
  }

  // Handle if function type, all operand types did not match result type.
  if (auto fnType = dyn_cast<FunctionType>(type)) {
    return assignFromFunctionType(parser, loc, operands, result, fnType);
  }

  // Handle bare types. ` : type` indicating all input/output types match.
  for (Type* t : operands) {
    *t = type;
  }
  result = type;
  return success();
}
}  // namespace detail

void printVariadicSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                            OperandRange operands,
                                            TypeRange opTypes, Type result) {
  (void)operands;
  return detail::printSameOperandsAndResultTypeImpl(p, op, opTypes, result);
}

ParseResult parseVariadicSameOperandsAndResultType(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
    SmallVectorImpl<Type>& opTypes, Type& result) {
  // Insert a type for each operand. Need to do this since passing the type of
  // a variadic op gives no indication of how many operands were provided.
  opTypes.resize(operands.size());

  // Make a pointer list to the operands
  SmallVector<Type*> typePtrs;
  typePtrs.reserve(opTypes.size());
  for (Type& t : opTypes) {
    typePtrs.push_back(&t);
  }

  return detail::parseSameOperandsAndResultTypeImpl(parser, typePtrs, result);
}

void printInOutSpecifier(mlir::AsmPrinter& p, bool isInput) {
  if (isInput) {
    p << "in";
  } else {
    p << "out";
  }
}

ParseResult parseInOutSpecifier(mlir::AsmParser& parser, bool& isInput) {
  if (succeeded(parser.parseOptionalKeyword("in"))) {
    isInput = true;
    return success();
  }
  if (failed(parser.parseKeyword("out"))) {
    return failure();
  }
  isInput = false;
  return success();
}

void printArrayUpdateSliceBrackets(mlir::AsmPrinter& p, Operation* op,
                                   Type arrayType, IntegerAttr width,
                                   Type sliceType) {}
ParseResult parseArrayUpdateSliceBrackets(mlir::AsmParser& parser,
                                          Type& arrayType, IntegerAttr& width,
                                          Type& sliceType) {
  // We must derive sliceType based on array and width.
  auto arrayTypeAsArray = dyn_cast<ArrayType>(arrayType);
  if (!arrayTypeAsArray) {
    return failure();
  }
  sliceType = ArrayType::get(parser.getContext(), width.getInt(),
                             arrayTypeAsArray.getElementType());
  return ParseResult::success();
}

void printZippedSymbols(mlir::AsmPrinter& p, Operation*, ArrayAttr globalRefs,
                        ArrayAttr localRefs) {
  p << "(";
  llvm::interleaveComma(llvm::zip(globalRefs, localRefs), p.getStream(),
                        [&](auto globalLocal) {
                          p.printAttribute(std::get<1>(globalLocal));
                          p << " as ";
                          p.printAttribute(std::get<0>(globalLocal));
                        });
  p << ")";
}
ParseResult parseZippedSymbols(mlir::AsmParser& parser, ArrayAttr& globalRefs,
                               ArrayAttr& localRefs) {
  SmallVector<Attribute> globals;
  SmallVector<Attribute> locals;

  if (parser.parseLParen()) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    if (failed(parser.parseCommaSeparatedList([&]() {
          Attribute global, local;
          if (parser.parseAttribute(local) || parser.parseKeyword("as") ||
              parser.parseAttribute(global)) {
            return failure();
          }
          globals.push_back(global);
          locals.push_back(local);
          return success();
        }))) {
      return failure();
    }
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }
  globalRefs = ArrayAttr::get(parser.getContext(), globals);
  localRefs = ArrayAttr::get(parser.getContext(), locals);
  return success();
}

void printChannelNamesAndTypes(mlir::AsmPrinter& p, Operation*,
                               ArrayAttr channelNames, ArrayAttr channelTypes) {
  p << "(";
  llvm::interleaveComma(llvm::zip(channelNames, channelTypes), p.getStream(),
                        [&](auto nameType) {
                          auto name = cast<StringAttr>(std::get<0>(nameType));
                          p << name.getValue() << ": ";
                          p.printAttribute(std::get<1>(nameType));
                        });
  p << ")";
}
ParseResult parseChannelNamesAndTypes(mlir::AsmParser& parser,
                                      ArrayAttr& channelNames,
                                      ArrayAttr& channelTypes) {
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;

  if (parser.parseLParen()) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    if (failed(parser.parseCommaSeparatedList([&]() {
          std::string name;
          TypeAttr type;
          if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
              parser.parseAttribute(type)) {
            return failure();
          }
          names.push_back(StringAttr::get(parser.getContext(), name));
          types.push_back(type);
          return success();
        }))) {
      return failure();
    }
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }
  channelNames = ArrayAttr::get(parser.getContext(), names);
  channelTypes = ArrayAttr::get(parser.getContext(), types);
  return success();
}

}  // namespace mlir::xls

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

#include "xls/contrib/mlir/IR/xls_ops.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

// Some of these need the keep IWYU pragma as they are required by *.inc files

#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/include/llvm/Support/ErrorHandling.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Diagnostics.h"
#include "mlir/include/mlir/IR/Dialect.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/OpImplementation.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Support/TypeID.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Transforms/InliningUtils.h"
#include "xls/contrib/mlir/IR/assembly_format.h"  // IWYU pragma: keep

// Generate enum printer/parsers.
#include "xls/contrib/mlir/IR/xls_ops_enums.cc.inc"  // IWYU pragma: keep

using namespace mlir;  // NOLINT

namespace mlir::xls {
namespace {
// Holds either a Shape or a "bad" value. The bad value is poison.
class ShapeOrBad {
 public:
  static ShapeOrBad getShaped(llvm::ArrayRef<int64_t> shape) {
    return ShapeOrBad(shape, false);
  }
  static ShapeOrBad getBad() { return ShapeOrBad({}, true); }

  // Compares two ShapeOrBads. "bad" is poison - if either this or other are bad
  // then false is returned.
  bool operator==(const ShapeOrBad& other) const {
    if (bad || other.bad) {
      return false;
    }
    return shape == other.shape;
  }

 private:
  ShapeOrBad(llvm::ArrayRef<int64_t> shape, bool bad)
      : shape(shape), bad(bad) {}

  llvm::ArrayRef<int64_t> shape;
  bool bad;
};

// Returns the shape of `type`, or the empty shape if the type is not shaped.
// Never returns Bad().
ShapeOrBad getShapeSplat(Type type) {
  if (auto t = dyn_cast<ArrayType>(type)) {
    return getShapeSplat(t.getElementType());
  }
  if (auto t = dyn_cast<ShapedType>(type)) {
    return ShapeOrBad::getShaped(t.getShape());
  }
  return ShapeOrBad::getShaped({});
}

// Returns the shape of all types in `range`, as given by GetShapeSplat(Type).
// If not all types share the same shape, Bad() is returned.
ShapeOrBad getShapeSplat(Operation::operand_type_range range) {
  std::optional<ShapeOrBad> shapeSplat;
  for (Type type : range) {
    auto result = getShapeSplat(type);
    if (!shapeSplat.has_value()) {
      shapeSplat = result;
    } else if (result != *shapeSplat) {
      return ShapeOrBad::getBad();
    }
  }
  return shapeSplat.value_or(ShapeOrBad::getBad());
}

// Verify that the Verilog import function arguments and results have names.
LogicalResult verifyImportedVerilogFunctionArgumentsAndResults(
    func::FuncOp& func) {
  // Make sure that all the arguments are named.
  unsigned numArguments = func.getNumArguments();
  for (unsigned i = 0; i < numArguments; ++i) {
    DictionaryAttr dictAttr = func.getArgAttrDict(i);
    if (!dictAttr) {
      return func->emitError()
             << "for Verilog imported function all arguments should "
                "be named (use xls.name attribute)";
    }
    StringRef attrName = "xls.name";
    if (!dictAttr.contains(attrName)) {
      func->emitError() << "for Verilog imported function all arguments"
                           " should be named (use xls.name attribute)";
      return failure();
    }
  }

  // Make sure that all the results are named.
  unsigned numResults = func.getNumResults();
  for (unsigned i = 0; i < numResults; ++i) {
    DictionaryAttr dictAttr = func.getResultAttrDict(i);
    if (!dictAttr) {
      return func->emitError()
             << "for Verilog imported function all results should "
                "be named (use xls.name attribute)";
    }
    StringRef attrName = "xls.name";
    if (!dictAttr.contains(attrName)) {
      func->emitError() << "for Verilog imported function all results should "
                           "be named (use xls.name attribute)";
      return failure();
    }
  }
  return success();
}
}  // namespace

ParseResult parseNextValuePair(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& predicates,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& values) {
  do {
    if (parser.parseLSquare() ||
        parser.parseOperand(predicates.emplace_back()) || parser.parseComma() ||
        parser.parseOperand(values.emplace_back()) || parser.parseRSquare()) {
      return failure();
    }
  } while (!parser.parseOptionalComma());
  return success();
}

void printNextValuePair(OpAsmPrinter& printer, NextValueOp /*op*/,
                        OperandRange predicates, OperandRange values) {
  bool first = true;
  for (auto [predicate, value] : llvm::zip(predicates, values)) {
    if (!first) {
      printer << ", ";
    }
    first = false;
    printer << '[';
    printer.printOperand(predicate);
    printer << ", ";
    printer.printOperand(value);
    printer << ']';
  }
}

void CallDslxOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects) {
  if (getIsPure()) {
    return;
  }

  // By default, conservatively assume all side effects.
  effects.emplace_back(mlir::MemoryEffects::Allocate::get());
  effects.emplace_back(mlir::MemoryEffects::Free::get());
  effects.emplace_back(mlir::MemoryEffects::Read::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get());
}

OpFoldResult ConstantTensorOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

OpFoldResult ConstantScalarOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

void CountedForOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getToApplyAttr()) {
    auto symRef = callee.get<SymbolRefAttr>();
    setToApplyAttr(cast<FlatSymbolRefAttr>(symRef));
    return;
  }
  // Indirect call, callee Value is the first operand.
  setOperand(0, callee.get<Value>());
}

// Verify that the xls.linkage attribute for importing Verilog is valid
// and the Verilog import function arguments and results have names.
LogicalResult verifyImportedVerilogFunction(func::FuncOp func,
                                            TranslationLinkage linkage) {
  // The Verilog import function has to have a foreign linkage attribute.
  if (linkage.getKind() != LinkageKind::kForeign) {
    return func->emitError()
           << "imported Verilog requires foreign linkage kind";
  }

  // Make sure a Verilog imported function has a body.
  if (func.getFunctionBody().getBlocks().empty()) {
    return func.emitError() << "imported verilog function should have a body";
  }
  // Now check if all the argument and resulta are.
  // TODO (lubol): This is probably better done in the
  // verifyOperationAttribute. Investigate if it can be moved there. The
  // issue is that we can't get a SymbolTable to get the import statement
  // there. It is needed for correctness and more precise checks to make
  // sure this is a Verilog import function, so we can enforce the naming of
  // arguments and results.
  if (failed(verifyImportedVerilogFunctionArgumentsAndResults(func))) {
    return failure();
  }
  return success();
}

LogicalResult ImportVerilogFileOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  SymbolTable moduleSymbolTable = symbolTable.getSymbolTable(module);
  auto allUses = moduleSymbolTable.getSymbolUses(*this, module);
  if (!allUses) {
    return success();
  }

  for (SymbolTable::SymbolUse use : *allUses) {
    auto func = dyn_cast<func::FuncOp>(use.getUser());
    if (!func) {
      continue;
    }
    auto linkage = func->getAttrOfType<TranslationLinkage>("xls.linkage");
    if (!linkage) {
      continue;
    }

    return verifyImportedVerilogFunction(func, linkage);
  }

  return success();
}

LogicalResult CountedForOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  // Check that the callee references a valid function.
  auto funcOp = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
      *this, getToApplyAttr());
  if (!funcOp) {
    return emitError("'") << getToApply()
                          << "' does not reference a valid function";
  }

  if (funcOp.getFunctionType().getInputs().drop_front() != getOperandTypes()) {
    return emitOpError("input argument mismatch between op and for body");
  }
  if (funcOp.getFunctionType().getResults().front() != getType()) {
    return emitOpError("result mismatch between op and for body");
  }

  return success();
}

LogicalResult ArrayIndexStaticOp::canonicalize(ArrayIndexStaticOp op,
                                               PatternRewriter& rewriter) {
  // (array_index (array $a $b $c), 0) -> $a
  if (ArrayOp arrayOp = op.getArray().getDefiningOp<ArrayOp>()) {
    rewriter.replaceOp(op, arrayOp.getOperand(op.getIndex()));
    return LogicalResult::success();
  }
  return LogicalResult::failure();
}

void VectorizedCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = callee.get<SymbolRefAttr>();
    return setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, callee.get<Value>());
}

LogicalResult VectorizedCallOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  // Check that the callee references a valid function.
  auto funcOp =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!funcOp) {
    return emitError("'") << getCallee()
                          << "' does not reference a valid function";
  }

  for (auto [outerType, innerType] :
       llvm::zip(getOperandTypes(), funcOp.getFunctionType().getInputs())) {
    Type expectedInnerType;
    if (auto tensorType = dyn_cast<TensorType>(outerType)) {
      expectedInnerType = tensorType.getElementType();
    } else {
      continue;
    }
    if (innerType != expectedInnerType) {
      return emitError(
                 "expected argument in callee to match scalarized call "
                 "operand type, got: ")
             << innerType << " vs expected " << expectedInnerType;
    }
  }

  for (auto [outerType, innerType] :
       llvm::zip(getResultTypes(), funcOp.getFunctionType().getResults())) {
    Type expectedInnerType;
    if (auto tensorType = dyn_cast<TensorType>(outerType)) {
      expectedInnerType = tensorType.getElementType();
    } else {
      continue;
    }
    if (innerType != expectedInnerType) {
      return emitError(
                 "Expected return type in callee to match scalarized "
                 "call result type, got: ")
             << innerType << " vs expected " << expectedInnerType;
    }
  }
  return success();
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (lhs && rhs) {
    auto result = lhs.getInt() + rhs.getInt();
    return IntegerAttr::get(getType(), result);
  }
  if (rhs && rhs.getInt() == 0) {
    return lhs;
  }
  return {};
}

OpFoldResult UmulOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (lhs && rhs) {
    auto result = lhs.getInt() * rhs.getInt();
    return IntegerAttr::get(getType(), result);
  }
  if (rhs && rhs.getInt() == 1) {
    return lhs;
  }
  return {};
}

LogicalResult UmulOp::canonicalize(UmulOp op, PatternRewriter& rewriter) {
  // (umul $a, 2^x) -> (shll $a, x)
  if (auto rhs = op.getRhs().getDefiningOp<ConstantScalarOp>()) {
    int32_t x = cast<IntegerAttr>(rhs.getValue()).getValue().exactLogBase2();
    if (x != -1) {
      rewriter.replaceOpWithNewOp<ShllOp>(
          op, op.getLhs(),
          rewriter.createOrFold<ConstantScalarOp>(
              op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(x)));
      return LogicalResult::success();
    }
  }
  return LogicalResult::failure();
}

LogicalResult ArrayIndexOp::canonicalize(ArrayIndexOp op,
                                         PatternRewriter& rewriter) {
  // (array_index $a, (constant_scalar $i)) -> (array_index_static $a, $s)
  if (auto s = op.getIndex().getDefiningOp<ConstantScalarOp>()) {
    rewriter.replaceOpWithNewOp<ArrayIndexStaticOp>(
        op, op.getType(), op.getArray(),
        cast<IntegerAttr>(s.getValue()).getInt());
    return LogicalResult::success();
  }
  return LogicalResult::failure();
}

LogicalResult TupleIndexOp::canonicalize(TupleIndexOp op,
                                         PatternRewriter& rewriter) {
  // tuple_index(tuple($x1, $x2, ...), index=N) = $xN
  if (auto defining_op = op.getOperand().getDefiningOp<TupleOp>()) {
    rewriter.replaceAllOpUsesWith(op, defining_op->getOperand(op.getIndex()));
    return success();
  }
  return failure();
}

LogicalResult InstantiateEprocOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  // Check that the callee references a valid function.
  auto eprocOp =
      symbolTable.lookupNearestSymbolFrom<EprocOp>(*this, getEprocAttr());
  if (!eprocOp) {
    return emitError("'") << getEproc() << "' does not reference a valid eproc";
  }

  for (auto ref : getGlobalChannels()) {
    auto chanOp = symbolTable.lookupNearestSymbolFrom<ChanOp>(
        *this, cast<FlatSymbolRefAttr>(ref));
    if (!chanOp) {
      return emitOpError("'") << ref << "' does not reference a valid channel";
    }
  }
  for (auto ref : getLocalChannels()) {
    auto chanOp = symbolTable.lookupNearestSymbolFrom<ChanOp>(
        *this, cast<FlatSymbolRefAttr>(ref));
    if (!chanOp) {
      return emitOpError("'") << ref << "' does not reference a valid channel";
    }
  }
  return success();
}

}  // namespace mlir::xls

namespace {
SmallVector<Type> vectorizeTypesAs(TypeRange types, Value example) {
  if (!isa<TensorType>(example.getType())) {
    return SmallVector<Type>(types.begin(), types.end());
  }
  auto shape = cast<TensorType>(example.getType()).getShape();
  SmallVector<Type> result;
  for (Type type : types) {
    result.push_back(RankedTensorType::get(shape, type));
  }
  return result;
}

Type getI1TypeOf(Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    return RankedTensorType::get(tensorType.getShape(),
                                 IntegerType::get(type.getContext(), 1));
  }
  return IntegerType::get(type.getContext(), 1);
}

}  // namespace

#define GET_OP_CLASSES
#include "xls/contrib/mlir/IR/xls_ops.cc.inc"  // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "xls/contrib/mlir/IR/xls_ops_attrs.cc.inc"  // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "xls/contrib/mlir/IR/xls_ops_typedefs.cc.inc"  // IWYU pragma: keep

#define GET_INTERFACE_CLASSES
#include "xls/contrib/mlir/IR/interfaces.cc.inc"  // IWYU pragma: keep

namespace mlir::xls {

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;
  inferredReturnTypes.push_back(TupleType::get(context, operands.getTypes()));
  return success();
}

void ForOp::getAsmBlockArgumentNames(Region& region,
                                     OpAsmSetValueNameFn setNameFn) {
  for (auto& block : region.getBlocks()) {
    const size_t initsCount = getInits().size();
    const size_t invariantCount = getInvariants().size();
    if (block.getArguments().size() != 1 + initsCount + invariantCount) {
      // Validation error.
      return;
    }
    int argNum = 0;
    setNameFn(block.getArgument(argNum++), "indvar");
    for (size_t i = 0; i < initsCount; ++i) {
      setNameFn(block.getArgument(argNum++), "carry");
    }
    for (size_t i = 0; i < invariantCount; ++i) {
      setNameFn(block.getArgument(argNum++), "invariant");
    }
  }
}

void EprocOp::print(OpAsmPrinter& printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << '(';
  llvm::interleaveComma(getBody().getArguments(), printer.getStream(),
                        [&](auto arg) { printer.printRegionArgument(arg); });
  printer << ") zeroinitializer";
  if (getDiscardable()) {
    printer << " discardable";
  }
  SmallVector<StringRef> elideAttrNames = {"sym_name", "discardable"};
  if (getMinPipelineStages() == 1) {
    elideAttrNames.push_back("min_pipeline_stages");
  }
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elideAttrNames);
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

ParseResult EprocOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true)) {
    return failure();
  }

  if (parser.parseKeyword("zeroinitializer")) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("discardable"))) {
    result.addAttribute("discardable", UnitAttr::get(parser.getContext()));
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }

  Region* body = result.addRegion();
  return parser.parseRegion(*body, args);
}

LogicalResult EprocOp::verify() {
  TupleType stateType =
      TupleType::get(getContext(), TypeRange(getStateArguments()));
  TupleType yieldedType =
      TupleType::get(getContext(), getYieldedArguments().getTypes());
  if (yieldedType != stateType) {
    return emitOpError()
           << "yielded state type does not match carried state type ("
           << yieldedType << " vs " << stateType << ")";
  }
  return success();
}

ParseResult SchanOp::parse(OpAsmParser& parser, OperationState& result) {
  Type type;
  if (parser.parseLess() || parser.parseType(type) || parser.parseGreater() ||
      parser.parseLParen()) {
    return failure();
  }
  StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes)) {
    return failure();
  }
  if (parser.parseRParen()) {
    return failure();
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.types.push_back(SchanType::get(parser.getContext(), type, false));
  result.types.push_back(SchanType::get(parser.getContext(), type, true));
  return success();
}

void SchanOp::print(OpAsmPrinter& printer) {
  printer << '<';
  printer.printType(cast<SchanType>(getResult(0).getType()).getElementType());
  printer << '>';
  printer << '(';
  printer.printString(getName());
  printer << ')';
}

void SprocOp::print(OpAsmPrinter& printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << '(';
  llvm::interleaveComma(getSpawns().getArguments(), printer.getStream(),
                        [&](auto arg) { printer.printRegionArgument(arg); });
  printer << ")";
  if (getIsTop()) {
    printer << " top";
  }
  SmallVector<StringRef> elideAttrNames = {"sym_name", "is_top"};
  if (getMinPipelineStages() == 1) {
    elideAttrNames.push_back("min_pipeline_stages");
  }
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elideAttrNames);
  printer << " {";
  printer.increaseIndent();
  printer.printNewline();
  printer << "spawns ";
  printer.printRegion(getSpawns(), /*printEntryBlockArgs=*/false);
  printer.printNewline();
  printer << "next (";
  llvm::interleaveComma(getNext().getArguments(), printer.getStream(),
                        [&](auto arg) { printer.printRegionArgument(arg); });
  printer << ") zeroinitializer ";
  printer.printRegion(getNext(), /*printEntryBlockArgs=*/false);
  printer.decreaseIndent();
  printer.printNewline();
  printer << '}';
}

ParseResult SprocOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> spawnsArgs;
  if (parser.parseArgumentList(spawnsArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true)) {
    return failure();
  }

  bool top = false;
  if (succeeded(parser.parseOptionalKeyword("top"))) {
    top = true;
  }
  result.addAttribute("is_top", BoolAttr::get(parser.getContext(), top));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }

  Region* spawns = result.addRegion();
  if (parser.parseLBrace() || parser.parseKeyword("spawns") ||
      parser.parseRegion(*spawns, spawnsArgs)) {
    return failure();
  }

  Region* next = result.addRegion();
  SmallVector<OpAsmParser::Argument> nextArgs;
  if (parser.parseKeyword("next") ||
      parser.parseArgumentList(nextArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true) ||
      parser.parseKeyword("zeroinitializer") ||
      parser.parseRegion(*next, nextArgs) || parser.parseRBrace()) {
    return failure();
  }
  return success();
}

LogicalResult SprocOp::verify() {
  TupleType stateType =
      TupleType::get(getContext(), TypeRange(getStateArguments()));
  TupleType yieldedType =
      TupleType::get(getContext(), getYieldedArguments().getTypes());
  if (yieldedType != stateType) {
    return emitOpError()
           << "yielded state type does not match carried state type ("
           << yieldedType << " vs " << stateType << ")";
  }

  ValueRange nextChannels = getNextChannels();
  ValueRange yieldedChannels = getYieldedChannels();
  if (nextChannels.size() != yieldedChannels.size()) {
    return emitOpError() << "next expects " << nextChannels.size()
                         << " channels but spawns yields "
                         << yieldedChannels.size();
  }
  for (auto [next, yielded] : llvm::zip(nextChannels, yieldedChannels)) {
    if (next.getType() != yielded.getType()) {
      return emitOpError() << "next expects channel of type " << next.getType()
                           << " but spawns yields channel of type "
                           << yielded.getType();
    }
  }

  return success();
}

SprocOp SpawnOp::resolveCallee(SymbolTableCollection* symbolTable) {
  if (symbolTable) {
    return symbolTable->lookupNearestSymbolFrom<SprocOp>(getOperation(),
                                                         getCallee());
  }
  return SymbolTable::lookupNearestSymbolFrom<SprocOp>(getOperation(),
                                                       getCallee());
}

ExternSprocOp SpawnOp::resolveExternCallee(SymbolTableCollection* symbolTable) {
  if (symbolTable) {
    return symbolTable->lookupNearestSymbolFrom<ExternSprocOp>(getOperation(),
                                                               getCallee());
  }
  return SymbolTable::lookupNearestSymbolFrom<ExternSprocOp>(getOperation(),
                                                             getCallee());
}

namespace {
template <typename T>
LogicalResult verifySpawnOpSymbolUses(SpawnOp op, T callee) {
  if (callee.getChannelArgumentTypes().size() != op.getChannels().size()) {
    return op.emitOpError()
           << "callee expects " << callee.getChannelArgumentTypes().size()
           << " channels but spawn has " << op.getChannels().size()
           << " arguments";
  }
  return success();
}
}  // namespace

LogicalResult SpawnOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  Operation* callee =
      symbolTable.lookupNearestSymbolFrom(getOperation(), getCallee());
  if (!callee) {
    return emitOpError() << "callee not found: " << getCallee();
  }
  if (auto sproc = dyn_cast<SprocOp>(callee)) {
    return verifySpawnOpSymbolUses(*this, sproc);
  }
  if (auto extern_sproc = dyn_cast<ExternSprocOp>(callee)) {
    return verifySpawnOpSymbolUses(*this, extern_sproc);
  }
  return emitOpError() << "callee is not a SprocOp or ExternSprocOp";
}

namespace {
LogicalResult verifyChannelUsingOp(Operation* op, SymbolRefAttr channelAttr,
                                   Type elementType,
                                   SymbolTableCollection& symbolTable) {
  auto chanOp = symbolTable.lookupNearestSymbolFrom<ChanOp>(op, channelAttr);
  if (!chanOp) {
    return op->emitOpError("channel symbol not found: ") << channelAttr;
  }
  if (chanOp.getType() != elementType) {
    return op->emitOpError("channel element type does not match element type (")
           << chanOp.getType() << " vs " << elementType << ")";
  }
  return success();
}

LogicalResult verifyStructuredChannelUsingOp(Operation* op, Value channel,
                                             Type elementType) {
  Type channelElementType = cast<SchanType>(channel.getType()).getElementType();
  if (channelElementType != elementType) {
    return op->emitOpError("channel element type does not match element type (")
           << channelElementType << " vs " << elementType << ")";
  }
  return success();
}

}  // namespace

LogicalResult BlockingReceiveOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  return verifyChannelUsingOp(getOperation(), getChannelAttr(),
                              getResult().getType(), symbolTable);
}

LogicalResult NonblockingReceiveOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  return verifyChannelUsingOp(getOperation(), getChannelAttr(),
                              getResult().getType(), symbolTable);
}

LogicalResult SendOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  return verifyChannelUsingOp(getOperation(), getChannelAttr(),
                              getData().getType(), symbolTable);
}

LogicalResult SBlockingReceiveOp::verify() {
  if (!cast<SchanType>(getChannel().getType()).getIsInput()) {
    return emitOpError() << "channel is not an input channel";
  }
  return verifyStructuredChannelUsingOp(getOperation(), getChannel(),
                                        getResult().getType());
}

LogicalResult SNonblockingReceiveOp::verify() {
  if (!cast<SchanType>(getChannel().getType()).getIsInput()) {
    return emitOpError() << "channel is not an input channel";
  }
  return verifyStructuredChannelUsingOp(getOperation(), getChannel(),
                                        getResult().getType());
}

LogicalResult SSendOp::verify() {
  if (!cast<SchanType>(getChannel().getType()).getIsOutput()) {
    return emitOpError() << "channel is not an output channel";
  }
  return verifyStructuredChannelUsingOp(getOperation(), getChannel(),
                                        getData().getType());
}

LogicalResult NextValueOp::verify() {
  auto types = getValues().getTypes();
  if (types.empty()) {
    return emitOpError()
           << "at least one type-predicate tuple must be present";
  }
  if (!llvm::all_of(types, [&](Type elem) { return elem == types.front(); })) {
    return emitOpError() << "all input values must have the same type";
  }
  if (types.front() != getResult().getType()) {
    return emitOpError()
           << "the type of the input values and return type must match";
  }
  return success();
}

Region& EprocOp::getBodyRegion() { return getBody(); }
::llvm::StringRef EprocOp::getName() { return getSymName(); }
Operation* EprocOp::buildTerminator(Location loc, OpBuilder& builder,
                                    ValueRange operands) {
  return builder.create<YieldOp>(loc, operands);
}

// Signature conversion for an EprocOp. Exposed as part of the
// XlsRegionOpInterface. Rewrites region arguments using the given type
// converter.
struct EprocOpSignatureConversion : public ConversionPattern {
  EprocOpSignatureConversion(mlir::MLIRContext* ctx,
                             const TypeConverter& converter)
      : ConversionPattern(EprocOp::getOperationName(), /*benefit=*/1, ctx),
        regionTypeConverter(converter) {}

  LogicalResult matchAndRewrite(
      Operation* op, llvm::ArrayRef<Value> /*operands*/,
      ConversionPatternRewriter& rewriter) const override {
    Region& region = op->getRegion(0);
    TypeConverter::SignatureConversion result(region.getNumArguments());
    if (failed(regionTypeConverter.convertSignatureArgs(
            region.getArgumentTypes(), result))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&region, regionTypeConverter,
                                           &result))) {
      return failure();
    }

    // Perform a no-op modification to inform the rewriter that we did actually
    // modify the op successfully (convertRegionTypes modifies the region).
    rewriter.modifyOpInPlace(op, [&] {});
    return success();
  }

  const TypeConverter& regionTypeConverter;  // NOLINT
};

void EprocOp::addSignatureConversionPatterns(RewritePatternSet& patterns,
                                             TypeConverter& typeConverter,
                                             ConversionTarget& target) {
  patterns.add<EprocOpSignatureConversion>(patterns.getContext(),
                                           typeConverter);
  target.addDynamicallyLegalOp(
      OperationName(EprocOp::getOperationName(), patterns.getContext()),
      [&](Operation* op) {
        return typeConverter.isLegal(op->getRegion(0).getArgumentTypes());
      });
}

Region& SprocOp::getBodyRegion() {
  // getBodyRegion is part of the XlsRegionOpInterface, so we return the region
  // that contains ops to be translated to XLS - the "next" region.
  return getNext();
}
::llvm::StringRef SprocOp::getName() { return getSymName(); }
Operation* SprocOp::buildTerminator(Location loc, OpBuilder& builder,
                                    ValueRange operands) {
  return builder.create<YieldOp>(loc, operands);
}

void SprocOp::addSignatureConversionPatterns(RewritePatternSet& patterns,
                                             TypeConverter& typeConverter,
                                             ConversionTarget& target) {
  // We can't easily convert an SprocOp's signature because it has two regions,
  // so signature conversion is non trivial. This is only needed for ArrayToBits
  // and Scalarize which always run after proc elaboration.
  llvm_unreachable("SprocOp::addSignatureConversionPatterns not implemented");
}

namespace {
// func::FuncOp adheres to the XlsRegionOpInterface if it has the attribute "
// xls".
//
// This allows us to write funcs in tests and expect XLS lowering passes to
// work on them.
struct FuncIsXlsRegionOpAdaptor
    : public XlsRegionOpInterface::ExternalModel<FuncIsXlsRegionOpAdaptor,
                                                 ::mlir::func::FuncOp> {
  static bool isSupportedRegion(Operation* op) { return op->hasAttr("xls"); }

  static Region& getBodyRegion(Operation* op) { return op->getRegion(0); }

  static StringRef getName(Operation* op) {
    return cast<mlir::func::FuncOp>(op).getName();
  }

  static Operation* buildTerminator(Location loc, OpBuilder& builder,
                                    ValueRange operands) {
    return builder.create<::mlir::func::ReturnOp>(loc, operands);
  }

  static void addSignatureConversionPatterns(RewritePatternSet& patterns,
                                             TypeConverter& typeConverter,
                                             ConversionTarget& target) {
    mlir::populateFunctionOpInterfaceTypeConversionPattern(
        mlir::func::FuncOp::getOperationName(), patterns, typeConverter);
    target.addDynamicallyLegalOp(
        OperationName(mlir::func::FuncOp::getOperationName(),
                      patterns.getContext()),
        [&](Operation* op) {
          auto funcOp = cast<mlir::func::FuncOp>(op);
          auto isLegal = [&](Type type) { return typeConverter.isLegal(type); };
          return llvm::all_of(funcOp.getArgumentTypes(), isLegal) &&
                 llvm::all_of(funcOp.getResultTypes(), isLegal);
        });
  }
};

// Interface for dialects that support unconditional inlining of function calls
// and regions contained in region-based ops from these dialects, or functions
// that contain ops of these dialects.
struct SupportsUnconditionalInliner : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation*, Operation*, bool) const final {
    return true;
  }

  bool isLegalToInline(Region*, Region*, bool, mlir::IRMapping&) const final {
    return true;
  }

  bool isLegalToInline(Operation*, Region*, bool,
                       mlir::IRMapping&) const final {
    return true;
  }
};
}  // namespace

XlsDialect::XlsDialect(mlir::MLIRContext* ctx)
    : Dialect("xls", ctx, TypeID::get<XlsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "xls/contrib/mlir/IR/xls_ops.cc.inc"  // IWYU pragma: keep
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xls/contrib/mlir/IR/xls_ops_attrs.cc.inc"  // IWYU pragma: keep
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xls/contrib/mlir/IR/xls_ops_typedefs.cc.inc"  // IWYU pragma: keep
      >();
  addInterface<SupportsUnconditionalInliner>();
  // Ensure dialect is loaded before attaching interfaces.
  (void)ctx->loadDialect<mlir::func::FuncDialect>();
  ::mlir::func::FuncOp::attachInterface<FuncIsXlsRegionOpAdaptor>(*ctx);
}

Operation* XlsDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                           Type type, Location loc) {
  return builder.create<ConstantScalarOp>(loc, type, value);
}

}  // namespace mlir::xls

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
#include <functional>
#include <optional>
#include <string>

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
#include "mlir/include/mlir/IR/OpAsmSupport.h"
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

// Generate enum printer/parsers.
#include "xls/contrib/mlir/IR/xls_ops_enums.cc.inc"  // IWYU pragma: keep

using namespace mlir;  // NOLINT

namespace mlir::xls {
namespace {

void printDimensionList(AsmPrinter& printer, ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
  printer.printKeywordOrString("x");
}

ParseResult parseDimensionList(AsmParser& parser,
                               SmallVectorImpl<int64_t>& shape) {
  return parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                   /*withTrailingX=*/true);
}

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
void printChannelNamesAndTypes(mlir::AsmPrinter& p, Operation* op,
                               ArrayAttr channelNames, ArrayAttr channelTypes);
ParseResult parseChannelNamesAndTypes(mlir::AsmParser& parser,
                                      ArrayAttr& channelNames,
                                      ArrayAttr& channelTypes);

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
                 "return type in callee mismatch with scalarized call result: ")
             << innerType << " vs expected " << expectedInnerType;
    }
  }
  return success();
}

LogicalResult VectorizedCallOp::canonicalize(VectorizedCallOp op,
                                             PatternRewriter& rewriter) {
  bool tensor =
      llvm::any_of(op.getOperands(),
                   [](Value v) { return isa<TensorType>(v.getType()); }) ||
      llvm::any_of(op.getResultTypes(),
                   [](Type t) { return isa<TensorType>(t); });
  if (!tensor) {
    // If no tensor operands or results, replace with plain call.
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getCalleeAttr(), op->getResultTypes(), op.getOperands());
    return success();
  }
  return failure();
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

LogicalResult LiteralOp::verifyRegions() {
  Block& b = getInitializerBlock();

  // Verify return op & typing:
  auto ret = dyn_cast<YieldOp>(b.getTerminator());
  if (!ret) {
    return emitOpError(
        "initializer region must be terminated by an xls.yield op");
  }
  if (ret->getNumOperands() != 1) {
    return emitOpError("initializer region must return exactly one value");
  }
  if (ret->getOperand(0).getType() != getType()) {
    return emitOpError("initializer region type ")
           << ret->getOperand(0).getType() << " does not match literal type "
           << getType();
  }
  // Verify that operations inside are allowed inside a literal region:
  for (Operation& op : b) {
    auto iface = dyn_cast<LiteralMemberOpInterface>(op);
    if (!iface || !iface.isPermissibleInsideLiteral()) {
      return op.emitError() << "op not permissible inside literal region";
    }
  }
  return success();
}

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

  return failure(
      parser.parseOptionalAttrDictWithKeyword(result.attributes).failed());
}

void SchanOp::print(OpAsmPrinter& printer) {
  printer << '<';
  printer.printType(cast<SchanType>(getResult(0).getType()).getElementType());
  printer << '>';
  printer << '(';
  printer.printString(getName());
  printer << ')';
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           {"type", "name"});
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
    return emitOpError() << "at least one type-predicate tuple must be present";
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
  return YieldOp::create(builder, loc, operands);
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
  return YieldOp::create(builder, loc, operands);
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
    return ::mlir::func::ReturnOp::create(builder, loc, operands);
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
  return ConstantScalarOp::create(builder, loc, type, value);
}

LogicalResult ArrayOp::verify() {
  int64_t num_operands = getNumOperands();
  if (num_operands == 0) {
    return emitOpError("requires at least one argument");
  }

  auto arg_type = getOperandTypes().front();
  auto result_type = cast<ArrayType>(getResult().getType());

  std::function<int64_t(Type)> get_num_elements = [&](Type x) -> int64_t {
    if (auto shape_type = dyn_cast<mlir::ShapedType>(x)) {
      return shape_type.getNumElements() *
             get_num_elements(shape_type.getElementType());
    }
    return 1;
  };

  int64_t num_elements_in_arguments =
      get_num_elements(arg_type) * getNumOperands();
  int64_t num_elements_in_result = get_num_elements(result_type);

  if (num_elements_in_arguments != num_elements_in_result) {
    return emitOpError("return mismatch between op element counts; got ")
           << num_elements_in_result << " expected "
           << num_elements_in_arguments;
  }

  return success();
}

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

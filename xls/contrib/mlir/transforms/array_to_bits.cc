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

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Threading.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

#define DEBUG_TYPE "array-to-bits"

namespace mlir::xls {

#define GEN_PASS_DEF_ARRAYTOBITSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

// Converts !xls.array<...xT> to iN.
class TensorTypeConverter : public TypeConverter {
 public:
  explicit TensorTypeConverter() {
    addConversion([](ArrayType type) -> std::optional<Type> {
      if (!isa<IntegerType, FloatType>(type.getElementType())) {
        return std::nullopt;
      }
      return IntegerType::get(
          type.getContext(),
          type.getNumElements() * type.getElementTypeBitWidth());
    });
    addConversion([this](TupleType type) {
      bool isIdentityType = true;
      SmallVector<Type> convertedTypes;
      for (Type t : type.getTypes()) {
        Type ct = convertType(t);
        if (ct != t) {
          isIdentityType = false;
        }
        convertedTypes.push_back(ct);
      }
      if (isIdentityType) {
        return type;
      }
      return TupleType::get(type.getContext(), convertedTypes);
    });
    // All types other than ArrayTypes are legal.
    addConversion([](Type ty) {
      bool b = isa<ArrayType, TupleType>(ty);
      return b ? std::nullopt : std::optional<Type>(ty);
    });
  }
};

class ConvertForOpTypes : public OpConversionPattern<ForOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ForOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter))) {
      return failure();
    }

    SmallVector<Type> resultTypes;
    if (failed(
            typeConverter->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }

    ForOp newOp = ForOp::create(rewriter, op.getLoc(), resultTypes,
                                adaptor.getOperands(), op->getAttrs());
    // Inline the type converted region from the original operation.
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class LegalizeGenericOpPattern : public ConversionPattern {
 public:
  LegalizeGenericOpPattern(TypeConverter& tyConverter, MLIRContext* context)
      : ConversionPattern(tyConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (isa<ArrayOp, ArrayUpdateOp, ArrayIndexOp, ArraySliceOp, ArrayZeroOp,
            ArrayIndexStaticOp>(op)) {
      return failure();
    }
    SmallVector<Type> resultTypes;
    if (failed(
            typeConverter->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&] {
      for (auto [result, newType] : zip(op->getResults(), resultTypes)) {
        result.setType(newType);
      }
      op->setOperands(operands);
    });
    return success();
  }
};

class LegalizeChanOpPattern : public OpConversionPattern<ChanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    rewriter.replaceOpWithNewOp<ChanOp>(
        op, op.getSymName(), typeConverter->convertType(op.getType()),
        op.getFifoConfigAttr(), op.getInputFlopKindAttr(),
        op.getOutputFlopKindAttr(), op.getSendSupported(),
        op.getRecvSupported());
    return success();
  }
};

SmallVector<Value> CoerceFloats(ValueRange operands,
                                ConversionPatternRewriter& rewriter,
                                Operation* op) {
  SmallVector<Value> result;
  for (Value v : operands) {
    if (isa<FloatType>(v.getType())) {
      result.push_back(arith::BitcastOp::create(
          rewriter, v.getLoc(),
          rewriter.getIntegerType(v.getType().getIntOrFloatBitWidth()), v));
    } else {
      if (!isa<IntegerType>(v.getType())) {
        (void)rewriter.notifyMatchFailure(op, "Unsupported array element type");
        return {};
      }
      result.push_back(v);
    }
  }
  return result;
}

class LegalizeArrayPattern : public OpConversionPattern<ArrayOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArrayOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    SmallVector<Value> operands =
        CoerceFloats(adaptor.getOperands(), rewriter, op);
    if (operands.empty() && !adaptor.getOperands().empty()) {
      return failure();
    }
    // Concat's first operand becomes the most significant bits in the result
    // so we need to reverse the operands.
    rewriter.replaceOpWithNewOp<ConcatOp>(
        op, typeConverter->convertType(op.getType()),
        llvm::to_vector(llvm::reverse(operands)));
    return success();
  }
};

Value MultiplyByBitwidth(Value value, int64_t bitwidth,
                         ConversionPatternRewriter& rewriter) {
  return UmulOp::create(
      rewriter, value.getLoc(), value.getType(), value,
      ConstantScalarOp::create(rewriter, value.getLoc(), value.getType(),
                               rewriter.getI32IntegerAttr(bitwidth)));
}

Value MultiplyByBitwidth(Value value, ArrayType array_type,
                         ConversionPatternRewriter& rewriter) {
  int64_t bitwidth = array_type.getElementTypeBitWidth();
  return MultiplyByBitwidth(value, bitwidth, rewriter);
}

class LegalizeArrayUpdatePattern : public OpConversionPattern<ArrayUpdateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArrayUpdateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Value value = CoerceFloats({adaptor.getValue()}, rewriter, op)[0];
    auto type = cast<ArrayType>(op.getType());
    Value start = MultiplyByBitwidth(adaptor.getIndex(), type, rewriter);
    rewriter.replaceOpWithNewOp<BitSliceUpdateOp>(
        op, adaptor.getArray().getType(), /*operand=*/adaptor.getArray(),
        /*start=*/start, /*update_value=*/value);
    return success();
  }
};

Type BitcastTypeToInt(Type type) {
  if (isa<FloatType>(type)) {
    return IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth());
  }
  return type;
}

class LegalizeArraySlicePattern : public OpConversionPattern<ArraySliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArraySliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Type idxType = rewriter.getIndexType();

    // Compute the `width` of `dynamic_bit_slice`, which is in given bits.
    int64_t elementBitWidth =
        cast<ArrayType>(op.getType()).getElementTypeBitWidth();
    int64_t widthInBits = adaptor.getWidth() * elementBitWidth;

    // Convert the `start` value of the `xls.array_slice` op, which is the
    // number of array elements from the left, to the `start` value of the
    // `dynamic_bit_slice` op, which is given in bits from the *right*.
    //
    // We use the following equation to do the computation:
    //
    //    array_width = left + slice_width + right
    //
    // and do all computations in number of bits, not number of elements.
    //
    //                 array_width
    //        .----------------------------.
    //        |                            |
    //        |        slice_width         |
    //        v         .-------.          v
    // array: __________XXXXXXXXX___________
    //        ^        ^         ^         ^
    //        `--------´         `---------'
    //           left               right

    // Do the static part of the computation: from the equation above, we have
    // `left + right = array_width - slice_width`, where both terms on the right
    // side are known statically.
    auto arrayBitType = cast<IntegerType>(adaptor.getArray().getType());
    int64_t arrayBitWidth = arrayBitType.getIntOrFloatBitWidth();
    int64_t leftPlusRight = arrayBitWidth - widthInBits;
    Value leftPlusRightVal =
        arith::ConstantIndexOp::create(rewriter, loc, leftPlusRight);

    // Now, we compute `right = (left + right) - left`, where the `left` term is
    // dynamic and needs to be converted to bits first.
    Value leftVal =
        MultiplyByBitwidth(adaptor.getStart(), elementBitWidth, rewriter);
    Value rightVal =
        xls::SubOp::create(rewriter, loc, idxType, leftPlusRightVal, leftVal);

    // Create the `dynamic_bit_slice` op from the arguments computed above.
    Operation* bitslice =
        DynamicBitSliceOp::create(rewriter, loc, BitcastTypeToInt(op.getType()),
                                  adaptor.getArray(), rightVal, widthInBits);
    if (bitslice->getResult(0).getType() != op.getType()) {
      bitslice = arith::BitcastOp::create(rewriter, loc, op.getType(),
                                          bitslice->getResult(0));
    }
    rewriter.replaceOp(op, bitslice->getResults());
    return success();
  }
};

class LegalizeArrayUpdateSlicePattern
    : public OpConversionPattern<ArrayUpdateSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ArrayUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Value slice = CoerceFloats({adaptor.getSlice()}, rewriter, op)[0];
    Value start = MultiplyByBitwidth(adaptor.getStart(),
                                     cast<ArrayType>(op.getType()), rewriter);

    rewriter.replaceOpWithNewOp<BitSliceUpdateOp>(
        op, adaptor.getArray().getType(), /*operand=*/adaptor.getArray(),
        /*start=*/start, /*update_value=*/slice);
    return success();
  }
};

class LegalizeArrayIndexPattern : public OpConversionPattern<ArrayIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArrayIndexOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Type type = BitcastTypeToInt(typeConverter->convertType(op.getType()));
    int64_t bitwidth = type.getIntOrFloatBitWidth();
    Value index = MultiplyByBitwidth(adaptor.getIndex(), bitwidth, rewriter);
    Operation* bitslice = DynamicBitSliceOp::create(
        rewriter, op->getLoc(), type, adaptor.getArray(), index, bitwidth);
    if (bitslice->getResult(0).getType() != op.getType()) {
      bitslice = arith::BitcastOp::create(rewriter, op->getLoc(), op.getType(),
                                          bitslice->getResult(0));
    }
    rewriter.replaceOp(op, bitslice->getResults());
    return success();
  }
};

class LegalizeArrayIndexStaticPattern
    : public OpConversionPattern<ArrayIndexStaticOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArrayIndexStaticOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Type type = BitcastTypeToInt(typeConverter->convertType(op.getType()));
    int64_t bitwidth = type.getIntOrFloatBitWidth();
    Operation* bitslice =
        BitSliceOp::create(rewriter, op->getLoc(), type, adaptor.getArray(),
                           adaptor.getIndex() * bitwidth, bitwidth);
    if (bitslice->getResult(0).getType() != op.getType()) {
      bitslice = arith::BitcastOp::create(rewriter, op->getLoc(), op.getType(),
                                          bitslice->getResult(0));
    }
    rewriter.replaceOp(op, bitslice->getResults());

    return success();
  }
};

class LegalizeArrayZeroPattern : public OpConversionPattern<ArrayZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArrayZeroOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Type type = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<ConstantScalarOp>(
        op, type,
        rewriter.getIntegerAttr(type,
                                APInt::getZero(type.getIntOrFloatBitWidth())));
    return success();
  }
};

class LegalizeArrayConcatPattern : public OpConversionPattern<ArrayConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ArrayConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    SmallVector<Value> operands =
        CoerceFloats(adaptor.getOperands(), rewriter, op);
    if (operands.empty() && !adaptor.getOperands().empty()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ConcatOp>(
        op, typeConverter->convertType(op.getType()), operands);
    return success();
  }
};

class ArrayToBitsPass : public impl::ArrayToBitsPassBase<ArrayToBitsPass> {
 public:
  void runOnOperation() final {
    TensorTypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<ChanOp>(
        [&](ChanOp op) { return typeConverter.isLegal(op.getType()); });
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      auto is_legal = [&](auto type) { return typeConverter.isLegal(type); };
      return all_of(op->getOperandTypes(), is_legal) &&
             all_of(op->getResultTypes(), is_legal);
    });
    target.addDynamicallyLegalOp<ForOp>([&](ForOp op) {
      auto is_legal = [&](auto type) { return typeConverter.isLegal(type); };
      return all_of(op->getOperandTypes(), is_legal) &&
             all_of(op->getResultTypes(), is_legal) &&
             all_of(op.getRegion().getArgumentTypes(), is_legal);
    });
    target.addIllegalOp<VectorizedCallOp, ArrayOp, ArrayUpdateOp, ArraySliceOp,
                        ArrayUpdateSliceOp, ArrayIndexOp, ArrayIndexStaticOp,
                        ArrayZeroOp, ArrayConcatOp>();
    RewritePatternSet chanPatterns(&getContext());
    chanPatterns.add<LegalizeChanOpPattern>(typeConverter, &getContext());
    FrozenRewritePatternSet frozenChanPatterns(std::move(chanPatterns));

    RewritePatternSet regionPatterns(&getContext());
    regionPatterns.add<
        // clang-format off
        ConvertForOpTypes,
        LegalizeArrayPattern,
        LegalizeArrayUpdatePattern,
        LegalizeArraySlicePattern,
        LegalizeArrayUpdateSlicePattern,
        LegalizeArrayIndexPattern,
        LegalizeArrayIndexStaticPattern,
        LegalizeArrayZeroPattern,
        LegalizeArrayConcatPattern,
        LegalizeGenericOpPattern
        // clang-format on
        >(typeConverter, &getContext());
    mlir::populateReturnOpTypeConversionPattern(regionPatterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(regionPatterns, typeConverter);

    DenseSet<OperationName> seen;
    getOperation()->walk([&](XlsRegionOpInterface op) {
      if (seen.insert(op->getName()).second) {
        op.addSignatureConversionPatterns(regionPatterns, typeConverter,
                                          target);
      }
    });
    FrozenRewritePatternSet frozenRegionPatterns(std::move(regionPatterns));

    SmallVector<XlsRegionOpInterface> regions;
    getOperation()->walk([&](Operation* op) {
      if (auto interface = dyn_cast<XlsRegionOpInterface>(op)) {
        if (interface.isSupportedRegion()) {
          regions.push_back(interface);
          return WalkResult::skip();
        }
      } else if (auto chanOp = dyn_cast<ChanOp>(op)) {
        runOnOperation(chanOp, target, frozenChanPatterns);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });

    mlir::parallelForEach(
        &getContext(), regions, [&](XlsRegionOpInterface interface) {
          runOnOperation(interface, target, frozenRegionPatterns);
        });
  }

  void runOnOperation(ChanOp operation, ConversionTarget& target,
                      FrozenRewritePatternSet& patterns) {
    if (failed(mlir::applyFullConversion(operation, target, patterns))) {
      signalPassFailure();
    }
  }

  void runOnOperation(XlsRegionOpInterface operation, ConversionTarget& target,
                      FrozenRewritePatternSet& patterns) {
    if (failed(mlir::applyFullConversion(operation, target, patterns))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::xls

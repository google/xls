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

    ForOp newOp = rewriter.create<ForOp>(op.getLoc(), resultTypes,
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
    auto newOp = rewriter.replaceOpWithNewOp<ChanOp>(
        op, op.getSymName(), typeConverter->convertType(op.getType()));
    newOp.setSendSupported(op.getSendSupported());
    newOp.setRecvSupported(op.getRecvSupported());
    return success();
  }
};

SmallVector<Value> CoerceFloats(ValueRange operands,
                                ConversionPatternRewriter& rewriter,
                                Operation* op) {
  SmallVector<Value> result;
  for (Value v : operands) {
    if (isa<FloatType>(v.getType())) {
      result.push_back(rewriter.create<arith::BitcastOp>(
          v.getLoc(),
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
  return rewriter.create<UmulOp>(
      value.getLoc(), value.getType(), value,
      rewriter.create<ConstantScalarOp>(value.getLoc(), value.getType(),
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
    auto type = cast<ArrayType>(op.getArray().getType());
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
    (void)adaptor;
    int64_t elementWidth =
        cast<ArrayType>(op.getType()).getElementTypeBitWidth();
    Value start = MultiplyByBitwidth(adaptor.getStart(),
                                     cast<ArrayType>(op.getType()), rewriter);

    Operation* bitslice = rewriter.create<DynamicBitSliceOp>(
        op->getLoc(), BitcastTypeToInt(op.getType()), adaptor.getArray(), start,
        adaptor.getWidth() * elementWidth);
    if (bitslice->getResult(0).getType() != op.getType()) {
      bitslice = rewriter.create<arith::BitcastOp>(op->getLoc(), op.getType(),
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
    Operation* bitslice = rewriter.create<DynamicBitSliceOp>(
        op->getLoc(), type, adaptor.getArray(), index, bitwidth);
    if (bitslice->getResult(0).getType() != op.getType()) {
      bitslice = rewriter.create<arith::BitcastOp>(op->getLoc(), op.getType(),
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
        rewriter.create<BitSliceOp>(op->getLoc(), type, adaptor.getArray(),
                                    adaptor.getIndex() * bitwidth, bitwidth);
    if (bitslice->getResult(0).getType() != op.getType()) {
      bitslice = rewriter.create<arith::BitcastOp>(op->getLoc(), op.getType(),
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
  void runOnOperation() override {
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

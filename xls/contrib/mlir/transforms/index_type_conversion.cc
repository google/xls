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
#include <optional>
#include <utility>

// Some of these need the keep IWYU pragma as they are required by *.inc files

#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Support/TypeID.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DECL_INDEXTYPECONVERSIONPASS
#define GEN_PASS_DEF_INDEXTYPECONVERSIONPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

class IndexTypeConverter : public TypeConverter {
 public:
  explicit IndexTypeConverter(MLIRContext &ctx, unsigned indexTypeBitWidth)
      : ctx(ctx), indexTypeBitWidth(indexTypeBitWidth) {
    convertedIndexType = IntegerType::get(&ctx, getIndexTypeBitwidth());
    addConversion(
        [this](IndexType /*type*/) { return getConvertedIndexType(); });

    addConversion([&](ArrayType type) {
      Type elementType = type.getElementType();
      Type convertedType = convertType(elementType);
      return elementType == convertedType
                 ? type
                 : ArrayType::get(&ctx, type.getNumElements(), convertedType);
    });

    addConversion([&](TupleType type) {
      bool conversionRequired = false;
      SmallVector<Type> convertedTypes;
      for (Type t : type.getTypes()) {
        Type ct = convertType(t);
        if (ct != t) {
          conversionRequired = true;
        }
        convertedTypes.push_back(ct);
      }
      if (!conversionRequired) {
        return type;
      }
      return TupleType::get(&ctx, convertedTypes);
    });

    // All other types are legal.
    addConversion([](Type ty) {
      bool b = isa<IndexType, ArrayType, TupleType>(ty);
      return b ? std::nullopt : std::optional<Type>(ty);
    });
  }

  using TypeConverter::isLegal;

  bool isLegal(mlir::Operation *op) const {
    // An non-function XlsRegionOp is legal always (reminder: meaning that it
    // has legal operand and result types) as we will not do any conversion on
    // these. Functions or any other op needs to have their types converted.
    if (auto interface = dyn_cast<XlsRegionOpInterface>(op)) {
      if (interface.isSupportedRegion() && !mlir::isa<mlir::func::FuncOp>(op)) {
        return true;
      }
    }
    return TypeConverter::isLegal(op);
  }

  MLIRContext &getContext() const { return ctx; }

  Type getConvertedIndexType() const { return convertedIndexType; }

  // Get the bit width of the index type when converted to xls.
  unsigned getIndexTypeBitwidth() const { return indexTypeBitWidth; };

 private:
  MLIRContext &ctx;
  IntegerType convertedIndexType;
  unsigned indexTypeBitWidth;
};

// Converts constant ops with index typed values/results.
// Note: this also results in converting arith constant ops to xls constant
// scalar ops if its using index type.
class LegalizeConstantIndex : public ConversionPattern {
 public:
  LegalizeConstantIndex(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/2, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IntegerAttr intAttr;
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    } else if (auto cst = dyn_cast<xls::ConstantScalarOp>(op)) {
      intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    } else {
      // Note: just returning here as this is serving same purpose as the
      // OpConversionPattern filtering so not notifying on failure here.
      return failure();
    }
    if (!intAttr) {
      return rewriter.notifyMatchFailure(
          op, "all other types should have been converted by this point");
    }
    Type resultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    rewriter.replaceOpWithNewOp<xls::ConstantScalarOp>(
        op, resultType, IntegerAttr::get(resultType, intAttr.getInt()));
    return success();
  }
};

template <typename IndexCastOpTy, typename XlsCastOpTy>
class LegalizeIndexCastOp : public OpConversionPattern<IndexCastOpTy> {
  using OpConversionPattern<IndexCastOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IndexCastOpTy op, OpConversionPattern<IndexCastOpTy>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type outType = op.getOut().getType();
    if (!isa<IndexType, IntegerType>(outType)) {
      return rewriter.notifyMatchFailure(op, "only scalar type is supported");
    }

    bool isCastToIndex = outType.isIndex();
    unsigned srcBitWidth;
    unsigned resBitWidth;
    const IndexTypeConverter *typeConverter =
        static_cast<const IndexTypeConverter *>(this->getTypeConverter());
    if (isCastToIndex) {
      srcBitWidth = op.getIn().getType().getIntOrFloatBitWidth();
      resBitWidth = typeConverter->getIndexTypeBitwidth();
    } else {
      assert(isa<IntegerType>(outType));
      srcBitWidth = typeConverter->getIndexTypeBitwidth();
      resBitWidth = outType.getIntOrFloatBitWidth();
    }

    Value in = adaptor.getIn();
    if (srcBitWidth < resBitWidth) {
      rewriter.replaceOpWithNewOp<XlsCastOpTy>(
          op, IntegerType::get(op.getContext(), resBitWidth), in);
    } else if (srcBitWidth > resBitWidth) {
      rewriter.replaceOpWithNewOp<xls::BitSliceOp>(
          op, IntegerType::get(op.getContext(), resBitWidth), in,
          /*start=*/0, /*width=*/resBitWidth);
    } else {
      rewriter.replaceOp(op, in);
    }

    return success();
  }
};

using LegalizeIndexCast =
    LegalizeIndexCastOp<mlir::arith::IndexCastOp, xls::SignExtOp>;
using LegalizeIndexCastUI =
    LegalizeIndexCastOp<mlir::arith::IndexCastUIOp, xls::ZeroExtOp>;

class LegalizeGeneralOps : public ConversionPattern {
 public:
  LegalizeGeneralOps(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Dialect *opDialect = op->getDialect();
    if (!opDialect ||
        opDialect->getTypeID() != TypeID::get<xls::XlsDialect>()) {
      return rewriter.notifyMatchFailure(
          op, "only support XlsDialect ops conversion");
    }

    // Convert result types.
    llvm::SmallVector<Type, 4> newResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           newResultTypes))) {
      return failure();
    }

    // Create a new op using the converted operands and result
    // types. If the existing op has regions, we move them to the new op and
    // convert their signature.
    OperationState newOpState(op->getLoc(), op->getName().getStringRef(),
                              operands, newResultTypes, op->getAttrs(),
                              op->getSuccessors());

    for (Region &region : op->getRegions()) {
      Region *newRegion = newOpState.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());

      TypeConverter::SignatureConversion signatureConv(
          newRegion->getNumArguments());
      if (failed(typeConverter->convertSignatureArgs(
              newRegion->getArgumentTypes(), signatureConv))) {
        return failure();
      }
      rewriter.applySignatureConversion(&newRegion->front(), signatureConv);
    }

    Operation *newOp = rewriter.create(newOpState);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class LegalizeChanOp : public OpConversionPattern<ChanOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChanOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<xls::ChanOp>(op, op.getSymName(), resultType,
                                             op.getSendSupported(),
                                             op.getRecvSupported());
    return success();
  }
};

class IndexTypeConversionPass
    : public impl::IndexTypeConversionPassBase<IndexTypeConversionPass> {
 public:
  using IndexTypeConversionPassBase::IndexTypeConversionPassBase;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    IndexTypeConverter typeConverter(ctx, indexTypeBitWidth);
    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([typeConverter](Operation *op) {
      return typeConverter.isLegal(op) &&
             llvm::all_of(op->getRegions(), [&](Region &region) {
               return typeConverter.isLegal(&region);
             });
    });
    target.addDynamicallyLegalOp<ChanOp>([&](ChanOp op) {
      return typeConverter.convertType(op.getType()) == op.getType();
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeIndexCast, LegalizeIndexCastUI, LegalizeConstantIndex,
                 LegalizeGeneralOps, LegalizeChanOp>(typeConverter, &ctx);
    if (failed(mlir::applyFullConversion(getOperation(), target,
                                         std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::xls

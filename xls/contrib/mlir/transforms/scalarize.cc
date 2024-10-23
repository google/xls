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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Support/TypeID.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

#define DEBUG_TYPE "scalarize"

namespace mlir::xls {

#define GEN_PASS_DEF_SCALARIZEPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

// Converts tensor<...xT> to !xls.array<...xT>.
class TensorTypeConverter : public TypeConverter {
 public:
  explicit TensorTypeConverter() {
    addConversion([](TensorType type) {
      return ArrayType::get(type.getContext(), type.getNumElements(),
                            type.getElementType());
    });
    addConversion([this](ArrayType type) {
      if (!isa<TensorType>(type.getElementType())) {
        return type;
      }
      return ArrayType::get(type.getContext(), type.getNumElements(),
                            convertType(type.getElementType()));
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
    // All types other than TensorTypes Or ArrayTypes are legal.
    addConversion([](Type ty) {
      bool b = isa<TensorType, ArrayType, TupleType>(ty);
      return b ? std::nullopt : std::optional<Type>(ty);
    });
  }
};

// Legalizes constant_tensor<1,...> to array<constant_scalar<1>, ...>.
class LegalizeConstantTensorPattern
    : public OpConversionPattern<ConstantTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    int size = op.getValue().getNumElements();
    Type etype = op.getValue().getElementType();
    std::vector<Value> results;
    for (int i = 0; i < size; ++i) {
      auto attr = op.getValue().getValues<Attribute>()[i];
      results.push_back(
          rewriter.create<ConstantScalarOp>(op.getLoc(), etype, attr)
              .getResult());
    }
    rewriter.replaceOpWithNewOp<ArrayOp>(
        op, ArrayType::get(op.getContext(), size, etype), results);
    return success();
  }
};

Value castTo(OpBuilder& builder, Type type, Value value) {
  if (value.getType() == type) {
    return value;
  }
  if (type.isIndex() || value.getType().isIndex()) {
    return builder.createOrFold<mlir::arith::IndexCastOp>(value.getLoc(), type,
                                                          value);
  }
  if (type.isIntOrIndexOrFloat()) {
    uint32_t dstBitWidth = type.getIntOrFloatBitWidth();
    uint32_t srcBitWidth = value.getType().getIntOrFloatBitWidth();
    if (dstBitWidth == srcBitWidth) {
      return builder.createOrFold<mlir::arith::BitcastOp>(value.getLoc(), type,
                                                          value);
    }
    if (dstBitWidth < srcBitWidth) {
      return builder.createOrFold<mlir::arith::TruncIOp>(value.getLoc(), type,
                                                         value);
    }
    assert(dstBitWidth > srcBitWidth);
    if (type.isSignedInteger()) {
      return builder.createOrFold<mlir::arith::ExtSIOp>(value.getLoc(), type,
                                                        value);
    }
    assert(type.isUnsignedInteger());
    return builder.createOrFold<mlir::arith::ExtUIOp>(value.getLoc(), type,
                                                      value);
  }
  llvm_unreachable("unsupported cast target type");
}

Value getFlattenedIndex(OpBuilder& builder, TensorType type,
                        ValueRange values) {
  Location firstLoc =
      values.empty() ? builder.getUnknownLoc() : values.front().getLoc();
  Type idxType = builder.getIndexType();
  Value acc = builder.create<mlir::arith::ConstantOp>(firstLoc, idxType,
                                                      builder.getIndexAttr(0));
  int dimMultiplier = 1;
  for (int i = type.getRank() - 1; i >= 0; --i) {
    Location loc = values[i].getLoc();
    Value multiplier = builder.create<mlir::arith::ConstantOp>(
        loc, idxType, builder.getIndexAttr(dimMultiplier));
    Value addend = builder.createOrFold<UmulOp>(
        loc, idxType, castTo(builder, idxType, values[i]), multiplier);
    acc = builder.createOrFold<AddOp>(loc, idxType, ValueRange{acc, addend});
    dimMultiplier *= type.getDimSize(i);
  }
  return acc;
}

void buildOffsetValues(ValueRange dynamicOffsets,
                       ArrayRef<int64_t> staticOffsets, Location loc,
                       RewriterBase& rewriter,
                       llvm::SmallVectorImpl<Value>& offsets) {
  offsets.reserve(offsets.size() + staticOffsets.size());
  auto dynamicOffsetsIt = dynamicOffsets.begin();
  Type idxType = rewriter.getIndexType();
  for (auto [i, offset] : llvm::enumerate(staticOffsets)) {
    Value value;
    if (ShapedType::isDynamic(offset)) {
      value = *dynamicOffsetsIt++;
    } else {
      value = rewriter.create<mlir::arith::ConstantOp>(
          loc, idxType, rewriter.getIndexAttr(offset));
    }
    offsets.push_back(value);
  }
}

// Legalizes tensor.concat to xls concat where it matches trivially given
// tensor layout assumption.
class LegalizeTensorConcatPattern
    : public OpConversionPattern<mlir::tensor::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getDim() != 0) {
      return rewriter.notifyMatchFailure(op, "dim != 0 not supported");
    }

    if (isa<IntegerType>(op.getResult().getType())) {
      rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getResult().getType(),
                                            adaptor.getOperands());
      return success();
    }

    if (isa<TensorType>(op.getResult().getType())) {
      rewriter.replaceOpWithNewOp<ArrayConcatOp>(op, op.getResult().getType(),
                                                 adaptor.getOperands());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported type");
  }
};

// Legalizes tensor.insert to array_update.
class LegalizeTensorInsertPattern
    : public OpConversionPattern<mlir::tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto flattened = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getI32Type(),
        getFlattenedIndex(rewriter, op.getType(), adaptor.getIndices()));
    rewriter.replaceOpWithNewOp<ArrayUpdateOp>(op, adaptor.getDest().getType(),
                                               adaptor.getDest(),
                                               adaptor.getScalar(), flattened);
    return success();
  }
};

// Legalizes tensor.extract to array_index.
class LegalizeTensorExtractPattern
    : public OpConversionPattern<mlir::tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto flattened = getFlattenedIndex(rewriter, op.getTensor().getType(),
                                       adaptor.getIndices());
    rewriter.replaceOpWithNewOp<ArrayIndexOp>(
        op, cast<ShapedType>(op.getTensor().getType()).getElementType(),
        adaptor.getTensor(), flattened);
    return success();
  }
};

// Returns true if the given array of sizes is contiguous.
bool isSingleContiguousSlice(ArrayRef<int64_t> sizes,
                             ArrayRef<int64_t> sourceSizes) {
  // Find and drop leading and trailing unit sizes.
  ArrayRef<int64_t> range = sizes;
  const auto* nonUnitFrontIt =
      llvm::find_if(range, [](int64_t size) { return size != 1; });
  size_t numUnitFrontSizes = std::distance(range.begin(), nonUnitFrontIt);
  int64_t numDroppedFrontDims = std::min(numUnitFrontSizes + 1, range.size());
  range = range.drop_front(numDroppedFrontDims);
  const auto nonUnitBackIt = llvm::find_if(
      llvm::reverse(range), [](int64_t size) { return size != 1; });
  size_t numUnitBackSizes =
      std::distance(llvm::reverse(range).begin(), nonUnitBackIt);
  int64_t numDroppedBackDims = std::min(numUnitBackSizes, range.size());
  range = range.drop_back(numDroppedBackDims);

  auto drop = [numDroppedFrontDims, numDroppedBackDims](auto range) {
    return range.drop_front(numDroppedFrontDims).drop_back(numDroppedBackDims);
  };
  return range == drop(sourceSizes);
}

// Legalizes `tensor.extract_slice` to `array_slice` if the extracted slice is a
// single contiguous slice in the flattened array. This is the case iff the
// sizes are of the form `(1, ..., 1, k, N, M, ...)`, where `k` is an arbitrary
// value and `N`, `M`, ... are the sizes of the corresponding dimensions in the
// input. This pattern is preferred over (and, thus, has a higher benefit than)
// `LegalizeTensorExtractSliceUnrollPattern` because it is produces a single
// `array_slice` op instead of one op per element.
class LegalizeTensorExtractSingleSlicePattern
    : public OpConversionPattern<mlir::tensor::ExtractSliceOp> {
  explicit LegalizeTensorExtractSingleSlicePattern(mlir::MLIRContext* context)
      : OpConversionPattern(context, /*benefit=*/2) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ExtractSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Bail on unsupported cases.
    for (int64_t size : op.getStaticSizes()) {
      if (ShapedType::isDynamic(size)) {
        return rewriter.notifyMatchFailure(op, "dynamic sizes not supported");
      }
    }

    for (int64_t size : op.getStaticStrides()) {
      if (size != 1) {
        return rewriter.notifyMatchFailure(op, "only unit strides supported");
      }
    }

    mlir::RankedTensorType sourceType = op.getSourceType();

    // Bail if the extracted slice is not a single contiguous slice.
    if (!isSingleContiguousSlice(op.getStaticSizes(), sourceType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "does not extract a single contiguous slice");
    }

    Location loc = op.getLoc();
    mlir::MLIRContext* context = getContext();

    // Assemble `Value`s dynamic offsets, including for the static ones.
    SmallVector<Value> offsets;
    buildOffsetValues(adaptor.getOffsets(), op.getStaticOffsets(), loc,
                      rewriter, offsets);

    // Calculate the `start` index. `getFlattenIndex` does the right thing:
    // `offsets` corresponds to the index of the first element in the slice,
    // which gives us the first index in the flattened array.
    Value start = getFlattenedIndex(rewriter, sourceType, offsets);

    // Calculate the `width`. This is simply the product of the sizes of the
    // dimensions that we are extracting since we only have a single slice.
    int64_t width = 1;
    for (int64_t size : op.getStaticSizes()) {
      width *= size;
    }

    // Replace the op with a single `array_slice` op.
    auto resultType =
        ArrayType::get(context, width, sourceType.getElementType());
    rewriter.replaceOpWithNewOp<ArraySliceOp>(
        op, resultType, adaptor.getSource(), start, width);

    return success();
  }
};

class LegalizeTensorInsertSingleSlicePattern
    : public OpConversionPattern<mlir::tensor::InsertSliceOp> {
  explicit LegalizeTensorInsertSingleSlicePattern(mlir::MLIRContext* context)
      : OpConversionPattern(context, /*benefit=*/2) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::InsertSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Bail on unsupported cases.
    for (int64_t size : op.getStaticSizes()) {
      if (ShapedType::isDynamic(size)) {
        return rewriter.notifyMatchFailure(op, "dynamic sizes not supported");
      }
    }

    for (int64_t size : op.getStaticStrides()) {
      if (size != 1) {
        return rewriter.notifyMatchFailure(op, "only unit strides supported");
      }
    }

    mlir::RankedTensorType sourceType = op.getSourceType();

    // Bail if the extracted slice is not a single contiguous slice.
    if (!isSingleContiguousSlice(op.getStaticSizes(), sourceType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "does not extract a single contiguous slice");
    }

    Location loc = op.getLoc();

    // Assemble `Value`s dynamic offsets, including for the static ones.
    SmallVector<Value> offsets;
    buildOffsetValues(adaptor.getOffsets(), op.getStaticOffsets(), loc,
                      rewriter, offsets);

    // Calculate the `start` index. `getFlattenIndex` does the right thing:
    // `offsets` corresponds to the index of the first element in the slice,
    // which gives us the first index in the flattened array.
    Value start = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(),
        getFlattenedIndex(rewriter, sourceType, offsets));

    // Replace the op with a single `array_slice` op.
    rewriter.replaceOpWithNewOp<ArrayUpdateSliceOp>(op, adaptor.getDest(),
                                                    adaptor.getSource(), start);

    return success();
  }
};

// Rewrites `tensor.extract_slice` to `tensor.extract`s and
// `tensor.from_elements`, which can then be legalized by other patterns. This
// is more general than `LegalizeTensorExtractSingleSlicePattern` but produces
// more ops and, thus, has a lower benefit.
class LegalizeTensorExtractSliceUnrollPattern
    : public OpConversionPattern<mlir::tensor::ExtractSliceOp> {
  explicit LegalizeTensorExtractSliceUnrollPattern(mlir::MLIRContext* context)
      : OpConversionPattern(context, /*benefit=*/1) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ExtractSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Bail on unsupported cases.
    for (int64_t size : op.getStaticSizes()) {
      if (ShapedType::isDynamic(size)) {
        return rewriter.notifyMatchFailure(op, "dynamic sizes not supported");
      }
    }

    for (int64_t size : op.getStaticStrides()) {
      if (size != 1) {
        return rewriter.notifyMatchFailure(op, "only unit strides supported");
      }
    }

    Location loc = op.getLoc();

    // Assemble `Value`s dynamic offsets, including static ones.
    SmallVector<Value> offsets;
    buildOffsetValues(adaptor.getOffsets(), op.getStaticOffsets(), loc,
                      rewriter, offsets);

    // Extract individual elements from the source tensor.
    SmallVector<Value> elements;
    SmallVector<Value> indices;
    indices.reserve(offsets.size());
    buildExtractOps(loc, op.getSource(), offsets, op.getStaticSizes(), rewriter,
                    indices, elements);

    // Assemble a new tensor from the extracted elements.
    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
        op, op.getResultType(), elements);

    return success();
  }

 private:
  // Builds `tensor.extract` ops for all elements of `source` in the slice
  // defined by `offsets` and `sizes`. The extracted elements are appended to
  // `elements`. The function is implemented recursively removing one dimension
  // from the front of `offsets` and `sizes` per recursion step until they are
  // empty. In the recursive case, the function iterators over all values in the
  // first dimension and calls itself recursively for the next dimension,
  // building up `indices` along the way. In the base case, the function
  // extracts the element at position given by `indices` and appends it to
  // `elements`.
  void buildExtractOps(Location loc, Value source, ValueRange offsets,
                       mlir::ArrayRef<int64_t> sizes,
                       ConversionPatternRewriter& rewriter,
                       SmallVector<Value>& indices,
                       SmallVector<Value>& elements) const {
    assert(offsets.size() == sizes.size() &&
           "offsets and sizes must have the same size");

    // Base case: extract one element at the given indices.
    if (offsets.empty()) {
      Value element =
          rewriter.create<mlir::tensor::ExtractOp>(loc, source, indices);
      elements.push_back(element);
      return;
    }

    // Recursive case: co-iterate over the first level of offsets and sizes and
    // call the function recursively for each index between offset and offset +
    // size.
    Type idxType = rewriter.getIndexType();
    for (int64_t i = 0; i < sizes[0]; ++i) {
      Value offset = castTo(rewriter, idxType, offsets[0]);
      if (i != 0) {
        Value iVal = rewriter.create<mlir::arith::ConstantOp>(
            loc, idxType, rewriter.getIndexAttr(i));
        offset = rewriter.create<xls::AddOp>(loc, offset, iVal);
      }
      indices.push_back(offset);
      buildExtractOps(loc, source, offsets.drop_front(1), sizes.drop_front(1),
                      rewriter, indices, elements);
      indices.pop_back();
    }
  }
};

// Legalizes `tensor.from_elements` to `xls.array`.
class LegalizeTensorFromElementsPattern
    : public OpConversionPattern<mlir::tensor::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::FromElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    size_t size = op.getElements().size();
    Type elementType = mlir::getElementTypeOrSelf(op.getResult());
    auto arrayType = ArrayType::get(op.getContext(), size, elementType);
    rewriter.replaceOpWithNewOp<ArrayOp>(op, arrayType, adaptor.getElements());
    return success();
  }
};

class LegalizeTensorArrayTypeFungiblePattern
    : public OpTraitConversionPattern<TensorArrayTypeFungible> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           newResultTypes))) {
      return failure();
    }

    auto* newOp = rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                                  operands, newResultTypes, op->getAttrs());
    rewriter.replaceOp(op, newOp);
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

// Legalizes tensor.empty to (array (constant_scalar<0>...))..
class LegalizeTensorEmptyPattern
    : public OpConversionPattern<mlir::tensor::EmptyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::EmptyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    rewriter.replaceOpWithNewOp<ArrayZeroOp>(
        op, ArrayType::get(op.getContext(), op.getType().getNumElements(),
                           op.getType().getElementType()));

    return success();
  }
};

class LegalizeConcatPattern : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (typeConverter->isLegal(op.getType())) {
      return failure();
    }
    auto arrayType = typeConverter->convertType(op.getType());
    // Concat is concatenating the tensor operands (not elements of the tensors
    // pointwise). Moreover, since Concat is always on the outermost dimension
    // of tensors, it is correct to just use a single Concat here on the arrays
    // corresponding to the flattened operands.
    rewriter.replaceOpWithNewOp<ConcatOp>(op, arrayType, adaptor.getOperands());
    return success();
  }
};

class LegalizePrioritySelPattern : public OpConversionPattern<PrioritySelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PrioritySelOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<PrioritySelOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getOperands());
    return success();
  }
};

// Returns the array size of the operands. Assumes that a well-formed op
// verifies that all operands that could be arrays have the same size, so
// just picks the first array operand.
//
// Returns zero if all operands are scalar.
int getArraySize(ValueRange operands) {
  for (Value operand : operands) {
    if (ArrayType atype = dyn_cast<ArrayType>(operand.getType())) {
      return atype.getNumElements();
    }
  }
  return 0;
}

// Legalizes any scalarizable op.
class LegalizeScalarizableOpPattern
    : public OpTraitConversionPattern<OpTrait::Scalarizable> {
 public:
  using OpTraitConversionPattern::OpTraitConversionPattern;

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // Generic pattern but only apply it to XLS ops.
    if (op->getDialect()->getTypeID() != TypeID::get<XlsDialect>()) {
      return failure();
    }

    int size = getArraySize(operands);
    if (size == 0) {
      // No vectors to unwrap.
      return failure();
    }

    std::map<int, SmallVector<Value>> resultArrayMembers;
    for (int i = 0; i < size; ++i) {
      SmallVector<Type> resultTypes;
      for (auto type : op->getResultTypes()) {
        resultTypes.push_back(mlir::getElementTypeOrSelf(type));
      }

      SmallVector<Value> newOperands;
      for (Value operand : operands) {
        if (ArrayType vtype = dyn_cast<ArrayType>(operand.getType())) {
          newOperands.push_back(rewriter.create<ArrayIndexStaticOp>(
              op->getLoc(), vtype.getElementType(), operand,
              rewriter.getI64IntegerAttr(i)));
        } else {
          newOperands.push_back(operand);
        }
      }

      Operation* newOp =
          rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                          newOperands, resultTypes, op->getAttrs());
      for (int j = 0, e = resultTypes.size(); j < e; ++j) {
        resultArrayMembers[j].push_back(newOp->getResult(j));
      }
    }

    SmallVector<Value> newResultArrays;
    for (auto [i, result_array] : resultArrayMembers) {
      newResultArrays.push_back(rewriter.create<ArrayOp>(
          op->getLoc(),
          ArrayType::get(rewriter.getContext(), result_array.size(),
                         result_array.front().getType()),
          result_array));
    }

    rewriter.replaceOp(op, newResultArrays);

    return success();
  }
};

// Propagates type legalization through ForOp.
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

// Rewrites return(tensor<...>) to return(array<...>).
//
// This rewrite does no checking - it is the terminus of the chain of
// legalization rewrites and its function is to accept the proposed operand
// replacements and return success.
class ReturnLikeOpPattern
    : public OpInterfaceConversionPattern<RegionBranchTerminatorOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      RegionBranchTerminatorOpInterface op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(operands); });
    return success();
  }
};

class LegalizeVectorizedCallPattern
    : public OpConversionPattern<VectorizedCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      VectorizedCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto callee =
        op->getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::func::FuncOp>(
            adaptor.getCallee());
    if (!callee) {
      return failure();
    }

    int size = getArraySize(adaptor.getOperands());
    if (size == 0) {
      // No vectors to unwrap - just call.
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, callee,
                                                      adaptor.getOperands());
      return success();
    }

    std::map<int, SmallVector<Value>> resultArrayMembers;
    for (int i = 0; i < size; ++i) {
      mlir::IRMapping mapping;
      SmallVector<Value> newOperands;
      for (Value operand : adaptor.getOperands()) {
        if (ArrayType vtype = dyn_cast<ArrayType>(operand.getType())) {
          newOperands.push_back(rewriter.create<ArrayIndexStaticOp>(
              op->getLoc(), vtype.getElementType(), operand,
              rewriter.getI64IntegerAttr(i)));
        } else {
          newOperands.push_back(operand);
        }
      }
      Operation* newOp = rewriter.create<mlir::func::CallOp>(
          op->getLoc(), callee, newOperands);
      for (int j = 0, e = newOp->getResultTypes().size(); j < e; ++j) {
        resultArrayMembers[j].push_back(newOp->getResult(j));
      }
    }

    SmallVector<Value> newResultArrays;
    for (auto [i, result_array] : resultArrayMembers) {
      newResultArrays.push_back(rewriter.create<ArrayOp>(
          op->getLoc(),
          ArrayType::get(rewriter.getContext(), result_array.size(),
                         result_array.front().getType()),
          result_array));
    }

    rewriter.replaceOp(op, newResultArrays);

    return success();
  }
};
class LegalizeCallDslxPattern : public OpConversionPattern<CallDslxOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallDslxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            typeConverter->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }

    int size = getArraySize(adaptor.getOperands());
    if (size == 0) {
      // No vectors to unwrap - just call.
      rewriter.replaceOpWithNewOp<CallDslxOp>(
          op, resultTypes, adaptor.getOperands(), op->getAttrs());
      return success();
    }

    std::map<int, SmallVector<Value>> resultArrayMembers;
    for (int i = 0; i < size; ++i) {
      mlir::IRMapping mapping;
      SmallVector<Value> newOperands;
      for (Value operand : adaptor.getOperands()) {
        if (ArrayType vtype = dyn_cast<ArrayType>(operand.getType())) {
          newOperands.push_back(rewriter.create<ArrayIndexStaticOp>(
              op->getLoc(), vtype.getElementType(), operand,
              rewriter.getI64IntegerAttr(i)));
        } else {
          newOperands.push_back(operand);
        }
      }
      SmallVector<Type> thisResultTypes;
      for (Type resultType : resultTypes) {
        if (ArrayType vtype = dyn_cast<ArrayType>(resultType)) {
          thisResultTypes.push_back(vtype.getElementType());
        } else {
          thisResultTypes.push_back(resultType);
        }
      }
      Operation* newOp = rewriter.create<CallDslxOp>(
          op->getLoc(), thisResultTypes, newOperands, op->getAttrs());
      for (int j = 0, e = newOp->getResultTypes().size(); j < e; ++j) {
        resultArrayMembers[j].push_back(newOp->getResult(j));
      }
    }

    SmallVector<Value> newResultArrays;
    for (auto [i, result_array] : resultArrayMembers) {
      newResultArrays.push_back(rewriter.create<ArrayOp>(
          op->getLoc(),
          ArrayType::get(rewriter.getContext(), result_array.size(),
                         result_array.front().getType()),
          result_array));
    }

    rewriter.replaceOp(op, newResultArrays);
    return success();
  }
};

class ScalarizePass : public impl::ScalarizePassBase<ScalarizePass> {
 public:
  void runOnOperation() override {
    getOperation()->walk([&](Operation* op) {
      if (auto interface = dyn_cast<XlsRegionOpInterface>(op)) {
        if (interface.isSupportedRegion()) {
          runOnOperation(interface);
          return WalkResult::skip();
        }
      } else if (auto chanOp = dyn_cast<ChanOp>(op)) {
        runOnOperation(chanOp);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }

  void runOnOperation(ChanOp operation) {
    TensorTypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<ChanOp>(
        [&](ChanOp op) { return typeConverter.isLegal(op.getType()); });
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeChanOpPattern>(typeConverter, &getContext());
    if (failed(mlir::applyFullConversion(operation, target,
                                         std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void runOnOperation(XlsRegionOpInterface operation) {
    TensorTypeConverter typeConverter;
    ConversionTarget target(getContext());
    // TODO(jpienaar,jmolloy): This target definition seems to broad: it allows
    // to create ops (such as `arith.addi`) that the remainder of the
    // `xls-lower` pipeline does not handle.
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      auto is_legal = [&](auto type) { return typeConverter.isLegal(type); };
      return absl::c_all_of(op->getOperandTypes(), is_legal) &&
             absl::c_all_of(op->getResultTypes(), is_legal);
    });
    target.addIllegalOp<VectorizedCallOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<
        // clang-format off
        ConvertForOpTypes,
        LegalizeCallDslxPattern,
        LegalizeChanOpPattern,
        LegalizeConcatPattern,
        LegalizeConstantTensorPattern,
        LegalizeScalarizableOpPattern,
        LegalizeTensorArrayTypeFungiblePattern,
        LegalizeTensorConcatPattern,
        LegalizeTensorEmptyPattern,
        LegalizeTensorExtractPattern,
        LegalizeTensorExtractSingleSlicePattern,
        LegalizeTensorExtractSliceUnrollPattern,
        LegalizeTensorFromElementsPattern,
        LegalizeTensorInsertSingleSlicePattern,
        LegalizeTensorInsertPattern,
        LegalizeVectorizedCallPattern,
        ReturnLikeOpPattern
        // clang-format on
        >(typeConverter, &getContext());
    operation.addSignatureConversionPatterns(patterns, typeConverter, target);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    if (failed(mlir::applyFullConversion(operation, target,
                                         std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::xls

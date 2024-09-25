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

#include <optional>
#include <string>
#include <utility>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_ARITHTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::SmallVector;
using ::mlir::func::FuncOp;

FuncOp maybeDeclareDslxFn(SymbolTable& symtab, OpBuilder builder,
                          const std::string& symbolName,
                          const std::string& dslxName,
                          xls::ImportDslxFilePackageOp importOp,
                          TypeRange results, TypeRange operands) {
  if (auto fn = symtab.lookup<FuncOp>(symbolName)) {
    return fn;
  }

  FuncOp fn = builder.create<FuncOp>(importOp->getLoc(), symbolName,
                                     builder.getFunctionType(operands, results),
                                     std::nullopt);
  fn.setVisibility(SymbolTable::Visibility::Private);
  fn->setAttr("xls.linkage",
              xls::TranslationLinkage::get(builder.getContext(),
                                           SymbolRefAttr::get(importOp),
                                           builder.getStringAttr(dslxName)));
  return fn;
}

xls::ImportDslxFilePackageOp maybeImportDslxFilePackage(
    SymbolTable& symtab, OpBuilder builder, std::string_view packageName,
    std::string_view symbolName) {
  if (auto importOp = symtab.lookup<xls::ImportDslxFilePackageOp>(symbolName)) {
    return importOp;
  }
  return builder.create<xls::ImportDslxFilePackageOp>(
      builder.getUnknownLoc(), builder.getStringAttr(packageName),
      builder.getStringAttr(symbolName));
}

struct FloatLib {
  std::string_view packageName;
  std::string_view symbolName;
};

constexpr FloatLib kExtTruncLib = {
    .packageName = "xls/contrib/mlir/stdlib/fp_ext_trunc.x",
    .symbolName = "ext_trunclib"};

constexpr std::string_view kUnknownType = "UNKNOWN_TYPE";

FloatLib getFloatLib(Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    type = tensorType.getElementType();
  }
  return llvm::TypeSwitch<Type, FloatLib>(type)
      .Case<mlir::Float32Type>([](mlir::Float32Type) {
        return FloatLib{.packageName = "xls/dslx/stdlib/float32.x",
                        .symbolName = "f32lib"};
      })
      .Case<BFloat16Type>([](BFloat16Type) {
        return FloatLib{.packageName = "xls/dslx/stdlib/bfloat16.x",
                        .symbolName = "bf16lib"};
      })
      .Case<mlir::Float64Type>([](mlir::Float64Type) {
        return FloatLib{.packageName = "xls/dslx/stdlib/float64.x",
                        .symbolName = "f64lib"};
      })
      .Default([](Type) {
        return FloatLib{.packageName = kUnknownType,
                        .symbolName = kUnknownType};
      });
}

FuncOp getOrCreateFloatLibcallSymbol(PatternRewriter& rewriter, StringAttr name,
                                     OpResult opResult,
                                     bool returnsBool = false,
                                     bool useExtTruncStdlib = false) {
  (void)rewriter;
  Operation* op = opResult.getOwner();
  auto module = op->getParentOfType<mlir::ModuleOp>();

  auto floatLib =
      useExtTruncStdlib ? kExtTruncLib : getFloatLib(opResult.getType());
  std::string symName = absl::StrCat(floatLib.symbolName, "_", name.str());

  OpBuilder builder(module.getBodyRegion());
  SymbolTable symtab(module);
  xls::ImportDslxFilePackageOp importOp = maybeImportDslxFilePackage(
      symtab, builder, floatLib.packageName, floatLib.symbolName);
  builder.setInsertionPointAfter(importOp);

  // We scalarize all operands and the result type.
  SmallVector<Type> operandTypes;
  for (Type operand : op->getOperandTypes()) {
    operandTypes.push_back(mlir::getElementTypeOrSelf(operand));
  }
  Type resultType = mlir::getElementTypeOrSelf(opResult.getType());
  if (returnsBool) {
    resultType = builder.getI1Type();
  }

  std::string dslxName = name.str();
  return maybeDeclareDslxFn(symtab, builder, symName, dslxName, importOp,
                            {resultType}, operandTypes);
}

#include "xls/contrib/mlir/transforms/arith_to_xls_patterns.inc"

class ArithToXlsPass : public impl::ArithToXlsPassBase<ArithToXlsPass> {
 public:
  void runOnOperation() override {
    getOperation()->walk([&](Operation* op) {
      if (auto interface = dyn_cast<XlsRegionOpInterface>(op)) {
        if (interface.isSupportedRegion()) {
          runOnOperation(interface);
          return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    });
  }

  void runOnOperation(XlsRegionOpInterface operation) {
    ConversionTarget target(getContext());
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                           XlsDialect>();
    target.addLegalOp<mlir::arith::BitcastOp, mlir::arith::IndexCastOp>();
    // `ConstantOp` with index value is allowed, as it is required by
    // `tensor.extract`/`insert`.
    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [](mlir::arith::ConstantOp op) {
          return mlir::getElementTypeOrSelf(op.getValue()).isIndex();
        });
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    if (failed(mlir::applyPartialConversion(operation, target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "arith-to-xls"; }

  StringRef getDescription() const override { return ""; }
};

}  // namespace
}  // namespace mlir::xls

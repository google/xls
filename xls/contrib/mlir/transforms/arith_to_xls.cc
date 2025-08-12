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
#include <string_view>
#include <utility>

#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Matchers.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Support/WalkResult.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_ARITHTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::SmallVector;
using ::mlir::StringAttr;
using ::mlir::func::FuncOp;

// clang-tidy fails to see that these are needed by the *.inc file below
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
FuncOp maybeDeclareDslxFn(SymbolTable& symtab, OpBuilder builder,
                          const std::string& symbolName,
                          const std::string& dslxName,
                          xls::ImportDslxFilePackageOp importOp,
                          TypeRange results, TypeRange operands) {
  if (auto fn = symtab.lookup<FuncOp>(symbolName)) {
    return fn;
  }

  FuncOp fn = FuncOp::create(builder, importOp->getLoc(), symbolName,
                             builder.getFunctionType(operands, results), {});
  fn.setVisibility(SymbolTable::Visibility::Private);
  fn->setAttr("xls.linkage",
              xls::TranslationLinkage::get(
                  builder.getContext(), SymbolRefAttr::get(importOp),
                  builder.getStringAttr(dslxName), /*kind=*/{}));
  return fn;
}

// clang-tidy fails to see that these are needed by the *.inc file below
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
xls::ImportDslxFilePackageOp maybeImportDslxFilePackage(
    SymbolTable& symtab, OpBuilder builder, std::string_view packageName,
    std::string_view symbolName) {
  if (auto importOp = symtab.lookup<xls::ImportDslxFilePackageOp>(symbolName)) {
    return importOp;
  }
  return xls::ImportDslxFilePackageOp::create(
      builder, builder.getUnknownLoc(), builder.getStringAttr(packageName),
      builder.getStringAttr(symbolName));
}

StringAttr getFloatLib(Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    type = tensorType.getElementType();
  }
  std::string s = llvm::TypeSwitch<Type, std::string>(type)
                      .Case<mlir::Float32Type>([](mlir::Float32Type) {
                        return "xls/dslx/stdlib/float32.x";
                      })
                      .Case<BFloat16Type>([](BFloat16Type) {
                        return "xls/dslx/stdlib/bfloat16.x";
                      })
                      .Case<mlir::Float64Type>([](mlir::Float64Type) {
                        return "xls/dslx/stdlib/float64.x";
                      })
                      .Default([](Type) { return ""; });
  if (!s.empty()) {
    return StringAttr::get(type.getContext(), s);
  }
  return {};
}

// clang-tidy fails to see that these are needed by the *.inc file below
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static Type boolLike(Operation* op) {
  Type type = op->getResultTypes().front();
  mlir::Builder builder(op->getContext());
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.clone(builder.getI1Type());
  }
  return builder.getI1Type();
}

#include "xls/contrib/mlir/transforms/arith_to_xls_patterns.inc"

class ArithToXlsPass : public impl::ArithToXlsPassBase<ArithToXlsPass> {
 public:
  void runOnOperation() final {
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
    target.addLegalOp<mlir::arith::BitcastOp, mlir::arith::IndexCastOp,
                      mlir::arith::IndexCastUIOp>();
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

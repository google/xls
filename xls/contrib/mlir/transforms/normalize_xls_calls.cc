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

#include <filesystem>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/ADT/StringMap.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/WalkResult.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_NORMALIZEXLSCALLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::dyn_cast;
using ::llvm::StringRef;

class NormalizeXlsCallsPass
    : public impl::NormalizeXlsCallsPassBase<NormalizeXlsCallsPass> {
 public:
  void runOnOperation() override;
};

}  // namespace

void NormalizeXlsCallsPass::runOnOperation() {
  llvm::StringMap<std::vector<ImportDslxFilePackageOp>> packageImports;
  llvm::DenseMap<std::pair<StringRef, StringRef>,
                 std::vector<mlir::func::FuncOp>>
      fnOps;
  SymbolTable symbolTable(getOperation());
  OpBuilder builder(getOperation().getBodyRegion());
  getOperation()->walk([&](Operation* op) {
    if (auto import = dyn_cast<ImportDslxFilePackageOp>(op)) {
      packageImports[import.getFilename()].push_back(import);
      return WalkResult::advance();
    }
    if (auto call = dyn_cast<CallDslxOp>(op)) {
      auto [fIt, fnInserted] =
          fnOps.insert({{call.getFilename(), call.getFunction()}, {}});
      if (fnInserted) {
        auto [it, inserted] = packageImports.insert({call.getFilename(), {}});
        std::filesystem::path path(call.getFilename().str());
        if (inserted) {
          auto pkgImport = ImportDslxFilePackageOp::create(
              builder, op->getLoc(), call.getFilenameAttr(),
              builder.getStringAttr(path.stem().string()));
          // Ensure unique symbol name.
          symbolTable.insert(pkgImport);
          it->second.push_back(pkgImport);
        }

        FunctionType newType;
        if (call.getIsVectorCall()) {
          auto scalarize = [](TypeRange r) {
            return llvm::to_vector(llvm::map_range(
                r, [](Type t) { return getElementTypeOrSelf(t); }));
          };
          newType = builder.getFunctionType(scalarize(call->getOperandTypes()),
                                            scalarize(call->getResultTypes()));
        } else {
          newType = builder.getFunctionType(call->getOperandTypes(),
                                            call->getResultTypes());
        }

        auto func = mlir::func::FuncOp::create(
            builder, op->getLoc(),
            llvm::formatv("{0}_{1}", path.stem(), call.getFunction()).str(),
            newType);
        func.setVisibility(SymbolTable::Visibility::Private);
        func->setAttr("xls.linkage", xls::TranslationLinkage::get(
                                         builder.getContext(),
                                         SymbolRefAttr::get(it->second.front()),
                                         call.getFunctionAttr(), /*kind=*/{}));
        // Ensure unique symbol name.
        symbolTable.insert(func);
        fIt->second.push_back(func);
      }

      OpBuilder b(op);
      Operation* fnCall;
      if (call.getIsVectorCall()) {
        fnCall = mlir::xls::VectorizedCallOp::create(
            b, op->getLoc(), fIt->second.front(), call.getOperands());
      } else {
        fnCall = mlir::func::CallOp::create(
            b, op->getLoc(), fIt->second.front(), call.getOperands());
      }
      op->replaceAllUsesWith(fnCall->getResults());
      op->erase();
    }
    return WalkResult::advance();
  });

  // TODO(jpienaar): Remove all duplicate package imports.
}

}  // namespace mlir::xls

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

#include <utility>

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"
#include "mlir/include/mlir/IR/AttrTypeSubElements.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::xls {

#define GEN_PASS_DEF_INSTANTIATEEPROCSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {
using ::llvm::StringRef;

class InstantiateEprocsPass
    : public impl::InstantiateEprocsPassBase<InstantiateEprocsPass> {
 public:
  void runOnOperation() override;
};

class InstantiateEprocPattern : public OpRewritePattern<InstantiateEprocOp> {
 public:
  InstantiateEprocPattern(MLIRContext* context, SymbolTable& symbolTable,
                          DenseMap<EprocOp, StringAttr>& eprocToOriginalName)
      : OpRewritePattern<InstantiateEprocOp>(context),

        symbolTable(symbolTable),
        eprocToOriginalName(eprocToOriginalName) {}

  LogicalResult matchAndRewrite(InstantiateEprocOp op,
                                PatternRewriter& rewriter) const override {
    StringAttr eprocName = op.getEprocAttr().getLeafReference();
    EprocOp eproc = symbolTable.lookup<EprocOp>(eprocName);
    if (!eproc) {
      return failure();
    }

    int instantiationIndex =
        eprocToInstantiationIndex[{eproc, op.getNameAttr()}]++;
    Operation* cloned = rewriter.clone(*eproc);

    StringAttr name = eprocToOriginalName.contains(eproc)
                          ? eprocToOriginalName.at(eproc)
                          : eprocName;
    if (op.getNameAttr()) {
      name = op.getNameAttr();
    }
    if (instantiationIndex > 0) {
      name = rewriter.getStringAttr(
          llvm::formatv("{0}_{1}", name, instantiationIndex));
    }
    cast<EprocOp>(cloned).setSymName(name);
    // All instantiated eprocs now must be kept, unlike the template they have
    // been cloned from.
    cast<EprocOp>(cloned).setDiscardable(false);
    symbolTable.insert(cloned);

    DenseMap<StringRef, StringRef> localToGlobal;
    for (auto [global, local] :
         llvm::zip(op.getGlobalChannels(), op.getLocalChannels())) {
      localToGlobal[cast<FlatSymbolRefAttr>(local).getValue()] =
          cast<FlatSymbolRefAttr>(global).getValue();
    }
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](SymbolRefAttr attr) {
      return SymbolRefAttr::get(op.getContext(),
                                localToGlobal[attr.getLeafReference()]);
    });
    replacer.recursivelyReplaceElementsIn(cloned);
    op.erase();
    return success();
  }

 private:
  SymbolTable& symbolTable;
  DenseMap<EprocOp, StringAttr> eprocToOriginalName;
  mutable DenseMap<std::pair<EprocOp, StringAttr>, int>
      eprocToInstantiationIndex;
};

}  // namespace

void InstantiateEprocsPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);
  OpBuilder builder(module);

  // Rename all discardable eprocs by adding "_template" to their name. This
  // frees up the original name for the first instantiation.
  DenseMap<EprocOp, StringAttr> eprocToOriginalName;
  module.walk([&](EprocOp eproc) {
    if (eproc.getDiscardable()) {
      eprocToOriginalName[eproc] = eproc.getSymNameAttr();
      StringAttr newSymName = builder.getStringAttr(
          llvm::formatv("{0}_template", eproc.getSymName()));
      if (failed(symbolTable.rename(eproc, newSymName))) {
        eproc.emitOpError("failed to rename during instantiation");
      }
    }
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<InstantiateEprocPattern>(patterns.getContext(), symbolTable,
                                        eprocToOriginalName);
  if (failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::xls

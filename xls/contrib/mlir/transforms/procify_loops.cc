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

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Transforms/FoldUtils.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/passes.h"  // IWYU pragma: keep
#include "xls/contrib/mlir/util/proc_utils.h"

namespace mlir::xls {

#define GEN_PASS_DEF_PROCIFYLOOPSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {
namespace scf = ::mlir::scf;

class ProcifyLoopsPass : public impl::ProcifyLoopsPassBase<ProcifyLoopsPass> {
 public:
  using ProcifyLoopsPassBase::ProcifyLoopsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    // We need to transform in pre-order, but MLIR's walkers require that if we
    // erase forOp we must interrupt the walk. So we run the walk to fixpoint.
    bool changed = true;
    while (changed) {
      changed = false;
      module->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) mutable {
        changed |= runOnOperation(forOp, symbolTable);
        return changed ? WalkResult::interrupt() : WalkResult::advance();
      });
    }
  }

  bool runOnOperation(scf::ForOp forOp, SymbolTable& symbolTable) {
    if (!shouldTransform(forOp)) {
      return false;
    }
    SprocOp sproc = forOp->getParentOfType<SprocOp>();
    if (!sproc) {
      forOp.emitError("procify_loops expected op to have a parent sproc");
      return false;
    }
    return succeeded(convertForOpToSprocCall(forOp, symbolTable));
  }

  bool shouldTransform(scf::ForOp forOp) {
    if (auto attr = forOp->getAttrOfType<BoolAttr>("xls.unroll")) {
      return !attr.getValue();
    }
    return apply_by_default;
  }
};

}  // namespace
}  // namespace mlir::xls

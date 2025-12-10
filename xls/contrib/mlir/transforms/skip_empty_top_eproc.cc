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

#include "llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::xls {

#define GEN_PASS_DEF_SKIPEMPTYTOPEPROCPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

bool IsEmptyEproc(EprocOp eproc) {
  return eproc.getBody().front().without_terminator().empty();
}

class SkipEmptyTopEprocPass
    : public impl::SkipEmptyTopEprocPassBase<SkipEmptyTopEprocPass> {
  using SkipEmptyTopEprocPassBase::SkipEmptyTopEprocPassBase;

 public:
  void runOnOperation() override {
    if (!top_proc_name.has_value()) {
      return;
    }

    ModuleOp module_op = getOperation();
    auto eprocs = llvm::to_vector(module_op.getOps<EprocOp>());
    if (eprocs.size() != 2) {
      return;
    }

    EprocOp new_top_eproc = eprocs[0];
    EprocOp current_empty_top_eproc = eprocs[1];
    assert(!(new_top_eproc.getSymName() == *top_proc_name &&
             current_empty_top_eproc.getSymName() == *top_proc_name) &&
           "Only one eproc must have the top-level name");
    if (new_top_eproc.getSymName() == *top_proc_name) {
      std::swap(new_top_eproc, current_empty_top_eproc);
    } else if (current_empty_top_eproc.getSymName() != *top_proc_name) {
      // Nothing to do here then.
      return;
    }

    if (!IsEmptyEproc(current_empty_top_eproc)) {
      // Nothing to do here. This top proc is not empty.
      return;
    }

    // Step 1: Erase all instances of this empty eproc.
    auto uses = *current_empty_top_eproc.getSymbolUses(module_op);
    for (auto& use : uses) {
      use.getUser()->erase();
    }

    // Step 2: Change the name of the non-empty non-top eproc to be the name of
    // the empty one. Both eproc and instances need to be updated.
    uses = *new_top_eproc.getSymbolUses(module_op);
    for (auto& use : uses) {
      cast<InstantiateEprocOp>(use.getUser())
          .setEproc(current_empty_top_eproc.getSymName());
    }
    new_top_eproc.setSymName(current_empty_top_eproc.getSymName());

    // Step 3: Erase the empty eproc.
    current_empty_top_eproc.erase();
  }
};

}  // namespace
}  // namespace mlir::xls

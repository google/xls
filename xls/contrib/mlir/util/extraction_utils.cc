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

#include "xls/contrib/mlir/util/extraction_utils.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>

#include "llvm/include/llvm/Support/FormatVariadic.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Analysis/CallGraph.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/TypeID.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {
namespace {
ModuleOp extractOpAsTopLevelModule(CallableOpInterface op,
                                   mlir::CallGraph& callGraph,
                                   StringRef symName) {
  CallGraphNode* node = callGraph.lookupNode(op.getCallableRegion());
  if (!node) {
    op->emitOpError("failed to find call graph node");
    return {};
  }
  SmallVector<CallGraphNode*> worklist(1, node);
  SetVector<CallGraphNode*> visited;
  while (!worklist.empty()) {
    CallGraphNode* node = worklist.back();
    worklist.pop_back();
    if (node->isExternal() || !visited.insert(node)) {
      continue;
    }
    for (auto& edge : *node) {
      worklist.push_back(edge.getTarget());
    }
  }

  // CallGraph categorizes all declaration nodes (ChanOp, FuncOp declarations)
  // as external, so go through all the regions we found and look inside them
  // for declarations.
  SetVector<Operation*> needed;
  SymbolTableCollection symbolTable;
  for (CallGraphNode* node : visited) {
    needed.insert(node->getCallableRegion()->getParentOp());
    node->getCallableRegion()->walk([&](CallOpInterface op) {
      Operation* callee = op.resolveCallableInTable(&symbolTable);
      if (!callee) {
        return;
      }
      needed.insert(callee);

      if (auto linkage =
              callee->getAttrOfType<TranslationLinkage>("xls.linkage")) {
        needed.insert(
            symbolTable.lookupNearestSymbolFrom(op, linkage.getPackage()));
      }
    });
  }

  // Now we know all the operations we need to copy into the new module.
  ModuleOp oldModule = op->getParentOfType<ModuleOp>();
  ModuleOp newModule = ModuleOp::create(op->getLoc(), symName);
  OpBuilder builder(newModule.getBodyRegion());
  for (Operation& op : oldModule.getOps()) {
    if (needed.contains(&op)) {
      builder.clone(op);
    }
  }
  return newModule;
}
}  // namespace

ModuleOp extractAsTopLevelModule(EprocOp op, mlir::CallGraph& callGraph) {
  ModuleOp newModule =
      extractOpAsTopLevelModule(op, callGraph, op.getSymName());
  if (!newModule) {
    return {};
  }

  // Fix up channel send/receive support.
  DenseSet<StringRef> sentToChans, receivedFromChans;
  newModule.walk([&](Operation* op) {
    if (auto send = dyn_cast<SendOp>(op)) {
      sentToChans.insert(send.getChannel().getLeafReference());
    } else if (auto recv = dyn_cast<BlockingReceiveOp>(op)) {
      receivedFromChans.insert(recv.getChannel().getLeafReference());
    } else if (auto recv = dyn_cast<NonblockingReceiveOp>(op)) {
      receivedFromChans.insert(recv.getChannel().getLeafReference());
    }
  });

  for (ChanOp chanOp : newModule.getOps<ChanOp>()) {
    bool sentTo = sentToChans.contains(chanOp.getSymName());
    bool receivedFrom = receivedFromChans.contains(chanOp.getSymName());
    chanOp.setSendSupported(sentTo);
    chanOp.setRecvSupported(receivedFrom);
  }
  return newModule;
}

namespace {
void addBoundaryChannelNames(
    SprocOp op, std::function<std::string(BlockArgument)> boundaryChannelName) {
  int64_t argIndex = 0;
  int64_t resultIndex = 0;
  if (!boundaryChannelName) {
    boundaryChannelName = [&](BlockArgument arg) {
      if (cast<SchanType>(arg.getType()).getIsInput()) {
        return llvm::formatv("arg{}", argIndex++);
      }
      return llvm::formatv("result{}", resultIndex++);
    };
  }

  if (op.getBoundaryChannelNames().has_value()) {
    return;
  }
  ::mlir::OpBuilder builder(op.getOperation());
  ::mlir::SmallVector<::mlir::Attribute> boundaryChannelNames;
  for (Value value : op.getChannelArguments()) {
    boundaryChannelNames.push_back(
        builder.getStringAttr(boundaryChannelName(cast<BlockArgument>(value))));
  }
  op.setBoundaryChannelNamesAttr(builder.getArrayAttr(boundaryChannelNames));
}
}  // namespace

ModuleOp extractAsTopLevelModule(
    SprocOp op, mlir::CallGraph& callGraph,
    std::function<std::string(BlockArgument)> boundaryChannelName) {
  ModuleOp newModule =
      extractOpAsTopLevelModule(op, callGraph, op.getSymName());
  if (!newModule) {
    return {};
  }
  newModule.walk([&](SprocOp newOp) {
    if (newOp.getSymName() != op.getSymName()) {
      return;
    }
    addBoundaryChannelNames(newOp, boundaryChannelName);
    newOp.setIsTop(true);
  });
  return newModule;
}

ModuleOp extractAsTopLevelModule(mlir::func::FuncOp op,
                                 mlir::CallGraph& callGraph) {
  return extractOpAsTopLevelModule(op, callGraph, op.getSymName());
}

namespace {
// Test pass for extractAsTopLevelModule, defined here to avoid exposing it in
// the passes.h header.
struct TestExtractAsTopLevelModulePass
    : public PassWrapper<TestExtractAsTopLevelModulePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestExtractAsTopLevelModulePass)

  StringRef getArgument() const final {
    return "test-extract-as-top-level-module";
  }
  StringRef getDescription() const final {
    return "Extracts the function in 'test.name' as a top level module.";
  }
  void runOnOperation() override {
    auto attr =
        cast<StringAttr>(getOperation()->getDiscardableAttr("test.name"));
    llvm::outs() << "Testing : " << attr << "\n";
    Operation* op = getOperation().lookupSymbol(attr.getValue());
    assert(op);
    ModuleOp newModule;
    if (auto eproc = dyn_cast<EprocOp>(op)) {
      newModule = extractAsTopLevelModule(eproc, getAnalysis<CallGraph>());
    } else if (auto sproc = dyn_cast<SprocOp>(op)) {
      newModule = extractAsTopLevelModule(sproc, getAnalysis<CallGraph>());
    } else if (auto func = dyn_cast<mlir::func::FuncOp>(op)) {
      newModule = extractAsTopLevelModule(func, getAnalysis<CallGraph>());
    } else {
      getOperation().emitError("Unknown op type for extraction in test");
      signalPassFailure();
      return;
    }
    getOperation().getBodyRegion().takeBody(newModule.getBodyRegion());
    newModule.erase();
  }
};
}  // namespace

namespace test {
void registerTestExtractAsTopLevelModulePass() {
  PassRegistration<TestExtractAsTopLevelModulePass>();
}
}  // namespace test

}  // namespace mlir::xls

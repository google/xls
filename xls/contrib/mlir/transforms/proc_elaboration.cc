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
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"
#include "llvm/include/llvm/ADT/StringMap.h"
#include "llvm/include/llvm/ADT/StringRef.h"  // IWYU pragma: keep
#include "llvm/include/llvm/Support/FormatVariadic.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/passes.h"  // IWYU pragma: keep
#include "xls/contrib/mlir/util/interpreter.h"

namespace mlir::xls {

#define GEN_PASS_DEF_PROCELABORATIONPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::ArrayRef;

// Deliberately chosen to be a suffix that is unlikely to appear in a user
// defined symbol name.
constexpr StringLiteral suffix = "___pre_elaboration___";

// Appends a string to the end of the given operation's symbol name. We do this
// to free up the original name of an sproc for an eproc.
void appendToSymbolName(Operation* op, SymbolUserMap& symbolUserMap) {
  auto symName = cast<StringAttr>(op->getAttr("sym_name"));
  auto newSymName = StringAttr::get(op->getContext(), Twine(symName) + suffix);
  symbolUserMap.replaceAllUsesWith(op, newSymName);
  op->setAttr("sym_name", newSymName);
}

// Returns the original symbol name, before appendToSymbolName.
StringRef originalSymbolName(Operation* op) {
  auto symName = cast<StringAttr>(op->getAttr("sym_name"));
  return symName.getValue().drop_back(suffix.size());
}

// Replaces all structured channel ops in a region with the corresponding
// unstructured channel op (ssend -> send, etc). The channel map is a mapping
// from the channel value in the structured op to the symbol ref of the
// corresponding chan op.
void replaceStructuredChannelOps(Region& region,
                                 llvm::DenseMap<Value, SymbolRefAttr> chanMap) {
  mlir::IRRewriter rewriter(region.getContext());
  region.walk([&](SBlockingReceiveOp srecv) {
    rewriter.setInsertionPoint(srecv);
    rewriter.replaceOpWithNewOp<BlockingReceiveOp>(
        srecv, srecv->getResultTypes(), srecv.getTkn(), srecv.getPredicate(),
        chanMap[srecv.getChannel()]);
  });
  region.walk([&](SNonblockingReceiveOp srecv) {
    rewriter.setInsertionPoint(srecv);
    rewriter.replaceOpWithNewOp<NonblockingReceiveOp>(
        srecv, srecv->getResultTypes(), srecv.getTkn(), srecv.getPredicate(),
        chanMap[srecv.getChannel()]);
  });
  region.walk([&](SSendOp ssend) {
    rewriter.setInsertionPoint(ssend);
    rewriter.replaceOpWithNewOp<SendOp>(ssend, ssend.getTkn(), ssend.getData(),
                                        ssend.getPredicate(),
                                        chanMap[ssend.getChannel()]);
  });
}

class ProcElaborationPass
    : public impl::ProcElaborationPassBase<ProcElaborationPass> {
 public:
  void runOnOperation() override;
};

struct EprocAndChannels {
  // A discardable eproc.
  EprocOp eproc;
  // The channels used by the eproc.
  std::vector<ChanOp> channels;
};

class ElaborationContext
    : public InterpreterContext<ElaborationContext, ChanOp> {
 public:
  explicit ElaborationContext(OpBuilder& builder, SymbolTable& symbolTable,
                              DenseMap<SprocOp, EprocAndChannels>& procCache)
      : builder(builder), symbolTable(symbolTable), procCache(procCache) {}

  OpBuilder& getBuilder() { return builder; }

  // Creates an eproc for the given sproc if none has yet been created. New
  // local channels are created for the eproc and are returned.
  //
  // Eprocs are cached such that a sproc is only elaborated once.
  EprocAndChannels createEproc(SprocOp sproc) {
    if (auto it = procCache.find(sproc); it != procCache.end()) {
      return it->second;
    }

    StringRef originalName = originalSymbolName(sproc);
    EprocOp eproc = builder.create<EprocOp>(sproc.getLoc(), originalName,
                                            /*discardable=*/true,
                                            sproc.getMinPipelineStages());
    symbolTable.insert(eproc);
    IRMapping mapping;
    sproc.getNext().cloneInto(&eproc.getBody(), mapping);
    llvm::DenseMap<Value, SymbolRefAttr> chanMap;
    std::vector<value_type> eprocChannels;
    for (auto [i, arg] : llvm::enumerate(sproc.getNextChannels())) {
      auto chan = builder.create<ChanOp>(
          sproc.getLoc(), absl::StrFormat("%s_arg%d", originalName.str(), i),
          cast<SchanType>(arg.getType()).getElementType());
      eprocChannels.push_back(chan);
      chanMap[mapping.lookup(arg)] = SymbolRefAttr::get(chan.getSymNameAttr());
    }
    replaceStructuredChannelOps(eproc.getBody(), chanMap);
    eproc.getBody().front().eraseArguments(0, sproc.getNextChannels().size());

    EprocAndChannels result = {eproc, std::move(eprocChannels)};
    procCache[sproc] = result;
    return result;
  }

  void instantiateEproc(const EprocAndChannels& eprocAndChannels,
                        ArrayRef<value_type> globalChannels,
                        StringAttr nameHint = {}) {
    EprocOp eproc = eprocAndChannels.eproc;
    ArrayRef<value_type> localChannels = eprocAndChannels.channels;
    assert(globalChannels.size() == localChannels.size());
    auto flatchan = [](value_type chan) -> Attribute {
      return FlatSymbolRefAttr::get(chan.getSymNameAttr());
    };
    SmallVector<Attribute> globalSymbols =
        llvm::map_to_vector(globalChannels, flatchan);
    SmallVector<Attribute> localSymbols =
        llvm::map_to_vector(localChannels, flatchan);
    builder.create<InstantiateEprocOp>(
        eproc.getLoc(), eproc.getSymName(), builder.getArrayAttr(globalSymbols),
        builder.getArrayAttr(localSymbols), nameHint);
  }

  void instantiateExternEproc(ExternSprocOp externSproc,
                              ArrayRef<value_type> globalChannels,
                              StringAttr nameHint = {}) {
    auto flatchan = [](value_type chan) -> Attribute {
      return FlatSymbolRefAttr::get(chan.getSymNameAttr());
    };
    SmallVector<Attribute> globalSymbols =
        llvm::map_to_vector(globalChannels, flatchan);
    StringAttr symName = externSproc.getSymNameAttr();
    builder.create<InstantiateExternEprocOp>(
        externSproc.getLoc(), symName, builder.getArrayAttr(globalSymbols),
        externSproc.getBoundaryChannelNames(), nameHint);
  }

  SymbolTable& getSymbolTable() { return symbolTable; }

  // Tries to make the given name unique by appending a number to the end. This
  // generates better names than SymbolTable::insert as it uses a different
  // uniquer for each base name.
  StringAttr Uniquify(StringAttr name) {
    int& uniqueId = uniqueIds[name];
    ++uniqueId;
    if (uniqueId == 1) {
      return name;
    }
    return builder.getStringAttr(llvm::formatv("{0}_{1}", name, uniqueId));
  }

 private:
  OpBuilder& builder;
  SymbolTable& symbolTable;
  DenseMap<SprocOp, EprocAndChannels>& procCache;
  llvm::StringMap<int> uniqueIds;
};

class ElaborationInterpreter
    : public Interpreter<ElaborationInterpreter, ElaborationContext, ChanOp,
                         SchanOp, YieldOp, SpawnOp> {
 public:
  using Interpreter::Interpret;

  template <typename... T>
  absl::Status InterpretTop(SprocOp sproc,
                            ArrayRef<ChanOp> boundaryChannels = {},
                            T&&... contextArgs) {  // NOLINT
    ElaborationContext ctx(std::forward<T>(contextArgs)...);
    ctx.PushLiveness(GetOrCreateLiveness(sproc));
    return Interpret(sproc, ctx, boundaryChannels);
  }

  absl::Status Interpret(SchanOp op, ElaborationContext& ctx) {
    std::string name = op.getName().str();
    auto uniqueName = ctx.Uniquify(op.getNameAttr());
    ChanOp chan =
        ctx.getBuilder().create<ChanOp>(op.getLoc(), uniqueName, op.getType());
    ctx.getSymbolTable().insert(chan);
    ctx.Set(op.getResult(0), chan);
    ctx.Set(op.getResult(1), chan);

    return absl::OkStatus();
  }

  absl::Status Interpret(YieldOp, ElaborationContext&) {
    return absl::OkStatus();
  }

  absl::Status Interpret(SpawnOp op, ElaborationContext& ctx) {
    ExternSprocOp externSproc = op.resolveExternCallee();
    if (externSproc) {
      return InterpretSpawnOfExtern(op, externSproc, ctx);
    }
    SprocOp sproc = op.resolveCallee();
    if (!sproc) {
      return absl::InvalidArgumentError("failed to resolve callee");
    }

    ctx.PushLiveness(GetOrCreateLiveness(sproc));
    absl::Cleanup popper = [&] { ctx.PopLiveness(); };

    XLS_ASSIGN_OR_RETURN(auto arguments, ctx.Get(op.getChannels()));
    if (arguments.size() != sproc.getChannelArguments().size()) {
      return absl::InternalError(absl::StrFormat(
          "Call to %s requires %d arguments but got %d",
          op.getCallee().getLeafReference().str(),
          sproc.getChannelArguments().size(), arguments.size()));
    }
    XLS_ASSIGN_OR_RETURN(auto results,
                         Interpret(sproc.getSpawns(), arguments, ctx));
    auto eproc_channels = ctx.createEproc(sproc);
    ctx.instantiateEproc(eproc_channels, results, op.getNameHintAttr());
    return absl::OkStatus();
  }

  absl::Status InterpretSpawnOfExtern(SpawnOp op, ExternSprocOp externSproc,
                                      ElaborationContext& ctx) {
    XLS_ASSIGN_OR_RETURN(auto arguments, ctx.Get(op.getChannels()));
    if (arguments.size() != externSproc.getChannelArgumentTypes().size()) {
      return absl::InternalError(absl::StrFormat(
          "Call to %s requires %d arguments but got %d",
          op.getCallee().getLeafReference().str(),
          externSproc.getChannelArgumentTypes().size(), arguments.size()));
    }
    ctx.instantiateExternEproc(externSproc, arguments, op.getNameHintAttr());
    return absl::OkStatus();
  }

  absl::Status Interpret(SprocOp op, ElaborationContext& ctx,
                         ArrayRef<ChanOp> boundaryChannels = {}) {
    XLS_ASSIGN_OR_RETURN(auto results,
                         Interpret(op.getSpawns(), boundaryChannels, ctx));
    auto eproc_channels = ctx.createEproc(op);
    ctx.instantiateEproc(eproc_channels, results);
    return absl::OkStatus();
  }
};

}  // namespace

void ProcElaborationPass::runOnOperation() {
  ModuleOp module = getOperation();

  SymbolTableCollection symbolTableCollection;
  SymbolUserMap symbolUserMap(symbolTableCollection, module);
  for (auto sproc : module.getOps<SprocOp>()) {
    assert(sproc.getSymName().find(suffix) == StringRef::npos &&
           "Magic suffix already present in sproc name");
    appendToSymbolName(sproc, symbolUserMap);
  }

  DenseMap<SprocOp, EprocAndChannels> procCache;
  SymbolTable symbolTable(module);

  // Elaborate all sprocs marked "top". Elaboration traverses a potentially
  // cyclical graph of sprocs, so we delay removing the sprocs until the end.
  for (auto sproc : module.getOps<SprocOp>()) {
    if (!sproc.getIsTop()) {
      continue;
    }

    OpBuilder builder(sproc);
    SmallVector<ChanOp> boundaryChannels;
    if (sproc.getBoundaryChannelNames().has_value()) {
      for (auto [arg, name] : llvm::zip(sproc.getChannelArguments(),
                                        *sproc.getBoundaryChannelNames())) {
        SchanType schan = cast<SchanType>(arg.getType());
        auto nameAttr = cast<StringAttr>(name);
        auto echan = builder.create<ChanOp>(sproc.getLoc(), nameAttr.str(),
                                            schan.getElementType());
        if (schan.getIsInput()) {
          echan.setSendSupported(false);
        } else {
          echan.setRecvSupported(false);
        }
        boundaryChannels.push_back(echan);
      }
    }

    ElaborationInterpreter interpreter;
    auto result = interpreter.InterpretTop(sproc, boundaryChannels, builder,
                                           symbolTable, procCache);
    if (!result.ok()) {
      sproc.emitError() << "failed to elaborate: " << result.message();
    }
  }
  module.walk([&](Operation* op) {
    if (isa<SprocOp, ExternSprocOp>(op)) {
      op->erase();
    }
  });
}

}  // namespace mlir::xls

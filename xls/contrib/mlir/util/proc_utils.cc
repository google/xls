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

#include "xls/contrib/mlir/util/proc_utils.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/include/mlir/Analysis/CallGraph.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/Matchers.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/TypeID.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Transforms/RegionUtils.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {
namespace {
namespace arith = ::mlir::arith;
using arith::AddIOp;
using arith::CmpIOp;
using arith::CmpIPredicate;
using arith::ConstantOp;
using arith::IndexCastOp;
using ::llvm::enumerate;
using ::llvm::zip;

// Wraps a type in a SchanType.
SchanType makeSchanType(Type type, bool isInput) {
  return SchanType::get(type.getContext(), type, isInput);
}

// Returns the concatenation of the given ValueRanges.
template <typename... RangeT>
auto concat(RangeT... ranges) {
  return llvm::to_vector(llvm::concat<Value>(ranges...));
}

// Creates a skeleton for an SprocOp. The sproc has input and output channels
// corresponding to inputs and results. inputs/results should NOT be channel
// types, they should be the underlying types. An i32 state is created.
SprocOp createSprocSkeleton(ImplicitLocOpBuilder& builder, TypeRange inputs,
                            TypeRange results, TypeRange stateTypes, Twine name,
                            SymbolTable& symbolTable) {
  OpBuilder::InsertionGuard guard(builder);
  auto sproc = builder.create<SprocOp>(builder.getStringAttr(name),
                                       /*is_top=*/false,
                                       /*boundary_channel_names=*/nullptr);
  symbolTable.insert(sproc);
  Block& spawns = sproc.getSpawns().emplaceBlock();
  Block& next = sproc.getNext().emplaceBlock();
  for (Type input : inputs) {
    Type type = makeSchanType(input, true);
    spawns.addArgument(type, builder.getLoc());
    next.addArgument(type, builder.getLoc());
  }
  for (Type result : results) {
    Type type = makeSchanType(result, false);
    spawns.addArgument(type, builder.getLoc());
    next.addArgument(type, builder.getLoc());
  }
  SmallVector<Location> locs(stateTypes.size(), builder.getLoc());
  auto stateArgs = next.addArguments(stateTypes, locs);
  SmallVector<Value> stateArgsVec(stateArgs.begin(), stateArgs.end());
  builder.setInsertionPointToEnd(&spawns);
  builder.create<YieldOp>(builder.getLoc(), spawns.getArguments());
  builder.setInsertionPointToEnd(&next);
  builder.create<YieldOp>(builder.getLoc(), stateArgsVec);
  return sproc;
}

// Creates a skeleton for the body Sproc. The body has input and output channels
// corresponding to the ForOp's region arguments plus invariants, and results.
SprocOp createBodySkeleton(scf::ForOp forOp, SprocOp parent,
                           TypeRange invariants, SymbolTable& symbolTable) {
  ImplicitLocOpBuilder builder(forOp.getLoc(), parent);
  Operation* forTerminator = forOp.getBody()->getTerminator();
  SmallVector<Type> inputs;
  inputs.insert(inputs.end(), forOp.getBody()->getArgumentTypes().begin(),
                forOp.getBody()->getArgumentTypes().end());
  inputs.insert(inputs.end(), invariants.begin(), invariants.end());
  // TODO(jmolloy): If XLS can handle it, we can just use an empty tuple as the
  // state type.
  return createSprocSkeleton(
      builder, inputs, forTerminator->getOperandTypes(), builder.getI32Type(),
      llvm::Twine(parent.getSymName()) + "_for_body", symbolTable);
}

// Creates a skeleton for the controller Sproc. The controller has input and
// output channels corresponding to the ForOp's init operands + invariants and
// results.
SprocOp createControllerSkeleton(scf::ForOp forOp, SprocOp parent,
                                 TypeRange invariants,
                                 SymbolTable& symbolTable) {
  ImplicitLocOpBuilder builder(forOp.getLoc(), parent);
  SmallVector<Type> inputs;
  inputs.insert(inputs.end(), forOp.getInits().getTypes().begin(),
                forOp.getInits().getTypes().end());
  inputs.insert(inputs.end(), invariants.begin(), invariants.end());
  SmallVector<Type> stateTypes;
  stateTypes.push_back(builder.getI32Type());
  stateTypes.insert(stateTypes.end(), invariants.begin(), invariants.end());
  return createSprocSkeleton(
      builder, inputs, forOp.getResultTypes(), stateTypes,
      llvm::Twine(parent.getSymName()) + "_for_controller", symbolTable);
}

// Creates SchanOps for the given types. Returns [outChannels, inChannels].
std::pair<SmallVector<Value>, SmallVector<Value>> createChannels(
    ImplicitLocOpBuilder& builder, StringRef name, TypeRange types) {
  SmallVector<Value> inChannels;
  SmallVector<Value> outChannels;
  for (auto [i, type] : enumerate(types)) {
    auto chanName = (llvm::Twine(name) + "_" + llvm::Twine(i)).str();
    SchanOp chanOp = builder.create<SchanOp>(chanName, type);
    inChannels.push_back(chanOp.getIn());
    outChannels.push_back(chanOp.getOut());
  }
  return {outChannels, inChannels};
}

// Populates the controller SprocOp with the control flow logic. Creates
// SchanOps and SpawnOps in-place in the `next` region to ease code generation
// (resulting in an invalid SprocOp).
//
// The controller algorithm is:
// state <- 0
// invariants <- zeros!
// while True:
//   if state == 0:
//      iter_args <- recv(init_args_chans)
//      invariants <- recv(invariant_args_chans)
//   else:
//      iter_args <- recv(body_results_chans)
//  if state == kTripCount:
//     send(iter_args to results_chans)
//     state <- 0
//  else:
//      send(iter_args to body_iter_args_chans)
//      send(invariants to body_invariants_chans)
//      send(state to body_indvar_chan)
//      state <- state+1
void populateController(SprocOp controller, scf::ForOp forOp,
                        TypeRange invariants, SprocOp body, int64_t tripCount) {
  Block& next = controller.getNext().front();
  auto b = ImplicitLocOpBuilder::atBlockBegin(forOp.getLoc(), &next);

  // As we're accepting loop requests from "outside" and dispatching loop
  // iteration requests to the body, we adopt a naming convention of "inner"
  // and "outer" to refer to the body and controller respectively.

  // The init argument channels, including invariants.
  ValueRange nextArgs = next.getArguments();
  ValueRange outerInitArgs = nextArgs.take_front(forOp.getInits().size());
  ValueRange outerInvariantArgs =
      nextArgs.slice(forOp.getInits().size(), invariants.size());
  // The channels used to send final results back.
  ValueRange outerResults = nextArgs.slice(
      forOp.getInits().size() + invariants.size(), forOp.getNumResults());
  Value state = controller.getStateArguments().front();
  ValueRange invariantsState = controller.getStateArguments().drop_front();

  // Create channels to talk to the body (inner channels).
  SmallVector<Type> innerChanTypes;
  innerChanTypes.insert(innerChanTypes.end(),
                        forOp.getBody()->getArgumentTypes().begin(),
                        forOp.getBody()->getArgumentTypes().end());
  innerChanTypes.insert(innerChanTypes.end(), invariants.begin(),
                        invariants.end());
  auto [innerOutArgChans, innerInArgChans] =
      createChannels(b, "body_arg", innerChanTypes);
  innerChanTypes.clear();
  innerChanTypes.insert(innerChanTypes.end(), forOp.getResultTypes().begin(),
                        forOp.getResultTypes().end());
  auto [innerOutResultChans, innerInResultChans] =
      createChannels(b, "body_result", innerChanTypes);

  auto spawnArgs = concat(innerInArgChans, innerOutResultChans);
  b.create<SpawnOp>(spawnArgs, SymbolRefAttr::get(body.getSymNameAttr()),
                    nullptr);

  Value innerIndvarArg = innerOutArgChans.front();
  ValueRange innerIterArgs =
      ArrayRef(innerOutArgChans)
          .slice(1, forOp.getBody()->getArgumentTypes().size() - 1);
  ValueRange innerInvariantArgs =
      ArrayRef(innerOutArgChans)
          .slice(forOp.getBody()->getArgumentTypes().size(), invariants.size());
  ValueRange innerResults = innerInResultChans;

  assert(innerInvariantArgs.size() == outerInvariantArgs.size());
  assert(innerIterArgs.size() == outerInitArgs.size());
  assert(innerResults.size() == outerResults.size());

  Type i32 = b.getI32Type();
  Value c0 = b.create<ConstantOp>(b.getI32IntegerAttr(0));
  Value c1 = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value cTripCount = b.create<ConstantOp>(b.getI32IntegerAttr(tripCount));

  auto readChans = [&](ValueRange chans, ImplicitLocOpBuilder& b) {
    SmallVector<Value> data;
    for (Value chan : chans) {
      data.push_back(b.create<xls::SBlockingReceiveOp>(chan).getResult());
    }
    return data;
  };

  auto ifOp = b.create<scf::IfOp>(
      b.create<CmpIOp>(CmpIPredicate::eq, state, c0),
      [&](OpBuilder& builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(concat(readChans(outerInitArgs, b),
                                      readChans(outerInvariantArgs, b)));
      },
      [&](OpBuilder& builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(
            concat(readChans(innerResults, b), invariantsState));
      });
  auto iterArgs = ifOp.getResults().drop_back(invariants.size());
  auto newInvariantsState = ifOp.getResults().take_back(invariants.size());

  auto sendToChans = [&](ValueRange chans) {
    for (auto [from, to] : zip(iterArgs, chans)) {
      b.create<xls::SSendOp>(from, to);
    }
  };

  Value finalState =
      b.create<scf::IfOp>(
           b.create<CmpIOp>(CmpIPredicate::eq, state, cTripCount),
           [&](OpBuilder& builder, Location loc) {
             ImplicitLocOpBuilder b(loc, builder);
             sendToChans(outerResults);
             b.create<scf::YieldOp>(c0);
           },
           [&](OpBuilder& builder, Location loc) {
             ImplicitLocOpBuilder b(loc, builder);
             sendToChans(innerIterArgs);
             for (auto [from, to] :
                  zip(newInvariantsState, innerInvariantArgs)) {
               b.create<xls::SSendOp>(from, to);
             }
             b.create<xls::SSendOp>(
                 b.create<IndexCastOp>(b.getIndexType(), state),
                 innerIndvarArg);
             b.create<scf::YieldOp>(
                 b.create<AddIOp>(i32, state, c1).getResult());
           })
          .getResult(0);

  next.getTerminator()->setOperand(0, finalState);
  next.getTerminator()->setOperands(1, newInvariantsState.size(),
                                    newInvariantsState);
}

// Clones the body of `forOp` into the next region of `body`. The region already
// has argument channels for the body's arguments and invariants, as:
//
//   arg[0:]  = forOp.getBody().getArguments()
//   arg[...] = invariants
//   arg[...] = forOp.getResultTypes()
//   arg[-1]  = state
void populateBody(SprocOp body, scf::ForOp forOp, ValueRange invariants,
                  ValueRange toClone) {
  Block& next = body.getNext().front();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(forOp.getLoc(), &next);
  IRMapping mapper;
  ValueRange forBodyArguments = forOp.getBody()->getArguments();
  Value token = builder.create<AfterAllOp>();
  for (auto [from, to] : zip(forBodyArguments, next.getArguments())) {
    mapper.map(from, builder.create<SBlockingReceiveOp>(token, to).getResult());
  }
  for (auto [from, to] :
       zip(invariants, next.getArguments().slice(forBodyArguments.size(),
                                                 invariants.size()))) {
    mapper.map(from, builder.create<SBlockingReceiveOp>(token, to).getResult());
  }
  for (Value value : toClone) {
    builder.clone(*value.getDefiningOp(), mapper);
  }
  Operation* yieldTerminator = next.getTerminator();
  IRRewriter rewriter(forOp.getContext());
  rewriter.cloneRegionBefore(forOp.getBodyRegion(), body.getNext(),
                             body.getNext().end(), mapper);
  rewriter.mergeBlocks(&body.getNext().back(), &body.getNext().front());
  mapper.lookup(forOp.getBody()->getTerminator())->erase();
  yieldTerminator->moveBefore(&next, next.end());

  ValueRange yielded = forOp.getBody()->getTerminator()->getOperands();
  token = builder.create<AfterAllOp>();
  for (auto [from, to] : zip(yielded, next.getArguments().drop_back().take_back(
                                          yielded.size()))) {
    builder.create<SSendOp>(token, mapper.lookup(from), to);
  }
}

// Fixes up an SprocOp that may have SchanOps and SpawnOps in the `next` region.
// Moves them into the `spawns` region, and adds arguments to the `next` region
// for any SchanOps that are used outside of the `spawns` region.
void fixupSproc(SprocOp sproc) {
  Block& spawns = sproc.getSpawns().front();
  Block& next = sproc.getNext().front();
  Operation* terminator = sproc.getSpawns().front().getTerminator();
  sproc.getNext().walk([&](Operation* op) {
    if (!isa<SchanOp, SpawnOp>(op)) {
      return;
    }
    op->moveBefore(terminator);
  });

  sproc.getSpawns().walk([&](Operation* op) {
    if (!isa<SchanOp, SpawnOp>(op)) {
      return;
    }
    for (OpResult result : op->getResults()) {
      if (!result.isUsedOutsideOfBlock(&sproc.getSpawns().front())) {
        continue;
      }
      int insertPoint = terminator->getNumOperands();
      terminator->insertOperands(terminator->getNumOperands(), result);
      Value arg = next.insertArgument(std::next(next.args_begin(), insertPoint),
                                      result.getType(), result.getLoc());
      result.replaceUsesWithIf(arg, [&](OpOperand& opOperand) {
        return opOperand.getOwner()->getBlock() != &spawns;
      });
    }
  });
}

// Tries to determine the trip count of the ForOp. Returns failure if the trip
// count cannot be determined.
FailureOr<int64_t> getTripCount(scf::ForOp forOp) {
  if (auto step = forOp.getConstantStep();
      !step.has_value() || !step->isOne()) {
    if (step.has_value()) {
      return mlir::failure();
    }
    return mlir::failure();
  }
  APInt lowerBound;
  if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&lowerBound))) {
    return mlir::failure();
  }

  APInt upperBound;
  if (!matchPattern(forOp.getUpperBound(), m_ConstantInt(&upperBound))) {
    return mlir::failure();
  }

  return (upperBound - lowerBound).getLimitedValue();
}

// Splits the given invariants into two groups: those that must be captured and
// those that can be cloned.
//
// The cloned values have a defining op that has zero arguments.
std::pair<SmallVector<Value>, SmallVector<Value>> splitOutInvariantsToClone(
    ValueRange invariants) {
  SmallVector<Value> toKeep;
  SmallVector<Value> toClone;
  for (Value invariant : invariants) {
    if (isa_and_present<arith::ConstantOp, xls::ConstantScalarOp>(
            invariant.getDefiningOp())) {
      toClone.push_back(invariant);
    } else {
      toKeep.push_back(invariant);
    }
  }
  return {toKeep, toClone};
}

}  // namespace

LogicalResult convertForOpToSprocCall(scf::ForOp forOp,
                                      SymbolTable& symbolTable) {
  auto tripCount = getTripCount(forOp);
  if (failed(tripCount)) {
    return failure();
  }
  SprocOp parent = forOp->getParentOfType<SprocOp>();

  SetVector<Value> invariantsAsSetVector;
  mlir::getUsedValuesDefinedAbove(forOp.getBodyRegion(), invariantsAsSetVector);
  auto [toCapture, toClone] =
      splitOutInvariantsToClone(invariantsAsSetVector.getArrayRef());

  SprocOp body =
      createBodySkeleton(forOp, parent, ValueRange(toCapture), symbolTable);
  SprocOp controller = createControllerSkeleton(
      forOp, parent, ValueRange(toCapture), symbolTable);
  populateController(controller, forOp, ValueRange(toCapture), body,
                     *tripCount);
  populateBody(body, forOp, toCapture, toClone);

  ImplicitLocOpBuilder builder(forOp.getLoc(), forOp);
  SmallVector<Value> sendArgs;
  sendArgs.insert(sendArgs.end(), forOp.getInits().begin(),
                  forOp.getInits().end());
  sendArgs.insert(sendArgs.end(), toCapture.begin(), toCapture.end());
  auto [sendChanOuts, sendChanIns] =
      createChannels(builder, "for_arg", ValueRange(sendArgs));
  auto [recvChanOuts, recvChanIns] =
      createChannels(builder, "for_result", forOp.getResultTypes());
  auto spawnArgs = concat(sendChanIns, recvChanOuts);
  builder.create<SpawnOp>(
      spawnArgs, SymbolRefAttr::get(controller.getSymNameAttr()), nullptr);

  Value token = builder.create<AfterAllOp>();
  SmallVector<Value> tokens;
  for (auto [chan, arg] : zip(sendChanOuts, sendArgs)) {
    tokens.push_back(builder.create<SSendOp>(token, arg, chan));
  }
  token = builder.create<AfterAllOp>(tokens);
  SmallVector<Value> results;
  for (auto [chan, arg] : zip(recvChanIns, forOp.getResultTypes())) {
    results.push_back(
        builder.create<SBlockingReceiveOp>(token, chan).getResult());
  }
  forOp->replaceAllUsesWith(results);
  forOp.erase();

  fixupSproc(parent);
  fixupSproc(controller);

  // In the parent we have added a receive(send(...)) sequence, so it now won't
  // schedule unless it has at least one more pipeline stage.
  parent.setMinPipelineStages(parent.getMinPipelineStages() + 1);

  return success();
}

namespace {
// Test pass for convertForOpToSprocCall, defined here to avoid exposing it in
// the passes.h header.
struct TestConvertForOpToSprocCallPass
    : public PassWrapper<TestConvertForOpToSprocCallPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertForOpToSprocCallPass)

  StringRef getArgument() const final { return "test-convert-for-op-to-sproc"; }
  StringRef getDescription() const final {
    return "Testing only: Converts a ForOp to a SprocOp call.";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<xls::XlsDialect>();
  }
  void runOnOperation() override {
    // We need to transform in pre-order, but MLIR's walkers require that if we
    // erase forOp we must interrupt the walk. So we run the walk to fixpoint.
    SymbolTable symbolTable(getOperation());
    bool changed = true;
    while (changed) {
      changed = false;
      getOperation()->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
        bool thisChanged =
            succeeded(convertForOpToSprocCall(forOp, symbolTable));
        changed |= thisChanged;
        return thisChanged ? WalkResult::interrupt() : WalkResult::advance();
      });
    }
  }
};
}  // namespace

namespace test {
void registerTestConvertForOpToSprocCallPass() {
  PassRegistration<TestConvertForOpToSprocCallPass>();
}
}  // namespace test

}  // namespace mlir::xls

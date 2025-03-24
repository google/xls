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
#include <cstdint>
#include <utility>

#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_OPTIMIZESPAWNSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

class SendOfBlockingReceiveOp : public OpRewritePattern<SSendOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SSendOp send,
                                PatternRewriter& rewriter) const override {
    auto recv = send.getData().getDefiningOp<SBlockingReceiveOp>();
    // We assume that nobody uses the token output of the send or the token
    // output of the receive.
    if (!recv || recv.getPredicate() || send.getPredicate() ||
        !recv.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          send, "not a recv, or predicated, or recv has multiple uses");
    }
    // Only proceed if the token is trivial for the send, or is the output token
    // of the receive.
    if (!isTrivialToken(send.getTkn()) && send.getTkn() != recv.getTknOut()) {
      return rewriter.notifyMatchFailure(send, "send token is not trivial");
    }

    // It is safe to remove the receive and send.
    SprocOp sproc = send->getParentOfType<SprocOp>();
    int64_t sendChanIdx = cast<BlockArgument>(send.getChannel()).getArgNumber();
    int64_t recvChanIdx = cast<BlockArgument>(recv.getChannel()).getArgNumber();
    Value sendChan = sproc.getYieldedChannels()[sendChanIdx];
    Value recvChan = sproc.getYieldedChannels()[recvChanIdx];

    if (auto sendChanOp = sendChan.getDefiningOp<SchanOp>()) {
      // The send channel is an interior channel, so replace uses of its "in"
      // port with the recv channel (recv can either be interior or argument).
      Value sendChanReceiver = sendChanOp.getIn();
      rewriter.replaceUsesWithIf(sendChanReceiver, recvChan,
                                 [&](OpOperand& opOperand) {
                                   return !isa<YieldOp>(opOperand.getOwner());
                                 });
    } else if (auto recvChanOp = recvChan.getDefiningOp<SchanOp>()) {
      // The recv channel is an interior channel, so replace uses of its "out"
      // port with the send channel (send can either be interior or a result).
      Value recvChanSender = recvChanOp.getOut();
      rewriter.replaceUsesWithIf(recvChanSender, sendChan,
                                 [&](OpOperand& opOperand) {
                                   return !isa<YieldOp>(opOperand.getOwner());
                                 });
    } else {
      // The recv channel is an argument and the send channel is a result. This
      // needs a send/recv pair so we can't optimize this.
      return failure();
    }

    rewriter.replaceAllUsesWith(recv.getTknOut(), recv.getTkn());
    rewriter.replaceAllUsesWith(send.getResult(), send.getTkn());
    rewriter.eraseOp(send);
    rewriter.eraseOp(recv);

    // When erasing arguments, we need to erase them from highest to lowest so
    // that the indices are not invalidated.
    SmallVector<int64_t> channelIndices = {sendChanIdx, recvChanIdx};
    std::sort(channelIndices.begin(), channelIndices.end());
    std::reverse(channelIndices.begin(), channelIndices.end());

    for (int64_t chanIdx : channelIndices) {
      sproc.getNext().eraseArgument(chanIdx);
      sproc.getSpawns().front().getTerminator()->eraseOperand(chanIdx);
    }
    return success();
  }

  bool isTrivialToken(Value token) const {
    AfterAllOp afterAll = token.getDefiningOp<AfterAllOp>();
    return afterAll && afterAll.getOperands().empty();
  }
};

void RemoveUnusedArguments(SprocOp sproc) {
  // Erase unused channel arguments from the next region.
  for (int i = sproc.getYieldedChannels().size() - 1; i >= 0; --i) {
    Value next_chan = sproc.getNextChannels()[i];
    if (next_chan.use_empty()) {
      sproc.getNext().eraseArgument(i);
      sproc.getSpawns().front().getTerminator()->eraseOperand(i);
    }
  }

  // Erase state arguments that are only passed to the terminator from the next
  // region.
  Operation* terminator = sproc.getNext().front().getTerminator();
  for (int i = terminator->getNumOperands() - 1; i >= 0; --i) {
    BlockArgument arg = sproc.getStateArguments()[i];
    OpOperand& yield = terminator->getOpOperand(i);

    if (arg == yield.get() && arg.hasOneUse()) {
      terminator->eraseOperand(i);
      sproc.getNext().eraseArgument(arg.getArgNumber());
    }
  }
}

class OptimizeSpawnsPass
    : public impl::OptimizeSpawnsPassBase<OptimizeSpawnsPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SendOfBlockingReceiveOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation().getNext(),
                                     std::move(patterns)))) {
      signalPassFailure();
    }

    // Ensure that all spawns regions are topologically sorted; given the order
    // in which the pattern is applied, this is not currently guaranteed.
    sortTopologically(&getOperation().getSpawns().front());

    // Optimize away unused arguments to the next region.
    RemoveUnusedArguments(getOperation());
  }
};

}  // namespace
}  // namespace mlir::xls

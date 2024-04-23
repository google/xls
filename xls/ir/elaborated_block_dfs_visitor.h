// Copyright 2020 The XLS Authors
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

#ifndef XLS_IR_ELABORATED_BLOCK_DFS_VISITOR_H_
#define XLS_IR_ELABORATED_BLOCK_DFS_VISITOR_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

// Abstract interface for a DFS post-order node visitor. See Node::Accept() and
// Function::Accept. Dispatches on Op.
class ElaboratedBlockDfsVisitor {
 public:
  virtual ~ElaboratedBlockDfsVisitor() = default;

  virtual absl::Status HandleAdd(BinOp* add, BlockInstance* instance) = 0;
  virtual absl::Status HandleAfterAll(AfterAll* after_all,
                                      BlockInstance* instance) = 0;
  virtual absl::Status HandleMinDelay(MinDelay* min_delay,
                                      BlockInstance* instance) = 0;
  virtual absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce,
                                       BlockInstance* instance) = 0;
  virtual absl::Status HandleArray(Array* array, BlockInstance* instance) = 0;
  virtual absl::Status HandleArrayConcat(ArrayConcat* array_concat,
                                         BlockInstance* instance) = 0;
  virtual absl::Status HandleAssert(Assert* assert_op,
                                    BlockInstance* instance) = 0;
  virtual absl::Status HandleBitSlice(BitSlice* bit_slice,
                                      BlockInstance* instance) = 0;
  virtual absl::Status HandleBitSliceUpdate(BitSliceUpdate* update,
                                            BlockInstance* instance) = 0;
  virtual absl::Status HandleConcat(Concat* concat,
                                    BlockInstance* instance) = 0;
  virtual absl::Status HandleCountedFor(CountedFor* counted_for,
                                        BlockInstance* instance) = 0;
  virtual absl::Status HandleCover(Cover* cover, BlockInstance* instance) = 0;
  virtual absl::Status HandleDecode(Decode* decode,
                                    BlockInstance* instance) = 0;
  virtual absl::Status HandleDynamicBitSlice(DynamicBitSlice* dynamic_bit_slice,
                                             BlockInstance* instance) = 0;
  virtual absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for, BlockInstance* instance) = 0;
  virtual absl::Status HandleEncode(Encode* encode,
                                    BlockInstance* instance) = 0;
  virtual absl::Status HandleEq(CompareOp* eq, BlockInstance* instance) = 0;
  virtual absl::Status HandleGate(Gate* gate, BlockInstance* instance) = 0;
  virtual absl::Status HandleIdentity(UnOp* identity,
                                      BlockInstance* instance) = 0;
  virtual absl::Status HandleInputPort(InputPort* input_port,
                                       BlockInstance* instance) = 0;
  virtual absl::Status HandleInvoke(Invoke* invoke,
                                    BlockInstance* instance) = 0;
  virtual absl::Status HandleInstantiationInput(
      InstantiationInput* instantiation_input, BlockInstance* instance) = 0;
  virtual absl::Status HandleInstantiationOutput(
      InstantiationOutput* instantiation_output, BlockInstance* instance) = 0;
  virtual absl::Status HandleLiteral(Literal* literal,
                                     BlockInstance* instance) = 0;
  virtual absl::Status HandleMap(Map* map, BlockInstance* instance) = 0;
  virtual absl::Status HandleArrayIndex(ArrayIndex* index,
                                        BlockInstance* instance) = 0;
  virtual absl::Status HandleArraySlice(ArraySlice* update,
                                        BlockInstance* instance) = 0;
  virtual absl::Status HandleArrayUpdate(ArrayUpdate* update,
                                         BlockInstance* instance) = 0;
  virtual absl::Status HandleNaryAnd(NaryOp* and_op,
                                     BlockInstance* instance) = 0;
  virtual absl::Status HandleNaryNand(NaryOp* nand_op,
                                      BlockInstance* instance) = 0;
  virtual absl::Status HandleNaryNor(NaryOp* nor_op,
                                     BlockInstance* instance) = 0;
  virtual absl::Status HandleNaryOr(NaryOp* or_op, BlockInstance* instance) = 0;
  virtual absl::Status HandleNaryXor(NaryOp* xor_op,
                                     BlockInstance* instance) = 0;
  virtual absl::Status HandleNe(CompareOp* ne, BlockInstance* instance) = 0;
  virtual absl::Status HandleNeg(UnOp* neg, BlockInstance* instance) = 0;
  virtual absl::Status HandleNot(UnOp* not_op, BlockInstance* instance) = 0;
  virtual absl::Status HandleOneHot(OneHot* one_hot,
                                    BlockInstance* instance) = 0;
  virtual absl::Status HandleOneHotSel(OneHotSelect* sel,
                                       BlockInstance* instance) = 0;
  virtual absl::Status HandlePrioritySel(PrioritySelect* sel,
                                         BlockInstance* instance) = 0;
  virtual absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce,
                                      BlockInstance* instance) = 0;
  virtual absl::Status HandleOutputPort(OutputPort* output_port,
                                        BlockInstance* instance) = 0;
  virtual absl::Status HandleParam(Param* param, BlockInstance* instance) = 0;
  virtual absl::Status HandleNext(Next* next, BlockInstance* instance) = 0;
  virtual absl::Status HandleReceive(Receive* receive,
                                     BlockInstance* instance) = 0;
  virtual absl::Status HandleRegisterRead(RegisterRead* reg_read,
                                          BlockInstance* instance) = 0;
  virtual absl::Status HandleRegisterWrite(RegisterWrite* reg_write,
                                           BlockInstance* instance) = 0;
  virtual absl::Status HandleReverse(UnOp* reverse,
                                     BlockInstance* instance) = 0;
  virtual absl::Status HandleSDiv(BinOp* div, BlockInstance* instance) = 0;
  virtual absl::Status HandleSGe(CompareOp* ge, BlockInstance* instance) = 0;
  virtual absl::Status HandleSGt(CompareOp* gt, BlockInstance* instance) = 0;
  virtual absl::Status HandleSLe(CompareOp* le, BlockInstance* instance) = 0;
  virtual absl::Status HandleSLt(CompareOp* lt, BlockInstance* instance) = 0;
  virtual absl::Status HandleSMod(BinOp* mod, BlockInstance* instance) = 0;
  virtual absl::Status HandleSMul(ArithOp* mul, BlockInstance* instance) = 0;
  virtual absl::Status HandleSMulp(PartialProductOp* mul,
                                   BlockInstance* instance) = 0;
  virtual absl::Status HandleSel(Select* sel, BlockInstance* instance) = 0;
  virtual absl::Status HandleSend(Send* send, BlockInstance* instance) = 0;
  virtual absl::Status HandleShll(BinOp* shll, BlockInstance* instance) = 0;
  virtual absl::Status HandleShra(BinOp* shra, BlockInstance* instance) = 0;
  virtual absl::Status HandleShrl(BinOp* shrl, BlockInstance* instance) = 0;
  virtual absl::Status HandleSignExtend(ExtendOp* sign_ext,
                                        BlockInstance* instance) = 0;
  virtual absl::Status HandleSub(BinOp* sub, BlockInstance* instance) = 0;
  virtual absl::Status HandleTrace(Trace* trace_op,
                                   BlockInstance* instance) = 0;
  virtual absl::Status HandleTuple(Tuple* tuple, BlockInstance* instance) = 0;
  virtual absl::Status HandleTupleIndex(TupleIndex* index,
                                        BlockInstance* instance) = 0;
  virtual absl::Status HandleUDiv(BinOp* div, BlockInstance* instance) = 0;
  virtual absl::Status HandleUGe(CompareOp* ge, BlockInstance* instance) = 0;
  virtual absl::Status HandleUGt(CompareOp* gt, BlockInstance* instance) = 0;
  virtual absl::Status HandleULe(CompareOp* le, BlockInstance* instance) = 0;
  virtual absl::Status HandleULt(CompareOp* lt, BlockInstance* instance) = 0;
  virtual absl::Status HandleUMod(BinOp* mod, BlockInstance* instance) = 0;
  virtual absl::Status HandleUMul(ArithOp* mul, BlockInstance* instance) = 0;
  virtual absl::Status HandleUMulp(PartialProductOp* mul,
                                   BlockInstance* instance) = 0;
  virtual absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce,
                                       BlockInstance* instance) = 0;
  virtual absl::Status HandleZeroExtend(ExtendOp* zero_ext,
                                        BlockInstance* instance) = 0;

  // Returns true if the given node has been visited.
  bool IsVisited(const ElaboratedNode& node) const {
    return visited_.contains(node);
  }

  // Marks the given node as visited.
  void MarkVisited(const ElaboratedNode& node) { visited_.insert(node); }

  // Returns whether the given node is on path from the root of the traversal
  // to the currently visited node. Used to identify cycles in the graph.
  bool IsTraversing(const ElaboratedNode& node) const {
    return traversing_.contains(node);
  }

  // Sets/unsets whether this node is being traversed through.
  void SetTraversing(const ElaboratedNode& node) { traversing_.insert(node); }
  void UnsetTraversing(const ElaboratedNode& node) { traversing_.erase(node); }

  // Resets traversal state.
  // This is for cases where creating a new DfsVisitor is not feasible or
  // convenient, e.g., during Z3 LEC. There, we need to replace certain IR
  // translations with constant Z3 nodes, and creating a new object would
  // destroy the replacements. By enabling re-traversal, the process is much
  // cleaner.
  void ResetVisitedState() {
    visited_.clear();
    traversing_.clear();
  }

  // Return the total number of nodes visited.
  int64_t GetVisitedCount() const { return visited_.size(); }

 private:
  // Set of nodes which have been visited.
  absl::flat_hash_set<ElaboratedNode> visited_;

  // Set of nodes which are being traversed through.
  absl::flat_hash_set<ElaboratedNode> traversing_;
};

// Visitor with a default action. If the Handle<Op> method is not overridden
// in the derived class then DefaultHandler is called when visiting nodes of
// type <Op>.
class ElaboratedBlockDfsVisitorWithDefault : public ElaboratedBlockDfsVisitor {
 public:
  virtual absl::Status DefaultHandler(const ElaboratedNode& node) = 0;

  absl::Status HandleAdd(BinOp* add, BlockInstance* instance) override;
  absl::Status HandleAfterAll(AfterAll* after_all,
                              BlockInstance* instance) override;
  absl::Status HandleMinDelay(MinDelay* min_delay,
                              BlockInstance* instance) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce,
                               BlockInstance* instance) override;
  absl::Status HandleArray(Array* array, BlockInstance* instance) override;
  absl::Status HandleArrayConcat(ArrayConcat* array_concat,
                                 BlockInstance* instance) override;
  absl::Status HandleAssert(Assert* assert_op,
                            BlockInstance* instance) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice,
                              BlockInstance* instance) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update,
                                    BlockInstance* instance) override;
  absl::Status HandleConcat(Concat* concat, BlockInstance* instance) override;
  absl::Status HandleCountedFor(CountedFor* counted_for,
                                BlockInstance* instance) override;
  absl::Status HandleCover(Cover* cover, BlockInstance* instance) override;
  absl::Status HandleDecode(Decode* decode, BlockInstance* instance) override;
  absl::Status HandleDynamicBitSlice(DynamicBitSlice* dynamic_bit_slice,
                                     BlockInstance* instance) override;
  absl::Status HandleDynamicCountedFor(DynamicCountedFor* dynamic_counted_for,
                                       BlockInstance* instance) override;
  absl::Status HandleEncode(Encode* encode, BlockInstance* instance) override;
  absl::Status HandleEq(CompareOp* eq, BlockInstance* instance) override;
  absl::Status HandleGate(Gate* gate, BlockInstance* instance) override;
  absl::Status HandleIdentity(UnOp* identity, BlockInstance* instance) override;
  absl::Status HandleInputPort(InputPort* input_port,
                               BlockInstance* instance) override;
  absl::Status HandleInvoke(Invoke* invoke, BlockInstance* instance) override;
  absl::Status HandleInstantiationInput(InstantiationInput* instantiation_input,
                                        BlockInstance* instance) override;
  absl::Status HandleInstantiationOutput(
      InstantiationOutput* instantiation_output,
      BlockInstance* instance) override;
  absl::Status HandleLiteral(Literal* literal,
                             BlockInstance* instance) override;
  absl::Status HandleMap(Map* map, BlockInstance* instance) override;
  absl::Status HandleArrayIndex(ArrayIndex* index,
                                BlockInstance* instance) override;
  absl::Status HandleArraySlice(ArraySlice* slice,
                                BlockInstance* instance) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update,
                                 BlockInstance* instance) override;
  absl::Status HandleNaryAnd(NaryOp* and_op, BlockInstance* instance) override;
  absl::Status HandleNaryNand(NaryOp* and_op, BlockInstance* instance) override;
  absl::Status HandleNaryNor(NaryOp* nor_op, BlockInstance* instance) override;
  absl::Status HandleNaryOr(NaryOp* or_op, BlockInstance* instance) override;
  absl::Status HandleNaryXor(NaryOp* xor_op, BlockInstance* instance) override;
  absl::Status HandleNe(CompareOp* ne, BlockInstance* instance) override;
  absl::Status HandleNeg(UnOp* neg, BlockInstance* instance) override;
  absl::Status HandleNot(UnOp* not_op, BlockInstance* instance) override;
  absl::Status HandleOneHot(OneHot* one_hot, BlockInstance* instance) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel,
                               BlockInstance* instance) override;
  absl::Status HandlePrioritySel(PrioritySelect* sel,
                                 BlockInstance* instance) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce,
                              BlockInstance* instance) override;
  absl::Status HandleOutputPort(OutputPort* output_port,
                                BlockInstance* instance) override;
  absl::Status HandleParam(Param* param, BlockInstance* instance) override;
  absl::Status HandleNext(Next* next, BlockInstance* instance) override;
  absl::Status HandleReceive(Receive* receive,
                             BlockInstance* instance) override;
  absl::Status HandleRegisterRead(RegisterRead* reg_read,
                                  BlockInstance* instance) override;
  absl::Status HandleRegisterWrite(RegisterWrite* reg_write,
                                   BlockInstance* instance) override;
  absl::Status HandleReverse(UnOp* reverse, BlockInstance* instance) override;
  absl::Status HandleSDiv(BinOp* div, BlockInstance* instance) override;
  absl::Status HandleSGe(CompareOp* ge, BlockInstance* instance) override;
  absl::Status HandleSGt(CompareOp* gt, BlockInstance* instance) override;
  absl::Status HandleSLe(CompareOp* le, BlockInstance* instance) override;
  absl::Status HandleSLt(CompareOp* lt, BlockInstance* instance) override;
  absl::Status HandleSMod(BinOp* mod, BlockInstance* instance) override;
  absl::Status HandleSMul(ArithOp* mul, BlockInstance* instance) override;
  absl::Status HandleSMulp(PartialProductOp* mul,
                           BlockInstance* instance) override;
  absl::Status HandleSel(Select* sel, BlockInstance* instance) override;
  absl::Status HandleSend(Send* send, BlockInstance* instance) override;
  absl::Status HandleShll(BinOp* shll, BlockInstance* instance) override;
  absl::Status HandleShra(BinOp* shra, BlockInstance* instance) override;
  absl::Status HandleShrl(BinOp* shrl, BlockInstance* instance) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext,
                                BlockInstance* instance) override;
  absl::Status HandleSub(BinOp* sub, BlockInstance* instance) override;
  absl::Status HandleTrace(Trace* trace_op, BlockInstance* instance) override;
  absl::Status HandleTuple(Tuple* tuple, BlockInstance* instance) override;
  absl::Status HandleTupleIndex(TupleIndex* index,
                                BlockInstance* instance) override;
  absl::Status HandleUDiv(BinOp* div, BlockInstance* instance) override;
  absl::Status HandleUGe(CompareOp* ge, BlockInstance* instance) override;
  absl::Status HandleUGt(CompareOp* gt, BlockInstance* instance) override;
  absl::Status HandleULe(CompareOp* le, BlockInstance* instance) override;
  absl::Status HandleULt(CompareOp* lt, BlockInstance* instance) override;
  absl::Status HandleUMod(BinOp* mod, BlockInstance* instance) override;
  absl::Status HandleUMul(ArithOp* mul, BlockInstance* instance) override;
  absl::Status HandleUMulp(PartialProductOp* mul,
                           BlockInstance* instance) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce,
                               BlockInstance* instance) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext,
                                BlockInstance* instance) override;
};

}  // namespace xls

#endif  // XLS_IR_ELABORATED_BLOCK_DFS_VISITOR_H_

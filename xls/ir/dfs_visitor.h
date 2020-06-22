// Copyright 2020 Google LLC
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

#ifndef XLS_IR_DFS_VISITOR_H_
#define XLS_IR_DFS_VISITOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

// Abstract interface for a DFS post-order node visitor. See Node::Accept() and
// Function::Accept. Dispatches on Op.
class DfsVisitor {
 public:
  virtual ~DfsVisitor() = default;

  virtual absl::Status HandleAdd(BinOp* add) = 0;
  virtual absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) = 0;
  virtual absl::Status HandleArray(Array* array) = 0;
  virtual absl::Status HandleArrayIndex(ArrayIndex* index) = 0;
  virtual absl::Status HandleArrayUpdate(ArrayUpdate* update) = 0;
  virtual absl::Status HandleBitSlice(BitSlice* bit_slice) = 0;
  virtual absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) = 0;
  virtual absl::Status HandleConcat(Concat* concat) = 0;
  virtual absl::Status HandleCountedFor(CountedFor* counted_for) = 0;
  virtual absl::Status HandleDecode(Decode* decode) = 0;
  virtual absl::Status HandleEncode(Encode* encode) = 0;
  virtual absl::Status HandleEq(CompareOp* eq) = 0;
  virtual absl::Status HandleIdentity(UnOp* identity) = 0;
  virtual absl::Status HandleInvoke(Invoke* invoke) = 0;
  virtual absl::Status HandleLiteral(Literal* literal) = 0;
  virtual absl::Status HandleMap(Map* map) = 0;
  virtual absl::Status HandleNaryAnd(NaryOp* and_op) = 0;
  virtual absl::Status HandleNaryNand(NaryOp* nand_op) = 0;
  virtual absl::Status HandleNaryNor(NaryOp* nor_op) = 0;
  virtual absl::Status HandleNaryOr(NaryOp* or_op) = 0;
  virtual absl::Status HandleNaryXor(NaryOp* xor_op) = 0;
  virtual absl::Status HandleNe(CompareOp* ne) = 0;
  virtual absl::Status HandleNeg(UnOp* neg) = 0;
  virtual absl::Status HandleNot(UnOp* not_op) = 0;
  virtual absl::Status HandleOneHot(OneHot* one_hot) = 0;
  virtual absl::Status HandleOneHotSel(OneHotSelect* sel) = 0;
  virtual absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) = 0;
  virtual absl::Status HandleParam(Param* param) = 0;
  virtual absl::Status HandleReverse(UnOp* reverse) = 0;
  virtual absl::Status HandleSDiv(BinOp* div) = 0;
  virtual absl::Status HandleSGe(CompareOp* ge) = 0;
  virtual absl::Status HandleSGt(CompareOp* gt) = 0;
  virtual absl::Status HandleSLe(CompareOp* le) = 0;
  virtual absl::Status HandleSLt(CompareOp* lt) = 0;
  virtual absl::Status HandleSel(Select* sel) = 0;
  virtual absl::Status HandleShll(BinOp* shll) = 0;
  virtual absl::Status HandleShra(BinOp* shra) = 0;
  virtual absl::Status HandleShrl(BinOp* shrl) = 0;
  virtual absl::Status HandleSignExtend(ExtendOp* sign_ext) = 0;
  virtual absl::Status HandleSMul(ArithOp* mul) = 0;
  virtual absl::Status HandleSub(BinOp* sub) = 0;
  virtual absl::Status HandleTuple(Tuple* tuple) = 0;
  virtual absl::Status HandleTupleIndex(TupleIndex* index) = 0;
  virtual absl::Status HandleUDiv(BinOp* div) = 0;
  virtual absl::Status HandleUGe(CompareOp* ge) = 0;
  virtual absl::Status HandleUGt(CompareOp* gt) = 0;
  virtual absl::Status HandleULe(CompareOp* le) = 0;
  virtual absl::Status HandleULt(CompareOp* lt) = 0;
  virtual absl::Status HandleUMul(ArithOp* mul) = 0;
  virtual absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) = 0;
  virtual absl::Status HandleZeroExtend(ExtendOp* zero_ext) = 0;

  // Returns true if the given node has been visited.
  bool IsVisited(Node* node) const { return visited_.contains(node); }

  // Marks the given node as visited.
  void MarkVisited(Node* node) { visited_.insert(node); }

  // Returns whether the given node is on path from the root of the traversal
  // to the currently visited node. Used to identify cycles in the graph.
  bool IsTraversing(Node* node) const { return traversing_.contains(node); }

  // Sets/unsets whether this node is being traversed through.
  void SetTraversing(Node* node) { traversing_.insert(node); }
  void UnsetTraversing(Node* node) { traversing_.erase(node); }

 private:
  // Set of nodes which have been visited.
  absl::flat_hash_set<Node*> visited_;

  // Set of nodes which are being traversed through.
  absl::flat_hash_set<Node*> traversing_;
};

// Visitor with a default action. If the Handle<Op> method is not overridden
// in the derived class then DefaultHandler is called when visiting nodes of
// type <Op>.
class DfsVisitorWithDefault : public DfsVisitor {
 public:
  virtual absl::Status DefaultHandler(Node* node) = 0;

  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* index) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleInvoke(Invoke* invoke) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleMap(Map* map) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryNand(NaryOp* and_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShra(BinOp* shra) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;
};

}  // namespace xls

#endif  // XLS_IR_DFS_VISITOR_H_

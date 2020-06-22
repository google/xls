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

#include "xls/ir/dfs_visitor.h"

namespace xls {

absl::Status DfsVisitorWithDefault::HandleAdd(BinOp* add) {
  return DefaultHandler(add);
}

absl::Status DfsVisitorWithDefault::HandleAndReduce(
    BitwiseReductionOp* and_reduce) {
  return DefaultHandler(and_reduce);
}

absl::Status DfsVisitorWithDefault::HandleOrReduce(
    BitwiseReductionOp* or_reduce) {
  return DefaultHandler(or_reduce);
}

absl::Status DfsVisitorWithDefault::HandleXorReduce(
    BitwiseReductionOp* xor_reduce) {
  return DefaultHandler(xor_reduce);
}

absl::Status DfsVisitorWithDefault::HandleNaryAnd(NaryOp* and_op) {
  return DefaultHandler(and_op);
}

absl::Status DfsVisitorWithDefault::HandleNaryNand(NaryOp* nand_op) {
  return DefaultHandler(nand_op);
}

absl::Status DfsVisitorWithDefault::HandleNaryNor(NaryOp* nor_op) {
  return DefaultHandler(nor_op);
}

absl::Status DfsVisitorWithDefault::HandleNaryOr(NaryOp* or_op) {
  return DefaultHandler(or_op);
}

absl::Status DfsVisitorWithDefault::HandleNaryXor(NaryOp* xor_op) {
  return DefaultHandler(xor_op);
}

absl::Status DfsVisitorWithDefault::HandleArray(Array* array) {
  return DefaultHandler(array);
}

absl::Status DfsVisitorWithDefault::HandleArrayIndex(ArrayIndex* index) {
  return DefaultHandler(index);
}

absl::Status DfsVisitorWithDefault::HandleArrayUpdate(ArrayUpdate* update) {
  return DefaultHandler(update);
}

absl::Status DfsVisitorWithDefault::HandleBitSlice(BitSlice* bit_slice) {
  return DefaultHandler(bit_slice);
}

absl::Status DfsVisitorWithDefault::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  return DefaultHandler(dynamic_bit_slice);
}

absl::Status DfsVisitorWithDefault::HandleConcat(Concat* concat) {
  return DefaultHandler(concat);
}

absl::Status DfsVisitorWithDefault::HandleCountedFor(CountedFor* counted_for) {
  return DefaultHandler(counted_for);
}

absl::Status DfsVisitorWithDefault::HandleDecode(Decode* decode) {
  return DefaultHandler(decode);
}
absl::Status DfsVisitorWithDefault::HandleEncode(Encode* encode) {
  return DefaultHandler(encode);
}

absl::Status DfsVisitorWithDefault::HandleEq(CompareOp* eq) {
  return DefaultHandler(eq);
}

absl::Status DfsVisitorWithDefault::HandleIdentity(UnOp* identity) {
  return DefaultHandler(identity);
}

absl::Status DfsVisitorWithDefault::HandleInvoke(Invoke* invoke) {
  return DefaultHandler(invoke);
}

absl::Status DfsVisitorWithDefault::HandleLiteral(Literal* literal) {
  return DefaultHandler(literal);
}

absl::Status DfsVisitorWithDefault::HandleMap(Map* map) {
  return DefaultHandler(map);
}

absl::Status DfsVisitorWithDefault::HandleNe(CompareOp* ne) {
  return DefaultHandler(ne);
}

absl::Status DfsVisitorWithDefault::HandleNeg(UnOp* neg) {
  return DefaultHandler(neg);
}

absl::Status DfsVisitorWithDefault::HandleNot(UnOp* not_op) {
  return DefaultHandler(not_op);
}

absl::Status DfsVisitorWithDefault::HandleOneHot(OneHot* one_hot) {
  return DefaultHandler(one_hot);
}

absl::Status DfsVisitorWithDefault::HandleOneHotSel(OneHotSelect* sel) {
  return DefaultHandler(sel);
}

absl::Status DfsVisitorWithDefault::HandleParam(Param* param) {
  return DefaultHandler(param);
}

absl::Status DfsVisitorWithDefault::HandleReverse(UnOp* reverse) {
  return DefaultHandler(reverse);
}

absl::Status DfsVisitorWithDefault::HandleSDiv(BinOp* div) {
  return DefaultHandler(div);
}

absl::Status DfsVisitorWithDefault::HandleSGe(CompareOp* ge) {
  return DefaultHandler(ge);
}

absl::Status DfsVisitorWithDefault::HandleSGt(CompareOp* gt) {
  return DefaultHandler(gt);
}

absl::Status DfsVisitorWithDefault::HandleSLe(CompareOp* le) {
  return DefaultHandler(le);
}

absl::Status DfsVisitorWithDefault::HandleSLt(CompareOp* lt) {
  return DefaultHandler(lt);
}

absl::Status DfsVisitorWithDefault::HandleUMul(ArithOp* mul) {
  return DefaultHandler(mul);
}

absl::Status DfsVisitorWithDefault::HandleSel(Select* sel) {
  return DefaultHandler(sel);
}

absl::Status DfsVisitorWithDefault::HandleShll(BinOp* shll) {
  return DefaultHandler(shll);
}

absl::Status DfsVisitorWithDefault::HandleShra(BinOp* shra) {
  return DefaultHandler(shra);
}

absl::Status DfsVisitorWithDefault::HandleShrl(BinOp* shrl) {
  return DefaultHandler(shrl);
}

absl::Status DfsVisitorWithDefault::HandleSignExtend(ExtendOp* sign_ext) {
  return DefaultHandler(sign_ext);
}

absl::Status DfsVisitorWithDefault::HandleSMul(ArithOp* mul) {
  return DefaultHandler(mul);
}

absl::Status DfsVisitorWithDefault::HandleSub(BinOp* sub) {
  return DefaultHandler(sub);
}

absl::Status DfsVisitorWithDefault::HandleTuple(Tuple* tuple) {
  return DefaultHandler(tuple);
}

absl::Status DfsVisitorWithDefault::HandleTupleIndex(TupleIndex* index) {
  return DefaultHandler(index);
}

absl::Status DfsVisitorWithDefault::HandleUDiv(BinOp* div) {
  return DefaultHandler(div);
}

absl::Status DfsVisitorWithDefault::HandleUGe(CompareOp* ge) {
  return DefaultHandler(ge);
}

absl::Status DfsVisitorWithDefault::HandleUGt(CompareOp* gt) {
  return DefaultHandler(gt);
}

absl::Status DfsVisitorWithDefault::HandleULe(CompareOp* le) {
  return DefaultHandler(le);
}

absl::Status DfsVisitorWithDefault::HandleULt(CompareOp* lt) {
  return DefaultHandler(lt);
}

absl::Status DfsVisitorWithDefault::HandleZeroExtend(ExtendOp* zero_ext) {
  return DefaultHandler(zero_ext);
}

}  // namespace xls

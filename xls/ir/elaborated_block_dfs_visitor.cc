// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License",
// .instance=instance}); you may not use this file except in compliance with the
// License. You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/ir/elaborated_block_dfs_visitor.h"

#include "absl/status/status.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/nodes.h"

namespace xls {

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleAdd(
    BinOp* add, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = add, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleAndReduce(
    BitwiseReductionOp* and_reduce, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = and_reduce, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleOrReduce(
    BitwiseReductionOp* or_reduce, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = or_reduce, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleXorReduce(
    BitwiseReductionOp* xor_reduce, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = xor_reduce, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNaryAnd(
    NaryOp* and_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = and_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNaryNand(
    NaryOp* nand_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = nand_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNaryNor(
    NaryOp* nor_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = nor_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNaryOr(
    NaryOp* or_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = or_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNaryXor(
    NaryOp* xor_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = xor_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleAfterAll(
    AfterAll* after_all, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = after_all, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleMinDelay(
    MinDelay* min_delay, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = min_delay, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleArray(
    Array* array, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = array, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleArrayIndex(
    ArrayIndex* index, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = index, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleArraySlice(
    ArraySlice* slice, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = slice, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleArrayUpdate(
    ArrayUpdate* update, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = update, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleArrayConcat(
    ArrayConcat* array_concat, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = array_concat, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleAssert(
    Assert* assert_op, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = assert_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleTrace(
    Trace* trace_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = trace_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleBitSlice(
    BitSlice* bit_slice, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = bit_slice, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleBitSliceUpdate(
    BitSliceUpdate* update, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = update, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleCover(
    Cover* cover, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = cover, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleReceive(
    Receive* receive, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = receive, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSend(
    Send* send, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = send, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = dynamic_bit_slice, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleConcat(
    Concat* concat, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = concat, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleCountedFor(
    CountedFor* counted_for, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = counted_for, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleDecode(
    Decode* decode, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = decode, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = dynamic_counted_for, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleEncode(
    Encode* encode, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = encode, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleEq(
    CompareOp* eq, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = eq, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleGate(
    Gate* gate, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = gate, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleIdentity(
    UnOp* identity, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = identity, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleInputPort(
    InputPort* input_port, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = input_port, .instance = instance});
}
absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleOutputPort(
    OutputPort* output_port, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = output_port, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleInvoke(
    Invoke* invoke, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = invoke, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleInstantiationInput(
    InstantiationInput* instantiation_input, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = instantiation_input, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleInstantiationOutput(
    InstantiationOutput* instantiation_output, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = instantiation_output, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleLiteral(
    Literal* literal, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = literal, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleMap(
    Map* map, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = map, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNe(
    CompareOp* ne, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = ne, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNeg(
    UnOp* neg, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = neg, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNot(
    UnOp* not_op, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = not_op, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleOneHot(
    OneHot* one_hot, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = one_hot, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleOneHotSel(
    OneHotSelect* sel, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = sel, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandlePrioritySel(
    PrioritySelect* sel, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = sel, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleParam(
    Param* param, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = param, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleStateRead(
    StateRead* state_read, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = state_read, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleNext(
    Next* next, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = next, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleRegisterRead(
    RegisterRead* reg_read, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = reg_read, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleRegisterWrite(
    RegisterWrite* reg_write, BlockInstance* instance) {
  return DefaultHandler(
      ElaboratedNode{.node = reg_write, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleReverse(
    UnOp* reverse, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = reverse, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSDiv(
    BinOp* div, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = div, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSMod(
    BinOp* mod, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = mod, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSGe(
    CompareOp* ge, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = ge, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSGt(
    CompareOp* gt, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = gt, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSLe(
    CompareOp* le, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = le, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSLt(
    CompareOp* lt, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = lt, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleUMul(
    ArithOp* mul, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = mul, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleUMulp(
    PartialProductOp* mul, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = mul, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSel(
    Select* sel, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = sel, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleShll(
    BinOp* shll, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = shll, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleShra(
    BinOp* shra, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = shra, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleShrl(
    BinOp* shrl, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = shrl, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSignExtend(
    ExtendOp* sign_ext, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = sign_ext, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSMul(
    ArithOp* mul, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = mul, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSMulp(
    PartialProductOp* mul, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = mul, .instance = instance});
}
absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleSub(
    BinOp* sub, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = sub, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleTuple(
    Tuple* tuple, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = tuple, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleTupleIndex(
    TupleIndex* index, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = index, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleUDiv(
    BinOp* div, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = div, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleUMod(
    BinOp* mod, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = mod, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleUGe(
    CompareOp* ge, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = ge, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleUGt(
    CompareOp* gt, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = gt, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleULe(
    CompareOp* le, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = le, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleULt(
    CompareOp* lt, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = lt, .instance = instance});
}

absl::Status ElaboratedBlockDfsVisitorWithDefault::HandleZeroExtend(
    ExtendOp* zero_ext, BlockInstance* instance) {
  return DefaultHandler(ElaboratedNode{.node = zero_ext, .instance = instance});
}

}  // namespace xls

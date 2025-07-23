// Copyright 2025 The XLS Authors
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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_visitor.h"

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"

namespace xls {

// Given a FuzzOpProto, call the corresponding Handle* function based on the
// FuzzOpProto type. This is a common visitor selection function.
void IrFuzzVisitor::VisitFuzzOp(const FuzzOpProto& fuzz_op) {
  switch (fuzz_op.fuzz_op_case()) {
    case FuzzOpProto::kParam:
      HandleParam(fuzz_op.param());
      break;
    case FuzzOpProto::kShra:
      HandleShra(fuzz_op.shra());
      break;
    case FuzzOpProto::kShll:
      HandleShll(fuzz_op.shll());
      break;
    case FuzzOpProto::kShrl:
      HandleShrl(fuzz_op.shrl());
      break;
    case FuzzOpProto::kOrOp:
      HandleOr(fuzz_op.or_op());
      break;
    case FuzzOpProto::kNor:
      HandleNor(fuzz_op.nor());
      break;
    case FuzzOpProto::kXorOp:
      HandleXor(fuzz_op.xor_op());
      break;
    case FuzzOpProto::kAndOp:
      HandleAnd(fuzz_op.and_op());
      break;
    case FuzzOpProto::kNand:
      HandleNand(fuzz_op.nand());
      break;
    case FuzzOpProto::kAndReduce:
      HandleAndReduce(fuzz_op.and_reduce());
      break;
    case FuzzOpProto::kOrReduce:
      HandleOrReduce(fuzz_op.or_reduce());
      break;
    case FuzzOpProto::kXorReduce:
      HandleXorReduce(fuzz_op.xor_reduce());
      break;
    case FuzzOpProto::kUmul:
      HandleUMul(fuzz_op.umul());
      break;
    case FuzzOpProto::kSmul:
      HandleSMul(fuzz_op.smul());
      break;
    case FuzzOpProto::kUmulp:
      HandleUMulp(fuzz_op.umulp());
      break;
    case FuzzOpProto::kSmulp:
      HandleSMulp(fuzz_op.smulp());
      break;
    case FuzzOpProto::kUdiv:
      HandleUDiv(fuzz_op.udiv());
      break;
    case FuzzOpProto::kSdiv:
      HandleSDiv(fuzz_op.sdiv());
      break;
    case FuzzOpProto::kUmod:
      HandleUMod(fuzz_op.umod());
      break;
    case FuzzOpProto::kSmod:
      HandleSMod(fuzz_op.smod());
      break;
    case FuzzOpProto::kSubtract:
      HandleSubtract(fuzz_op.subtract());
      break;
    case FuzzOpProto::kAdd:
      HandleAdd(fuzz_op.add());
      break;
    case FuzzOpProto::kConcat:
      HandleConcat(fuzz_op.concat());
      break;
    case FuzzOpProto::kUle:
      HandleULe(fuzz_op.ule());
      break;
    case FuzzOpProto::kUlt:
      HandleULt(fuzz_op.ult());
      break;
    case FuzzOpProto::kUge:
      HandleUGe(fuzz_op.uge());
      break;
    case FuzzOpProto::kUgt:
      HandleUGt(fuzz_op.ugt());
      break;
    case FuzzOpProto::kSle:
      HandleSLe(fuzz_op.sle());
      break;
    case FuzzOpProto::kSlt:
      HandleSLt(fuzz_op.slt());
      break;
    case FuzzOpProto::kSge:
      HandleSGe(fuzz_op.sge());
      break;
    case FuzzOpProto::kSgt:
      HandleSGt(fuzz_op.sgt());
      break;
    case FuzzOpProto::kEq:
      HandleEq(fuzz_op.eq());
      break;
    case FuzzOpProto::kNe:
      HandleNe(fuzz_op.ne());
      break;
    case FuzzOpProto::kNegate:
      HandleNegate(fuzz_op.negate());
      break;
    case FuzzOpProto::kNotOp:
      HandleNot(fuzz_op.not_op());
      break;
    case FuzzOpProto::kLiteral:
      HandleLiteral(fuzz_op.literal());
      break;
    case FuzzOpProto::kSelect:
      HandleSelect(fuzz_op.select());
      break;
    case FuzzOpProto::kOneHot:
      HandleOneHot(fuzz_op.one_hot());
      break;
    case FuzzOpProto::kOneHotSelect:
      HandleOneHotSelect(fuzz_op.one_hot_select());
      break;
    case FuzzOpProto::kPrioritySelect:
      HandlePrioritySelect(fuzz_op.priority_select());
      break;
    case FuzzOpProto::kClz:
      HandleClz(fuzz_op.clz());
      break;
    case FuzzOpProto::kCtz:
      HandleCtz(fuzz_op.ctz());
      break;
    case FuzzOpProto::kMatch:
      HandleMatch(fuzz_op.match());
      break;
    case FuzzOpProto::kMatchTrue:
      HandleMatchTrue(fuzz_op.match_true());
      break;
    case FuzzOpProto::kTuple:
      HandleTuple(fuzz_op.tuple());
      break;
    case FuzzOpProto::kArray:
      HandleArray(fuzz_op.array());
      break;
    case FuzzOpProto::kTupleIndex:
      HandleTupleIndex(fuzz_op.tuple_index());
      break;
    case FuzzOpProto::kArrayIndex:
      HandleArrayIndex(fuzz_op.array_index());
      break;
    case FuzzOpProto::kArraySlice:
      HandleArraySlice(fuzz_op.array_slice());
      break;
    case FuzzOpProto::kArrayUpdate:
      HandleArrayUpdate(fuzz_op.array_update());
      break;
    case FuzzOpProto::kArrayConcat:
      HandleArrayConcat(fuzz_op.array_concat());
      break;
    case FuzzOpProto::kReverse:
      HandleReverse(fuzz_op.reverse());
      break;
    case FuzzOpProto::kIdentity:
      HandleIdentity(fuzz_op.identity());
      break;
    case FuzzOpProto::kSignExtend:
      HandleSignExtend(fuzz_op.sign_extend());
      break;
    case FuzzOpProto::kZeroExtend:
      HandleZeroExtend(fuzz_op.zero_extend());
      break;
    case FuzzOpProto::kBitSlice:
      HandleBitSlice(fuzz_op.bit_slice());
      break;
    case FuzzOpProto::kBitSliceUpdate:
      HandleBitSliceUpdate(fuzz_op.bit_slice_update());
      break;
    case FuzzOpProto::kDynamicBitSlice:
      HandleDynamicBitSlice(fuzz_op.dynamic_bit_slice());
      break;
    case FuzzOpProto::kEncode:
      HandleEncode(fuzz_op.encode());
      break;
    case FuzzOpProto::kDecode:
      HandleDecode(fuzz_op.decode());
      break;
    case FuzzOpProto::kGate:
      HandleGate(fuzz_op.gate());
      break;
    case FuzzOpProto::FUZZ_OP_NOT_SET:
      break;
  }
}

}  // namespace xls

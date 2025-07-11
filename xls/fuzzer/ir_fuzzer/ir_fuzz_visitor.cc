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
void IrFuzzVisitor::VisitFuzzOp(FuzzOpProto* fuzz_op) {
  switch (fuzz_op->fuzz_op_case()) {
    case FuzzOpProto::kParam:
      HandleParam(fuzz_op->mutable_param());
      break;
    case FuzzOpProto::kShra:
      HandleShra(fuzz_op->mutable_shra());
      break;
    case FuzzOpProto::kShll:
      HandleShll(fuzz_op->mutable_shll());
      break;
    case FuzzOpProto::kShrl:
      HandleShrl(fuzz_op->mutable_shrl());
      break;
    case FuzzOpProto::kOrOp:
      HandleOr(fuzz_op->mutable_or_op());
      break;
    case FuzzOpProto::kNor:
      HandleNor(fuzz_op->mutable_nor());
      break;
    case FuzzOpProto::kXorOp:
      HandleXor(fuzz_op->mutable_xor_op());
      break;
    case FuzzOpProto::kAndOp:
      HandleAnd(fuzz_op->mutable_and_op());
      break;
    case FuzzOpProto::kNand:
      HandleNand(fuzz_op->mutable_nand());
      break;
    case FuzzOpProto::kAndReduce:
      HandleAndReduce(fuzz_op->mutable_and_reduce());
      break;
    case FuzzOpProto::kOrReduce:
      HandleOrReduce(fuzz_op->mutable_or_reduce());
      break;
    case FuzzOpProto::kXorReduce:
      HandleXorReduce(fuzz_op->mutable_xor_reduce());
      break;
    case FuzzOpProto::kUmul:
      HandleUMul(fuzz_op->mutable_umul());
      break;
    case FuzzOpProto::kSmul:
      HandleSMul(fuzz_op->mutable_smul());
      break;
    case FuzzOpProto::kUdiv:
      HandleUDiv(fuzz_op->mutable_udiv());
      break;
    case FuzzOpProto::kSdiv:
      HandleSDiv(fuzz_op->mutable_sdiv());
      break;
    case FuzzOpProto::kUmod:
      HandleUMod(fuzz_op->mutable_umod());
      break;
    case FuzzOpProto::kSmod:
      HandleSMod(fuzz_op->mutable_smod());
      break;
    case FuzzOpProto::kSubtract:
      HandleSubtract(fuzz_op->mutable_subtract());
      break;
    case FuzzOpProto::kAdd:
      HandleAdd(fuzz_op->mutable_add());
      break;
    case FuzzOpProto::kConcat:
      HandleConcat(fuzz_op->mutable_concat());
      break;
    case FuzzOpProto::kUle:
      HandleULe(fuzz_op->mutable_ule());
      break;
    case FuzzOpProto::kUlt:
      HandleULt(fuzz_op->mutable_ult());
      break;
    case FuzzOpProto::kUge:
      HandleUGe(fuzz_op->mutable_uge());
      break;
    case FuzzOpProto::kUgt:
      HandleUGt(fuzz_op->mutable_ugt());
      break;
    case FuzzOpProto::kSle:
      HandleSLe(fuzz_op->mutable_sle());
      break;
    case FuzzOpProto::kSlt:
      HandleSLt(fuzz_op->mutable_slt());
      break;
    case FuzzOpProto::kSge:
      HandleSGe(fuzz_op->mutable_sge());
      break;
    case FuzzOpProto::kSgt:
      HandleSGt(fuzz_op->mutable_sgt());
      break;
    case FuzzOpProto::kEq:
      HandleEq(fuzz_op->mutable_eq());
      break;
    case FuzzOpProto::kNe:
      HandleNe(fuzz_op->mutable_ne());
      break;
    case FuzzOpProto::kNegate:
      HandleNegate(fuzz_op->mutable_negate());
      break;
    case FuzzOpProto::kNotOp:
      HandleNot(fuzz_op->mutable_not_op());
      break;
    case FuzzOpProto::kLiteral:
      HandleLiteral(fuzz_op->mutable_literal());
      break;
    case FuzzOpProto::kSelect:
      HandleSelect(fuzz_op->mutable_select());
      break;
    case FuzzOpProto::kOneHot:
      HandleOneHot(fuzz_op->mutable_one_hot());
      break;
    case FuzzOpProto::kOneHotSelect:
      HandleOneHotSelect(fuzz_op->mutable_one_hot_select());
      break;
    case FuzzOpProto::kPrioritySelect:
      HandlePrioritySelect(fuzz_op->mutable_priority_select());
      break;
    case FuzzOpProto::kClz:
      HandleClz(fuzz_op->mutable_clz());
      break;
    case FuzzOpProto::kCtz:
      HandleCtz(fuzz_op->mutable_ctz());
      break;
    case FuzzOpProto::kMatch:
      HandleMatch(fuzz_op->mutable_match());
      break;
    case FuzzOpProto::kMatchTrue:
      HandleMatchTrue(fuzz_op->mutable_match_true());
      break;
    case FuzzOpProto::kReverse:
      HandleReverse(fuzz_op->mutable_reverse());
      break;
    case FuzzOpProto::kIdentity:
      HandleIdentity(fuzz_op->mutable_identity());
      break;
    case FuzzOpProto::kSignExtend:
      HandleSignExtend(fuzz_op->mutable_sign_extend());
      break;
    case FuzzOpProto::kZeroExtend:
      HandleZeroExtend(fuzz_op->mutable_zero_extend());
      break;
    case FuzzOpProto::kBitSlice:
      HandleBitSlice(fuzz_op->mutable_bit_slice());
      break;
    case FuzzOpProto::kBitSliceUpdate:
      HandleBitSliceUpdate(fuzz_op->mutable_bit_slice_update());
      break;
    case FuzzOpProto::kDynamicBitSlice:
      HandleDynamicBitSlice(fuzz_op->mutable_dynamic_bit_slice());
      break;
    case FuzzOpProto::kEncode:
      HandleEncode(fuzz_op->mutable_encode());
      break;
    case FuzzOpProto::kDecode:
      HandleDecode(fuzz_op->mutable_decode());
      break;
    case FuzzOpProto::kGate:
      HandleGate(fuzz_op->mutable_gate());
      break;
    case FuzzOpProto::FUZZ_OP_NOT_SET:
      break;
  }
}

}  // namespace xls

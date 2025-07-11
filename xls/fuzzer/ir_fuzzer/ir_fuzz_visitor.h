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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"

namespace xls {

// Interface visitor class inherited by classes which perform a pass over the
// FuzzProgramProto. This class is almost identical to the dfs_visitor.h class,
// but handles FuzzOpProtos instead of IR nodes.
class IrFuzzVisitor {
 public:
  virtual ~IrFuzzVisitor() = default;

  // These functions correlate to an IR Node.
  virtual void HandleParam(FuzzParamProto* param) = 0;
  virtual void HandleShra(FuzzShraProto* shra) = 0;
  virtual void HandleShrl(FuzzShrlProto* shrl) = 0;
  virtual void HandleShll(FuzzShllProto* shll) = 0;
  virtual void HandleOr(FuzzOrProto* or_op) = 0;
  virtual void HandleNor(FuzzNorProto* nor) = 0;
  virtual void HandleXor(FuzzXorProto* xor_op) = 0;
  virtual void HandleAnd(FuzzAndProto* and_op) = 0;
  virtual void HandleNand(FuzzNandProto* nand) = 0;
  virtual void HandleAndReduce(FuzzAndReduceProto* and_reduce) = 0;
  virtual void HandleOrReduce(FuzzOrReduceProto* or_reduce) = 0;
  virtual void HandleXorReduce(FuzzXorReduceProto* xor_reduce) = 0;
  virtual void HandleUMul(FuzzUMulProto* umul) = 0;
  virtual void HandleSMul(FuzzSMulProto* smul) = 0;
  virtual void HandleUDiv(FuzzUDivProto* udiv) = 0;
  virtual void HandleSDiv(FuzzSDivProto* sdiv) = 0;
  virtual void HandleUMod(FuzzUModProto* umod) = 0;
  virtual void HandleSMod(FuzzSModProto* smod) = 0;
  virtual void HandleSubtract(FuzzSubtractProto* subtract) = 0;
  virtual void HandleAdd(FuzzAddProto* add) = 0;
  virtual void HandleConcat(FuzzConcatProto* concat) = 0;
  virtual void HandleULe(FuzzULeProto* ule) = 0;
  virtual void HandleULt(FuzzULtProto* ult) = 0;
  virtual void HandleUGe(FuzzUGeProto* uge) = 0;
  virtual void HandleUGt(FuzzUGtProto* ugt) = 0;
  virtual void HandleSLe(FuzzSLeProto* sle) = 0;
  virtual void HandleSLt(FuzzSLtProto* slt) = 0;
  virtual void HandleSGe(FuzzSGeProto* sge) = 0;
  virtual void HandleSGt(FuzzSGtProto* sgt) = 0;
  virtual void HandleEq(FuzzEqProto* eq) = 0;
  virtual void HandleNe(FuzzNeProto* ne) = 0;
  virtual void HandleNegate(FuzzNegateProto* negate) = 0;
  virtual void HandleNot(FuzzNotProto* not_op) = 0;
  virtual void HandleLiteral(FuzzLiteralProto* literal) = 0;
  virtual void HandleSelect(FuzzSelectProto* select) = 0;
  virtual void HandleOneHot(FuzzOneHotProto* one_hot) = 0;
  virtual void HandleOneHotSelect(FuzzOneHotSelectProto* one_hot_select) = 0;
  virtual void HandlePrioritySelect(
      FuzzPrioritySelectProto* priority_select) = 0;
  virtual void HandleClz(FuzzClzProto* clz) = 0;
  virtual void HandleCtz(FuzzCtzProto* ctz) = 0;
  virtual void HandleMatch(FuzzMatchProto* match) = 0;
  virtual void HandleMatchTrue(FuzzMatchTrueProto* match_true) = 0;
  virtual void HandleReverse(FuzzReverseProto* reverse) = 0;
  virtual void HandleIdentity(FuzzIdentityProto* identity) = 0;
  virtual void HandleSignExtend(FuzzSignExtendProto* sign_extend) = 0;
  virtual void HandleZeroExtend(FuzzZeroExtendProto* zero_extend) = 0;
  virtual void HandleBitSlice(FuzzBitSliceProto* bit_slice) = 0;
  virtual void HandleBitSliceUpdate(
      FuzzBitSliceUpdateProto* bit_slice_update) = 0;
  virtual void HandleDynamicBitSlice(
      FuzzDynamicBitSliceProto* dynamic_bit_slice) = 0;
  virtual void HandleEncode(FuzzEncodeProto* encode) = 0;
  virtual void HandleDecode(FuzzDecodeProto* decode) = 0;
  virtual void HandleGate(FuzzGateProto* gate) = 0;

  void VisitFuzzOp(FuzzOpProto* fuzz_op);
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_

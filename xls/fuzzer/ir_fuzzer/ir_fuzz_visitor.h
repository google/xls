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
  virtual void HandleParam(const FuzzParamProto& param) = 0;
  virtual void HandleShra(const FuzzShraProto& shra) = 0;
  virtual void HandleShrl(const FuzzShrlProto& shrl) = 0;
  virtual void HandleShll(const FuzzShllProto& shll) = 0;
  virtual void HandleOr(const FuzzOrProto& or_op) = 0;
  virtual void HandleNor(const FuzzNorProto& nor) = 0;
  virtual void HandleXor(const FuzzXorProto& xor_op) = 0;
  virtual void HandleAnd(const FuzzAndProto& and_op) = 0;
  virtual void HandleNand(const FuzzNandProto& nand) = 0;
  virtual void HandleAndReduce(const FuzzAndReduceProto& and_reduce) = 0;
  virtual void HandleOrReduce(const FuzzOrReduceProto& or_reduce) = 0;
  virtual void HandleXorReduce(const FuzzXorReduceProto& xor_reduce) = 0;
  virtual void HandleUMul(const FuzzUMulProto& umul) = 0;
  virtual void HandleSMul(const FuzzSMulProto& smul) = 0;
  virtual void HandleUMulp(const FuzzUMulpProto& umulp) = 0;
  virtual void HandleSMulp(const FuzzSMulpProto& smulp) = 0;
  virtual void HandleUDiv(const FuzzUDivProto& udiv) = 0;
  virtual void HandleSDiv(const FuzzSDivProto& sdiv) = 0;
  virtual void HandleUMod(const FuzzUModProto& umod) = 0;
  virtual void HandleSMod(const FuzzSModProto& smod) = 0;
  virtual void HandleSubtract(const FuzzSubtractProto& subtract) = 0;
  virtual void HandleAdd(const FuzzAddProto& add) = 0;
  virtual void HandleConcat(const FuzzConcatProto& concat) = 0;
  virtual void HandleULe(const FuzzULeProto& ule) = 0;
  virtual void HandleULt(const FuzzULtProto& ult) = 0;
  virtual void HandleUGe(const FuzzUGeProto& uge) = 0;
  virtual void HandleUGt(const FuzzUGtProto& ugt) = 0;
  virtual void HandleSLe(const FuzzSLeProto& sle) = 0;
  virtual void HandleSLt(const FuzzSLtProto& slt) = 0;
  virtual void HandleSGe(const FuzzSGeProto& sge) = 0;
  virtual void HandleSGt(const FuzzSGtProto& sgt) = 0;
  virtual void HandleEq(const FuzzEqProto& eq) = 0;
  virtual void HandleNe(const FuzzNeProto& ne) = 0;
  virtual void HandleNegate(const FuzzNegateProto& negate) = 0;
  virtual void HandleNot(const FuzzNotProto& not_op) = 0;
  virtual void HandleLiteral(const FuzzLiteralProto& literal) = 0;
  virtual void HandleSelect(const FuzzSelectProto& select) = 0;
  virtual void HandleOneHot(const FuzzOneHotProto& one_hot) = 0;
  virtual void HandleOneHotSelect(
      const FuzzOneHotSelectProto& one_hot_select) = 0;
  virtual void HandlePrioritySelect(
      const FuzzPrioritySelectProto& priority_select) = 0;
  virtual void HandleClz(const FuzzClzProto& clz) = 0;
  virtual void HandleCtz(const FuzzCtzProto& ctz) = 0;
  virtual void HandleMatch(const FuzzMatchProto& match) = 0;
  virtual void HandleMatchTrue(const FuzzMatchTrueProto& match_true) = 0;
  virtual void HandleTuple(const FuzzTupleProto& tuple) = 0;
  virtual void HandleArray(const FuzzArrayProto& array) = 0;
  virtual void HandleTupleIndex(const FuzzTupleIndexProto& tuple_index) = 0;
  virtual void HandleArrayIndex(const FuzzArrayIndexProto& array_index) = 0;
  virtual void HandleArraySlice(const FuzzArraySliceProto& array_slice) = 0;
  virtual void HandleArrayUpdate(const FuzzArrayUpdateProto& array_update) = 0;
  virtual void HandleArrayConcat(const FuzzArrayConcatProto& array_concat) = 0;
  virtual void HandleReverse(const FuzzReverseProto& reverse) = 0;
  virtual void HandleIdentity(const FuzzIdentityProto& identity) = 0;
  virtual void HandleSignExtend(const FuzzSignExtendProto& sign_extend) = 0;
  virtual void HandleZeroExtend(const FuzzZeroExtendProto& zero_extend) = 0;
  virtual void HandleBitSlice(const FuzzBitSliceProto& bit_slice) = 0;
  virtual void HandleBitSliceUpdate(
      const FuzzBitSliceUpdateProto& bit_slice_update) = 0;
  virtual void HandleDynamicBitSlice(
      const FuzzDynamicBitSliceProto& dynamic_bit_slice) = 0;
  virtual void HandleEncode(const FuzzEncodeProto& encode) = 0;
  virtual void HandleDecode(const FuzzDecodeProto& decode) = 0;
  virtual void HandleGate(const FuzzGateProto& gate) = 0;

  void VisitFuzzOp(const FuzzOpProto& fuzz_op);
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_

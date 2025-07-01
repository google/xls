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

#ifndef XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_
#define XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_

#include <cstdint>
#include <vector>

#include "google/protobuf/repeated_ptr_field.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_visitor.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls {

// Pass that iterates over the FuzzOpProtos within the FuzzProgramProto. Each
// FuzzOpProto gets instantiated into an IR node/BValue. This value is placed on
// the stack. BValues on the stack may be used as operands by future
// instantiated FuzzOps. We use a stack because it is a simple way to reference
// previous FuzzOps through indices.
class GenIrNodesPass : public IrFuzzVisitor {
 public:
  GenIrNodesPass(FuzzProgramProto* fuzz_program, Package* p,
                 FunctionBuilder* fb, std::vector<BValue>& stack)
      : fuzz_program_(fuzz_program), p_(p), fb_(fb), stack_(stack) {}

  void GenIrNodes();

  void HandleParam(FuzzParamProto* param) override;
  void HandleShra(FuzzShraProto* shra) override;
  void HandleShrl(FuzzShrlProto* shrl) override;
  void HandleShll(FuzzShllProto* shll) override;
  void HandleOr(FuzzOrProto* or_op) override;
  void HandleNor(FuzzNorProto* nor) override;
  void HandleXor(FuzzXorProto* xor_op) override;
  void HandleAnd(FuzzAndProto* and_op) override;
  void HandleNand(FuzzNandProto* nand) override;
  void HandleAndReduce(FuzzAndReduceProto* and_reduce) override;
  void HandleOrReduce(FuzzOrReduceProto* or_reduce) override;
  void HandleXorReduce(FuzzXorReduceProto* xor_reduce) override;
  void HandleUMul(FuzzUMulProto* umul) override;
  void HandleSMul(FuzzSMulProto* smul) override;
  void HandleUDiv(FuzzUDivProto* udiv) override;
  void HandleSDiv(FuzzSDivProto* sdiv) override;
  void HandleUMod(FuzzUModProto* umod) override;
  void HandleSMod(FuzzSModProto* smod) override;
  void HandleSubtract(FuzzSubtractProto* subtract) override;
  void HandleAdd(FuzzAddProto* add) override;
  void HandleConcat(FuzzConcatProto* concat) override;
  void HandleULe(FuzzULeProto* ule) override;
  void HandleULt(FuzzULtProto* ult) override;
  void HandleUGe(FuzzUGeProto* uge) override;
  void HandleUGt(FuzzUGtProto* ugt) override;
  void HandleSLe(FuzzSLeProto* sle) override;
  void HandleSLt(FuzzSLtProto* slt) override;
  void HandleSGe(FuzzSGeProto* sge) override;
  void HandleSGt(FuzzSGtProto* sgt) override;
  void HandleEq(FuzzEqProto* eq) override;
  void HandleNe(FuzzNeProto* ne) override;
  void HandleNegate(FuzzNegateProto* negate) override;
  void HandleNot(FuzzNotProto* not_op) override;
  void HandleLiteral(FuzzLiteralProto* literal) override;
  void HandleSelect(FuzzSelectProto* select) override;
  void HandleOneHot(FuzzOneHotProto* one_hot) override;
  void HandleOneHotSelect(FuzzOneHotSelectProto* one_hot_select) override;
  void HandlePrioritySelect(FuzzPrioritySelectProto* priority_select) override;
  void HandleClz(FuzzClzProto* clz) override;
  void HandleCtz(FuzzCtzProto* ctz) override;
  void HandleMatch(FuzzMatchProto* match) override;
  void HandleMatchTrue(FuzzMatchTrueProto* match_true) override;
  void HandleReverse(FuzzReverseProto* reverse) override;
  void HandleIdentity(FuzzIdentityProto* identity) override;
  void HandleSignExtend(FuzzSignExtendProto* sign_extend) override;
  void HandleZeroExtend(FuzzZeroExtendProto* zero_extend) override;
  void HandleBitSlice(FuzzBitSliceProto* bit_slice) override;
  void HandleBitSliceUpdate(FuzzBitSliceUpdateProto* bit_slice_update) override;
  void HandleDynamicBitSlice(
      FuzzDynamicBitSliceProto* dynamic_bit_slice) override;
  void HandleEncode(FuzzEncodeProto* encode) override;
  void HandleDecode(FuzzDecodeProto* decode) override;
  void HandleGate(FuzzGateProto* gate) override;

 private:
  std::vector<FunctionBuilder::Case> GetCases(
      google::protobuf::RepeatedPtrField<CaseProto>* case_protos, int64_t bit_width);

  BValue GetOperand(int64_t idx);
  std::vector<BValue> GetOperands(google::protobuf::RepeatedField<int64_t>* operand_idxs,
                                  int64_t min_operand_count = 0,
                                  int64_t max_operand_count = -1);
  BValue GetWidthFittedOperand(FittedOperandIdxProto* operand_idx,
                               int64_t bit_width);
  std::vector<BValue> GetWidthFittedOperands(
      google::protobuf::RepeatedPtrField<FittedOperandIdxProto>* operand_idxs,
      int64_t bit_width, int64_t min_operand_count = 0,
      int64_t max_operand_count = -1);

  FuzzProgramProto* fuzz_program_;
  Package* p_;
  FunctionBuilder* fb_;
  std::vector<BValue>& stack_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_

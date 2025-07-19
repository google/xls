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
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls {

// Pass that iterates over the FuzzOpProtos within the FuzzProgramProto. Each
// FuzzOpProto gets instantiated into an IR node/BValue. This value is placed on
// the context list. BValues on the context list may be used as operands by
// future instantiated FuzzOps. We use a context list because it is a simple way
// to reference previous FuzzOps through indices.
class GenIrNodesPass : public IrFuzzVisitor {
 public:
  GenIrNodesPass(const FuzzProgramProto& fuzz_program, Package* p,
                 FunctionBuilder* fb, IrNodeContextList& context_list)
      : fuzz_program_(fuzz_program),
        p_(p),
        fb_(fb),
        context_list_(context_list) {}

  void GenIrNodes();

  void HandleParam(const FuzzParamProto& param) override;
  void HandleShra(const FuzzShraProto& shra) override;
  void HandleShrl(const FuzzShrlProto& shrl) override;
  void HandleShll(const FuzzShllProto& shll) override;
  void HandleOr(const FuzzOrProto& or_op) override;
  void HandleNor(const FuzzNorProto& nor) override;
  void HandleXor(const FuzzXorProto& xor_op) override;
  void HandleAnd(const FuzzAndProto& and_op) override;
  void HandleNand(const FuzzNandProto& nand) override;
  void HandleAndReduce(const FuzzAndReduceProto& and_reduce) override;
  void HandleOrReduce(const FuzzOrReduceProto& or_reduce) override;
  void HandleXorReduce(const FuzzXorReduceProto& xor_reduce) override;
  void HandleUMul(const FuzzUMulProto& umul) override;
  void HandleSMul(const FuzzSMulProto& smul) override;
  void HandleUMulp(const FuzzUMulpProto& umulp) override;
  void HandleSMulp(const FuzzSMulpProto& smulp) override;
  void HandleUDiv(const FuzzUDivProto& udiv) override;
  void HandleSDiv(const FuzzSDivProto& sdiv) override;
  void HandleUMod(const FuzzUModProto& umod) override;
  void HandleSMod(const FuzzSModProto& smod) override;
  void HandleSubtract(const FuzzSubtractProto& subtract) override;
  void HandleAdd(const FuzzAddProto& add) override;
  void HandleConcat(const FuzzConcatProto& concat) override;
  void HandleULe(const FuzzULeProto& ule) override;
  void HandleULt(const FuzzULtProto& ult) override;
  void HandleUGe(const FuzzUGeProto& uge) override;
  void HandleUGt(const FuzzUGtProto& ugt) override;
  void HandleSLe(const FuzzSLeProto& sle) override;
  void HandleSLt(const FuzzSLtProto& slt) override;
  void HandleSGe(const FuzzSGeProto& sge) override;
  void HandleSGt(const FuzzSGtProto& sgt) override;
  void HandleEq(const FuzzEqProto& eq) override;
  void HandleNe(const FuzzNeProto& ne) override;
  void HandleNegate(const FuzzNegateProto& negate) override;
  void HandleNot(const FuzzNotProto& not_op) override;
  void HandleLiteral(const FuzzLiteralProto& literal) override;
  void HandleSelect(const FuzzSelectProto& select) override;
  void HandleOneHot(const FuzzOneHotProto& one_hot) override;
  void HandleOneHotSelect(const FuzzOneHotSelectProto& one_hot_select) override;
  void HandlePrioritySelect(
      const FuzzPrioritySelectProto& priority_select) override;
  void HandleClz(const FuzzClzProto& clz) override;
  void HandleCtz(const FuzzCtzProto& ctz) override;
  void HandleMatch(const FuzzMatchProto& match) override;
  void HandleMatchTrue(const FuzzMatchTrueProto& match_true) override;
  void HandleTuple(const FuzzTupleProto& tuple) override;
  void HandleArray(const FuzzArrayProto& array) override;
  void HandleTupleIndex(const FuzzTupleIndexProto& tuple_index) override;
  void HandleArrayIndex(const FuzzArrayIndexProto& array_index) override;
  void HandleArraySlice(const FuzzArraySliceProto& array_slice) override;
  void HandleArrayUpdate(const FuzzArrayUpdateProto& array_update) override;
  void HandleArrayConcat(const FuzzArrayConcatProto& array_concat) override;
  void HandleReverse(const FuzzReverseProto& reverse) override;
  void HandleIdentity(const FuzzIdentityProto& identity) override;
  void HandleSignExtend(const FuzzSignExtendProto& sign_extend) override;
  void HandleZeroExtend(const FuzzZeroExtendProto& zero_extend) override;
  void HandleBitSlice(const FuzzBitSliceProto& bit_slice) override;
  void HandleBitSliceUpdate(
      const FuzzBitSliceUpdateProto& bit_slice_update) override;
  void HandleDynamicBitSlice(
      const FuzzDynamicBitSliceProto& dynamic_bit_slice) override;
  void HandleEncode(const FuzzEncodeProto& encode) override;
  void HandleDecode(const FuzzDecodeProto& decode) override;
  void HandleGate(const FuzzGateProto& gate) override;

 private:
  BValue GetOperand(const OperandIdxProto& operand_idx);
  BValue GetBitsOperand(const BitsOperandIdxProto& operand_idx);
  BValue GetTupleOperand(const TupleOperandIdxProto& operand_idx);
  BValue GetArrayOperand(const ArrayOperandIdxProto& operand_idx);
  BValue GetCoercedOperand(const OperandIdxProto& operand_idx,
                           const CoercedTypeProto& coerced_type);
  BValue GetCoercedBitsOperand(const BitsOperandIdxProto& operand_idx,
                               const BitsCoercedTypeProto& coerced_type);
  BValue GetCoercedArrayOperand(const ArrayOperandIdxProto& operand_idx,
                                const ArrayCoercedTypeProto& coerced_type);
  BValue GetFittedOperand(const OperandIdxProto& operand_idx,
                          const CoercionMethodProto& coercion_method,
                          Type* type);
  BValue GetBitsFittedOperand(const BitsOperandIdxProto& operand_idx,
                              const BitsCoercionMethodProto& coercion_method,
                              int64_t bit_width);

  std::vector<BValue> GetOperands(
      const google::protobuf::RepeatedPtrField<OperandIdxProto>& operand_idxs,
      int64_t min_operand_count = 0, int64_t max_operand_count = -1);
  std::vector<BValue> GetBitsOperands(
      const google::protobuf::RepeatedPtrField<BitsOperandIdxProto>& operand_idxs,
      int64_t min_operand_count = 0, int64_t max_operand_count = -1);
  std::vector<BValue> GetArrayOperands(
      const google::protobuf::RepeatedPtrField<ArrayOperandIdxProto>& operand_idxs,
      int64_t min_operand_count = 0, int64_t max_operand_count = -1);
  std::vector<BValue> GetCoercedOperands(
      const google::protobuf::RepeatedPtrField<OperandIdxProto>& operand_idxs,
      const CoercedTypeProto& coerced_type, int64_t min_operand_count = 0,
      int64_t max_operand_count = -1);
  std::vector<BValue> GetCoercedBitsOperands(
      const google::protobuf::RepeatedPtrField<BitsOperandIdxProto>& operand_idxs,
      const BitsCoercedTypeProto& coerced_type, int64_t min_operand_count = 0,
      int64_t max_operand_count = -1);
  std::vector<BValue> GetCoercedArrayOperands(
      const google::protobuf::RepeatedPtrField<ArrayOperandIdxProto>& operand_idxs,
      const ArrayCoercedTypeProto& coerced_type, int64_t min_operand_count = 0,
      int64_t max_operand_count = -1);

  const FuzzProgramProto& fuzz_program_;
  Package* p_;
  FunctionBuilder* fb_;
  IrNodeContextList& context_list_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_

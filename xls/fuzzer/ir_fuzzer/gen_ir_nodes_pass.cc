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

#include "xls/fuzzer/ir_fuzzer/gen_ir_nodes_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "google/protobuf/repeated_field.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/lsb_or_msb.h"

namespace xls {

// Loops through all of the FuzzOpProtos in the FuzzProgramProto. Each
// FuzzOpProto is a randomly generated object that is used to
// instantiate/generate an IR node/BValue. Add these BValues to the list. Some
// FuzzOpProtos may require retrieving previous BValues from the list.
void GenIrNodesPass::GenIrNodes() {
  for (FuzzOpProto& fuzz_op : *fuzz_program_->mutable_fuzz_ops()) {
    VisitFuzzOp(&fuzz_op);
  }
}

void GenIrNodesPass::HandleParam(FuzzParamProto* param) {
  param->set_bit_width(BoundedWidth(param->bit_width()));
  // Params are named as "p" followed by the list index of the param.
  context_list_.AppendElement(
      fb_->Param("p" + std::to_string(context_list_.GetListSize()),
                 p_->GetBitsType(param->bit_width())));
}

void GenIrNodesPass::HandleShra(FuzzShraProto* shra) {
  BValue operand = GetOperand(shra->operand_idx());
  BValue amount = GetOperand(shra->amount_idx());
  context_list_.AppendElement(fb_->Shra(operand, amount));
}

void GenIrNodesPass::HandleShrl(FuzzShrlProto* shrl) {
  BValue operand = GetOperand(shrl->operand_idx());
  BValue amount = GetOperand(shrl->amount_idx());
  context_list_.AppendElement(fb_->Shrl(operand, amount));
}

void GenIrNodesPass::HandleShll(FuzzShllProto* shll) {
  BValue operand = GetOperand(shll->operand_idx());
  BValue amount = GetOperand(shll->amount_idx());
  context_list_.AppendElement(fb_->Shll(operand, amount));
}

void GenIrNodesPass::HandleOr(FuzzOrProto* or_op) {
  or_op->set_bit_width(BoundedWidth(or_op->bit_width()));
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetWidthFittedOperands(or_op->mutable_operand_idxs(), or_op->bit_width(),
                             /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Or(operands));
}

void GenIrNodesPass::HandleNor(FuzzNorProto* nor) {
  nor->set_bit_width(BoundedWidth(nor->bit_width()));
  std::vector<BValue> operands = GetWidthFittedOperands(
      nor->mutable_operand_idxs(), nor->bit_width(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Nor(operands));
}

void GenIrNodesPass::HandleXor(FuzzXorProto* xor_op) {
  xor_op->set_bit_width(BoundedWidth(xor_op->bit_width()));
  std::vector<BValue> operands =
      GetWidthFittedOperands(xor_op->mutable_operand_idxs(),
                             xor_op->bit_width(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Xor(operands));
}

void GenIrNodesPass::HandleAnd(FuzzAndProto* and_op) {
  and_op->set_bit_width(BoundedWidth(and_op->bit_width()));
  std::vector<BValue> operands =
      GetWidthFittedOperands(and_op->mutable_operand_idxs(),
                             and_op->bit_width(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->And(operands));
}

void GenIrNodesPass::HandleNand(FuzzNandProto* nand) {
  nand->set_bit_width(BoundedWidth(nand->bit_width()));
  std::vector<BValue> operands = GetWidthFittedOperands(
      nand->mutable_operand_idxs(), nand->bit_width(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Nand(operands));
}

void GenIrNodesPass::HandleAndReduce(FuzzAndReduceProto* and_reduce) {
  BValue operand = GetOperand(and_reduce->operand_idx());
  context_list_.AppendElement(fb_->AndReduce(operand));
}

void GenIrNodesPass::HandleOrReduce(FuzzOrReduceProto* or_reduce) {
  BValue operand = GetOperand(or_reduce->operand_idx());
  context_list_.AppendElement(fb_->OrReduce(operand));
}

void GenIrNodesPass::HandleXorReduce(FuzzXorReduceProto* xor_reduce) {
  BValue operand = GetOperand(xor_reduce->operand_idx());
  context_list_.AppendElement(fb_->XorReduce(operand));
}

void GenIrNodesPass::HandleUMul(FuzzUMulProto* umul) {
  BValue lhs = GetOperand(umul->lhs_idx());
  BValue rhs = GetOperand(umul->rhs_idx());
  umul->set_bit_width(BoundedWidth(
      umul->bit_width(), 1,
      std::min<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie())));
  context_list_.AppendElement(fb_->UMul(lhs, rhs, umul->bit_width()));
}

void GenIrNodesPass::HandleSMul(FuzzSMulProto* smul) {
  BValue lhs = GetOperand(smul->lhs_idx());
  BValue rhs = GetOperand(smul->rhs_idx());
  smul->set_bit_width(BoundedWidth(
      smul->bit_width(), 1,
      std::min<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie())));
  context_list_.AppendElement(fb_->SMul(lhs, rhs, smul->bit_width()));
}

void GenIrNodesPass::HandleUDiv(FuzzUDivProto* udiv) {
  udiv->set_bit_width(BoundedWidth(udiv->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(udiv->mutable_lhs_idx(), udiv->bit_width());
  BValue rhs =
      GetWidthFittedOperand(udiv->mutable_rhs_idx(), udiv->bit_width());
  context_list_.AppendElement(fb_->UDiv(lhs, rhs));
}

void GenIrNodesPass::HandleSDiv(FuzzSDivProto* sdiv) {
  sdiv->set_bit_width(BoundedWidth(sdiv->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(sdiv->mutable_lhs_idx(), sdiv->bit_width());
  BValue rhs =
      GetWidthFittedOperand(sdiv->mutable_rhs_idx(), sdiv->bit_width());
  context_list_.AppendElement(fb_->SDiv(lhs, rhs));
}

void GenIrNodesPass::HandleUMod(FuzzUModProto* umod) {
  umod->set_bit_width(BoundedWidth(umod->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(umod->mutable_lhs_idx(), umod->bit_width());
  BValue rhs =
      GetWidthFittedOperand(umod->mutable_rhs_idx(), umod->bit_width());
  context_list_.AppendElement(fb_->UMod(lhs, rhs));
}

void GenIrNodesPass::HandleSMod(FuzzSModProto* smod) {
  smod->set_bit_width(BoundedWidth(smod->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(smod->mutable_lhs_idx(), smod->bit_width());
  BValue rhs =
      GetWidthFittedOperand(smod->mutable_rhs_idx(), smod->bit_width());
  context_list_.AppendElement(fb_->SMod(lhs, rhs));
}

void GenIrNodesPass::HandleSubtract(FuzzSubtractProto* subtract) {
  subtract->set_bit_width(BoundedWidth(subtract->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(subtract->mutable_lhs_idx(), subtract->bit_width());
  BValue rhs =
      GetWidthFittedOperand(subtract->mutable_rhs_idx(), subtract->bit_width());
  context_list_.AppendElement(fb_->Subtract(lhs, rhs));
}

void GenIrNodesPass::HandleAdd(FuzzAddProto* add) {
  add->set_bit_width(BoundedWidth(add->bit_width()));
  BValue lhs = GetWidthFittedOperand(add->mutable_lhs_idx(), add->bit_width());
  BValue rhs = GetWidthFittedOperand(add->mutable_rhs_idx(), add->bit_width());
  context_list_.AppendElement(fb_->Add(lhs, rhs));
}

void GenIrNodesPass::HandleConcat(FuzzConcatProto* concat) {
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetOperands(concat->mutable_operand_idxs(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Concat(operands));
}

void GenIrNodesPass::HandleULe(FuzzULeProto* ule) {
  ule->set_bit_width(BoundedWidth(ule->bit_width()));
  BValue lhs = GetWidthFittedOperand(ule->mutable_lhs_idx(), ule->bit_width());
  BValue rhs = GetWidthFittedOperand(ule->mutable_rhs_idx(), ule->bit_width());
  context_list_.AppendElement(fb_->ULe(lhs, rhs));
}

void GenIrNodesPass::HandleULt(FuzzULtProto* ult) {
  ult->set_bit_width(BoundedWidth(ult->bit_width()));
  BValue lhs = GetWidthFittedOperand(ult->mutable_lhs_idx(), ult->bit_width());
  BValue rhs = GetWidthFittedOperand(ult->mutable_rhs_idx(), ult->bit_width());
  context_list_.AppendElement(fb_->ULt(lhs, rhs));
}

void GenIrNodesPass::HandleUGe(FuzzUGeProto* uge) {
  uge->set_bit_width(BoundedWidth(uge->bit_width()));
  BValue lhs = GetWidthFittedOperand(uge->mutable_lhs_idx(), uge->bit_width());
  BValue rhs = GetWidthFittedOperand(uge->mutable_rhs_idx(), uge->bit_width());
  context_list_.AppendElement(fb_->UGe(lhs, rhs));
}

void GenIrNodesPass::HandleUGt(FuzzUGtProto* ugt) {
  ugt->set_bit_width(BoundedWidth(ugt->bit_width()));
  BValue lhs = GetWidthFittedOperand(ugt->mutable_lhs_idx(), ugt->bit_width());
  BValue rhs = GetWidthFittedOperand(ugt->mutable_rhs_idx(), ugt->bit_width());
  context_list_.AppendElement(fb_->UGt(lhs, rhs));
}

void GenIrNodesPass::HandleSLe(FuzzSLeProto* sle) {
  sle->set_bit_width(BoundedWidth(sle->bit_width()));
  BValue lhs = GetWidthFittedOperand(sle->mutable_lhs_idx(), sle->bit_width());
  BValue rhs = GetWidthFittedOperand(sle->mutable_rhs_idx(), sle->bit_width());
  context_list_.AppendElement(fb_->SLe(lhs, rhs));
}

void GenIrNodesPass::HandleSLt(FuzzSLtProto* slt) {
  slt->set_bit_width(BoundedWidth(slt->bit_width()));
  BValue lhs = GetWidthFittedOperand(slt->mutable_lhs_idx(), slt->bit_width());
  BValue rhs = GetWidthFittedOperand(slt->mutable_rhs_idx(), slt->bit_width());
  context_list_.AppendElement(fb_->SLt(lhs, rhs));
}

void GenIrNodesPass::HandleSGe(FuzzSGeProto* sge) {
  sge->set_bit_width(BoundedWidth(sge->bit_width()));
  BValue lhs = GetWidthFittedOperand(sge->mutable_lhs_idx(), sge->bit_width());
  BValue rhs = GetWidthFittedOperand(sge->mutable_rhs_idx(), sge->bit_width());
  context_list_.AppendElement(fb_->SGe(lhs, rhs));
}

void GenIrNodesPass::HandleSGt(FuzzSGtProto* sgt) {
  sgt->set_bit_width(BoundedWidth(sgt->bit_width()));
  BValue lhs = GetWidthFittedOperand(sgt->mutable_lhs_idx(), sgt->bit_width());
  BValue rhs = GetWidthFittedOperand(sgt->mutable_rhs_idx(), sgt->bit_width());
  context_list_.AppendElement(fb_->SGt(lhs, rhs));
}

void GenIrNodesPass::HandleEq(FuzzEqProto* eq) {
  eq->set_bit_width(BoundedWidth(eq->bit_width()));
  BValue lhs = GetWidthFittedOperand(eq->mutable_lhs_idx(), eq->bit_width());
  BValue rhs = GetWidthFittedOperand(eq->mutable_rhs_idx(), eq->bit_width());
  context_list_.AppendElement(fb_->Eq(lhs, rhs));
}

void GenIrNodesPass::HandleNe(FuzzNeProto* ne) {
  ne->set_bit_width(BoundedWidth(ne->bit_width()));
  BValue lhs = GetWidthFittedOperand(ne->mutable_lhs_idx(), ne->bit_width());
  BValue rhs = GetWidthFittedOperand(ne->mutable_rhs_idx(), ne->bit_width());
  context_list_.AppendElement(fb_->Ne(lhs, rhs));
}

void GenIrNodesPass::HandleNegate(FuzzNegateProto* negate) {
  BValue operand = GetOperand(negate->operand_idx());
  context_list_.AppendElement(fb_->Negate(operand));
}

void GenIrNodesPass::HandleNot(FuzzNotProto* not_op) {
  BValue operand = GetOperand(not_op->operand_idx());
  context_list_.AppendElement(fb_->Not(operand));
}

void GenIrNodesPass::HandleLiteral(FuzzLiteralProto* literal) {
  literal->set_bit_width(BoundedWidth(literal->bit_width()));
  // Take the bytes protobuf datatype and convert it to a Bits object by making
  // a const uint8_t span. Any bytes that exceed the bit width of the literal
  // will be dropped.
  Bits value_bits =
      ChangeBytesBitWidth(literal->value_bytes(), literal->bit_width());
  context_list_.AppendElement(fb_->Literal(value_bits));
}

void GenIrNodesPass::HandleSelect(FuzzSelectProto* select) {
  select->set_bit_width(BoundedWidth(select->bit_width()));
  BValue selector = GetOperand(select->selector_idx());
  int64_t max_case_count = select->case_idxs_size();
  bool use_default_value = false;
  if (select->case_idxs_size() < selector.BitCountOrDie() ||
      select->case_idxs_size() < 1ULL << selector.BitCountOrDie()) {
    // If the number of cases is less than 2 ** selector_width, we must use a
    // default value otherwise there are not enough cases to cover all possible
    // selector values.
    use_default_value = true;
  } else if (select->case_idxs_size() > 1ULL << selector.BitCountOrDie()) {
    // If the number of cases is greater than 2 ** selector_width, we must
    // reduce the amount of cases to 2 ** selector_width.
    max_case_count = 1ULL << selector.BitCountOrDie();
  }
  // If the number of cases is equal to 2 ** selector_width, we cannot use a
  // default value because it is useless.
  // We need at least one case.
  std::vector<BValue> cases = GetWidthFittedOperands(
      select->mutable_case_idxs(), select->bit_width(), 1, max_case_count);
  if (use_default_value) {
    BValue default_value = GetWidthFittedOperand(
        select->mutable_default_value_idx(), select->bit_width());
    context_list_.AppendElement(fb_->Select(selector, cases, default_value));
  } else {
    context_list_.AppendElement(fb_->Select(selector, cases));
  }
}

void GenIrNodesPass::HandleOneHot(FuzzOneHotProto* one_hot) {
  BValue input = GetOperand(one_hot->input_idx());
  // Convert the LsbOrMsb proto enum to the C++ enum.
  LsbOrMsb priority;
  switch (one_hot->priority()) {
    case FuzzOneHotProto::MSB_PRIORITY:
      priority = LsbOrMsb::kMsb;
      break;
    case FuzzOneHotProto::LSB_PRIORITY:
    default:
      priority = LsbOrMsb::kLsb;
      break;
  }
  context_list_.AppendElement(fb_->OneHot(input, priority));
}

// Same as Select, but each selector_width bit corresponds to a case and there
// is no default value.
void GenIrNodesPass::HandleOneHotSelect(FuzzOneHotSelectProto* one_hot_select) {
  one_hot_select->set_bit_width(BoundedWidth(one_hot_select->bit_width()));
  BValue selector = GetOperand(one_hot_select->selector_idx());
  // Use a default value for the selector if there are no cases or if there are
  // less cases than the selector bit width.
  if (one_hot_select->case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = fb_->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > one_hot_select->case_idxs_size()) {
    selector = fb_->Literal(UBits(0, one_hot_select->case_idxs_size()));
  }
  // Use of GetWidthFittedOperands's min_operand_count and max_operand_count
  // arguments is to ensure that the number of cases is equal to the selector
  // bit width.
  std::vector<BValue> cases = GetWidthFittedOperands(
      one_hot_select->mutable_case_idxs(), one_hot_select->bit_width(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  context_list_.AppendElement(fb_->OneHotSelect(selector, cases));
}

// Same as OneHotSelect, but with a default value.
void GenIrNodesPass::HandlePrioritySelect(
    FuzzPrioritySelectProto* priority_select) {
  priority_select->set_bit_width(BoundedWidth(priority_select->bit_width()));
  BValue selector = GetOperand(priority_select->selector_idx());
  // Use a default value for the selector if there are no cases or if there are
  // less cases than the selector bit width.
  if (priority_select->case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = fb_->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > priority_select->case_idxs_size()) {
    selector = fb_->Literal(UBits(0, priority_select->case_idxs_size()));
  }
  // Use of GetWidthFittedOperands's min_operand_count and max_operand_count
  // arguments is to ensure that the number of cases is equal to the selector
  // bit width.
  std::vector<BValue> cases = GetWidthFittedOperands(
      priority_select->mutable_case_idxs(), priority_select->bit_width(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  BValue default_value =
      GetWidthFittedOperand(priority_select->mutable_default_value_idx(),
                            priority_select->bit_width());
  context_list_.AppendElement(
      fb_->PrioritySelect(selector, cases, default_value));
}

void GenIrNodesPass::HandleClz(FuzzClzProto* clz) {
  BValue operand = GetOperand(clz->operand_idx());
  context_list_.AppendElement(fb_->Clz(operand));
}

void GenIrNodesPass::HandleCtz(FuzzCtzProto* ctz) {
  BValue operand = GetOperand(ctz->operand_idx());
  context_list_.AppendElement(fb_->Ctz(operand));
}

// Retrieves a vector of Case objects based off of CaseProtos.
std::vector<FunctionBuilder::Case> GenIrNodesPass::GetCases(
    google::protobuf::RepeatedPtrField<CaseProto>* case_protos, int64_t bit_width) {
  std::vector<FunctionBuilder::Case> cases;
  for (CaseProto& case_proto : *case_protos) {
    cases.push_back(FunctionBuilder::Case{
        GetWidthFittedOperand(case_proto.mutable_clause_idx(), bit_width),
        GetWidthFittedOperand(case_proto.mutable_value_idx(), bit_width)});
  }
  // If there are no cases, add a default case, otherwise the interpreter
  // breaks.
  if (cases.empty()) {
    cases.push_back(FunctionBuilder::Case{fb_->Literal(UBits(0, bit_width)),
                                          fb_->Literal(UBits(0, bit_width))});
  }
  return cases;
}

void GenIrNodesPass::HandleMatch(FuzzMatchProto* match) {
  BValue condition = GetOperand(match->condition_idx());
  std::vector<FunctionBuilder::Case> cases =
      GetCases(match->mutable_case_protos(), condition.BitCountOrDie());
  BValue default_value = GetWidthFittedOperand(
      match->mutable_default_value_idx(), condition.BitCountOrDie());
  context_list_.AppendElement(fb_->Match(condition, cases, default_value));
}

void GenIrNodesPass::HandleMatchTrue(FuzzMatchTrueProto* match_true) {
  // MatchTrue only supports bit widths of 1.
  std::vector<FunctionBuilder::Case> cases =
      GetCases(match_true->mutable_case_protos(), 1);
  BValue default_value =
      GetWidthFittedOperand(match_true->mutable_default_value_idx(), 1);
  context_list_.AppendElement(fb_->MatchTrue(cases, default_value));
}

void GenIrNodesPass::HandleReverse(FuzzReverseProto* reverse) {
  BValue operand = GetOperand(reverse->operand_idx());
  context_list_.AppendElement(fb_->Reverse(operand));
}

void GenIrNodesPass::HandleIdentity(FuzzIdentityProto* identity) {
  BValue operand = GetOperand(identity->operand_idx());
  context_list_.AppendElement(fb_->Identity(operand));
}

void GenIrNodesPass::HandleSignExtend(FuzzSignExtendProto* sign_extend) {
  BValue operand = GetOperand(sign_extend->operand_idx());
  // The bit width cannot be less than the operand bit width because that is an
  // invalid extension.
  sign_extend->set_bit_width(
      BoundedWidth(sign_extend->bit_width(), operand.BitCountOrDie()));
  context_list_.AppendElement(
      fb_->SignExtend(operand, sign_extend->bit_width()));
}

void GenIrNodesPass::HandleZeroExtend(FuzzZeroExtendProto* zero_extend) {
  BValue operand = GetOperand(zero_extend->operand_idx());
  // The bit width cannot be less than the operand bit width because that is an
  // invalid extension.
  zero_extend->set_bit_width(
      BoundedWidth(zero_extend->bit_width(), operand.BitCountOrDie()));
  context_list_.AppendElement(
      fb_->ZeroExtend(operand, zero_extend->bit_width()));
}

void GenIrNodesPass::HandleBitSlice(FuzzBitSliceProto* bit_slice) {
  BValue operand = GetOperand(bit_slice->operand_idx());
  // The start value must be within the operand bit width and allow at least 1
  // bit to be sliced.
  bit_slice->set_start(Bounded(bit_slice->start(), /*left_bound=*/0,
                               operand.BitCountOrDie() - 1));
  // The bit width slice amount cannot exceed the operand bit width when
  // starting to slice from the start value.
  int64_t right_bound =
      std::max<int64_t>(1, operand.BitCountOrDie() - bit_slice->start());
  bit_slice->set_bit_width(
      Bounded(bit_slice->bit_width(), /*left_bound=*/1, right_bound));
  context_list_.AppendElement(
      fb_->BitSlice(operand, bit_slice->start(), bit_slice->bit_width()));
}

void GenIrNodesPass::HandleBitSliceUpdate(
    FuzzBitSliceUpdateProto* bit_slice_update) {
  BValue operand = GetOperand(bit_slice_update->operand_idx());
  BValue start = GetOperand(bit_slice_update->start_idx());
  BValue update_value = GetOperand(bit_slice_update->update_value_idx());
  context_list_.AppendElement(
      fb_->BitSliceUpdate(operand, start, update_value));
}

void GenIrNodesPass::HandleDynamicBitSlice(
    FuzzDynamicBitSliceProto* dynamic_bit_slice) {
  dynamic_bit_slice->set_bit_width(
      BoundedWidth(dynamic_bit_slice->bit_width()));
  BValue operand = GetWidthFittedOperand(
      dynamic_bit_slice->mutable_operand_idx(), dynamic_bit_slice->bit_width());
  BValue start = GetOperand(dynamic_bit_slice->start_idx());
  context_list_.AppendElement(
      fb_->DynamicBitSlice(operand, start, dynamic_bit_slice->bit_width()));
}

void GenIrNodesPass::HandleEncode(FuzzEncodeProto* encode) {
  BValue operand = GetOperand(encode->operand_idx());
  BValue encode_bvalue = fb_->Encode(operand);
  // Encode may result in a 0-bit value. If so, change the bit width to 1.
  if (encode_bvalue.BitCountOrDie() == 0) {
    encode_bvalue = ChangeBitWidth(fb_, encode_bvalue, 1);
  }
  context_list_.AppendElement(encode_bvalue);
}

void GenIrNodesPass::HandleDecode(FuzzDecodeProto* decode) {
  BValue operand = GetOperand(decode->operand_idx());
  // The decode bit width cannot exceed 2 ** operand_bit_width.
  int64_t right_bound = 1000;
  if (operand.BitCountOrDie() < 64) {
    right_bound = std::min<int64_t>(1000, 1ULL << operand.BitCountOrDie());
  }
  decode->set_bit_width(
      BoundedWidth(decode->bit_width(), /*left_bound=*/1, right_bound));
  context_list_.AppendElement(fb_->Decode(operand, decode->bit_width()));
}

void GenIrNodesPass::HandleGate(FuzzGateProto* gate) {
  // The Gate condition only supports bit widths of 1.
  BValue condition = GetWidthFittedOperand(gate->mutable_condition_idx(), 1);
  BValue data = GetOperand(gate->data_idx());
  context_list_.AppendElement(fb_->Gate(condition, data));
}

// Retrieves an operand from the list based off of a list index.
BValue GenIrNodesPass::GetOperand(int64_t stack_idx) {
  if (context_list_.IsEmpty()) {
    // If the list is empty, return a default value.
    return fb_->Literal(UBits(0, 64));
  } else {
    // Retrieve the operand from the list based off of the
    // randomly generated list index.
    return context_list_.GetElementAt(stack_idx);
  }
}

// Retrieves multiple operands from the list based off of list indices. If
// min_operand_count and max_operand_count are specified, the number of returned
// operands will be of the specified minimum or maximum constraint.
std::vector<BValue> GenIrNodesPass::GetOperands(
    google::protobuf::RepeatedField<int64_t>* operand_idxs, int64_t min_operand_count,
    int64_t max_operand_count) {
  // If the max operand count is less than the min operand count, assume invalid
  // or default inputs, so just ignore max_operand_count.
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  // Add operands up to the max operand count or up to the end of the vector.
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetOperand(operand_idxs->at(i)));
  }
  // Fill in any remaining operands with default values.
  while (operands.size() < min_operand_count) {
    operands.push_back(fb_->Literal(UBits(0, 64)));
  }
  return operands;
}

// Retrieves an operand from the stack based off of a FittedOperandIdxProto. The
// width of the operand will be changed to match the specified bit width.
BValue GenIrNodesPass::GetWidthFittedOperand(FittedOperandIdxProto* operand_idx,
                                             int64_t bit_width) {
  BValue operand = GetOperand(operand_idx->stack_idx());
  // Change the width using the specified increase/decrease methods.
  return ChangeBitWidth(fb_, operand, bit_width,
                        operand_idx->mutable_width_fitting_method());
}

// Retrieves multiple operands from the stack based off of
// FittedOperandIdxProtos. If min_operand_count and max_operand_count are
// specified, the number of returned operands will be of the specified minimum
// or maximum constraint.
std::vector<BValue> GenIrNodesPass::GetWidthFittedOperands(
    google::protobuf::RepeatedPtrField<FittedOperandIdxProto>* operand_idxs,
    int64_t bit_width, int64_t min_operand_count, int64_t max_operand_count) {
  // If the max operand count is less than the min operand count, assume invalid
  // or default inputs, so just ignore max_operand_count.
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  // Add operands up to the max operand count or up to the end of the vector.
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetWidthFittedOperand(&operand_idxs->at(i), bit_width));
  }
  // Fill in any remaining operands with default values of the same bit width.
  while (operands.size() < min_operand_count) {
    operands.push_back(fb_->Literal(UBits(0, bit_width)));
  }
  return operands;
}

}  // namespace xls

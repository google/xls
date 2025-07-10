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
  // Params are named as "p" followed by the list index of the param.
  std::string name = "p" + std::to_string(context_list_.GetListSize());
  Type* type = ConvertTypeProtoToType(p_, param->mutable_type());
  context_list_.AppendElement(fb_->Param(name, type));
}

void GenIrNodesPass::HandleShra(FuzzShraProto* shra) {
  BValue operand = GetBitsOperand(shra->mutable_operand_idx());
  BValue amount = GetBitsOperand(shra->mutable_amount_idx());
  context_list_.AppendElement(fb_->Shra(operand, amount));
}

void GenIrNodesPass::HandleShrl(FuzzShrlProto* shrl) {
  BValue operand = GetBitsOperand(shrl->mutable_operand_idx());
  BValue amount = GetBitsOperand(shrl->mutable_amount_idx());
  context_list_.AppendElement(fb_->Shrl(operand, amount));
}

void GenIrNodesPass::HandleShll(FuzzShllProto* shll) {
  BValue operand = GetBitsOperand(shll->mutable_operand_idx());
  BValue amount = GetBitsOperand(shll->mutable_amount_idx());
  context_list_.AppendElement(fb_->Shll(operand, amount));
}

void GenIrNodesPass::HandleOr(FuzzOrProto* or_op) {
  // Requires at least one operand.
  std::vector<BValue> operands = GetCoercedBitsOperands(
      or_op->mutable_operand_idxs(), or_op->mutable_operands_type(),
      /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Or(operands));
}

void GenIrNodesPass::HandleNor(FuzzNorProto* nor) {
  std::vector<BValue> operands = GetCoercedBitsOperands(
      nor->mutable_operand_idxs(), nor->mutable_operands_type(),
      /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Nor(operands));
}

void GenIrNodesPass::HandleXor(FuzzXorProto* xor_op) {
  std::vector<BValue> operands = GetCoercedBitsOperands(
      xor_op->mutable_operand_idxs(), xor_op->mutable_operands_type(),
      /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Xor(operands));
}

void GenIrNodesPass::HandleAnd(FuzzAndProto* and_op) {
  std::vector<BValue> operands = GetCoercedBitsOperands(
      and_op->mutable_operand_idxs(), and_op->mutable_operands_type(),
      /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->And(operands));
}

void GenIrNodesPass::HandleNand(FuzzNandProto* nand) {
  std::vector<BValue> operands = GetCoercedBitsOperands(
      nand->mutable_operand_idxs(), nand->mutable_operands_type(),
      /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Nand(operands));
}

void GenIrNodesPass::HandleAndReduce(FuzzAndReduceProto* and_reduce) {
  BValue operand = GetBitsOperand(and_reduce->mutable_operand_idx());
  context_list_.AppendElement(fb_->AndReduce(operand));
}

void GenIrNodesPass::HandleOrReduce(FuzzOrReduceProto* or_reduce) {
  BValue operand = GetBitsOperand(or_reduce->mutable_operand_idx());
  context_list_.AppendElement(fb_->OrReduce(operand));
}

void GenIrNodesPass::HandleXorReduce(FuzzXorReduceProto* xor_reduce) {
  BValue operand = GetBitsOperand(xor_reduce->mutable_operand_idx());
  context_list_.AppendElement(fb_->XorReduce(operand));
}

void GenIrNodesPass::HandleUMul(FuzzUMulProto* umul) {
  // If the bit width is set, use it. Otherwise, don't use it and coerce the
  // operands to be of the same type.
  if (umul->has_bit_width()) {
    BValue lhs = GetBitsOperand(umul->mutable_lhs_idx());
    BValue rhs = GetBitsOperand(umul->mutable_rhs_idx());
    int64_t right_bound =
        std::max<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        BoundedWidth(umul->bit_width(), /*left_bound=*/1, right_bound);
    context_list_.AppendElement(fb_->UMul(lhs, rhs, bit_width));
  } else {
    BValue lhs = GetCoercedBitsOperand(umul->mutable_lhs_idx(),
                                       umul->mutable_operands_type());
    BValue rhs = GetCoercedBitsOperand(umul->mutable_rhs_idx(),
                                       umul->mutable_operands_type());
    context_list_.AppendElement(fb_->UMul(lhs, rhs));
  }
}

void GenIrNodesPass::HandleSMul(FuzzSMulProto* smul) {
  if (smul->has_bit_width()) {
    BValue lhs = GetBitsOperand(smul->mutable_lhs_idx());
    BValue rhs = GetBitsOperand(smul->mutable_rhs_idx());
    int64_t right_bound =
        std::max<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        BoundedWidth(smul->bit_width(), /*left_bound=*/1, right_bound);
    context_list_.AppendElement(fb_->SMul(lhs, rhs, bit_width));
  } else {
    BValue lhs = GetCoercedBitsOperand(smul->mutable_lhs_idx(),
                                       smul->mutable_operands_type());
    BValue rhs = GetCoercedBitsOperand(smul->mutable_rhs_idx(),
                                       smul->mutable_operands_type());
    context_list_.AppendElement(fb_->SMul(lhs, rhs));
  }
}

void GenIrNodesPass::HandleUDiv(FuzzUDivProto* udiv) {
  BValue lhs = GetCoercedBitsOperand(udiv->mutable_lhs_idx(),
                                     udiv->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(udiv->mutable_rhs_idx(),
                                     udiv->mutable_operands_type());
  context_list_.AppendElement(fb_->UDiv(lhs, rhs));
}

void GenIrNodesPass::HandleSDiv(FuzzSDivProto* sdiv) {
  BValue lhs = GetCoercedBitsOperand(sdiv->mutable_lhs_idx(),
                                     sdiv->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(sdiv->mutable_rhs_idx(),
                                     sdiv->mutable_operands_type());
  context_list_.AppendElement(fb_->SDiv(lhs, rhs));
}

void GenIrNodesPass::HandleUMod(FuzzUModProto* umod) {
  BValue lhs = GetCoercedBitsOperand(umod->mutable_lhs_idx(),
                                     umod->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(umod->mutable_rhs_idx(),
                                     umod->mutable_operands_type());
  context_list_.AppendElement(fb_->UMod(lhs, rhs));
}

void GenIrNodesPass::HandleSMod(FuzzSModProto* smod) {
  BValue lhs = GetCoercedBitsOperand(smod->mutable_lhs_idx(),
                                     smod->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(smod->mutable_rhs_idx(),
                                     smod->mutable_operands_type());
  context_list_.AppendElement(fb_->SMod(lhs, rhs));
}

void GenIrNodesPass::HandleSubtract(FuzzSubtractProto* subtract) {
  BValue lhs = GetCoercedBitsOperand(subtract->mutable_lhs_idx(),
                                     subtract->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(subtract->mutable_rhs_idx(),
                                     subtract->mutable_operands_type());
  context_list_.AppendElement(fb_->Subtract(lhs, rhs));
}

void GenIrNodesPass::HandleAdd(FuzzAddProto* add) {
  BValue lhs = GetCoercedBitsOperand(add->mutable_lhs_idx(),
                                     add->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(add->mutable_rhs_idx(),
                                     add->mutable_operands_type());
  context_list_.AppendElement(fb_->Add(lhs, rhs));
}

void GenIrNodesPass::HandleConcat(FuzzConcatProto* concat) {
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetBitsOperands(concat->mutable_operand_idxs(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Concat(operands));
}

void GenIrNodesPass::HandleULe(FuzzULeProto* ule) {
  BValue lhs = GetCoercedBitsOperand(ule->mutable_lhs_idx(),
                                     ule->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(ule->mutable_rhs_idx(),
                                     ule->mutable_operands_type());
  context_list_.AppendElement(fb_->ULe(lhs, rhs));
}

void GenIrNodesPass::HandleULt(FuzzULtProto* ult) {
  BValue lhs = GetCoercedBitsOperand(ult->mutable_lhs_idx(),
                                     ult->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(ult->mutable_rhs_idx(),
                                     ult->mutable_operands_type());
  context_list_.AppendElement(fb_->ULt(lhs, rhs));
}

void GenIrNodesPass::HandleUGe(FuzzUGeProto* uge) {
  BValue lhs = GetCoercedBitsOperand(uge->mutable_lhs_idx(),
                                     uge->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(uge->mutable_rhs_idx(),
                                     uge->mutable_operands_type());
  context_list_.AppendElement(fb_->UGe(lhs, rhs));
}

void GenIrNodesPass::HandleUGt(FuzzUGtProto* ugt) {
  BValue lhs = GetCoercedBitsOperand(ugt->mutable_lhs_idx(),
                                     ugt->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(ugt->mutable_rhs_idx(),
                                     ugt->mutable_operands_type());
  context_list_.AppendElement(fb_->UGt(lhs, rhs));
}

void GenIrNodesPass::HandleSLe(FuzzSLeProto* sle) {
  BValue lhs = GetCoercedBitsOperand(sle->mutable_lhs_idx(),
                                     sle->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(sle->mutable_rhs_idx(),
                                     sle->mutable_operands_type());
  context_list_.AppendElement(fb_->SLe(lhs, rhs));
}

void GenIrNodesPass::HandleSLt(FuzzSLtProto* slt) {
  BValue lhs = GetCoercedBitsOperand(slt->mutable_lhs_idx(),
                                     slt->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(slt->mutable_rhs_idx(),
                                     slt->mutable_operands_type());
  context_list_.AppendElement(fb_->SLt(lhs, rhs));
}

void GenIrNodesPass::HandleSGe(FuzzSGeProto* sge) {
  BValue lhs = GetCoercedBitsOperand(sge->mutable_lhs_idx(),
                                     sge->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(sge->mutable_rhs_idx(),
                                     sge->mutable_operands_type());
  context_list_.AppendElement(fb_->SGe(lhs, rhs));
}

void GenIrNodesPass::HandleSGt(FuzzSGtProto* sgt) {
  BValue lhs = GetCoercedBitsOperand(sgt->mutable_lhs_idx(),
                                     sgt->mutable_operands_type());
  BValue rhs = GetCoercedBitsOperand(sgt->mutable_rhs_idx(),
                                     sgt->mutable_operands_type());
  context_list_.AppendElement(fb_->SGt(lhs, rhs));
}

void GenIrNodesPass::HandleEq(FuzzEqProto* eq) {
  BValue lhs =
      GetCoercedBitsOperand(eq->mutable_lhs_idx(), eq->mutable_operands_type());
  BValue rhs =
      GetCoercedBitsOperand(eq->mutable_rhs_idx(), eq->mutable_operands_type());
  context_list_.AppendElement(fb_->Eq(lhs, rhs));
}

void GenIrNodesPass::HandleNe(FuzzNeProto* ne) {
  BValue lhs =
      GetCoercedBitsOperand(ne->mutable_lhs_idx(), ne->mutable_operands_type());
  BValue rhs =
      GetCoercedBitsOperand(ne->mutable_rhs_idx(), ne->mutable_operands_type());
  context_list_.AppendElement(fb_->Ne(lhs, rhs));
}

void GenIrNodesPass::HandleNegate(FuzzNegateProto* negate) {
  BValue operand = GetBitsOperand(negate->mutable_operand_idx());
  context_list_.AppendElement(fb_->Negate(operand));
}

void GenIrNodesPass::HandleNot(FuzzNotProto* not_op) {
  BValue operand = GetBitsOperand(not_op->mutable_operand_idx());
  context_list_.AppendElement(fb_->Not(operand));
}

BValue GenIrNodesPass::GetValueFromValueTypeProto(ValueTypeProto* value_type) {
  switch (value_type->type_case()) {
    case ValueTypeProto::kBits: {
      // Take the bytes protobuf datatype and convert it to a Bits object by
      // making a const uint8_t span. Any bytes that exceed the bit width of the
      // literal will be dropped.
      auto bits_type = value_type->mutable_bits();
      int64_t bit_width = BoundedWidth(bits_type->bit_width());
      Bits value_bits =
          ChangeBytesBitWidth(bits_type->value_bytes(), bit_width);
      return fb_->Literal(value_bits);
    }
    case ValueTypeProto::kTuple: {
      auto tuple_type = value_type->mutable_tuple();
      std::vector<BValue> elements;
      for (auto& element_value_type : *tuple_type->mutable_tuple_elements()) {
        elements.push_back(GetValueFromValueTypeProto(&element_value_type));
      }
      return fb_->Tuple(elements);
    }
    case ValueTypeProto::kArray: {
      auto array_type = value_type->mutable_array();
      std::vector<BValue> elements;
      BValue element =
          GetValueFromValueTypeProto(array_type->mutable_array_element());
      for (int64_t i = 0; i < array_type->array_size(); i += 1) {
        elements.push_back(element);
      }
      Type* element_type = element.GetType();
      if (array_type->array_size() == 0) {
        element_type =
            ConvertTypeProtoToType(p_, array_type->mutable_array_element());
      }
      return fb_->Array(elements, element_type);
    }
    default:
      return DefaultValue(p_, fb_);
  }
}

void GenIrNodesPass::HandleLiteral(FuzzLiteralProto* literal) {
  context_list_.AppendElement(
      GetValueFromValueTypeProto(literal->mutable_value_type()));
}

void GenIrNodesPass::HandleSelect(FuzzSelectProto* select) {
  BValue selector = GetBitsOperand(select->mutable_selector_idx());
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
  auto cases_and_default_type = select->mutable_cases_and_default_type();
  std::vector<BValue> cases =
      GetCoercedOperands(select->mutable_case_idxs(), cases_and_default_type,
                         /*min_operand_count=*/1, max_case_count);
  if (use_default_value) {
    BValue default_value = GetCoercedOperand(
        select->mutable_default_value_idx(), cases_and_default_type);
    context_list_.AppendElement(fb_->Select(selector, cases, default_value));
  } else {
    context_list_.AppendElement(fb_->Select(selector, cases));
  }
}

void GenIrNodesPass::HandleOneHot(FuzzOneHotProto* one_hot) {
  BValue input = GetBitsOperand(one_hot->mutable_input_idx());
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
  BValue selector = GetBitsOperand(one_hot_select->mutable_selector_idx());
  // Use a default value for the selector if there are no cases or if there are
  // less cases than the selector bit width.
  if (one_hot_select->case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = fb_->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > one_hot_select->case_idxs_size()) {
    selector = fb_->Literal(UBits(0, one_hot_select->case_idxs_size()));
  }
  // Use of GetCoercedOperands's min_operand_count and max_operand_count
  // arguments to ensure that the number of cases is equal to the selector
  // bit width.
  std::vector<BValue> cases = GetCoercedOperands(
      one_hot_select->mutable_case_idxs(), one_hot_select->mutable_cases_type(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  context_list_.AppendElement(fb_->OneHotSelect(selector, cases));
}

// Same as OneHotSelect, but with a default value.
void GenIrNodesPass::HandlePrioritySelect(
    FuzzPrioritySelectProto* priority_select) {
  BValue selector = GetBitsOperand(priority_select->mutable_selector_idx());
  if (priority_select->case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = fb_->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > priority_select->case_idxs_size()) {
    selector = fb_->Literal(UBits(0, priority_select->case_idxs_size()));
  }
  std::vector<BValue> cases =
      GetCoercedOperands(priority_select->mutable_case_idxs(),
                         priority_select->mutable_cases_and_default_type(),
                         selector.BitCountOrDie(), selector.BitCountOrDie());
  BValue default_value =
      GetCoercedOperand(priority_select->mutable_default_value_idx(),
                        priority_select->mutable_cases_and_default_type());
  context_list_.AppendElement(
      fb_->PrioritySelect(selector, cases, default_value));
}

void GenIrNodesPass::HandleClz(FuzzClzProto* clz) {
  BValue operand = GetBitsOperand(clz->mutable_operand_idx());
  context_list_.AppendElement(fb_->Clz(operand));
}

void GenIrNodesPass::HandleCtz(FuzzCtzProto* ctz) {
  BValue operand = GetBitsOperand(ctz->mutable_operand_idx());
  context_list_.AppendElement(fb_->Ctz(operand));
}

void GenIrNodesPass::HandleMatch(FuzzMatchProto* match) {
  BValue condition = GetCoercedBitsOperand(match->mutable_condition_idx(),
                                           match->mutable_operands_type());
  // Retrieves a vector of Case objects based off of CaseProtos.
  std::vector<FunctionBuilder::Case> cases;
  for (CaseProto& case_proto : *match->mutable_case_protos()) {
    cases.push_back(FunctionBuilder::Case{
        GetCoercedBitsOperand(case_proto.mutable_clause_idx(),
                              match->mutable_operands_type()),
        GetCoercedBitsOperand(case_proto.mutable_value_idx(),
                              match->mutable_operands_type())});
  }
  // If there are no cases, add a default case.
  if (cases.empty()) {
    BValue default_case_value =
        DefaultFromBitsTypeProto(p_, fb_, match->mutable_operands_type());
    cases.push_back(
        FunctionBuilder::Case{default_case_value, default_case_value});
  }
  BValue default_value = GetCoercedBitsOperand(
      match->mutable_default_value_idx(), match->mutable_operands_type());
  context_list_.AppendElement(fb_->Match(condition, cases, default_value));
}

void GenIrNodesPass::HandleMatchTrue(FuzzMatchTrueProto* match_true) {
  // MatchTrue only supports bit widths of 1.
  std::vector<FunctionBuilder::Case> cases;
  for (CaseProto& case_proto : *match_true->mutable_case_protos()) {
    cases.push_back(FunctionBuilder::Case{
        GetBitsFittedOperand(case_proto.mutable_clause_idx(), /*bit_width=*/1,
                             match_true->mutable_operands_coercion_method()),
        GetBitsFittedOperand(case_proto.mutable_clause_idx(), /*bit_width=*/1,
                             match_true->mutable_operands_coercion_method())});
  }
  if (cases.empty()) {
    BValue default_case_value = fb_->Literal(UBits(0, 1));
    cases.push_back(
        FunctionBuilder::Case{default_case_value, default_case_value});
  }
  BValue default_value = GetBitsFittedOperand(
      match_true->mutable_default_value_idx(), /*bit_width=*/1,
      match_true->mutable_operands_coercion_method());
  context_list_.AppendElement(fb_->MatchTrue(cases, default_value));
}

void GenIrNodesPass::HandleReverse(FuzzReverseProto* reverse) {
  BValue operand = GetBitsOperand(reverse->mutable_operand_idx());
  context_list_.AppendElement(fb_->Reverse(operand));
}

void GenIrNodesPass::HandleIdentity(FuzzIdentityProto* identity) {
  BValue operand = GetOperand(identity->mutable_operand_idx());
  context_list_.AppendElement(fb_->Identity(operand));
}

void GenIrNodesPass::HandleSignExtend(FuzzSignExtendProto* sign_extend) {
  BValue operand = GetBitsOperand(sign_extend->mutable_operand_idx());
  // The bit width cannot be less than the operand bit width because that is an
  // invalid extension.
  int64_t bit_width =
      BoundedWidth(sign_extend->bit_width(), operand.BitCountOrDie());
  context_list_.AppendElement(fb_->SignExtend(operand, bit_width));
}

void GenIrNodesPass::HandleZeroExtend(FuzzZeroExtendProto* zero_extend) {
  BValue operand = GetBitsOperand(zero_extend->mutable_operand_idx());
  // The bit width cannot be less than the operand bit width because that is an
  // invalid extension.
  int64_t bit_width =
      BoundedWidth(zero_extend->bit_width(), operand.BitCountOrDie());
  context_list_.AppendElement(fb_->ZeroExtend(operand, bit_width));
}

void GenIrNodesPass::HandleBitSlice(FuzzBitSliceProto* bit_slice) {
  BValue operand = GetBitsOperand(bit_slice->mutable_operand_idx());
  // The start value must be in the range [0, operand_width - bit_width]
  // otherwise you would be slicing past the end of the operand.
  int64_t bit_width = BoundedWidth(bit_slice->bit_width());
  int64_t start = Bounded(bit_slice->start(), /*left_bound=*/0,
                          operand.BitCountOrDie() - bit_width);
  context_list_.AppendElement(fb_->BitSlice(operand, start, bit_width));
}

void GenIrNodesPass::HandleBitSliceUpdate(
    FuzzBitSliceUpdateProto* bit_slice_update) {
  BValue operand = GetBitsOperand(bit_slice_update->mutable_operand_idx());
  BValue start = GetBitsOperand(bit_slice_update->mutable_start_idx());
  BValue update_value =
      GetBitsOperand(bit_slice_update->mutable_update_value_idx());
  context_list_.AppendElement(
      fb_->BitSliceUpdate(operand, start, update_value));
}

void GenIrNodesPass::HandleDynamicBitSlice(
    FuzzDynamicBitSliceProto* dynamic_bit_slice) {
  int64_t bit_width = BoundedWidth(dynamic_bit_slice->bit_width());
  BValue operand = GetBitsOperand(dynamic_bit_slice->mutable_operand_idx());
  // The operand must be of the same or greater bit width than the dynamic bit
  // slice bit width.
  if (operand.BitCountOrDie() < bit_width) {
    auto operand_coercion_method =
        dynamic_bit_slice->mutable_operand_coercion_method();
    operand = ChangeBitWidth(
        fb_, operand, bit_width,
        operand_coercion_method->mutable_change_bit_width_method());
  }
  BValue start = GetBitsOperand(dynamic_bit_slice->mutable_start_idx());
  context_list_.AppendElement(fb_->DynamicBitSlice(operand, start, bit_width));
}

void GenIrNodesPass::HandleEncode(FuzzEncodeProto* encode) {
  BValue operand = GetBitsOperand(encode->mutable_operand_idx());
  BValue encode_bvalue = fb_->Encode(operand);
  // Encode may result in a 0-bit value. If so, change the bit width to 1.
  if (encode_bvalue.BitCountOrDie() == 0) {
    encode_bvalue = ChangeBitWidth(fb_, encode_bvalue, /*new_bit_width=*/1);
  }
  context_list_.AppendElement(encode_bvalue);
}

void GenIrNodesPass::HandleDecode(FuzzDecodeProto* decode) {
  BValue operand = GetBitsOperand(decode->mutable_operand_idx());
  int64_t bit_width = BoundedWidth(decode->bit_width());
  context_list_.AppendElement(fb_->Decode(operand, bit_width));
}

void GenIrNodesPass::HandleGate(FuzzGateProto* gate) {
  // The Gate condition only supports bit widths of 1.
  BValue condition =
      GetBitsFittedOperand(gate->mutable_condition_idx(), /*bit_width=*/1,
                           gate->mutable_condition_coercion_method());
  BValue data = GetBitsOperand(gate->mutable_data_idx());
  context_list_.AppendElement(fb_->Gate(condition, data));
}

// Retrieves an operand from the list based off of a list index.
BValue GenIrNodesPass::GetOperand(OperandIdxProto* operand_idx) {
  if (context_list_.IsEmpty()) {
    // If the list is empty, return a default value.
    return DefaultValue(p_, fb_);
  } else {
    // Retrieve the operand from the list based off of the
    // randomly generated list index.
    return context_list_.GetElementAt(operand_idx->list_idx());
  }
}

BValue GenIrNodesPass::GetBitsOperand(BitsOperandIdxProto* operand_idx) {
  if (context_list_.IsEmpty(TypeCase::BITS)) {
    return DefaultValue(p_, fb_, TypeCase::BITS);
  } else {
    return context_list_.GetElementAt(operand_idx->list_idx(), TypeCase::BITS);
  }
}

BValue GenIrNodesPass::GetCoercedOperand(OperandIdxProto* operand_idx,
                                         CoercedTypeProto* coerced_type) {
  BValue operand = GetOperand(operand_idx);
  Type* type = ConvertTypeProtoToType(p_, coerced_type);
  return Coerced(p_, fb_, operand, coerced_type, type);
}

BValue GenIrNodesPass::GetCoercedBitsOperand(
    BitsOperandIdxProto* operand_idx, BitsCoercedTypeProto* coerced_type) {
  BValue operand = GetBitsOperand(operand_idx);
  Type* type = ConvertBitsTypeProtoToType(p_, coerced_type);
  return CoercedBits(p_, fb_, operand, coerced_type, type);
}

BValue GenIrNodesPass::GetBitsFittedOperand(
    BitsOperandIdxProto* operand_idx, int64_t bit_width,
    BitsCoercionMethodProto* coercion_method) {
  BValue operand = GetBitsOperand(operand_idx);
  return ChangeBitWidth(fb_, operand, bit_width,
                        coercion_method->mutable_change_bit_width_method());
}

// Retrieves multiple operands from the list based off of list indices. If
// min_operand_count and max_operand_count are specified, the number of returned
// operands will be of the specified minimum or maximum constraint.
std::vector<BValue> GenIrNodesPass::GetBitsOperands(
    google::protobuf::RepeatedPtrField<BitsOperandIdxProto>* operand_idxs,
    int64_t min_operand_count, int64_t max_operand_count) {
  // If the max operand count is less than the min operand count, assume invalid
  // or default inputs, so just ignore max_operand_count.
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  // Add operands up to the max operand count or up to the end of the vector.
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetBitsOperand(&operand_idxs->at(i)));
  }
  // Fill in any remaining operands with default values.
  BValue default_value = DefaultValue(p_, fb_, TypeCase::BITS);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

std::vector<BValue> GenIrNodesPass::GetCoercedOperands(
    google::protobuf::RepeatedPtrField<OperandIdxProto>* operand_idxs,
    CoercedTypeProto* coerced_type, int64_t min_operand_count,
    int64_t max_operand_count) {
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetCoercedOperand(&operand_idxs->at(i), coerced_type));
  }
  BValue default_value = DefaultFromTypeProto(p_, fb_, coerced_type);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

std::vector<BValue> GenIrNodesPass::GetCoercedBitsOperands(
    google::protobuf::RepeatedPtrField<BitsOperandIdxProto>* operand_idxs,
    BitsCoercedTypeProto* coerced_type, int64_t min_operand_count,
    int64_t max_operand_count) {
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(
        GetCoercedBitsOperand(&operand_idxs->at(i), coerced_type));
  }
  BValue default_value = DefaultFromBitsTypeProto(p_, fb_, coerced_type);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

}  // namespace xls

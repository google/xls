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

#include "absl/strings/str_format.h"
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
// instantiate/generate an IR node/BValue. Add these BValues to the context
// list. Some FuzzOpProtos may require retrieving previous BValues from the
// context list.
void GenIrNodesPass::GenIrNodes() {
  for (const FuzzOpProto& fuzz_op : fuzz_program_.fuzz_ops()) {
    VisitFuzzOp(fuzz_op);
  }
}

void GenIrNodesPass::HandleParam(const FuzzParamProto& param) {
  // Params are named as "p" followed by the combined context list index of the
  // param.
  std::string name = absl::StrFormat("p%d", context_list_.GetListSize());
  // Retrieve the Type object from the FuzzTypeProto.
  Type* type = ConvertTypeProtoToType(p_, param.type());
  // Append the param BValue to the combined context list and the context list
  // for its type.
  context_list_.AppendElement(fb_->Param(name, type));
}

void GenIrNodesPass::HandleShra(const FuzzShraProto& shra) {
  // Retrieve a bits operand from the bits context list based on the list_idx.
  BValue operand = GetBitsOperand(shra.operand_idx());
  BValue amount = GetBitsOperand(shra.amount_idx());
  context_list_.AppendElement(fb_->Shra(operand, amount));
}

void GenIrNodesPass::HandleShrl(const FuzzShrlProto& shrl) {
  BValue operand = GetBitsOperand(shrl.operand_idx());
  BValue amount = GetBitsOperand(shrl.amount_idx());
  context_list_.AppendElement(fb_->Shrl(operand, amount));
}

void GenIrNodesPass::HandleShll(const FuzzShllProto& shll) {
  BValue operand = GetBitsOperand(shll.operand_idx());
  BValue amount = GetBitsOperand(shll.amount_idx());
  context_list_.AppendElement(fb_->Shll(operand, amount));
}

void GenIrNodesPass::HandleOr(const FuzzOrProto& or_op) {
  // Requires at least one bits operand of a specific bit width, which is
  // defined by the operands_type.
  std::vector<BValue> operands =
      GetCoercedBitsOperands(or_op.operand_idxs(), or_op.operands_type(),
                             /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Or(operands));
}

void GenIrNodesPass::HandleNor(const FuzzNorProto& nor) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(nor.operand_idxs(), nor.operands_type(),
                             /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Nor(operands));
}

void GenIrNodesPass::HandleXor(const FuzzXorProto& xor_op) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(xor_op.operand_idxs(), xor_op.operands_type(),
                             /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Xor(operands));
}

void GenIrNodesPass::HandleAnd(const FuzzAndProto& and_op) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(and_op.operand_idxs(), and_op.operands_type(),
                             /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->And(operands));
}

void GenIrNodesPass::HandleNand(const FuzzNandProto& nand) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(nand.operand_idxs(), nand.operands_type(),
                             /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Nand(operands));
}

void GenIrNodesPass::HandleAndReduce(const FuzzAndReduceProto& and_reduce) {
  BValue operand = GetBitsOperand(and_reduce.operand_idx());
  context_list_.AppendElement(fb_->AndReduce(operand));
}

void GenIrNodesPass::HandleOrReduce(const FuzzOrReduceProto& or_reduce) {
  BValue operand = GetBitsOperand(or_reduce.operand_idx());
  context_list_.AppendElement(fb_->OrReduce(operand));
}

void GenIrNodesPass::HandleXorReduce(const FuzzXorReduceProto& xor_reduce) {
  BValue operand = GetBitsOperand(xor_reduce.operand_idx());
  context_list_.AppendElement(fb_->XorReduce(operand));
}

void GenIrNodesPass::HandleUMul(const FuzzUMulProto& umul) {
  if (umul.has_bit_width()) {
    // If the bit width is set, use it.
    BValue lhs = GetBitsOperand(umul.lhs_idx());
    BValue rhs = GetBitsOperand(umul.rhs_idx());
    // The bit width cannot exceed the sum of the bit widths of the operands.
    int64_t right_bound =
        std::min<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        BoundedWidth(umul.bit_width(), /*left_bound=*/1, right_bound);
    context_list_.AppendElement(fb_->UMul(lhs, rhs, bit_width));
  } else {
    // If the bit width is not set, don't use it and coerce the operands to be
    // of the same type.
    BValue lhs = GetCoercedBitsOperand(umul.lhs_idx(), umul.operands_type());
    BValue rhs = GetCoercedBitsOperand(umul.rhs_idx(), umul.operands_type());
    context_list_.AppendElement(fb_->UMul(lhs, rhs));
  }
}

// Same as UMul.
void GenIrNodesPass::HandleSMul(const FuzzSMulProto& smul) {
  if (smul.has_bit_width()) {
    BValue lhs = GetBitsOperand(smul.lhs_idx());
    BValue rhs = GetBitsOperand(smul.rhs_idx());
    int64_t right_bound =
        std::min<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        BoundedWidth(smul.bit_width(), /*left_bound=*/1, right_bound);
    context_list_.AppendElement(fb_->SMul(lhs, rhs, bit_width));
  } else {
    BValue lhs = GetCoercedBitsOperand(smul.lhs_idx(), smul.operands_type());
    BValue rhs = GetCoercedBitsOperand(smul.rhs_idx(), smul.operands_type());
    context_list_.AppendElement(fb_->SMul(lhs, rhs));
  }
}

void GenIrNodesPass::HandleUDiv(const FuzzUDivProto& udiv) {
  BValue lhs = GetCoercedBitsOperand(udiv.lhs_idx(), udiv.operands_type());
  BValue rhs = GetCoercedBitsOperand(udiv.rhs_idx(), udiv.operands_type());
  context_list_.AppendElement(fb_->UDiv(lhs, rhs));
}

void GenIrNodesPass::HandleSDiv(const FuzzSDivProto& sdiv) {
  BValue lhs = GetCoercedBitsOperand(sdiv.lhs_idx(), sdiv.operands_type());
  BValue rhs = GetCoercedBitsOperand(sdiv.rhs_idx(), sdiv.operands_type());
  context_list_.AppendElement(fb_->SDiv(lhs, rhs));
}

void GenIrNodesPass::HandleUMod(const FuzzUModProto& umod) {
  BValue lhs = GetCoercedBitsOperand(umod.lhs_idx(), umod.operands_type());
  BValue rhs = GetCoercedBitsOperand(umod.rhs_idx(), umod.operands_type());
  context_list_.AppendElement(fb_->UMod(lhs, rhs));
}

void GenIrNodesPass::HandleSMod(const FuzzSModProto& smod) {
  BValue lhs = GetCoercedBitsOperand(smod.lhs_idx(), smod.operands_type());
  BValue rhs = GetCoercedBitsOperand(smod.rhs_idx(), smod.operands_type());
  context_list_.AppendElement(fb_->SMod(lhs, rhs));
}

void GenIrNodesPass::HandleSubtract(const FuzzSubtractProto& subtract) {
  BValue lhs =
      GetCoercedBitsOperand(subtract.lhs_idx(), subtract.operands_type());
  BValue rhs =
      GetCoercedBitsOperand(subtract.rhs_idx(), subtract.operands_type());
  context_list_.AppendElement(fb_->Subtract(lhs, rhs));
}

void GenIrNodesPass::HandleAdd(const FuzzAddProto& add) {
  BValue lhs = GetCoercedBitsOperand(add.lhs_idx(), add.operands_type());
  BValue rhs = GetCoercedBitsOperand(add.rhs_idx(), add.operands_type());
  context_list_.AppendElement(fb_->Add(lhs, rhs));
}

void GenIrNodesPass::HandleConcat(const FuzzConcatProto& concat) {
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetBitsOperands(concat.operand_idxs(), /*min_operand_count=*/1);
  context_list_.AppendElement(fb_->Concat(operands));
}

void GenIrNodesPass::HandleULe(const FuzzULeProto& ule) {
  BValue lhs = GetCoercedBitsOperand(ule.lhs_idx(), ule.operands_type());
  BValue rhs = GetCoercedBitsOperand(ule.rhs_idx(), ule.operands_type());
  context_list_.AppendElement(fb_->ULe(lhs, rhs));
}

void GenIrNodesPass::HandleULt(const FuzzULtProto& ult) {
  BValue lhs = GetCoercedBitsOperand(ult.lhs_idx(), ult.operands_type());
  BValue rhs = GetCoercedBitsOperand(ult.rhs_idx(), ult.operands_type());
  context_list_.AppendElement(fb_->ULt(lhs, rhs));
}

void GenIrNodesPass::HandleUGe(const FuzzUGeProto& uge) {
  BValue lhs = GetCoercedBitsOperand(uge.lhs_idx(), uge.operands_type());
  BValue rhs = GetCoercedBitsOperand(uge.rhs_idx(), uge.operands_type());
  context_list_.AppendElement(fb_->UGe(lhs, rhs));
}

void GenIrNodesPass::HandleUGt(const FuzzUGtProto& ugt) {
  BValue lhs = GetCoercedBitsOperand(ugt.lhs_idx(), ugt.operands_type());
  BValue rhs = GetCoercedBitsOperand(ugt.rhs_idx(), ugt.operands_type());
  context_list_.AppendElement(fb_->UGt(lhs, rhs));
}

void GenIrNodesPass::HandleSLe(const FuzzSLeProto& sle) {
  BValue lhs = GetCoercedBitsOperand(sle.lhs_idx(), sle.operands_type());
  BValue rhs = GetCoercedBitsOperand(sle.rhs_idx(), sle.operands_type());
  context_list_.AppendElement(fb_->SLe(lhs, rhs));
}

void GenIrNodesPass::HandleSLt(const FuzzSLtProto& slt) {
  BValue lhs = GetCoercedBitsOperand(slt.lhs_idx(), slt.operands_type());
  BValue rhs = GetCoercedBitsOperand(slt.rhs_idx(), slt.operands_type());
  context_list_.AppendElement(fb_->SLt(lhs, rhs));
}

void GenIrNodesPass::HandleSGe(const FuzzSGeProto& sge) {
  BValue lhs = GetCoercedBitsOperand(sge.lhs_idx(), sge.operands_type());
  BValue rhs = GetCoercedBitsOperand(sge.rhs_idx(), sge.operands_type());
  context_list_.AppendElement(fb_->SGe(lhs, rhs));
}

void GenIrNodesPass::HandleSGt(const FuzzSGtProto& sgt) {
  BValue lhs = GetCoercedBitsOperand(sgt.lhs_idx(), sgt.operands_type());
  BValue rhs = GetCoercedBitsOperand(sgt.rhs_idx(), sgt.operands_type());
  context_list_.AppendElement(fb_->SGt(lhs, rhs));
}

void GenIrNodesPass::HandleEq(const FuzzEqProto& eq) {
  BValue lhs = GetCoercedBitsOperand(eq.lhs_idx(), eq.operands_type());
  BValue rhs = GetCoercedBitsOperand(eq.rhs_idx(), eq.operands_type());
  context_list_.AppendElement(fb_->Eq(lhs, rhs));
}

void GenIrNodesPass::HandleNe(const FuzzNeProto& ne) {
  BValue lhs = GetCoercedBitsOperand(ne.lhs_idx(), ne.operands_type());
  BValue rhs = GetCoercedBitsOperand(ne.rhs_idx(), ne.operands_type());
  context_list_.AppendElement(fb_->Ne(lhs, rhs));
}

void GenIrNodesPass::HandleNegate(const FuzzNegateProto& negate) {
  BValue operand = GetBitsOperand(negate.operand_idx());
  context_list_.AppendElement(fb_->Negate(operand));
}

void GenIrNodesPass::HandleNot(const FuzzNotProto& not_op) {
  BValue operand = GetBitsOperand(not_op.operand_idx());
  context_list_.AppendElement(fb_->Not(operand));
}

// Traverses the ValueTypeProto and returns a BValue representing the
// instantiated literal.
BValue GenIrNodesPass::GetValueFromValueTypeProto(
    const ValueTypeProto& value_type) {
  switch (value_type.type_case()) {
    case ValueTypeProto::kBits: {
      // Take the bytes protobuf datatype and convert it to a Bits object by
      // making a const uint8_t span. Any bytes that exceed the bit width of the
      // literal will be dropped.
      auto bits_type = value_type.bits();
      int64_t bit_width = BoundedWidth(bits_type.bit_width());
      Bits value_bits = ChangeBytesBitWidth(bits_type.value_bytes(), bit_width);
      return fb_->Literal(value_bits);
    }
    default:
      return DefaultValue(p_, fb_);
  }
}

void GenIrNodesPass::HandleLiteral(const FuzzLiteralProto& literal) {
  context_list_.AppendElement(GetValueFromValueTypeProto(literal.value_type()));
}

void GenIrNodesPass::HandleSelect(const FuzzSelectProto& select) {
  BValue selector = GetBitsOperand(select.selector_idx());
  int64_t max_case_count = select.case_idxs_size();
  bool use_default_value = false;
  if (select.case_idxs_size() < selector.BitCountOrDie() ||
      select.case_idxs_size() < 1ULL << selector.BitCountOrDie()) {
    // If the number of cases is less than 2 ** selector_width, we must use a
    // default value otherwise there are not enough cases to cover all possible
    // selector values.
    use_default_value = true;
  } else if (select.case_idxs_size() > 1ULL << selector.BitCountOrDie()) {
    // If the number of cases is greater than 2 ** selector_width, we must
    // reduce the amount of cases to 2 ** selector_width.
    max_case_count = 1ULL << selector.BitCountOrDie();
  }
  // If the number of cases is equal to 2 ** selector_width, we cannot use a
  // default value because it is useless.
  // We need at least one case.
  auto cases_and_default_type = select.cases_and_default_type();
  // Get operands of any type and coerce them to the specified type.
  std::vector<BValue> cases =
      GetCoercedOperands(select.case_idxs(), cases_and_default_type,
                         /*min_operand_count=*/1, max_case_count);
  if (use_default_value) {
    BValue default_value =
        GetCoercedOperand(select.default_value_idx(), cases_and_default_type);
    context_list_.AppendElement(fb_->Select(selector, cases, default_value));
  } else {
    context_list_.AppendElement(fb_->Select(selector, cases));
  }
}

void GenIrNodesPass::HandleOneHot(const FuzzOneHotProto& one_hot) {
  BValue input = GetBitsOperand(one_hot.input_idx());
  // Convert the LsbOrMsb proto enum to the FunctionBuilder enum.
  LsbOrMsb priority;
  switch (one_hot.priority()) {
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
void GenIrNodesPass::HandleOneHotSelect(
    const FuzzOneHotSelectProto& one_hot_select) {
  BValue selector = GetBitsOperand(one_hot_select.selector_idx());
  // Use a default value for the selector if there are no cases or if there are
  // less cases than the selector bit width.
  if (one_hot_select.case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = fb_->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > one_hot_select.case_idxs_size()) {
    selector = fb_->Literal(UBits(0, one_hot_select.case_idxs_size()));
  }
  // Use of GetCoercedOperands's min_operand_count and max_operand_count
  // arguments to ensure that the number of cases is equal to the selector
  // bit width.
  std::vector<BValue> cases = GetCoercedOperands(
      one_hot_select.case_idxs(), one_hot_select.cases_type(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  context_list_.AppendElement(fb_->OneHotSelect(selector, cases));
}

// Same as OneHotSelect, but with a default value.
void GenIrNodesPass::HandlePrioritySelect(
    const FuzzPrioritySelectProto& priority_select) {
  BValue selector = GetBitsOperand(priority_select.selector_idx());
  if (priority_select.case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = fb_->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > priority_select.case_idxs_size()) {
    selector = fb_->Literal(UBits(0, priority_select.case_idxs_size()));
  }
  std::vector<BValue> cases = GetCoercedOperands(
      priority_select.case_idxs(), priority_select.cases_and_default_type(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  BValue default_value =
      GetCoercedOperand(priority_select.default_value_idx(),
                        priority_select.cases_and_default_type());
  context_list_.AppendElement(
      fb_->PrioritySelect(selector, cases, default_value));
}

void GenIrNodesPass::HandleClz(const FuzzClzProto& clz) {
  BValue operand = GetBitsOperand(clz.operand_idx());
  context_list_.AppendElement(fb_->Clz(operand));
}

void GenIrNodesPass::HandleCtz(const FuzzCtzProto& ctz) {
  BValue operand = GetBitsOperand(ctz.operand_idx());
  context_list_.AppendElement(fb_->Ctz(operand));
}

void GenIrNodesPass::HandleMatch(const FuzzMatchProto& match) {
  BValue condition =
      GetCoercedBitsOperand(match.condition_idx(), match.operands_type());
  // Retrieves a vector of Case objects based off of CaseProtos.
  std::vector<FunctionBuilder::Case> cases;
  for (const CaseProto& case_proto : match.case_protos()) {
    cases.push_back(FunctionBuilder::Case{
        GetCoercedBitsOperand(case_proto.clause_idx(), match.operands_type()),
        GetCoercedBitsOperand(case_proto.value_idx(), match.operands_type())});
  }
  // If there are no cases, add a default case.
  if (cases.empty()) {
    Type* default_case_type =
        ConvertBitsTypeProtoToType(p_, match.operands_type());
    BValue default_case_value =
        DefaultValueOfBitsType(p_, fb_, default_case_type);
    cases.push_back(
        FunctionBuilder::Case{default_case_value, default_case_value});
  }
  BValue default_value =
      GetCoercedBitsOperand(match.default_value_idx(), match.operands_type());
  context_list_.AppendElement(fb_->Match(condition, cases, default_value));
}

void GenIrNodesPass::HandleMatchTrue(const FuzzMatchTrueProto& match_true) {
  // MatchTrue only supports bit widths of 1.
  std::vector<FunctionBuilder::Case> cases;
  for (const CaseProto& case_proto : match_true.case_protos()) {
    cases.push_back(FunctionBuilder::Case{
        GetBitsFittedOperand(case_proto.clause_idx(), /*bit_width=*/1,
                             match_true.operands_coercion_method()),
        GetBitsFittedOperand(case_proto.clause_idx(), /*bit_width=*/1,
                             match_true.operands_coercion_method())});
  }
  if (cases.empty()) {
    BValue default_case_value = fb_->Literal(UBits(0, 1));
    cases.push_back(
        FunctionBuilder::Case{default_case_value, default_case_value});
  }
  BValue default_value =
      GetBitsFittedOperand(match_true.default_value_idx(), /*bit_width=*/1,
                           match_true.operands_coercion_method());
  context_list_.AppendElement(fb_->MatchTrue(cases, default_value));
}

void GenIrNodesPass::HandleReverse(const FuzzReverseProto& reverse) {
  BValue operand = GetBitsOperand(reverse.operand_idx());
  context_list_.AppendElement(fb_->Reverse(operand));
}

void GenIrNodesPass::HandleIdentity(const FuzzIdentityProto& identity) {
  // Retrieves any operand type without coercion.
  BValue operand = GetOperand(identity.operand_idx());
  context_list_.AppendElement(fb_->Identity(operand));
}

void GenIrNodesPass::HandleSignExtend(const FuzzSignExtendProto& sign_extend) {
  BValue operand = GetBitsOperand(sign_extend.operand_idx());
  // The bit width cannot be less than the operand bit width because that is an
  // invalid extension.
  int64_t bit_width =
      BoundedWidth(sign_extend.bit_width(), operand.BitCountOrDie());
  context_list_.AppendElement(fb_->SignExtend(operand, bit_width));
}

// Same as SignExtend.
void GenIrNodesPass::HandleZeroExtend(const FuzzZeroExtendProto& zero_extend) {
  BValue operand = GetBitsOperand(zero_extend.operand_idx());
  int64_t bit_width =
      BoundedWidth(zero_extend.bit_width(), operand.BitCountOrDie());
  context_list_.AppendElement(fb_->ZeroExtend(operand, bit_width));
}

void GenIrNodesPass::HandleBitSlice(const FuzzBitSliceProto& bit_slice) {
  BValue operand = GetBitsOperand(bit_slice.operand_idx());
  // The start value must be within the operand bit width and allow at least 1
  // bit to be sliced.
  int64_t start =
      Bounded(bit_slice.start(), /*left_bound=*/0, operand.BitCountOrDie() - 1);
  // The bit width slice amount cannot exceed the operand bit width when
  // starting to slice from the start value.
  int64_t right_bound = std::max<int64_t>(1, operand.BitCountOrDie() - start);
  int64_t bit_width =
      Bounded(bit_slice.bit_width(), /*left_bound=*/1, right_bound);
  context_list_.AppendElement(fb_->BitSlice(operand, start, bit_width));
}

void GenIrNodesPass::HandleBitSliceUpdate(
    const FuzzBitSliceUpdateProto& bit_slice_update) {
  BValue operand = GetBitsOperand(bit_slice_update.operand_idx());
  BValue start = GetBitsOperand(bit_slice_update.start_idx());
  BValue update_value = GetBitsOperand(bit_slice_update.update_value_idx());
  context_list_.AppendElement(
      fb_->BitSliceUpdate(operand, start, update_value));
}

void GenIrNodesPass::HandleDynamicBitSlice(
    const FuzzDynamicBitSliceProto& dynamic_bit_slice) {
  BValue operand = GetBitsOperand(dynamic_bit_slice.operand_idx());
  int64_t bit_width = BoundedWidth(dynamic_bit_slice.bit_width());
  // The operand must be of the same or greater bit width than the dynamic bit
  // slice bit width.
  if (operand.BitCountOrDie() < bit_width) {
    auto operand_coercion_method = dynamic_bit_slice.operand_coercion_method();
    operand = ChangeBitWidth(fb_, operand, bit_width,
                             operand_coercion_method.change_bit_width_method());
  }
  BValue start = GetBitsOperand(dynamic_bit_slice.start_idx());
  context_list_.AppendElement(fb_->DynamicBitSlice(operand, start, bit_width));
}

void GenIrNodesPass::HandleEncode(const FuzzEncodeProto& encode) {
  BValue operand = GetBitsOperand(encode.operand_idx());
  BValue encode_bvalue = fb_->Encode(operand);
  // Encode may result in a 0-bit value. If so, change the bit width to 1.
  if (encode_bvalue.BitCountOrDie() == 0) {
    encode_bvalue = ChangeBitWidth(fb_, encode_bvalue, /*new_bit_width=*/1);
  }
  context_list_.AppendElement(encode_bvalue);
}

void GenIrNodesPass::HandleDecode(const FuzzDecodeProto& decode) {
  BValue operand = GetBitsOperand(decode.operand_idx());
  // The decode bit width cannot exceed 2 ** operand_bit_width.
  int64_t right_bound = 1000;
  if (operand.BitCountOrDie() < 64) {
    right_bound = std::min<int64_t>(1000, 1ULL << operand.BitCountOrDie());
  }
  int64_t bit_width =
      BoundedWidth(decode.bit_width(), /*left_bound=*/1, right_bound);
  context_list_.AppendElement(fb_->Decode(operand, bit_width));
}

void GenIrNodesPass::HandleGate(const FuzzGateProto& gate) {
  // The Gate condition only supports bit widths of 1.
  BValue condition = GetBitsFittedOperand(gate.condition_idx(), /*bit_width=*/1,
                                          gate.condition_coercion_method());
  BValue data = GetBitsOperand(gate.data_idx());
  context_list_.AppendElement(fb_->Gate(condition, data));
}

// Retrieves an operand from the combined context list based off of a list
// index.
BValue GenIrNodesPass::GetOperand(const OperandIdxProto& operand_idx) {
  // Retrieve an operand from the context list based off of the randomly
  // generated list index.
  return context_list_.GetElementAt(operand_idx.list_idx());
}

// Retrieves an operand from the bits context list.
BValue GenIrNodesPass::GetBitsOperand(const BitsOperandIdxProto& operand_idx) {
  return context_list_.GetElementAt(operand_idx.list_idx(),
                                    ContextListType::BITS_LIST);
}

// Retrieves an operand from the combined context list and then coerces it to
// the specified type.
BValue GenIrNodesPass::GetCoercedOperand(const OperandIdxProto& operand_idx,
                                         const CoercedTypeProto& coerced_type) {
  BValue operand = GetOperand(operand_idx);
  Type* type = ConvertTypeProtoToType(p_, coerced_type);
  return Coerced(p_, fb_, operand, coerced_type, type);
}

// Retrieves an operand from the bits context list and then coerces it to the
// specified bits type.
BValue GenIrNodesPass::GetCoercedBitsOperand(
    const BitsOperandIdxProto& operand_idx,
    const BitsCoercedTypeProto& coerced_type) {
  BValue operand = GetBitsOperand(operand_idx);
  Type* type = ConvertBitsTypeProtoToType(p_, coerced_type);
  return CoercedBits(p_, fb_, operand, coerced_type, type);
}

// Retrieves an operand from the bits context list and then changes the bit
// width to the specified bit width.
BValue GenIrNodesPass::GetBitsFittedOperand(
    const BitsOperandIdxProto& operand_idx, int64_t bit_width,
    const BitsCoercionMethodProto& coercion_method) {
  BValue operand = GetBitsOperand(operand_idx);
  return ChangeBitWidth(fb_, operand, bit_width,
                        coercion_method.change_bit_width_method());
}

// Retrieves multiple operands from the bits context list. If min_operand_count
// and max_operand_count are specified, the number of returned operands will be
// constrained to be between the specified minimum/maximum.
std::vector<BValue> GenIrNodesPass::GetBitsOperands(
    const google::protobuf::RepeatedPtrField<BitsOperandIdxProto>& operand_idxs,
    int64_t min_operand_count, int64_t max_operand_count) {
  // If the max operand count is less than the min operand count, assume invalid
  // or default inputs, so just ignore max_operand_count.
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs.size();
  }
  std::vector<BValue> operands;
  // Add operands up to the max operand count or up to the end of the vector.
  for (int64_t i = 0; i < operand_idxs.size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetBitsOperand(operand_idxs.at(i)));
  }
  // Fill in any remaining operands with default values.
  BValue default_value = DefaultValue(p_, fb_, TypeCase::BITS_CASE);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

// Retrieves multiple operands from the combined context list and then coerces
// them to the specified type.
std::vector<BValue> GenIrNodesPass::GetCoercedOperands(
    const google::protobuf::RepeatedPtrField<OperandIdxProto>& operand_idxs,
    const CoercedTypeProto& coerced_type, int64_t min_operand_count,
    int64_t max_operand_count) {
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs.size();
  }
  std::vector<BValue> operands;
  for (int64_t i = 0; i < operand_idxs.size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetCoercedOperand(operand_idxs.at(i), coerced_type));
  }
  Type* type = ConvertTypeProtoToType(p_, coerced_type);
  BValue default_value = DefaultValueOfType(p_, fb_, type);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

// Retrieves multiple operands from the bits context list and then coerces
// them to the specified bits type.
std::vector<BValue> GenIrNodesPass::GetCoercedBitsOperands(
    const google::protobuf::RepeatedPtrField<BitsOperandIdxProto>& operand_idxs,
    const BitsCoercedTypeProto& coerced_type, int64_t min_operand_count,
    int64_t max_operand_count) {
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs.size();
  }
  std::vector<BValue> operands;
  for (int64_t i = 0; i < operand_idxs.size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetCoercedBitsOperand(operand_idxs.at(i), coerced_type));
  }
  Type* type = ConvertBitsTypeProtoToType(p_, coerced_type);
  BValue default_value = DefaultValueOfBitsType(p_, fb_, type);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

}  // namespace xls

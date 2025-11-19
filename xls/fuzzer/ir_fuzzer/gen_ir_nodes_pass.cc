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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xls/fuzzer/ir_fuzzer/combine_context_list.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_flattening.h"

namespace xls {

// Loops through all of the FuzzOpProtos in the FuzzProgramProto. Each
// FuzzOpProto is a randomly generated object that is used to
// instantiate/generate an IR node/BValue. Add these BValues to the context
// list. Some FuzzOpProtos may require retrieving previous BValues from the
// context list.
void GenIrNodesPass::GenIrNodes() {
  for (const FuzzOpProto& fuzz_op : fuzz_program_.fuzz_ops()) {
    VisitFuzzOp(fuzz_op);
    if (std::optional<int64_t>& nodes_to_consume = state().nodes_to_consume();
        nodes_to_consume.has_value()) {
      --*nodes_to_consume;
      if (*nodes_to_consume <= 0) {
        function_states_.pop_back();
      }
    }
  }
  CHECK(!function_states_.empty());
  Function* top = function_states_.front().fb()->function()->AsFunctionOrDie();
  CHECK_OK(p_->SetTop(top));
  function_states_.clear();
}

void GenIrNodesPass::HandleParam(const FuzzParamProto& param) {
  // Params are named as "p" followed by the combined context list index of the
  // param.
  std::string name =
      absl::StrFormat("p%d", state().context_list().GetListSize());
  // Retrieve the Type object from the FuzzTypeProto.
  Type* type = helpers_.ConvertTypeProtoToType(p_, param.type());
  // Append the param BValue to the combined context list and the context list
  // for its type.
  state().context_list().AppendElement(state().fb()->Param(name, type));
}

void GenIrNodesPass::HandleShra(const FuzzShraProto& shra) {
  // Retrieve a bits operand from the bits context list based on the list_idx.
  BValue operand = GetBitsOperand(shra.operand_idx());
  BValue amount = GetBitsOperand(shra.amount_idx());
  state().context_list().AppendElement(state().fb()->Shra(operand, amount));
}

void GenIrNodesPass::HandleShrl(const FuzzShrlProto& shrl) {
  BValue operand = GetBitsOperand(shrl.operand_idx());
  BValue amount = GetBitsOperand(shrl.amount_idx());
  state().context_list().AppendElement(state().fb()->Shrl(operand, amount));
}

void GenIrNodesPass::HandleShll(const FuzzShllProto& shll) {
  BValue operand = GetBitsOperand(shll.operand_idx());
  BValue amount = GetBitsOperand(shll.amount_idx());
  state().context_list().AppendElement(state().fb()->Shll(operand, amount));
}

void GenIrNodesPass::HandleOr(const FuzzOrProto& or_op) {
  // Requires at least one bits operand of a specific bit width, which is
  // defined by the operands_type.
  std::vector<BValue> operands =
      GetCoercedBitsOperands(or_op.operand_idxs(), or_op.operands_type(),
                             /*min_operand_count=*/1);
  state().context_list().AppendElement(state().fb()->Or(operands));
}

void GenIrNodesPass::HandleNor(const FuzzNorProto& nor) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(nor.operand_idxs(), nor.operands_type(),
                             /*min_operand_count=*/1);
  state().context_list().AppendElement(state().fb()->Nor(operands));
}

void GenIrNodesPass::HandleXor(const FuzzXorProto& xor_op) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(xor_op.operand_idxs(), xor_op.operands_type(),
                             /*min_operand_count=*/1);
  state().context_list().AppendElement(state().fb()->Xor(operands));
}

void GenIrNodesPass::HandleAnd(const FuzzAndProto& and_op) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(and_op.operand_idxs(), and_op.operands_type(),
                             /*min_operand_count=*/1);
  state().context_list().AppendElement(state().fb()->And(operands));
}

void GenIrNodesPass::HandleNand(const FuzzNandProto& nand) {
  std::vector<BValue> operands =
      GetCoercedBitsOperands(nand.operand_idxs(), nand.operands_type(),
                             /*min_operand_count=*/1);
  state().context_list().AppendElement(state().fb()->Nand(operands));
}

void GenIrNodesPass::HandleAndReduce(const FuzzAndReduceProto& and_reduce) {
  BValue operand = GetBitsOperand(and_reduce.operand_idx());
  state().context_list().AppendElement(state().fb()->AndReduce(operand));
}

void GenIrNodesPass::HandleOrReduce(const FuzzOrReduceProto& or_reduce) {
  BValue operand = GetBitsOperand(or_reduce.operand_idx());
  state().context_list().AppendElement(state().fb()->OrReduce(operand));
}

void GenIrNodesPass::HandleXorReduce(const FuzzXorReduceProto& xor_reduce) {
  BValue operand = GetBitsOperand(xor_reduce.operand_idx());
  state().context_list().AppendElement(state().fb()->XorReduce(operand));
}

void GenIrNodesPass::HandleUMul(const FuzzUMulProto& umul) {
  if (umul.has_bit_width()) {
    // If the bit width is set, use it.
    BValue lhs = GetBitsOperand(umul.lhs_idx());
    BValue rhs = GetBitsOperand(umul.rhs_idx());
    // The bit width cannot exceed the sum of the bit widths of the operands.
    int64_t right_bound =
        std::min<int64_t>(IrFuzzHelpers::kMaxFuzzBitWidth,
                          lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        helpers_.BoundedWidth(umul.bit_width(), /*left_bound=*/1, right_bound);
    state().context_list().AppendElement(
        state().fb()->UMul(lhs, rhs, bit_width));
  } else {
    // If the bit width is not set, don't use it and coerce the operands to be
    // of the same type.
    BValue lhs = GetCoercedBitsOperand(umul.lhs_idx(), umul.operands_type());
    BValue rhs = GetCoercedBitsOperand(umul.rhs_idx(), umul.operands_type());
    state().context_list().AppendElement(state().fb()->UMul(lhs, rhs));
  }
}

// Same as UMul.
void GenIrNodesPass::HandleSMul(const FuzzSMulProto& smul) {
  if (smul.has_bit_width()) {
    BValue lhs = GetBitsOperand(smul.lhs_idx());
    BValue rhs = GetBitsOperand(smul.rhs_idx());
    int64_t right_bound =
        std::min<int64_t>(IrFuzzHelpers::kMaxFuzzBitWidth,
                          lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        helpers_.BoundedWidth(smul.bit_width(), /*left_bound=*/1, right_bound);
    state().context_list().AppendElement(
        state().fb()->SMul(lhs, rhs, bit_width));
  } else {
    BValue lhs = GetCoercedBitsOperand(smul.lhs_idx(), smul.operands_type());
    BValue rhs = GetCoercedBitsOperand(smul.rhs_idx(), smul.operands_type());
    state().context_list().AppendElement(state().fb()->SMul(lhs, rhs));
  }
}

// Same as UMul.
void GenIrNodesPass::HandleUMulp(const FuzzUMulpProto& umulp) {
  if (umulp.has_bit_width()) {
    BValue lhs = GetBitsOperand(umulp.lhs_idx());
    BValue rhs = GetBitsOperand(umulp.rhs_idx());
    int64_t right_bound =
        std::min<int64_t>(IrFuzzHelpers::kMaxFuzzBitWidth,
                          lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        helpers_.BoundedWidth(umulp.bit_width(), /*left_bound=*/1, right_bound);
    state().context_list().AppendElement(
        state().fb()->UMulp(lhs, rhs, bit_width));
  } else {
    BValue lhs = GetCoercedBitsOperand(umulp.lhs_idx(), umulp.operands_type());
    BValue rhs = GetCoercedBitsOperand(umulp.rhs_idx(), umulp.operands_type());
    state().context_list().AppendElement(state().fb()->UMulp(lhs, rhs));
  }
}

// Same as UMul.
void GenIrNodesPass::HandleSMulp(const FuzzSMulpProto& smulp) {
  if (smulp.has_bit_width()) {
    BValue lhs = GetBitsOperand(smulp.lhs_idx());
    BValue rhs = GetBitsOperand(smulp.rhs_idx());
    int64_t right_bound =
        std::min<int64_t>(IrFuzzHelpers::kMaxFuzzBitWidth,
                          lhs.BitCountOrDie() + rhs.BitCountOrDie());
    int64_t bit_width =
        helpers_.BoundedWidth(smulp.bit_width(), /*left_bound=*/1, right_bound);
    state().context_list().AppendElement(
        state().fb()->SMulp(lhs, rhs, bit_width));
  } else {
    BValue lhs = GetCoercedBitsOperand(smulp.lhs_idx(), smulp.operands_type());
    BValue rhs = GetCoercedBitsOperand(smulp.rhs_idx(), smulp.operands_type());
    state().context_list().AppendElement(state().fb()->SMulp(lhs, rhs));
  }
}

void GenIrNodesPass::HandleUDiv(const FuzzUDivProto& udiv) {
  BValue lhs = GetCoercedBitsOperand(udiv.lhs_idx(), udiv.operands_type());
  BValue rhs = GetCoercedBitsOperand(udiv.rhs_idx(), udiv.operands_type());
  state().context_list().AppendElement(state().fb()->UDiv(lhs, rhs));
}

void GenIrNodesPass::HandleSDiv(const FuzzSDivProto& sdiv) {
  BValue lhs = GetCoercedBitsOperand(sdiv.lhs_idx(), sdiv.operands_type());
  BValue rhs = GetCoercedBitsOperand(sdiv.rhs_idx(), sdiv.operands_type());
  state().context_list().AppendElement(state().fb()->SDiv(lhs, rhs));
}

void GenIrNodesPass::HandleUMod(const FuzzUModProto& umod) {
  BValue lhs = GetCoercedBitsOperand(umod.lhs_idx(), umod.operands_type());
  BValue rhs = GetCoercedBitsOperand(umod.rhs_idx(), umod.operands_type());
  state().context_list().AppendElement(state().fb()->UMod(lhs, rhs));
}

void GenIrNodesPass::HandleSMod(const FuzzSModProto& smod) {
  BValue lhs = GetCoercedBitsOperand(smod.lhs_idx(), smod.operands_type());
  BValue rhs = GetCoercedBitsOperand(smod.rhs_idx(), smod.operands_type());
  state().context_list().AppendElement(state().fb()->SMod(lhs, rhs));
}

void GenIrNodesPass::HandleSubtract(const FuzzSubtractProto& subtract) {
  BValue lhs =
      GetCoercedBitsOperand(subtract.lhs_idx(), subtract.operands_type());
  BValue rhs =
      GetCoercedBitsOperand(subtract.rhs_idx(), subtract.operands_type());
  state().context_list().AppendElement(state().fb()->Subtract(lhs, rhs));
}

void GenIrNodesPass::HandleAdd(const FuzzAddProto& add) {
  BValue lhs = GetCoercedBitsOperand(add.lhs_idx(), add.operands_type());
  BValue rhs = GetCoercedBitsOperand(add.rhs_idx(), add.operands_type());
  state().context_list().AppendElement(state().fb()->Add(lhs, rhs));
}

void GenIrNodesPass::HandleConcat(const FuzzConcatProto& concat) {
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetBitsOperands(concat.operand_idxs(), /*min_operand_count=*/1);
  // Only use operands such that their summed bit width is less than or equal
  // to IrFuzzHelpers::kMaxFuzzBitWidth.
  int64_t bit_width_sum = 0;
  int64_t operand_count = 0;
  for (const auto& operand : operands) {
    bit_width_sum += operand.BitCountOrDie();
    if (bit_width_sum > IrFuzzHelpers::kMaxFuzzBitWidth) {
      break;
    }
    operand_count += 1;
  }
  // Drop the remaining operands.
  operands.resize(operand_count);
  CHECK(!operands.empty());
  state().context_list().AppendElement(state().fb()->Concat(operands));
}

void GenIrNodesPass::HandleULe(const FuzzULeProto& ule) {
  BValue lhs = GetCoercedBitsOperand(ule.lhs_idx(), ule.operands_type());
  BValue rhs = GetCoercedBitsOperand(ule.rhs_idx(), ule.operands_type());
  state().context_list().AppendElement(state().fb()->ULe(lhs, rhs));
}

void GenIrNodesPass::HandleULt(const FuzzULtProto& ult) {
  BValue lhs = GetCoercedBitsOperand(ult.lhs_idx(), ult.operands_type());
  BValue rhs = GetCoercedBitsOperand(ult.rhs_idx(), ult.operands_type());
  state().context_list().AppendElement(state().fb()->ULt(lhs, rhs));
}

void GenIrNodesPass::HandleUGe(const FuzzUGeProto& uge) {
  BValue lhs = GetCoercedBitsOperand(uge.lhs_idx(), uge.operands_type());
  BValue rhs = GetCoercedBitsOperand(uge.rhs_idx(), uge.operands_type());
  state().context_list().AppendElement(state().fb()->UGe(lhs, rhs));
}

void GenIrNodesPass::HandleUGt(const FuzzUGtProto& ugt) {
  BValue lhs = GetCoercedBitsOperand(ugt.lhs_idx(), ugt.operands_type());
  BValue rhs = GetCoercedBitsOperand(ugt.rhs_idx(), ugt.operands_type());
  state().context_list().AppendElement(state().fb()->UGt(lhs, rhs));
}

void GenIrNodesPass::HandleSLe(const FuzzSLeProto& sle) {
  BValue lhs = GetCoercedBitsOperand(sle.lhs_idx(), sle.operands_type());
  BValue rhs = GetCoercedBitsOperand(sle.rhs_idx(), sle.operands_type());
  state().context_list().AppendElement(state().fb()->SLe(lhs, rhs));
}

void GenIrNodesPass::HandleSLt(const FuzzSLtProto& slt) {
  BValue lhs = GetCoercedBitsOperand(slt.lhs_idx(), slt.operands_type());
  BValue rhs = GetCoercedBitsOperand(slt.rhs_idx(), slt.operands_type());
  state().context_list().AppendElement(state().fb()->SLt(lhs, rhs));
}

void GenIrNodesPass::HandleSGe(const FuzzSGeProto& sge) {
  BValue lhs = GetCoercedBitsOperand(sge.lhs_idx(), sge.operands_type());
  BValue rhs = GetCoercedBitsOperand(sge.rhs_idx(), sge.operands_type());
  state().context_list().AppendElement(state().fb()->SGe(lhs, rhs));
}

void GenIrNodesPass::HandleSGt(const FuzzSGtProto& sgt) {
  BValue lhs = GetCoercedBitsOperand(sgt.lhs_idx(), sgt.operands_type());
  BValue rhs = GetCoercedBitsOperand(sgt.rhs_idx(), sgt.operands_type());
  state().context_list().AppendElement(state().fb()->SGt(lhs, rhs));
}

void GenIrNodesPass::HandleEq(const FuzzEqProto& eq) {
  BValue lhs = GetCoercedBitsOperand(eq.lhs_idx(), eq.operands_type());
  BValue rhs = GetCoercedBitsOperand(eq.rhs_idx(), eq.operands_type());
  state().context_list().AppendElement(state().fb()->Eq(lhs, rhs));
}

void GenIrNodesPass::HandleNe(const FuzzNeProto& ne) {
  BValue lhs = GetCoercedBitsOperand(ne.lhs_idx(), ne.operands_type());
  BValue rhs = GetCoercedBitsOperand(ne.rhs_idx(), ne.operands_type());
  state().context_list().AppendElement(state().fb()->Ne(lhs, rhs));
}

void GenIrNodesPass::HandleNegate(const FuzzNegateProto& negate) {
  BValue operand = GetBitsOperand(negate.operand_idx());
  state().context_list().AppendElement(state().fb()->Negate(operand));
}

void GenIrNodesPass::HandleNot(const FuzzNotProto& not_op) {
  BValue operand = GetBitsOperand(not_op.operand_idx());
  state().context_list().AppendElement(state().fb()->Not(operand));
}

void GenIrNodesPass::HandleLiteral(const FuzzLiteralProto& literal) {
  Type* type = helpers_.ConvertTypeProtoToType(p_, literal.type());
  Bits value_bits = helpers_.ChangeBytesBitWidth(literal.value_bytes(),
                                                 type->GetFlatBitCount());
  Value value = UnflattenBitsToValue(value_bits, type).value();
  state().context_list().AppendElement(state().fb()->Literal(value));
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
    state().context_list().AppendElement(
        state().fb()->Select(selector, cases, default_value));
  } else {
    state().context_list().AppendElement(state().fb()->Select(selector, cases));
  }
}

void GenIrNodesPass::HandleOneHot(const FuzzOneHotProto& one_hot) {
  BValue operand = GetBitsOperand(one_hot.operand_idx());
  // If the operand bit width is kMaxFuzzBitWidth, decrease it by 1 because
  // OneHot returns an operand with a bit width of 1 + the bit width of the
  // operand.
  if (operand.BitCountOrDie() == IrFuzzHelpers::kMaxFuzzBitWidth) {
    auto operand_coercion_method = one_hot.operand_coercion_method();
    operand = helpers_.ChangeBitWidth(
        state().fb(), operand, IrFuzzHelpers::kMaxFuzzBitWidth - 1,
        operand_coercion_method.change_bit_width_method());
  }
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
  state().context_list().AppendElement(state().fb()->OneHot(operand, priority));
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
      selector = state().fb()->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > one_hot_select.case_idxs_size()) {
    selector = state().fb()->Literal(UBits(0, one_hot_select.case_idxs_size()));
  }
  // Use of GetCoercedOperands's min_operand_count and max_operand_count
  // arguments to ensure that the number of cases is equal to the selector
  // bit width.
  std::vector<BValue> cases = GetCoercedOperands(
      one_hot_select.case_idxs(), one_hot_select.cases_type(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  state().context_list().AppendElement(
      state().fb()->OneHotSelect(selector, cases));
}

// Same as OneHotSelect, but with a default value.
void GenIrNodesPass::HandlePrioritySelect(
    const FuzzPrioritySelectProto& priority_select) {
  BValue selector = GetBitsOperand(priority_select.selector_idx());
  if (priority_select.case_idxs_size() == 0) {
    if (selector.BitCountOrDie() != 1) {
      selector = state().fb()->Literal(UBits(0, 1));
    }
  } else if (selector.BitCountOrDie() > priority_select.case_idxs_size()) {
    selector =
        state().fb()->Literal(UBits(0, priority_select.case_idxs_size()));
  }
  std::vector<BValue> cases = GetCoercedOperands(
      priority_select.case_idxs(), priority_select.cases_and_default_type(),
      selector.BitCountOrDie(), selector.BitCountOrDie());
  BValue default_value =
      GetCoercedOperand(priority_select.default_value_idx(),
                        priority_select.cases_and_default_type());
  state().context_list().AppendElement(
      state().fb()->PrioritySelect(selector, cases, default_value));
}

void GenIrNodesPass::HandleClz(const FuzzClzProto& clz) {
  BValue operand = GetBitsOperand(clz.operand_idx());
  state().context_list().AppendElement(state().fb()->Clz(operand));
}

void GenIrNodesPass::HandleCtz(const FuzzCtzProto& ctz) {
  BValue operand = GetBitsOperand(ctz.operand_idx());
  state().context_list().AppendElement(state().fb()->Ctz(operand));
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
        helpers_.ConvertBitsTypeProtoToType(p_, match.operands_type());
    BValue default_case_value =
        helpers_.DefaultValueOfBitsType(p_, state().fb(), default_case_type);
    cases.push_back(
        FunctionBuilder::Case{default_case_value, default_case_value});
  }
  BValue default_value =
      GetCoercedBitsOperand(match.default_value_idx(), match.operands_type());
  state().context_list().AppendElement(
      state().fb()->Match(condition, cases, default_value));
}

void GenIrNodesPass::HandleMatchTrue(const FuzzMatchTrueProto& match_true) {
  // MatchTrue only supports bit widths of 1.
  std::vector<FunctionBuilder::Case> cases;
  for (const CaseProto& case_proto : match_true.case_protos()) {
    cases.push_back(FunctionBuilder::Case{
        GetBitsFittedOperand(case_proto.clause_idx(),
                             match_true.operands_coercion_method(),
                             /*bit_width=*/1),
        GetBitsFittedOperand(case_proto.clause_idx(),
                             match_true.operands_coercion_method(),
                             /*bit_width=*/1)});
  }
  if (cases.empty()) {
    // If there are no cases, add a default case.
    BValue default_case_value = state().fb()->Literal(UBits(0, 1));
    cases.push_back(
        FunctionBuilder::Case{default_case_value, default_case_value});
  }
  BValue default_value = GetBitsFittedOperand(
      match_true.default_value_idx(), match_true.operands_coercion_method(),
      /*bit_width=*/1);
  state().context_list().AppendElement(
      state().fb()->MatchTrue(cases, default_value));
}

void GenIrNodesPass::HandleTuple(const FuzzTupleProto& tuple) {
  std::vector<BValue> operands = GetOperands(tuple.operand_idxs());
  state().context_list().AppendElement(state().fb()->Tuple(operands));
}

void GenIrNodesPass::HandleArray(const FuzzArrayProto& array) {
  // Arrays must have at least one element.
  std::vector<BValue> operands =
      GetCoercedOperands(array.operand_idxs(), array.operands_type(),
                         /*min_operand_count=*/1);
  Type* element_type =
      helpers_.ConvertTypeProtoToType(p_, array.operands_type());
  state().context_list().AppendElement(
      state().fb()->Array(operands, element_type));
}

void GenIrNodesPass::HandleTupleIndex(const FuzzTupleIndexProto& tuple_index) {
  BValue operand = GetTupleOperand(tuple_index.operand_idx());
  int64_t tuple_size = operand.GetType()->AsTupleOrDie()->size();
  // The tuple must have at least one element.
  if (tuple_size == 0) {
    operand = helpers_.DefaultValue(p_, state().fb(), TypeCase::TUPLE_CASE);
    tuple_size = operand.GetType()->AsTupleOrDie()->size();
  }
  int64_t index = helpers_.Bounded(tuple_index.index(),
                                   /*left_bound=*/0, tuple_size - 1);
  state().context_list().AppendElement(
      state().fb()->TupleIndex(operand, index));
}

void GenIrNodesPass::HandleArrayIndex(const FuzzArrayIndexProto& array_index) {
  BValue operand = GetArrayOperand(array_index.operand_idx());
  BValue indices = GetBitsOperand(array_index.indices_idx());
  state().context_list().AppendElement(state().fb()->ArrayIndex(
      operand, {indices}, /*assumed_in_bounds=*/false));
}

void GenIrNodesPass::HandleArraySlice(const FuzzArraySliceProto& array_slice) {
  BValue operand = GetArrayOperand(array_slice.operand_idx());
  BValue start = GetBitsOperand(array_slice.start_idx());
  int64_t width = helpers_.BoundedArraySize(array_slice.width());
  state().context_list().AppendElement(
      state().fb()->ArraySlice(operand, start, width));
}

void GenIrNodesPass::HandleArrayUpdate(
    const FuzzArrayUpdateProto& array_update) {
  BValue operand = GetArrayOperand(array_update.operand_idx());
  Type* element_type = operand.GetType()->AsArrayOrDie()->element_type();
  BValue update_value = GetFittedOperand(
      array_update.update_value_idx(),
      array_update.update_value_coercion_method(), element_type);
  BValue indices = GetBitsOperand(array_update.indices_idx());
  state().context_list().AppendElement(
      state().fb()->ArrayUpdate(operand, update_value, {indices}));
}

void GenIrNodesPass::HandleArrayConcat(
    const FuzzArrayConcatProto& array_concat) {
  std::vector<BValue> operands =
      GetArrayOperands(array_concat.operand_idxs(), /*min_operand_count=*/1);
  // Only use operands such that their summed array size is less than or equal
  // to IrFuzzHelpers::kMaxFuzzArraySize.
  int64_t array_size_sum = 0;
  int64_t operand_count = 0;
  for (const auto& operand : operands) {
    array_size_sum += operand.GetType()->AsArrayOrDie()->size();
    if (array_size_sum > IrFuzzHelpers::kMaxFuzzArraySize) {
      break;
    }
    operand_count += 1;
  }
  // Drop the remaining operands.
  operands.resize(operand_count);
  CHECK(!operands.empty());

  // Coerce the operands to share the same element type, keeping the sizes the
  // same.
  std::vector<BValue> coerced_operands;
  coerced_operands.reserve(operands.size());
  for (const auto& operand : operands) {
    ArrayCoercedTypeProto coerced_type;
    coerced_type.set_array_size(operand.GetType()->AsArrayOrDie()->size());
    *coerced_type.mutable_array_element() = array_concat.element_type();
    coerced_operands.push_back(helpers_.CoercedArray(
        p_, state().fb(), operand, coerced_type,
        helpers_.ConvertArrayTypeProtoToType(p_, coerced_type)));
  }

  state().context_list().AppendElement(
      state().fb()->ArrayConcat(coerced_operands));
}

void GenIrNodesPass::HandleReverse(const FuzzReverseProto& reverse) {
  BValue operand = GetBitsOperand(reverse.operand_idx());
  state().context_list().AppendElement(state().fb()->Reverse(operand));
}

void GenIrNodesPass::HandleIdentity(const FuzzIdentityProto& identity) {
  // Retrieves any operand type without coercion.
  BValue operand = GetOperand(identity.operand_idx());
  state().context_list().AppendElement(state().fb()->Identity(operand));
}

void GenIrNodesPass::HandleSignExtend(const FuzzSignExtendProto& sign_extend) {
  BValue operand = GetBitsOperand(sign_extend.operand_idx());
  // The bit width cannot be less than the operand bit width because that is an
  // invalid extension.
  int64_t bit_width =
      helpers_.BoundedWidth(sign_extend.bit_width(), operand.BitCountOrDie());
  state().context_list().AppendElement(
      state().fb()->SignExtend(operand, bit_width));
}

// Same as SignExtend.
void GenIrNodesPass::HandleZeroExtend(const FuzzZeroExtendProto& zero_extend) {
  BValue operand = GetBitsOperand(zero_extend.operand_idx());
  int64_t bit_width =
      helpers_.BoundedWidth(zero_extend.bit_width(), operand.BitCountOrDie());
  state().context_list().AppendElement(
      state().fb()->ZeroExtend(operand, bit_width));
}

void GenIrNodesPass::HandleBitSlice(const FuzzBitSliceProto& bit_slice) {
  BValue operand = GetBitsOperand(bit_slice.operand_idx());
  // The start value must be within the operand bit width and allow at least 1
  // bit to be sliced.
  int64_t start = helpers_.Bounded(bit_slice.start(), /*left_bound=*/0,
                                   operand.BitCountOrDie() - 1);
  // The bit width slice amount cannot exceed the operand bit width when
  // starting to slice from the start value.
  int64_t right_bound = std::max<int64_t>(1, operand.BitCountOrDie() - start);
  int64_t bit_width = helpers_.Bounded(bit_slice.bit_width(),
                                       /*left_bound=*/1, right_bound);
  state().context_list().AppendElement(
      state().fb()->BitSlice(operand, start, bit_width));
}

void GenIrNodesPass::HandleBitSliceUpdate(
    const FuzzBitSliceUpdateProto& bit_slice_update) {
  BValue operand = GetBitsOperand(bit_slice_update.operand_idx());
  BValue start = GetBitsOperand(bit_slice_update.start_idx());
  BValue update_value = GetBitsOperand(bit_slice_update.update_value_idx());
  state().context_list().AppendElement(
      state().fb()->BitSliceUpdate(operand, start, update_value));
}

void GenIrNodesPass::HandleDynamicBitSlice(
    const FuzzDynamicBitSliceProto& dynamic_bit_slice) {
  BValue operand = GetBitsOperand(dynamic_bit_slice.operand_idx());
  int64_t bit_width = helpers_.BoundedWidth(dynamic_bit_slice.bit_width());
  // The operand must be of the same or greater bit width than the dynamic bit
  // slice bit width.
  if (operand.BitCountOrDie() < bit_width) {
    auto operand_coercion_method = dynamic_bit_slice.operand_coercion_method();
    operand = helpers_.ChangeBitWidth(
        state().fb(), operand, bit_width,
        operand_coercion_method.change_bit_width_method());
  }
  BValue start = GetBitsOperand(dynamic_bit_slice.start_idx());
  state().context_list().AppendElement(
      state().fb()->DynamicBitSlice(operand, start, bit_width));
}

void GenIrNodesPass::HandleEncode(const FuzzEncodeProto& encode) {
  BValue operand = GetBitsOperand(encode.operand_idx());
  BValue encode_bvalue = state().fb()->Encode(operand);
  // Encode may result in a 0-bit value. If so, change the bit width to 1.
  if (encode_bvalue.BitCountOrDie() == 0) {
    encode_bvalue = helpers_.ChangeBitWidth(state().fb(), encode_bvalue,
                                            /*new_bit_width=*/1);
  }
  state().context_list().AppendElement(encode_bvalue);
}

void GenIrNodesPass::HandleDecode(const FuzzDecodeProto& decode) {
  BValue operand = GetBitsOperand(decode.operand_idx());
  // The decode bit width cannot exceed 2 ** operand_bit_width.
  int64_t right_bound = IrFuzzHelpers::kMaxFuzzBitWidth;
  // We could choose a larger size for this check, but we're clamping to
  // IrFuzzHelpers::kMaxFuzzBitWidth and 10 bits is sufficient for this while
  // completely avoiding potential overflow, unlike e.g. checking for 64 bits.
  if (operand.BitCountOrDie() < 10) {
    right_bound = std::min<int64_t>(IrFuzzHelpers::kMaxFuzzBitWidth,
                                    1ULL << operand.BitCountOrDie());
  }
  int64_t bit_width =
      helpers_.BoundedWidth(decode.bit_width(), /*left_bound=*/1, right_bound);
  state().context_list().AppendElement(
      state().fb()->Decode(operand, bit_width));
}

void GenIrNodesPass::HandleGate(const FuzzGateProto& gate) {
  // The Gate condition only supports bit widths of 1.
  BValue condition = GetBitsFittedOperand(gate.condition_idx(),
                                          gate.condition_coercion_method(),
                                          /*bit_width=*/1);
  BValue data = GetBitsOperand(gate.data_idx());
  state().context_list().AppendElement(state().fb()->Gate(condition, data));
}

void GenIrNodesPass::HandleDefineFunction(
    const FuzzDefineFunctionProto& define_function) {
  function_states_.emplace_back(
      p_,
      absl::StrCat(
          "child_function_",
          // need to add function_states_.size() because the builder doesn't add
          // to p_->functions() until after the function is fully built.
          p_->functions().size() + function_states_.size()),
      fuzz_program_.version(), define_function.combine_list_method(),
      helpers_.Bounded(define_function.next_nodes_consumed(), 0, 1000));
}

void GenIrNodesPass::HandleInvoke(const FuzzInvokeProto& invoke) {
  if (p_->functions().empty()) {
    return;
  }
  Function* to_invoke =
      p_->functions()[helpers_.Bounded(invoke.function_index(), 0,
                                       p_->functions().size() - 1)]
          .get();
  Function* caller = state().fb()->function()->AsFunctionOrDie();
  // Make a copy in case we rehash on caller.
  absl::flat_hash_set<Function*> callee_children = caller_to_callee_[to_invoke];
  DCHECK_NE(to_invoke, caller) << "Should be impossible to see caller as it "
                                  "shouldn't appear until finalized.";
  if (callee_children.contains(caller)) {
    // Don't allow a function to be invoked by itself or a function that invokes
    // itself.
    return;
  }
  CHECK_NE(to_invoke, nullptr);
  caller_to_callee_[caller].insert(to_invoke);
  caller_to_callee_[caller].insert(callee_children.begin(),
                                   callee_children.end());

  std::vector<BValue> operands =
      GetOperands(invoke.args_idxs(), /*min_operand_count=*/0,
                  /*max_operand_count=*/to_invoke->params().size());
  std::vector<BValue> invoke_args;
  invoke_args.reserve(to_invoke->params().size());
  int64_t param_idx = 0;
  auto coercion_method_itr = invoke.args_coercion_methods().begin();
  auto coercion_method_end = invoke.args_coercion_methods().end();
  CoercionMethodProto coercion_method = CoercionMethodProto::default_instance();
  for (Param* param : to_invoke->params()) {
    if (param_idx < operands.size()) {
      if (operands[param_idx].GetType() == param->GetType()) {
        invoke_args.push_back(operands[param_idx]);
      } else {
        // If we run out of coercion methods, just use the last one.
        // If there were no coercion methods, we set it to the default value
        // above and use it everywhere.
        if (coercion_method_itr != coercion_method_end) {
          coercion_method = *coercion_method_itr;
          ++coercion_method_itr;
        }
        invoke_args.push_back(
            helpers_.Fitted(p_, state().fb(), operands[param_idx],
                            coercion_method, param->GetType()));
      }
    } else {
      invoke_args.push_back(
          state().fb()->Literal(ZeroOfType(param->GetType())));
    }
    ++param_idx;
  }
  state().context_list().AppendElement(
      state().fb()->Invoke(invoke_args, to_invoke));
}

// Retrieves an operand from the combined context list based off of a list
// index.
BValue GenIrNodesPass::GetOperand(const OperandIdxProto& operand_idx) {
  // Retrieve an operand from the context list based off of the randomly
  // generated list index.
  return state().context_list().GetElementAt(operand_idx.list_idx());
}

// Retrieves an operand from the bits context list.
BValue GenIrNodesPass::GetBitsOperand(const BitsOperandIdxProto& operand_idx) {
  return state().context_list().GetElementAt(operand_idx.list_idx(),
                                             ContextListType::BITS_LIST);
}

// Retrieves an operand from the tuple context list.
BValue GenIrNodesPass::GetTupleOperand(
    const TupleOperandIdxProto& operand_idx) {
  return state().context_list().GetElementAt(operand_idx.list_idx(),
                                             ContextListType::TUPLE_LIST);
}

// Retrieves an operand from the array context list.
BValue GenIrNodesPass::GetArrayOperand(
    const ArrayOperandIdxProto& operand_idx) {
  return state().context_list().GetElementAt(operand_idx.list_idx(),
                                             ContextListType::ARRAY_LIST);
}

// Retrieves an operand from the combined context list and then coerces it to
// the specified type.
BValue GenIrNodesPass::GetCoercedOperand(const OperandIdxProto& operand_idx,
                                         const CoercedTypeProto& coerced_type) {
  BValue operand = GetOperand(operand_idx);
  Type* type = helpers_.ConvertTypeProtoToType(p_, coerced_type);
  return helpers_.Coerced(p_, state().fb(), operand, coerced_type, type);
}

// Retrieves an operand from the bits context list and then coerces it to the
// specified bits type.
BValue GenIrNodesPass::GetCoercedBitsOperand(
    const BitsOperandIdxProto& operand_idx,
    const BitsCoercedTypeProto& coerced_type) {
  BValue operand = GetBitsOperand(operand_idx);
  Type* type = helpers_.ConvertBitsTypeProtoToType(p_, coerced_type);
  return helpers_.CoercedBits(p_, state().fb(), operand, coerced_type, type);
}

// Same as GetCoercedOperand, but uses a CoercionMethodProto directly rather
// than a CoercedTypeProto. It coerces the operand by referencing a Type object
// rather than referencing a coerced type.
BValue GenIrNodesPass::GetFittedOperand(
    const OperandIdxProto& operand_idx,
    const CoercionMethodProto& coercion_method, Type* type) {
  BValue operand = GetOperand(operand_idx);
  return helpers_.Fitted(p_, state().fb(), operand, coercion_method, type);
}

// Same as GetCoercedBitsOperand, but uses a BitsCoercionMethodProto directly
// rather than a BitsCoercedTypeProto. It coerces the operand by referencing a
// Type object rather than referencing a coerced type.
BValue GenIrNodesPass::GetBitsFittedOperand(
    const BitsOperandIdxProto& operand_idx,
    const BitsCoercionMethodProto& coercion_method, int64_t bit_width) {
  BValue operand = GetBitsOperand(operand_idx);
  Type* type = p_->GetBitsType(bit_width);  // No change here.
  return helpers_.FittedBits(p_, state().fb(), operand, coercion_method, type);
}

// Retrieves multiple operands from the combined context list. If
// min_operand_count and max_operand_count are specified, the number of returned
// operands will be constrained to be between the specified minimum/maximum.
std::vector<BValue> GenIrNodesPass::GetOperands(
    const google::protobuf::RepeatedPtrField<OperandIdxProto>& operand_idxs,
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
    operands.push_back(GetOperand(operand_idxs.at(i)));
  }
  // Fill in any remaining operands with default values.
  BValue default_value = helpers_.DefaultValue(p_, state().fb());
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

std::vector<BValue> GenIrNodesPass::GetBitsOperands(
    const google::protobuf::RepeatedPtrField<BitsOperandIdxProto>& operand_idxs,
    int64_t min_operand_count, int64_t max_operand_count) {
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs.size();
  }
  std::vector<BValue> operands;
  for (int64_t i = 0; i < operand_idxs.size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetBitsOperand(operand_idxs.at(i)));
  }
  BValue default_value =
      helpers_.DefaultValue(p_, state().fb(), TypeCase::BITS_CASE);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

std::vector<BValue> GenIrNodesPass::GetArrayOperands(
    const google::protobuf::RepeatedPtrField<ArrayOperandIdxProto>& operand_idxs,
    int64_t min_operand_count, int64_t max_operand_count) {
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs.size();
  }
  std::vector<BValue> operands;
  for (int64_t i = 0; i < operand_idxs.size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetArrayOperand(operand_idxs.at(i)));
  }
  BValue default_value =
      helpers_.DefaultValue(p_, state().fb(), TypeCase::ARRAY_CASE);
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
  Type* type = helpers_.ConvertTypeProtoToType(p_, coerced_type);
  BValue default_value = helpers_.DefaultValueOfType(p_, state().fb(), type);
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
  Type* type = helpers_.ConvertBitsTypeProtoToType(p_, coerced_type);
  BValue default_value =
      helpers_.DefaultValueOfBitsType(p_, state().fb(), type);
  while (operands.size() < min_operand_count) {
    operands.push_back(default_value);
  }
  return operands;
}

GenIrNodesPass::FunctionState::~FunctionState() {
  if (fb_ == nullptr) {
    // We've been moved.
    return;
  }
  if (context_list_.IsEmpty()) {
    // BuildWithReturnValue requires a node to be a return value.
    CHECK_OK(fb_->BuildWithReturnValue(fb_->Tuple({})).status());
    return;
  }
  BValue combined_context_list =
      CombineContextList(combine_list_method_, fb_.get(), context_list_);
  CHECK_OK(fb_->BuildWithReturnValue(combined_context_list).status());
}

}  // namespace xls
